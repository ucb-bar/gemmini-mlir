"""Exact correction through a reused predictor ACC integer observation.

ABI A,B,C,tables,negative_scratch. Both writable owners are disjoint from each
other and all inputs. A/B/tables remain immutable; read-only inputs may alias.
C and negative_scratch are private until final completion.
Only identity-scale i32 ACC DMA RMW is used; no ACC execute read or FSM.
This family is explicit and does not change normal residual selection.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

from merlin.llvmlower.quantized_affine_rectifier import validate
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64

from .codegen.builder import PTR, FnBuilder
from .golden_gemm import GoldenGemm, Shape
from .golden_rectified_resadd import Capabilities as BasicCapabilities
from .golden_rectified_resadd import _chunks
from .rectifier_panel_batch import require_completion_sources
from .spad_fence_coalescing import OrderingContract, coalesce
from .tables import isa
from .tables import rtl_facts as F


@dataclass(frozen=True)
class Capabilities:
    primitive: BasicCapabilities
    identity_i32_acc_dma_accumulate: bool

    def require(self):
        if not isinstance(self.primitive, BasicCapabilities):
            raise ValueError("explicit primitive capability contract required")  # noqa: TRY004
        self.primitive.require()
        if self.identity_i32_acc_dma_accumulate is not True:
            raise ValueError(
                "identity-scale signed-i32 ACC DMA RMW capability required"
            )


@dataclass(frozen=True)
class Plan:
    m: int
    n: int
    certificate: dict
    capabilities: Capabilities
    panel_batch: int = 4
    max_key_fibres: int = 1

    def __post_init__(self):
        validate(self.certificate)
        self.capabilities.require()
        if (
            type(self.m) is not int
            or type(self.n) is not int
            or self.m <= 0
            or self.n <= 0
            or self.m % F.DIM
            or self.n % (4 * F.DIM)
        ):
            raise ValueError(
                "complete DIM row tiles and four-DIM column groups required"
            )
        if type(self.panel_batch) is not int or self.panel_batch not in (1, 4):
            raise ValueError("explicit bounded panel batch1 or4 required")
        if self.n >= 1 << 32 or self.m * self.n >= 1 << 63:
            raise ValueError("matrix extent or stride exceeds physical index fields")
        if F.DIM != 16 or F.OPERAND_DTYPE != "i8" or F.ACCUMULATOR_DTYPE != "i32":
            raise ValueError("signed-byte/i32 DIM16 primitive capability required")
        if self.certificate["indicator_family"] != "predictor_key":
            raise ValueError("complete predictor-key fibre certificate required")
        if (
            type(self.max_key_fibres) is int
            and self.max_key_fibres == 1
            and len(self.certificate["relation"]) != 1
        ):
            raise ValueError("exactly one complete correction-key fibre supported")
        if (
            type(self.max_key_fibres) is not int
            or self.max_key_fibres not in (1, 2)
            or not 1 <= len(self.certificate["relation"]) <= self.max_key_fibres
        ):
            raise ValueError(
                "explicit bounded complete correction-key fibre count required"
            )
        if any(
            not 1 <= self.certificate["predictor"][key] <= 32767 for key in ("p", "q")
        ):
            raise ValueError(
                "bounded positive diagonal coefficient decomposition required"
            )
        relation = self.relation
        if not -127 <= relation["correction"] <= 127 or relation["correction"] == 0:
            raise ValueError("nonzero signed-byte correction coefficient required")
        if max(map(abs, relation["key_range"])) >= 1 << 24:
            raise ValueError(
                "key conversion and unit scaled stores require exact binary32 integer range"
            )
        # Identity ACC DMA is not a universal full-i32 numeric guarantee. Bind
        # the actual seed to the binary32 exact-integer domain even when a
        # provider implements unit scaling through an integer/float conversion.
        if abs(relation["key"]["seed"]) >= 1 << 24:
            raise ValueError(
                "identity ACC DMA seed requires exact binary32 representation"
            )
        if max(map(abs, relation["predictor_integer_range"])) >= 1 << 24:
            raise ValueError(
                "predictor ACC integer conversion requires exact binary32 representation"
            )
        if any(
            relation["key"][name] != self.certificate["predictor"][field]
            for name, field in (("lhs_coefficient", "p"), ("rhs_coefficient", "q"))
        ):
            raise ValueError("key must reuse the actual predictor integer coefficients")
        for index, key_relation in enumerate(self.certificate["relation"]):
            if (
                not -127 <= key_relation["correction"] <= 127
                or key_relation["correction"] == 0
                or max(map(abs, key_relation["key_range"])) >= 1 << 24
                or max(map(abs, key_relation["predictor_integer_range"])) >= 1 << 24
                or abs(key_relation["key"]["seed"]) >= 1 << 24
                or abs(self.seed_delta(index)) >= 1 << 24
            ):
                raise ValueError(
                    "every key/seed delta requires exact binary32 integer bounds"
                )
            if any(
                key_relation["key"][name] != self.certificate["predictor"][field]
                for name, field in (("lhs_coefficient", "p"), ("rhs_coefficient", "q"))
            ):
                raise ValueError(
                    "every key must reuse the actual predictor coefficients"
                )
        intervals = [(base, base + rows) for base, rows in self.spad_intervals.values()]
        if any(begin < 0 or end > F.SPAD_ROWS for begin, end in intervals):
            raise ValueError("key rectifier SPAD reservation exceeds capacity")
        if any(
            a < d and c < b
            for i, (a, b) in enumerate(intervals)
            for c, d in intervals[i + 1 :]
        ):
            raise ValueError("key rectifier live SPAD extents overlap")
        if self.panel_batch * self.panel_rows > F.ACC_ROWS:
            raise ValueError("key rectifier ACC reservation exceeds capacity")
        if F.SPAD_BANKS < 4 or F.SPAD_BANK_ROWS * F.SPAD_BANKS != F.SPAD_ROWS:
            raise ValueError("four independent SPAD banks required")

    @property
    def panel_columns(self):
        return 4 * F.DIM

    @property
    def panel_rows(self):
        return self.panel_columns

    @property
    def relation(self):
        return self.certificate["relation"][0]

    @property
    def weights(self):
        predictor = self.certificate["predictor"]
        return list(
            dict.fromkeys(
                _chunks(predictor["p"])
                + _chunks(predictor["q"])
                + [1, -1]
                + [relation["correction"] for relation in self.certificate["relation"]]
            )
        )

    @property
    def spad_intervals(self):
        extent, bank = self.panel_batch * self.panel_rows, F.SPAD_BANK_ROWS
        fibres = len(self.certificate["relation"])
        if fibres != 1:
            reservations = {
                "lhs_positive": (0, extent),
                "prediction": (fibres * extent, extent),
                "rhs_negative": (bank, extent),
                "indicator0": (bank + fibres * extent, extent),
                "negative_rectified": (bank + (2 * fibres + 1) * extent, extent),
                "weights": (2 * bank, len(self.weights) * F.DIM),
                "ones": (3 * bank, self.panel_rows),
                "zero": (3 * bank + self.panel_rows, self.panel_rows),
            }
            for index in range(fibres):
                if index:
                    reservations[f"positive_key_{index}"] = (index * extent, extent)
                    reservations[f"negative_key_{index}"] = (
                        bank + index * extent,
                        extent,
                    )
                reservations[f"indicator_key_{index}"] = (
                    bank + (fibres + 1 + index) * extent,
                    extent,
                )
            return reservations
        return {
            "lhs_positive": (0, extent),
            "prediction": (extent, extent),
            "rhs_negative": (bank, extent),
            "indicator0": (bank + extent, extent),
            "indicator1": (bank + 2 * extent, extent),
            "negative_rectified": (bank + 3 * extent, extent),
            "weights": (2 * bank, len(self.weights) * F.DIM),
            "ones": (3 * bank, self.panel_rows),
            "zero": (3 * bank + self.panel_rows, self.panel_rows),
        }

    @property
    def scratch_bytes(self):
        return self.panel_batch * F.DIM * self.panel_columns

    @property
    def compute_passes(self):
        predictor = self.certificate["predictor"]
        return (
            len(_chunks(predictor["p"]))
            + len(_chunks(predictor["q"]))
            + 4 * len(self.certificate["relation"])
        )

    def seed_delta(self, index):
        seed = self.certificate["relation"][index]["key"]["seed"]
        return (
            seed
            if index == 0
            else seed - self.certificate["relation"][index - 1]["key"]["seed"]
        )

    @property
    def acc_seed_offset(self):
        return len(self.weights) * F.DIM * F.DIM + 2 * self.panel_columns

    def tables(self):
        data = bytearray()
        for value in self.weights:
            data.extend(
                (value if row == col else 0) & 255
                for row in range(F.DIM)
                for col in range(F.DIM)
            )
        data.extend([1] * self.panel_columns)
        data.extend([0] * self.panel_columns)
        for index in range(len(self.certificate["relation"])):
            data.extend(struct.pack("<i", self.seed_delta(index)) * F.DIM)
        return bytes(data)

    def attributes(self):
        count = self.m * self.n
        panels = count // (F.DIM * self.panel_columns)
        attributes = {
            "m": self.m,
            "n": self.n,
            "panel_batch": self.panel_batch,
            "compute_passes_per_tile": self.compute_passes,
            "spad_reservations": self.spad_intervals,
            "acc_reservation": [0, self.panel_batch * self.panel_rows],
            "negative_scratch_bytes": self.scratch_bytes,
            "identity_ACC_seed": self.relation["key"]["seed"],
            "identity_ACC_seed_conversion": "literal unit-scale seed is exactly representable in binary32; no universal full-i32 DMA scaling claim",
            "source_table_sha256": self.certificate["source_table_sha256"],
            "predictor_table_sha256": self.certificate["predictor_table_sha256"],
            "command_logical_dma_bytes_excluding_setup": (
                5 + 4 * len(self.certificate["relation"])
            )
            * count
            + panels * 4 * F.DIM * F.DIM * 4 * len(self.certificate["relation"]),
            "source_row_payload_bytes_excluding_setup": (
                5 + 4 * len(self.certificate["relation"])
            )
            * count
            + panels * 4 * F.DIM * 4 * len(self.certificate["relation"]),
            "stride0_payload_scope": "Seed command contains16repeatedrows; distinct source row64B pertile. Actual physical DMA/transfers require engine observation",
            "traffic_description": "A/B plus prediction/positive/negative loads; identityi32 seed rows; prediction/positive/negative/final stores",
            "ownership": "A/B/tables readonly; C/private scratch disjoint; prediction retained in SPAD before C overwritten; final fence before publication",
            "store_ordering": "Positive key store ReLU precedes scale1; negative key store has NO_ACT/scale-1 then exact SPAD unit ReLU; both readbacks complete before reload and no ACC overwrite between passes",
            "physical_dram_bytes": "UNKNOWN: requested payload is not physical traffic",
            "performance": "UNKNOWN: additional DMA/config/fences included in complete measured scope",
        }
        if len(self.certificate["relation"]) != 1:
            attributes.update(
                correction_key_fibres=len(self.certificate["relation"]),
                identity_ACC_seed_deltas=[
                    self.seed_delta(index)
                    for index in range(len(self.certificate["relation"]))
                ],
                ACC_lifetime="predictor retained through every key readback; indicators overwrite ACC only after all key observations reside in SPAD",
                DDR_reuse="every key readback/reload completes before C or private scratch is reused for the next key",
                traffic_description="A/B plus prediction load/store, positive/negative load/store per fibre, identity i32 seed delta per fibre and final store",
            )
        return attributes


def build(plan: Plan, *, ordering_contract: OrderingContract):
    plan.__post_init__()
    # Controller identity grants no schedule or numeric capability. This family
    # independently checks its full live extents and complete finite circuit.
    pins = require_completion_sources(ordering_contract)
    place = plan.spad_intervals
    e = GoldenGemm(Shape(plan.m, plan.n, F.DIM, bn=4))
    e.fb = FnBuilder([PTR] * 5)
    a, b, c, tables, negative = e.fb.entry.args
    rows, cols = F.DIM, plan.panel_columns
    weight_base, ones_base = place["weights"][0], place["ones"][0]

    def local(name, slot):
        return place[name][0] + slot * plan.panel_rows

    def product(source, coefficient, destination, *, real_d=None, accumulate=False):
        weight = weight_base + plan.weights.index(coefficient) * rows
        for tile in range(4):
            output = (
                isa.acc_addr((-destination - 1) + tile * rows, accumulate=accumulate)
                if destination < 0
                else destination + tile * rows
            )
            e._rocc(
                "preload",
                {
                    "bd": weight if tile == 0 else isa.GARBAGE_ADDR,
                    "c": output,
                    "bd_cols": rows,
                    "bd_rows": rows,
                    "c_cols": rows,
                    "c_rows": rows,
                },
            )
            attrs = {
                "a": source + tile * rows,
                "a_cols": rows,
                "a_rows": rows,
                "accumulate": tile != 0,
            }
            if real_d is not None:
                attrs.update(bd=real_d + tile * rows, bd_cols=rows, bd_rows=rows)
            e._rocc("compute", attrs)

    e._rocc("fence", {})
    e._rocc("flush", {})
    e._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY})
    e._rocc("config_ld", {"stride": rows, "load_id": 2})
    for index, _value in enumerate(plan.weights):
        e._rocc(
            "mvin",
            {
                "local": weight_base + index * rows,
                "rows": rows,
                "cols": rows,
                "load_id": 2,
            },
            e._ptr(tables, e.fb.const(0), 1, e.fb.const(index * rows * rows)),
        )
    e._rocc("config_ld", {"stride": 0, "load_id": 2})
    e._rocc(
        "mvin",
        {"local": ones_base, "rows": rows, "cols": cols, "load_id": 2},
        e._ptr(tables, e.fb.const(0), 1, e.fb.const(len(plan.weights) * rows * rows)),
    )
    e._rocc(
        "mvin",
        {"local": place["zero"][0], "rows": rows, "cols": cols, "load_id": 2},
        e._ptr(
            tables, e.fb.const(0), 1, e.fb.const(len(plan.weights) * rows * rows + cols)
        ),
    )
    e._rocc("fence", {})
    e._rocc("config_ld", {"stride": 0, "load_id": 2, "shrunk": 0, "scale": 1.0})

    def multiple_keys(destinations):
        # Preserve the predictor ACC throughout all key observations. Real-D
        # indicator products may overwrite it only after every key is in SPAD.
        # Each DDR readback is completed before the same C/scratch locations
        # are reused for another key; RS local-range hazards alone cannot
        # establish this external DDR alias dependency.
        e._rocc("fence", {})
        for slot, destination in enumerate(destinations):
            e._rocc(
                "mvin",
                {
                    "local": local("prediction", slot),
                    "rows": rows,
                    "cols": cols,
                    "load_id": 0,
                },
                destination,
            )
        for index, _relation in enumerate(plan.certificate["relation"]):
            seed = e._ptr(
                tables,
                e.fb.const(0),
                1,
                e.fb.const(plan.acc_seed_offset + index * rows * 4),
            )
            for slot in range(len(destinations)):
                for tile in range(4):
                    e._rocc(
                        "mvin",
                        {
                            "local": isa.acc_addr(
                                slot * cols + tile * rows, accumulate=True
                            ),
                            "rows": rows,
                            "cols": rows,
                            "load_id": 2,
                        },
                        seed,
                    )
            for scale, stride, output, activation in (
                (1.0, plan.n, c, isa.RELU),
                (-1.0, cols, negative, isa.NO_ACTIVATION),
            ):
                e._rocc(
                    "config_st",
                    {"stride": stride, "acc_act": activation, "acc_scale": scale},
                )
                for slot, destination in enumerate(destinations):
                    if output is negative:
                        destination = e._ptr(
                            negative, e.fb.const(slot * rows), cols, e.fb.const(0)
                        )
                    e._rocc(
                        "mvout",
                        {
                            "local": isa.acc_addr(slot * cols),
                            "rows": rows,
                            "cols": cols,
                        },
                        destination,
                    )
            e._rocc("fence", {})
            e._rocc("config_ld", {"stride": cols, "load_id": 1})
            for slot, destination in enumerate(destinations):
                positive_name = (
                    "lhs_positive" if index == 0 else f"positive_key_{index}"
                )
                negative_name = (
                    "rhs_negative" if index == 0 else f"negative_key_{index}"
                )
                e._rocc(
                    "mvin",
                    {
                        "local": local(positive_name, slot),
                        "rows": rows,
                        "cols": cols,
                        "load_id": 0,
                    },
                    destination,
                )
                e._rocc(
                    "mvin",
                    {
                        "local": local(negative_name, slot),
                        "rows": rows,
                        "cols": cols,
                        "load_id": 1,
                    },
                    e._ptr(negative, e.fb.const(slot * rows), cols, e.fb.const(0)),
                )
            e._rocc("fence", {})
        for slot, destination in enumerate(destinations):
            e._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.RELU})
            for index, _relation in enumerate(plan.certificate["relation"]):
                positive_name = (
                    "lhs_positive" if index == 0 else f"positive_key_{index}"
                )
                negative_name = (
                    "rhs_negative" if index == 0 else f"negative_key_{index}"
                )
                product(
                    local(negative_name, slot),
                    1,
                    local("negative_rectified", slot),
                    real_d=place["zero"][0],
                )
                e._rocc("fence", {"internal_spad_stage": 1})
                product(
                    local(positive_name, slot),
                    -1,
                    local("indicator0", slot),
                    real_d=ones_base,
                )
                e._rocc("fence", {"internal_spad_stage": 1})
                product(
                    local("negative_rectified", slot),
                    -1,
                    local(f"indicator_key_{index}", slot),
                    real_d=local("indicator0", slot),
                )
                e._rocc("fence", {"internal_spad_stage": 1})
            e._rocc(
                "config_ex",
                {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.NO_ACTIVATION},
            )
            for index, relation in enumerate(plan.certificate["relation"]):
                product(
                    local(f"indicator_key_{index}", slot),
                    relation["correction"],
                    -(slot * cols + 1),
                    real_d=local("prediction", slot) if index == 0 else None,
                    accumulate=index != 0,
                )
            e._rocc(
                "config_st",
                {
                    "stride": plan.n,
                    "acc_act": isa.RELU
                    if plan.certificate["source"]["relu"]
                    else isa.NO_ACTIVATION,
                    "acc_scale": 1.0,
                },
            )
            e._rocc(
                "mvout",
                {"local": isa.acc_addr(slot * cols), "rows": rows, "cols": cols},
                destination,
            )
        e._rocc("fence", {})

    def batch(mrow, count):
        def columns(ncol):
            e._rocc("config_ld", {"stride": plan.n, "load_id": 0})
            e._rocc("config_ld", {"stride": plan.n, "load_id": 1})
            e._rocc(
                "config_ex",
                {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.NO_ACTIVATION},
            )
            destinations = []
            for slot in range(count):
                row = mrow if slot == 0 else e.fb.add_i(mrow, e.fb.const(slot * rows))
                destination = e._ptr(c, row, plan.n, ncol)
                destinations.append(destination)
                for lid, pointer, name in (
                    (0, a, "lhs_positive"),
                    (1, b, "rhs_negative"),
                ):
                    e._rocc(
                        "mvin",
                        {
                            "local": local(name, slot),
                            "rows": rows,
                            "cols": cols,
                            "load_id": lid,
                        },
                        e._ptr(pointer, row, plan.n, ncol),
                    )
                first = True
                predictor = plan.certificate["predictor"]
                for name, value in (
                    ("lhs_positive", predictor["p"]),
                    ("rhs_negative", predictor["q"]),
                ):
                    for chunk in _chunks(value):
                        product(
                            local(name, slot),
                            chunk,
                            -(slot * cols + 1),
                            accumulate=not first,
                        )
                        first = False
                e._rocc(
                    "config_st",
                    {
                        "stride": plan.n,
                        "acc_act": isa.RELU
                        if plan.certificate["source"]["relu"]
                        else isa.NO_ACTIVATION,
                        "acc_scale": predictor["scale"],
                    },
                )
                e._rocc(
                    "mvout",
                    {"local": isa.acc_addr(slot * cols), "rows": rows, "cols": cols},
                    destination,
                )
            if len(plan.certificate["relation"]) != 1:
                multiple_keys(destinations)
                return
            e._rocc("fence", {})
            for slot, destination in enumerate(destinations):
                e._rocc(
                    "mvin",
                    {
                        "local": local("prediction", slot),
                        "rows": rows,
                        "cols": cols,
                        "load_id": 0,
                    },
                    destination,
                )
                seed = e._ptr(
                    tables, e.fb.const(0), 1, e.fb.const(plan.acc_seed_offset)
                )
                for tile in range(4):
                    e._rocc(
                        "mvin",
                        {
                            "local": isa.acc_addr(
                                slot * cols + tile * rows, accumulate=True
                            ),
                            "rows": rows,
                            "cols": rows,
                            "load_id": 2,
                        },
                        seed,
                    )
            # Config/store commands are ordered in the store queue. All key ACC
            # slots stay live until both passes and the retained DDR fence finish.
            for scale, stride, output, activation in (
                (1.0, plan.n, c, isa.RELU),
                (-1.0, cols, negative, isa.NO_ACTIVATION),
            ):
                e._rocc(
                    "config_st",
                    {"stride": stride, "acc_act": activation, "acc_scale": scale},
                )
                for slot, destination in enumerate(destinations):
                    if output is negative:
                        destination = e._ptr(
                            negative, e.fb.const(slot * rows), cols, e.fb.const(0)
                        )
                    e._rocc(
                        "mvout",
                        {
                            "local": isa.acc_addr(slot * cols),
                            "rows": rows,
                            "cols": cols,
                        },
                        destination,
                    )
            e._rocc("fence", {})
            e._rocc("config_ld", {"stride": cols, "load_id": 1})
            for slot, destination in enumerate(destinations):
                e._rocc(
                    "mvin",
                    {
                        "local": local("lhs_positive", slot),
                        "rows": rows,
                        "cols": cols,
                        "load_id": 0,
                    },
                    destination,
                )
                e._rocc(
                    "mvin",
                    {
                        "local": local("rhs_negative", slot),
                        "rows": rows,
                        "cols": cols,
                        "load_id": 1,
                    },
                    e._ptr(negative, e.fb.const(slot * rows), cols, e.fb.const(0)),
                )
                e._rocc(
                    "config_ex", {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.RELU}
                )
                product(
                    local("rhs_negative", slot),
                    1,
                    local("negative_rectified", slot),
                    real_d=place["zero"][0],
                )
                e._rocc("fence", {"internal_spad_stage": 1})
                product(
                    local("lhs_positive", slot),
                    -1,
                    local("indicator0", slot),
                    real_d=ones_base,
                )
                e._rocc("fence", {"internal_spad_stage": 1})
                product(
                    local("negative_rectified", slot),
                    -1,
                    local("indicator1", slot),
                    real_d=local("indicator0", slot),
                )
                e._rocc("fence", {"internal_spad_stage": 1})
                e._rocc(
                    "config_ex",
                    {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.NO_ACTIVATION},
                )
                product(
                    local("indicator1", slot),
                    plan.relation["correction"],
                    -(slot * cols + 1),
                    real_d=local("prediction", slot),
                )
                e._rocc(
                    "config_st",
                    {
                        "stride": plan.n,
                        "acc_act": isa.RELU
                        if plan.certificate["source"]["relu"]
                        else isa.NO_ACTIVATION,
                        "acc_scale": 1.0,
                    },
                )
                e._rocc(
                    "mvout",
                    {"local": isa.acc_addr(slot * cols), "rows": rows, "cols": cols},
                    destination,
                )
            e._rocc("fence", {})

        e.fb.for_loop(0, plan.n, cols, columns)

    extent = plan.panel_batch * rows
    end = plan.m // extent * extent
    if end:
        e.fb.for_loop(0, end, extent, lambda row: batch(row, plan.panel_batch))
    remaining = (plan.m - end) // rows
    if remaining:
        batch(e.fb.const(end), remaining)
    e._rocc("fence", {})
    module = ModuleOp(
        [
            llvm.FuncOp(
                "gemmini_golden_key_rectified_resadd",
                llvm.LLVMFunctionType([PTR] * 5),
                linkage=llvm.LinkageAttr("external"),
                body=e.fb.finish(),
            )
        ]
    )
    module.attributes["gemmini.dim"] = IntegerAttr(rows, i64)
    module.attributes["gemmini.key_rectified_plan"] = StringAttr(
        json.dumps(plan.attributes(), sort_keys=True)
    )
    module.verify()
    proof = coalesce(module, ordering_contract, place)
    module.attributes["gemmini.spad_fence_coalescing"] = StringAttr(
        json.dumps(proof, sort_keys=True)
    )
    module.attributes["gemmini.key_ordering_sources"] = StringAttr(
        json.dumps(pins, sort_keys=True)
    )
    return module
