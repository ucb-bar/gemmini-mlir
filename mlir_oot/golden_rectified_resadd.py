"""Explicit exact sparse affine correction with ordinary primitive commands.

ABI A,B,C,tables: dense signed-i8 MxN inputs/output, M divisible by DIM and N
by four DIM tiles. C is a fresh private owner, disjoint from immutable A/B and
readonly tables. Its predictor bytes are written then reloaded before final
publication. No CPU correction, ACC execute read, STORE_SPAD or FSM is used.

This implements a completely rederived Merlin axis-offset certificate. Target
execution capability and complete producer/transfer cost require independent
qualification; the default residual family and routing are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from merlin.llvmlower.quantized_affine_rectifier import validate
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64

from .codegen.builder import PTR, FnBuilder
from .golden_gemm import GoldenGemm, Shape
from .tables import isa
from .tables import rtl_facts as F


@dataclass(frozen=True)
class Capabilities:
    spad_execute_read: bool
    spad_execute_write: bool
    real_d_ws: bool
    signed_i32_accumulator: bool
    narrow_scaled_acc_dma_store: bool
    four_tile_i8_acc_dma_store: bool

    def require(self):
        if any(type(value) is not bool or not value for value in vars(self).values()):
            raise ValueError(
                "explicit supported primitive capability contract required"
            )


def _chunks(value):
    quotient, remainder = divmod(value, 127)
    return [127] * quotient + ([remainder] if remainder else [])


@dataclass(frozen=True)
class Plan:
    m: int
    n: int
    certificate: dict
    capabilities: Capabilities

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
                "complete DIM row tiles and four-DIM dense column groups required; tails refuse"
            )
        if self.n >= 1 << 32 or self.m * self.n >= 1 << 63:
            raise ValueError("matrix extent or stride exceeds physical index fields")
        if F.DIM != 16 or F.OPERAND_DTYPE != "i8" or F.ACCUMULATOR_DTYPE != "i32":
            raise ValueError("signed-byte/i32 16-lane diagonal capability required")
        if self.certificate["indicator_family"] != "axis_offsets":
            raise ValueError("this target family accepts axis-offset rectifiers only")
        if len(self.certificate["relation"]) > 1:
            raise ValueError(
                "multiple correction relations are not implemented by this target pipeline"
            )
        if any(
            not 1 <= self.certificate["predictor"][key] <= 32767 for key in ("p", "q")
        ):
            raise ValueError("positive bounded coefficient decomposition required")
        for row in self.certificate["relation"]:
            if not -127 <= row["correction"] <= 127 or row["correction"] == 0:
                raise ValueError(
                    "correction must fit a nonzero diagonal signed-byte coefficient"
                )
            if any(not -128 <= offset["seed"] <= 127 for offset in row["offsets"]):
                raise ValueError(
                    "rectifier constant seed does not fit real-D signed-byte lane"
                )
        # Reserve full extents even though the final logical row may not read all
        # slots at once. These intervals are pairwise disjoint, including bank gaps.
        intervals = [(base, base + rows) for base, rows in self.spad_intervals.values()]
        if any(begin < 0 or end > F.SPAD_ROWS for begin, end in intervals):
            raise ValueError("rectifier SPAD reservation exceeds capacity")
        if any(
            a < d and c < b
            for i, (a, b) in enumerate(intervals)
            for c, d in intervals[i + 1 :]
        ):
            raise ValueError("rectifier live SPAD reservations overlap")
        if self.panel_rows > F.ACC_ROWS:
            raise ValueError("private predictor/final ACC panel exceeds capacity")
        if F.SPAD_BANKS < 4 or F.SPAD_BANK_ROWS * F.SPAD_BANKS != F.SPAD_ROWS:
            raise ValueError("four independent SPAD banks required by real-D layout")

    @property
    def panel_columns(self):
        return 4 * F.DIM

    @property
    def panel_rows(self):
        return self.panel_columns

    @property
    def relation(self):
        return self.certificate["relation"][0] if self.certificate["relation"] else None

    @property
    def weights(self):
        values = []
        predictor = self.certificate["predictor"]
        for value in _chunks(predictor["p"]) + _chunks(predictor["q"]):
            if value not in values:
                values.append(value)
        if self.relation:
            for value in (1, -1, self.relation["correction"]):
                if value not in values:
                    values.append(value)
        return values

    @property
    def seeds(self):
        return (
            [offset["seed"] for offset in self.relation["offsets"]] + [1]
            if self.relation
            else []
        )

    @property
    def spad_intervals(self):
        p = self.panel_rows
        bank = F.SPAD_BANK_ROWS
        return {
            "lhs": (0, p),
            "temporary": (p, p),
            "prediction": (2 * p, p),
            "rhs": (bank, p),
            "indicator0": (bank + p, p),
            "indicator1": (bank + 2 * p, p),
            "weights": (2 * bank, len(self.weights) * F.DIM),
            "seeds": (3 * bank, len(self.seeds) * p),
        }

    @property
    def compute_passes(self):
        p = self.certificate["predictor"]
        return len(_chunks(p["p"])) + len(_chunks(p["q"])) + (9 if self.relation else 0)

    def tables(self):
        data = bytearray()
        for value in self.weights:
            data.extend(
                (value if row == column else 0) & 255
                for row in range(F.DIM)
                for column in range(F.DIM)
            )
        for value in self.seeds:
            data.extend([value & 255] * self.panel_columns)
        return bytes(data)

    def attributes(self):
        return {
            "m": self.m,
            "n": self.n,
            "compute_passes_per_tile": self.compute_passes,
            "input_loads_per_panel": 2,
            "predictor_store_and_reload_per_panel": int(self.relation is not None),
            "final_stores_per_panel": 1,
            "spad_reservations": self.spad_intervals,
            "acc_reservation": [0, self.panel_rows],
            "weights": self.weights,
            "seeds": self.seeds,
            "source_table_sha256": self.certificate["source_table_sha256"],
            "predictor_table_sha256": self.certificate["predictor_table_sha256"],
            "numeric": "Original full-domain signed-byte observation, separately proved physical prediction",
            "ownership": "A/B/tables immutable; C fresh private disjoint owner, unpublished until completion",
            "physical_dram_bytes": "UNKNOWN; all stores/reloads remain in the complete measured scope",
        }


def build(plan: Plan, *, coalesce_internal_spad: bool = False, ordering_contract=None):
    # Revalidate mutable nested certificate contents at the emission boundary.
    plan.__post_init__()
    if type(coalesce_internal_spad) is not bool:
        raise ValueError("SPAD fence policy must be an explicit boolean")
    if coalesce_internal_spad and ordering_contract is None:
        raise ValueError("SPAD fence coalescing requires a pinned ordering contract")
    e = GoldenGemm(Shape(plan.m, plan.n, F.DIM, bn=4))
    e.fb = FnBuilder([PTR] * 4)
    a, b, c, tables = e.fb.entry.args
    place = plan.spad_intervals
    cols = plan.panel_columns
    rows = F.DIM
    weight_base = place["weights"][0]
    seed_base = place["seeds"][0]
    e._rocc("fence", {})
    e._rocc("flush", {})
    e._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY})
    for lid in (0, 1):
        e._rocc("config_ld", {"stride": plan.n, "load_id": lid})
    e._rocc("config_ld", {"stride": F.DIM, "load_id": 2})
    for index, _value in enumerate(plan.weights):
        pointer = e._ptr(tables, e.fb.const(0), 1, e.fb.const(index * F.DIM * F.DIM))
        e._rocc(
            "mvin",
            {
                "local": weight_base + index * F.DIM,
                "rows": rows,
                "cols": rows,
                "load_id": 2,
            },
            pointer,
        )
    if plan.seeds:
        e._rocc("config_ld", {"stride": 0, "load_id": 2})
        start = len(plan.weights) * F.DIM * F.DIM
        for index, _value in enumerate(plan.seeds):
            pointer = e._ptr(tables, e.fb.const(0), 1, e.fb.const(start + index * cols))
            e._rocc(
                "mvin",
                {
                    "local": seed_base + index * cols,
                    "rows": rows,
                    "cols": cols,
                    "load_id": 2,
                },
                pointer,
            )
    e._rocc("fence", {})

    def product(source, coefficient, destination, *, real_d=None, accumulate=False):
        weight = weight_base + plan.weights.index(coefficient) * F.DIM
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

    def m_body(mrow):
        def n_body(ncol):
            for lid, pointer, local in (
                (0, a, place["lhs"][0]),
                (1, b, place["rhs"][0]),
            ):
                e._rocc(
                    "mvin",
                    {"local": local, "rows": rows, "cols": cols, "load_id": lid},
                    e._ptr(pointer, mrow, plan.n, ncol),
                )
            predictor = plan.certificate["predictor"]
            first = True
            e._rocc(
                "config_ex",
                {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.NO_ACTIVATION},
            )
            for source, value in (
                (place["lhs"][0], predictor["p"]),
                (place["rhs"][0], predictor["q"]),
            ):
                for chunk in _chunks(value):
                    product(source, chunk, -1, accumulate=not first)
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
            destination = e._ptr(c, mrow, plan.n, ncol)
            e._rocc(
                "mvout",
                {"local": isa.acc_addr(0), "rows": rows, "cols": cols},
                destination,
            )
            e._rocc("fence", {})
            if plan.relation:
                e._rocc(
                    "mvin",
                    {
                        "local": place["prediction"][0],
                        "rows": rows,
                        "cols": cols,
                        "load_id": 0,
                    },
                    destination,
                )
                e._rocc(
                    "config_ex", {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.RELU}
                )
                previous = None
                for index, offset in enumerate(plan.relation["offsets"]):
                    product(
                        place[offset["axis"]][0],
                        offset["coefficient"],
                        place["temporary"][0],
                        real_d=seed_base + index * cols,
                    )
                    e._rocc(
                        "fence",
                        {"internal_spad_stage": 1} if coalesce_internal_spad else {},
                    )
                    current = place["indicator" + str(index % 2)][0]
                    product(
                        place["temporary"][0],
                        -1,
                        current,
                        real_d=seed_base + 4 * cols if previous is None else previous,
                    )
                    e._rocc(
                        "fence",
                        {"internal_spad_stage": 1} if coalesce_internal_spad else {},
                    )
                    previous = current
                e._rocc(
                    "config_ex",
                    {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.NO_ACTIVATION},
                )
                product(
                    previous,
                    plan.relation["correction"],
                    -1,
                    real_d=place["prediction"][0],
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
                    {"local": isa.acc_addr(0), "rows": rows, "cols": cols},
                    destination,
                )
                e._rocc("fence", {})

        e.fb.for_loop(0, plan.n, cols, n_body)

    e.fb.for_loop(0, plan.m, rows, m_body)
    e._rocc("fence", {})
    function = llvm.FuncOp(
        "gemmini_golden_rectified_resadd",
        llvm.LLVMFunctionType([PTR] * 4),
        linkage=llvm.LinkageAttr("external"),
        body=e.fb.finish(),
    )
    result = ModuleOp([function])
    result.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
    result.attributes["gemmini.rectified_residual_plan"] = StringAttr(
        json.dumps(plan.attributes(), sort_keys=True)
    )
    result.verify()
    if coalesce_internal_spad:
        from .spad_fence_coalescing import coalesce

        proof = coalesce(result, ordering_contract, plan.spad_intervals)
        result.attributes["gemmini.spad_fence_coalescing"] = StringAttr(
            json.dumps(proof, sort_keys=True)
        )
    return result
