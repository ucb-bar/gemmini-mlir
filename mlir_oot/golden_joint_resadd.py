"""Two explicit diagonal affine readouts sharing exact resident input panels.

ABI A,B,C0,C1,coefficients. Both Mx64 signed-i8 predictions are independent
outputs, not an exact source result until the shared complete joint certificate
and portable decoder are bound by the caller. Inputs and all output storage
must be distinct and alive through both readouts and subsequent decoding.
"""

import math
from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import DictionaryAttr, IntegerAttr, ModuleOp, StringAttr, i64

from .codegen.builder import PTR, FnBuilder
from .golden_gemm import GoldenGemm, Shape
from .golden_wide_resadd import ResidualPrefetchPlan
from .golden_wide_resadd import tables as single_tables
from .tables import isa
from .tables import rtl_facts as F


@dataclass(frozen=True)
class JointResidualPlan:
    m: int
    coefficients: tuple[tuple[int, int], tuple[int, int]]

    def __post_init__(self):
        if not isinstance(self.coefficients, tuple) or len(self.coefficients) != 2:
            raise ValueError("two explicit coefficient pairs required")
        for pair in self.coefficients:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    "two explicit integer coefficients per prediction required"
                )
            ResidualPrefetchPlan(self.m, *pair, True)
        if 5 * F.DIM > F.SPAD_BANK_ROWS or 2 * self.panel_rows > F.ACC_BANK_ROWS:
            raise ValueError(
                "joint coefficients or disjoint readout slots exceed resources"
            )

    @property
    def panel_rows(self):
        return 4 * F.DIM

    @property
    def weight_base(self):
        return F.SPAD_BANK_ROWS

    def operand_base(self, slot, operand):
        if slot not in (0, 1) or operand not in (0, 1):
            raise ValueError("bounded input slot/operand required")
        return slot * 2 * F.SPAD_BANK_ROWS + operand * self.panel_rows

    def accumulator_base(self, prediction, slot):
        if prediction not in (0, 1) or slot not in (0, 1):
            raise ValueError("bounded prediction/panel slot required")
        return prediction * F.ACC_BANK_ROWS + slot * self.panel_rows

    def attributes(self):
        values = {
            "operand_slot_rows": 2 * self.panel_rows,
            "weight_base": self.weight_base,
            "weight_rows": 5 * F.DIM,
            "accumulator_reserved_rows": 4 * self.panel_rows,
            "panel_count": self.m // F.DIM,
        }
        for prediction, pair in enumerate(self.coefficients):
            values[f"chunks{prediction}_per_panel"] = sum(
                (value + 126) // 127 for value in pair
            )
            for slot in (0, 1):
                values[f"accumulator{prediction}_slot{slot}"] = self.accumulator_base(
                    prediction, slot
                )
        return DictionaryAttr(
            {key: IntegerAttr(value, i64) for key, value in values.items()}
        )


def tables(predictors):
    """One shared127 diagonal plus each predictor's two remainder diagonals."""
    pairs = tuple((row["p"], row["q"]) for row in predictors)
    JointResidualPlan(F.DIM, pairs)
    first, second = [single_tables(*pair) for pair in pairs]
    return first + second[F.DIM * F.DIM :]


def build(m, predictors):
    """Emit a fused pair only; caller retains full-domain proof/decoder obligations."""
    if not isinstance(predictors, (list, tuple)) or len(predictors) != 2:
        raise ValueError("two explicit predictor contracts required")
    contracts = []
    for supplied in predictors:
        if not isinstance(supplied, dict) or set(supplied) != {
            "p",
            "q",
            "scale",
            "relu",
        }:
            raise ValueError("explicit coefficient/scale/ReLU contracts required")
        if type(supplied["relu"]) is not bool:
            raise ValueError("explicit boolean ReLU contract required")
        if not math.isfinite(supplied["scale"]) or supplied["scale"] <= 0:
            raise ValueError("positive finite readout scale required")
        contracts.append(dict(supplied))
    plan = JointResidualPlan(m, tuple((row["p"], row["q"]) for row in contracts))
    e = GoldenGemm(Shape(m, plan.panel_rows, F.DIM, bn=4))
    e.fb = FnBuilder([PTR] * 5)
    a, b, c0, c1, coeff = e.fb.entry.args
    e._rocc("fence", {})
    e._rocc("flush", {})
    e._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY})
    for lid in (0, 1):
        e._rocc("config_ld", {"stride": plan.panel_rows, "load_id": lid})
    e._rocc("config_ld", {"stride": F.DIM, "load_id": 2})
    for index in range(5):
        ptr = e._ptr(coeff, e.fb.const(0), 1, e.fb.const(index * F.DIM * F.DIM))
        e._rocc(
            "mvin",
            {
                "local": plan.weight_base + index * F.DIM,
                "rows": F.DIM,
                "cols": F.DIM,
                "load_id": 2,
            },
            ptr,
        )

    def load_panel(row, slot):
        for operand, ptr in enumerate((a, b)):
            e._rocc(
                "mvin",
                {
                    "local": plan.operand_base(slot, operand),
                    "rows": F.DIM,
                    "cols": plan.panel_rows,
                    "load_id": operand,
                },
                e._ptr(ptr, row, plan.panel_rows, e.fb.const(0)),
            )

    def compute_panel(row, slot):
        for prediction, (contract, output) in enumerate(
            zip(contracts, (c0, c1), strict=True)
        ):
            acc_base, first = plan.accumulator_base(prediction, slot), True
            for operand, value in enumerate((contract["p"], contract["q"])):
                full, remainder = divmod(value, 127)
                base = plan.operand_base(slot, operand)

                def chunk(
                    weight, load_weight, initialize, acc_base=acc_base, base=base
                ):
                    for d in range(plan.panel_rows // F.DIM):
                        e._rocc(
                            "preload",
                            {
                                "bd": weight
                                if load_weight and d == 0
                                else isa.GARBAGE_ADDR,
                                "c": isa.acc_addr(
                                    acc_base + d * F.DIM, accumulate=not initialize
                                ),
                                "bd_cols": F.DIM,
                                "bd_rows": F.DIM,
                                "c_cols": F.DIM,
                                "c_rows": F.DIM,
                            },
                        )
                        e._rocc(
                            "compute",
                            {
                                "a": base + d * F.DIM,
                                "a_cols": F.DIM,
                                "a_rows": F.DIM,
                                "accumulate": not (load_weight and d == 0),
                            },
                        )

                if full:
                    chunk(plan.weight_base, True, first)
                    first = False
                    if full > 1:
                        e.fb.for_loop(
                            1,
                            full,
                            1,
                            lambda unused: chunk(plan.weight_base, False, False),
                        )
                if remainder:
                    chunk(
                        plan.weight_base + (1 + 2 * prediction + operand) * F.DIM,
                        True,
                        first,
                    )
                    first = False
            e._rocc(
                "config_st",
                {
                    "stride": plan.panel_rows,
                    "acc_act": isa.RELU if contract["relu"] else isa.NO_ACTIVATION,
                    "acc_scale": contract["scale"],
                },
            )
            e._rocc(
                "mvout",
                {
                    "local": isa.acc_addr(acc_base),
                    "rows": F.DIM,
                    "cols": plan.panel_rows,
                },
                e._ptr(output, row, plan.panel_rows, e.fb.const(0)),
            )

    load_panel(e.fb.const(0), 0)
    paired_rows = ((m // F.DIM - 1) // 2) * 2 * F.DIM

    def pair(row):
        following = e.fb.add_i(row, e.fb.const(F.DIM))
        load_panel(following, 1)
        compute_panel(row, 0)
        load_panel(e.fb.add_i(row, e.fb.const(2 * F.DIM)), 0)
        compute_panel(following, 1)

    e.fb.for_loop(0, paired_rows, 2 * F.DIM, pair)
    tail = e.fb.const(paired_rows)
    if m - paired_rows == 2 * F.DIM:
        last = e.fb.const(paired_rows + F.DIM)
        load_panel(last, 1)
        compute_panel(tail, 0)
        compute_panel(last, 1)
    else:
        compute_panel(tail, 0)
    e._rocc("fence", {})
    fn = llvm.FuncOp(
        "gemmini_golden_joint_resadd",
        llvm.LLVMFunctionType([PTR] * 5),
        linkage=llvm.LinkageAttr("external"),
        body=e.fb.finish(),
    )
    fn.attributes["gemmini.joint_residual_resources"] = plan.attributes()
    module = ModuleOp([fn])
    module.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
    module.attributes["gemmini.joint_residual_resources"] = plan.attributes()
    module.attributes["gemmini.joint_residual_predictions"] = StringAttr(str(contracts))
    module.verify()
    return module
