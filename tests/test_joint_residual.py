"""Independent ordered arithmetic/transfer witnesses for two readouts."""

import pytest
from test_wide_residual_prefetch import _executed

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_joint_resadd import JointResidualPlan, build, tables
from mlir_oot.golden_wide_resadd import build as single_build
from mlir_oot.golden_wide_resadd import tables as single_tables
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def arithmetic(commands, m, *, joint):
    """Resolve actual executed addresses to independent per-predictor operations."""
    spad, output, accumulator, store = {}, [[], []], None, None
    for op, inputs in commands:
        if op.name == "gemmini.mvin":
            operand, offset = inputs[0]
            limit = m * 64 if operand in (0, 1) else (5 if joint else 3) * 256
            assert 0 <= offset and offset + op.a("rows") * op.a("cols") <= limit
            for tile in range(op.a("cols") // F.DIM):
                spad[op.a("local") + tile * F.DIM] = (operand, offset + tile * F.DIM)
        elif op.name == "gemmini.preload":
            address = op.a("c") & ((1 << 30) - 1)
            prediction = address // F.ACC_BANK_ROWS if joint else 0
            row = address % F.ACC_BANK_ROWS
            if joint:
                row %= 64
            accumulator = prediction
            weight = "resident"
            if op.a("bd") != isa.GARBAGE_ADDR:
                operand, offset = spad[op.a("bd")]
                if joint:
                    assert operand == 4 and offset % 256 == 0
                    index = offset // 256
                    assert index in (0, 1 + 2 * prediction, 2 + 2 * prediction)
                    offset = (0 if index == 0 else index - 2 * prediction) * 256
                weight = (3, offset)
            output[prediction].append(
                (
                    "preload",
                    weight,
                    row,
                    bool(op.a("c") & (1 << 30)),
                    op.a("bd_rows"),
                    op.a("c_rows"),
                )
            )
        elif op.name == "gemmini.compute":
            output[accumulator].append(
                ("compute", spad[op.a("a")], op.a("accumulate"), op.a("a_rows"))
            )
        elif op.name == "gemmini.config_st":
            store = op.a("stride"), op.a("acc_act"), op.a("acc_scale")
        elif op.name == "gemmini.mvout":
            operand, offset = inputs[0]
            prediction = operand - 2 if joint else 0
            assert (
                prediction in (0, 1) and offset + op.a("rows") * op.a("cols") <= m * 64
            )
            assert (op.a("local") & ((1 << 30) - 1)) // F.ACC_BANK_ROWS == prediction
            output[prediction].append(
                ("store", offset, op.a("rows"), op.a("cols"), store)
            )
    return output


@pytest.mark.parametrize("m", [16, 32, 48, 80])
@pytest.mark.parametrize("pairs", [((73, 61), (523, 437)), ((1, 127), (254, 128))])
def test_actual_cfg_each_prediction_matches_independent_exact_command_order(m, pairs):
    predictors = [
        {"p": p, "q": q, "scale": 1.0 / (p + q), "relu": bool(i)}
        for i, (p, q) in enumerate(pairs)
    ]
    module = build(m, predictors)
    actual = arithmetic(_executed(module), m, joint=True)
    for index, predictor in enumerate(predictors):
        reference = arithmetic(_executed(single_build(m, **predictor)), m, joint=False)[
            0
        ]
        assert actual[index] == reference
    lower(module)


def test_complete_declared_storage_extents_and_coefficient_bytes():
    plan = JointResidualPlan(48, ((73, 61), (523, 437)))
    spad = [
        (plan.operand_base(slot, 0), plan.operand_base(slot, 0) + 128)
        for slot in (0, 1)
    ]
    spad.append((plan.weight_base, plan.weight_base + 5 * F.DIM))
    acc = [
        (
            plan.accumulator_base(prediction, slot),
            plan.accumulator_base(prediction, slot) + 64,
        )
        for prediction in (0, 1)
        for slot in (0, 1)
    ]
    for ranges, capacity in ((spad, F.SPAD_ROWS), (acc, F.ACC_ROWS)):
        for i, (start, end) in enumerate(ranges):
            assert 0 <= start < end <= capacity
            assert all(
                end <= other_start or other_end <= start
                for other_start, other_end in ranges[i + 1 :]
            )
    predictors = [{"p": 73, "q": 61}, {"p": 523, "q": 437}]
    coefficients = tables(predictors)
    assert len(coefficients) == 1280
    assert coefficients[:768] == single_tables(73, 61)
    assert coefficients[:256] + coefficients[768:] == single_tables(523, 437)


@pytest.mark.parametrize("m", [0, 15, 17, True])
def test_refuse_unproved_panel_extents(m):
    with pytest.raises(ValueError):
        JointResidualPlan(m, ((73, 61), (523, 437)))


def test_refuse_insufficient_independent_accumulators(monkeypatch):
    monkeypatch.setattr(F, "ACC_BANK_ROWS", 64)
    monkeypatch.setattr(F, "ACC_ROWS", 128)
    with pytest.raises(ValueError, match="disjoint readout"):
        JointResidualPlan(32, ((73, 61), (523, 437)))


def test_refuse_unknown_numerical_fields():
    predictors = [{"p": 73, "q": 61, "scale": 0.01, "relu": True}] * 2
    with pytest.raises(ValueError, match="contracts"):
        build(32, [dict(predictors[0], approximate=True), predictors[1]])
