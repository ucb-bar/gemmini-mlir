"""One exact integer producer serves independently configured readouts."""

import hashlib

import pytest
from test_wide_residual_prefetch import _executed

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_joint_resadd import JointResidualPlan, build, tables
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def assert_actual_symbolic_stores(module, m, predictors):
    coefficients = tables(predictors, share_affine=True)
    spad, acc, stores = {}, {}, []
    weight, current = None, None
    for op, inputs in _executed(module):
        if op.name == "gemmini.mvin":
            operand, offset = inputs[0]
            limit = len(coefficients) if operand == 4 else m * 64
            assert 0 <= offset and offset + op.a("rows") * op.a("cols") <= limit
            for tile in range(op.a("cols") // F.DIM):
                spad[op.a("local") + tile * F.DIM] = operand, offset + tile * F.DIM
        elif op.name == "gemmini.preload":
            if op.a("bd") != isa.GARBAGE_ADDR:
                operand, offset = spad[op.a("bd")]
                assert operand == 4
                diagonal = coefficients[offset]
                assert all(
                    coefficients[offset + row * 16 + col]
                    == (diagonal if row == col else 0)
                    for row in range(16)
                    for col in range(16)
                )
                weight = diagonal
            current = op.a("c") & ((1 << 30) - 1)
            if not op.a("c") & (1 << 30):
                acc[current] = {}
        elif op.name == "gemmini.compute":
            term = spad[op.a("a")]
            assert term[0] in (0, 1)
            acc[current][term] = acc[current].get(term, 0) + weight
        elif op.name == "gemmini.config_st":
            readout = op.a("acc_act"), op.a("acc_scale")
        elif op.name == "gemmini.mvout":
            operand, offset = inputs[0]
            prediction = operand - 2
            assert prediction in (0, 1) and 0 <= offset <= m * 64 - 1024
            base = op.a("local") & ((1 << 30) - 1)
            for tile in range(4):
                assert acc[base + 16 * tile] == {
                    (0, offset + 16 * tile): predictors[prediction]["p"],
                    (1, offset + 16 * tile): predictors[prediction]["q"],
                }
            assert readout == (
                isa.RELU if predictors[prediction]["relu"] else isa.NO_ACTIVATION,
                predictors[prediction]["scale"],
            )
            stores.append((operand, offset, base))
    assert len(stores) == 2 * m // F.DIM
    for first, second in zip(stores[::2], stores[1::2], strict=True):
        assert first[0] == 2 and second[0] == 3 and first[1:] == second[1:]


@pytest.mark.parametrize("m", [16, 32, 48, 80])
@pytest.mark.parametrize("p,q", [(298, 249), (129, 3)])
def test_actual_addresses_and_integer_expression_at_both_readouts(m, p, q):
    predictors = [
        {"p": p, "q": q, "scale": scale, "relu": bool(i)}
        for i, scale in enumerate((0.03125, 0.0625))
    ]
    module = build(m, predictors, share_affine=True)
    assert_actual_symbolic_stores(module, m, predictors)
    computes = sum(op.name == "gemmini.compute" for op, _ in _executed(module))
    assert computes == m // 16 * 4 * ((p + 126) // 127 + (q + 126) // 127)
    plan = JointResidualPlan(m, ((p, q), (p, q)), True)
    assert plan.weight_rows == 48
    assert plan.accumulator_base(0, 0) == plan.accumulator_base(1, 0) == 0
    assert plan.accumulator_base(0, 1) == plan.accumulator_base(1, 1) == F.ACC_BANK_ROWS
    lower(module)


def test_share_refuses_different_integer_producers():
    with pytest.raises(ValueError, match="identical integer"):
        JointResidualPlan(32, ((73, 61), (523, 437)), True)


def test_default_ir_is_byte_preserved_from_committed_independent_family():
    predictors = [
        {"p": 73, "q": 61, "scale": 0.013245166279375553, "relu": True},
        {"p": 523, "q": 437, "scale": 0.0018487992929294705, "relu": True},
    ]
    expected = "b49477ba5907b16daeaba02edf17d2649878cf18969441eade142ceccda06898"
    assert hashlib.sha256(str(build(32, predictors)).encode()).hexdigest() == expected
