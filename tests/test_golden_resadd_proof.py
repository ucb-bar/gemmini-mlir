import pytest
from pathlib import Path
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_resadd_proof import prove, match, census, op_name
from xdsl.dialects.builtin import IntegerAttr, TensorType, i8, i64


@pytest.mark.parametrize('relu', [False, True])
def test_identity_residual_complete_pair_domain(relu):
    report = prove(1.0, 1.0, 1.0, lhs_load=1.0, rhs_load=1.0, readout=1.0, relu=relu)
    assert report['exact']
    assert report['pairs'] == 65536
    assert report['max_output_lsb_error'] == 0


def test_two_load_rounds_are_not_a_single_source_round():
    report = prove(0.5, 0.5, 1.0, lhs_load=0.5, rhs_load=0.5, readout=1.0)
    assert not report['exact']
    assert report['witness']
    assert report['max_output_lsb_error'] == 1


def fixture():
    source = Path(__file__).with_name('fixtures').joinpath('captured_resadd.mlir').read_text()
    module = parse_module(source)
    q = next(op for op in module.walk() if op_name(op) == 'quant_ext.quantize_per_tensor')
    return source, q


def test_captured_residual_matches_complete_proof():
    source, q = fixture()
    assert match(q) == dict(lhs_scale=1.0, rhs_scale=1.0, output_scale=1.0, relu=False)
    report = census(source)
    assert not report['refused']
    assert len(report['accepted']) == 1
    assert report['accepted'][0]['pairs'] == 65536


def test_dequantize_output_shape_mismatch_refused():
    _, q = fixture()
    dq = q.operands[0].owner.operands[0].owner
    dq.operands[0]._type = TensorType(i8, [32, 32])
    with pytest.raises(ValueError, match='shapes or types'):
        match(q)


def test_dequantize_noncanonical_clamp_refused():
    _, q = fixture()
    dq = q.operands[0].owner.operands[0].owner
    dq.properties['quant_min'] = IntegerAttr(-127, i64)
    with pytest.raises(ValueError, match='clamp'):
        match(q)


def test_disconnected_scalar_add_refused():
    _, q = fixture()
    add = q.operands[0].owner
    scalar_add = add.regions[0].block.first_op
    scalar_add.operands = [scalar_add.operands[0], add.regions[0].block.args[-1]]
    with pytest.raises(ValueError, match='operands differ'):
        match(q)
