from dataclasses import asdict, replace

import pytest

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_resident_stripe_conv import GoldenResidentStripeConv, command_counts
from mlir_oot.ir.gemmini_dialect import ComputeOp, MvinOp
from mlir_oot.tables import rtl_facts as F


@pytest.mark.parametrize('shape', [
    ConvShape(5,23,32,19,bn=2),
    ConvShape(20,23,32,19,bn=2),
    ConvShape(3,31,48,33,bn=3,output_dtype='i8',scale=.002,relu=True),
    ConvShape(56,56,64,64,bn=4,output_dtype='i8',scale=.002,relu=True),
    ConvShape(28,28,128,128,bn=4,output_dtype='i8',scale=.002,relu=True),
])
def test_complete_input_partition_and_dynamic_execute_bounds(shape):
    g = GoldenResidentStripeConv(shape)
    module = g.build()
    module.verify()
    covered = set()
    for op in module.walk():
        if isinstance(op,MvinOp) and op.a('load_id') == 0:
            for block in range((op.a('cols')+F.DIM-1)//F.DIM):
                base = op.a('local') + block*g.plane
                addresses = set(range(base,base+op.a('rows')))
                assert covered.isdisjoint(addresses)
                assert max(addresses) < g.input_rows
                covered.update(addresses)
        if isinstance(op,ComputeOp):
            assert 0 <= op.a('a_max')
            assert op.a('a_max')+op.a('a_rows') <= op.a('a_reserved_rows') == g.input_rows
    assert covered == set(range(g.input_rows))
    assert g.input_rows <= g.bbase
    assert g.bbase+g.weight_rows == F.SPAD_ROWS
    lower(module).verify()


@pytest.mark.parametrize('bad', [
    ConvShape(56,56,128,128,bn=4),
    ConvShape(28,28,256,256,bn=4),
    ConvShape(56,56,64,64,bn=4,stride=2),
    ConvShape(5,23,31,19,bn=2),
    ConvShape(5,23,32,19,bn=2,explicit_halo=True),
])
def test_refuse_unsupported_semantics_or_residency(bad):
    with pytest.raises(ValueError):
        GoldenResidentStripeConv(bad)


@pytest.mark.parametrize('stripe', [0,17,True,1.5])
def test_refuse_invalid_stripe_extent(stripe):
    with pytest.raises(ValueError):
        GoldenResidentStripeConv(ConvShape(20,23,32,19,bn=2),stripe_rows=stripe)


def test_full_reduction_weights_remove_spatial_reload():
    s = ConvShape(56,56,64,64,bn=4,output_dtype='i8',scale=.002,relu=True)
    counts = command_counts(s)
    assert counts['stripe_rows'] == 4
    assert counts['resident_input_rows'] == 13456
    assert counts['resident_weight_rows'] == 2304
    assert counts['mvin_a'] == 344
    assert counts['mvin_b'] == 36
    assert counts['weight_dram_bytes'] == 9*64*64
    assert counts['compute'] == 32256
    assert counts['host_im2col_bytes'] == 0
    # Another N block reuses complete A, while loading only its own weights.
    wider = command_counts(replace(s,cout=128))
    assert wider['mvin_a'] == counts['mvin_a']
    assert wider['mvin_b'] == 2*counts['mvin_b']


@pytest.mark.parametrize('shape,applied', [
    (ConvShape(56,56,64,64,bn=4),True),
    (ConvShape(28,28,128,128,bn=4),True),
    (ConvShape(5,23,32,19,bn=2),True),
    (ConvShape(56,56,128,128,bn=4),False),
    (ConvShape(28,28,256,256,stride=2,bn=4),False),
    (ConvShape(14,14,256,256,bn=4),False),
])
def test_shape_resource_traffic_policy_and_fallback(shape,applied):
    from mlir_oot.conv_schedule import select_kernel
    control,control_kind = select_kernel(shape,flat_spatial=True,virtual_padding=True)
    selected,kind = select_kernel(shape,flat_spatial=True,virtual_padding=True,resident_stripes=True)
    decision = selected.resident_stripe_decision
    assert decision['applied'] == applied
    assert decision['timing_claim'] is False
    if applied:
        assert decision['refusal'] is None
        assert kind == 'resident_full_k_stripes'
        assert decision['candidate']['score'] < decision['control']['score']
        assert {k:v for k,v in asdict(selected.conv).items() if k not in ('bn','wide_b')} == {
            k:v for k,v in asdict(control.conv).items() if k not in ('bn','wide_b')}
    else:
        assert isinstance(decision['refusal'],str) and decision['refusal']
        assert kind == control_kind
        assert type(selected) is type(control)


def test_source_padding_contract_and_option_validation(tmp_path):
    from mlir_oot.captured_requant_bundle import build
    from mlir_oot.conv_schedule import select_kernel
    for flags in ({'resident_stripes':True},{'resident_stripes':True,'flat_spatial':True}):
        with pytest.raises(ValueError,match='resident stripe'):
            select_kernel(ConvShape(5,23,32,19,bn=2),**flags)
    for flags in ({'resident_stripes':1},{'resident_stripes':True},
                  {'resident_stripes':True,'flat_spatial':True}):
        with pytest.raises(ValueError,match='resident stripe'):
            build(tmp_path/'absent',tmp_path/'absent_tools',tmp_path/'output',**flags)
        assert not (tmp_path/'output').exists()
