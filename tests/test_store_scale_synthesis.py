import pytest

from mlir_oot.golden_requant import f32, quantized, synthesize_store_scale


def test_adjacent_store_scale_restores_exact_original_order():
    scales=(0.010870203375816345,0.003105518640950322,106.30004119873047)
    proof=synthesize_store_scale(scales,-1048576,1048576)
    assert proof['exact']
    assert proof['scale']==0.003588436171412468
    assert quantized(-14909,scales)==quantized(-14909,(proof['scale'],))==-53


@pytest.mark.parametrize('relu',[False,True])
@pytest.mark.parametrize('scales',[(.5,.25),(.1031,.7187),(1e-20,1e20)])
def test_solved_scale_matches_every_small_domain_value(scales,relu):
    scales=tuple(map(f32,scales))
    proof=synthesize_store_scale(scales,-512,512,relu)
    assert proof['exact']
    assert all(quantized(a,scales,relu)==quantized(a,(proof['scale'],),relu) for a in range(-512,513))


def test_conflicting_constraints_refuse_all_f32_scales():
    scales=(0.1525018811225891,0.0011392629239708185,4.235067367553711)
    proof=synthesize_store_scale(scales,-37748736,37748736,True)
    assert not proof['exact']
    assert proof['scale_bits_min']>proof['scale_bits_max']
    assert proof['witnesses']['lower'] and proof['witnesses']['upper']
