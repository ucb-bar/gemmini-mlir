import pytest

from mlir_oot.golden_requant import prove_scale_bound, quantized


@pytest.mark.parametrize('relu', [False, True])
@pytest.mark.parametrize('scales', [(0.5, 0.25), (0.1031, 0.7187), (0.00003457, 2314.17)])
def test_complete_transition_bound_matches_exhaustive_small_domain(scales, relu):
    low, high = -4096, 4096
    report = prove_scale_bound(scales, low, high, relu)
    actual = max(
        abs(quantized(acc, report['source_scales'], relu)-quantized(acc, (report['scale'],), relu))
        for acc in range(low, high+1)
    )
    assert report['max_output_lsb_error'] == actual
    assert report['exact'] == (actual == 0)


def test_full_domain_reassociation_refusal_retains_bound_and_witness():
    scales = (0.010870203375816345, 0.003105518640950322, 106.30004119873047)
    report = prove_scale_bound(scales, -1048576, 1048576)
    # The guarantee covers every signed i32, including values absent from a
    # calibration sample. A mismatch witness must reproduce the reported error.
    assert report['max_output_lsb_error'] == 1
    assert not report['exact']
    witness = report['witness']
    assert abs(witness['source']-witness['target']) == report['max_output_lsb_error']


@pytest.mark.parametrize('scales', [(), (0,), (-1,), (float('nan'),), (float('inf'),)])
def test_invalid_scales_refused(scales):
    with pytest.raises(ValueError, match='finite positive'):
        prove_scale_bound(scales)
