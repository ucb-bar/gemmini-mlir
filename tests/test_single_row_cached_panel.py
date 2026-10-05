from mlir_oot.contraction_patterns import IntegerGemm
from mlir_oot.golden_contraction_upstream import choose_shape
from mlir_oot.golden_tuning import estimate


def test_captured_classifier_tail_selects_verified_opt_in_schedule():
    dimensions=IntegerGemm(1,1,1000,2048)
    baseline=choose_shape(dimensions)
    selected=choose_shape(dimensions,large_n=True)
    assert (baseline.bm,baseline.bn,baseline.cache_a,baseline.wide_b)==(1,63,False,False)
    assert (selected.bm,selected.bn,selected.cache_a,selected.wide_b)==(1,63,True,True)
    selected.validate()
    assert estimate(selected)['primitive_command_count']<estimate(baseline)['primitive_command_count']


def test_small_attention_and_multirow_single_panel_keep_previous_policy():
    for dimensions in (IntegerGemm(1,8,8,64),IntegerGemm(1,8,1000,2048)):
        selected=choose_shape(dimensions,large_n=True)
        assert not selected.cache_a
