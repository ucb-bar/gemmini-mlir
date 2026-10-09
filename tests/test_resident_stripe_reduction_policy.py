"""Keep the modifier bound to the exact typed resident resources and ABI."""

from dataclasses import replace

import pytest
from test_resident_conv_command_loops import executed_commands

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.golden_resident_stripe_conv import GoldenResidentStripeConv
from mlir_oot.resident_stripe_reduction_policy import retain_reduction_commands


def test_explicit_modifier_revalidates_resources_and_preserves_commands():
    source = GoldenResidentStripeConv(ConvShape(5, 23, 48, 33, bn=3), stripe_rows=3)
    source.resident_stripe_decision = {"applied": True, "timing_claim": False}
    selected, decision = retain_reduction_commands(source)
    assert decision["applied"] and not decision["timing_claim"]
    assert selected.compact_inner_commands and selected.compact_reduction_commands
    assert selected.resident_stripe_decision == source.resident_stripe_decision
    assert (
        decision["resource_proof"]["resident_a_rows"]
        <= decision["resource_proof"]["b_base_row"]
    )
    assert executed_commands(source.build()) == executed_commands(selected.build())


def test_other_family_retains_exact_object():
    source = GoldenFlatConv(ConvShape(5, 23, 48, 33, bn=3), virtual_padding=True)
    selected, decision = retain_reduction_commands(source)
    assert selected is source and not decision["applied"] and decision["refusal"]


def test_no_remaining_k_loop_retains_current_object():
    source = GoldenResidentStripeConv(ConvShape(3, 5, 16, 17, bn=2))
    selected, decision = retain_reduction_commands(source)
    assert selected is source and not decision["applied"]


@pytest.mark.parametrize(
    "field,value", [("bbase", 0), ("input_rows", 1), ("weight_rows", 1)]
)
def test_contradictory_resource_state_refuses(field, value):
    source = GoldenResidentStripeConv(ConvShape(5, 23, 48, 33, bn=3))
    setattr(source, field, value)
    with pytest.raises(ValueError, match="typed reconstruction"):
        retain_reduction_commands(source)


def test_contradictory_numeric_state_refuses():
    source = GoldenResidentStripeConv(ConvShape(5, 23, 48, 33, bn=3))
    source.shape = replace(source.shape, scale=0.125)
    with pytest.raises(ValueError, match="typed reconstruction"):
        retain_reduction_commands(source)


def test_unsupported_store_contract_is_not_dropped():
    source = GoldenResidentStripeConv(ConvShape(5, 23, 48, 33, bn=3))
    source.store_plan = object()
    selected, decision = retain_reduction_commands(source)
    assert selected is source and not decision["applied"]


@pytest.mark.parametrize("option", [1, None, "yes"])
def test_normal_api_refuses_untyped_selection_before_reading_source(tmp_path, option):
    from mlir_oot.captured_requant_bundle import build

    with pytest.raises(ValueError, match="boolean"):
        build(
            tmp_path / "missing-source",
            tmp_path / "missing-tools",
            tmp_path / "output",
            resident_stripe_reduction_loops=option,
        )
    assert not (tmp_path / "output").exists()
