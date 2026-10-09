"""Typed normal compiler admission for stationary-B spatial tail placement."""

from collections import Counter
from pathlib import Path

import pytest
from resident_weight_packet_trace_probe import prove
from test_resident_conv_command_loops import executed_commands

from mlir_oot.captured_requant_bundle import build
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.golden_resident_conv import (
    GoldenResidentConv,
    issue_resident_weight_packets,
    place_resident_spatial_tail,
    retain_resident_commands,
)
from mlir_oot.readout_store_plan import PairedReadoutPlan


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_selector_preserves_source_resources_and_prior_choices(paired, prefetch):
    shape = ConvShape(5, 7, 32, 67, bn=4, output_dtype="i32" if paired else "i8")
    bound = 9 * shape.cin * 128**2
    plan = (
        PairedReadoutPlan(
            (1.0, 1.0, 1 / 1024), (1 / 1024, 1 / 1024), -bound, bound, False
        )
        if paired
        else None
    )
    control = GoldenResidentConv(
        shape,
        flat_spatial_planes=True,
        compact_commands=True,
        prefetch_b=prefetch,
        weight_issue_tiles=2 if prefetch else None,
        store_plan=plan,
    )
    control.flat_resident_decision = {"applied": True, "independent_source": "typed"}
    selected, decision = place_resident_spatial_tail(control)
    assert decision["applied"] and decision["selected_spatial_order"] == [0, 2, 1]
    assert selected.conv == control.conv and selected.store_plan is plan
    assert selected.bases == control.bases and selected.plane == control.plane
    assert selected.flat_resident_decision is control.flat_resident_decision
    a, b = control.build(), selected.build()
    assert prove(control, a) == prove(selected, b)
    assert Counter(executed_commands(a)) == Counter(executed_commands(b))
    repeated, repeated_decision = place_resident_spatial_tail(selected)
    assert repeated is selected and not repeated_decision["applied"]
    # Other explicit modifiers cannot silently discard a selected tail order.
    retained, _ = retain_resident_commands(selected)
    assert retained.tail_before_last_full
    packet, _ = issue_resident_weight_packets(selected, tiles=2)
    assert packet.tail_before_last_full


@pytest.mark.parametrize(
    "control",
    [
        GoldenFlatConv(ConvShape(5, 7, 16, 16), virtual_padding=True),
        GoldenResidentConv(ConvShape(5, 7, 16, 16)),
        GoldenResidentConv(
            ConvShape(4, 8, 16, 16), flat_spatial_planes=True, compact_commands=True
        ),
        GoldenResidentConv(
            ConvShape(3, 3, 16, 16), flat_spatial_planes=True, compact_commands=True
        ),
    ],
)
def test_unsupported_family_or_tile_geometry_preserves_control(control):
    selected, decision = place_resident_spatial_tail(control)
    assert selected is control and not decision["applied"] and decision["refusal"]


def test_normal_API_rejects_nonboolean_before_capture_access(tmp_path):
    with pytest.raises(ValueError, match="spatial tail selection must be boolean"):
        build(
            Path("missing"),
            Path("missing"),
            tmp_path / "unused",
            resident_tail_before_last_full=1,
        )
    assert not (tmp_path / "unused").exists()


def test_pair_binding_preserves_admitted_tail_choice():
    from mlir_oot.paired_readout_binding import choose

    shape = ConvShape(5, 7, 32, 67, bn=4, output_dtype="i32")
    control = GoldenResidentConv(shape, flat_spatial_planes=True, compact_commands=True)
    selected, decision = place_resident_spatial_tail(control)
    bound = 9 * shape.cin * 128**2
    paired, plan, pair_decision = choose(
        selected,
        {
            "source_scales": [1.0, 1.0, 1 / 1024],
            "accumulator_min": -bound,
            "accumulator_max": bound,
            "output_min": -128,
        },
    )
    assert pair_decision["applied"] and plan is not None
    assert paired.tail_before_last_full
    assert (
        paired.resident_spatial_tail_decision is selected.resident_spatial_tail_decision
    )
    assert paired.build().attributes["gemmini.resident_conv_tail_before_last_full"]
    assert decision["applied"]
