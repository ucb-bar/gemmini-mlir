"""Explicit admitted-family binding preserves all resources and source work."""

from pathlib import Path

import pytest
from test_resident_conv_command_loops import executed_commands

from mlir_oot.captured_requant_bundle import build as build_capture
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_resident_conv import GoldenResidentConv, retain_resident_commands


@pytest.mark.parametrize(
    "shape,options",
    [
        (ConvShape(5, 7, 32, 33, bn=2), {"rows_per_tile": 2}),
        (ConvShape(9, 5, 112, 35, bn=2), {"rows_per_tile": 2, "prefetch_b": True}),
        (
            ConvShape(9, 9, 32, 19, stride=2, bn=2),
            {"source_stride": True, "row_residue": True, "weight_base": 1024},
        ),
    ],
)
def test_explicit_binding_keeps_complete_admitted_layout_and_commands(shape, options):
    control = GoldenResidentConv(shape, **options)
    control.source_stride_decision = {"applied": True, "resources": "pinned"}
    selected, decision = retain_resident_commands(control)
    assert selected is not control and selected.compact_commands
    assert decision["applied"] and not decision["timing_claim"]
    assert selected.conv == control.conv
    for name in (
        "rows_per_tile",
        "prefetch_b",
        "explicit_weight_base",
        "source_stride",
        "row_residue",
        "row_tiles",
        "bases",
        "plane",
        "source_stride_decision",
    ):
        assert getattr(selected, name) == getattr(control, name)
    assert executed_commands(control.build()) == executed_commands(selected.build())


@pytest.mark.parametrize(
    "control",
    [
        GoldenGemm(Shape(9, 33, 17)),
        GoldenFlatConv(ConvShape(5, 7, 20, 33, bn=2, explicit_halo=True)),
    ],
)
def test_unadmitted_family_retains_actual_original_generator(control):
    selected, decision = retain_resident_commands(control)
    assert selected is control and not decision["applied"]
    assert not decision["timing_claim"]


@pytest.mark.parametrize("selection", [None, 1, "yes", True])
def test_source_route_refuses_untyped_or_unproved_selection_before_io(selection):
    with pytest.raises(ValueError, match="boolean selection and proved virtual"):
        build_capture(
            Path("absent"),
            Path("absent"),
            Path("absent"),
            compact_resident_commands=selection,
        )


@pytest.mark.parametrize("spatial,padding", [(False, True), (True, False)])
def test_source_route_requires_both_layout_permissions(spatial, padding):
    with pytest.raises(ValueError, match="proved virtual"):
        build_capture(
            Path("absent"),
            Path("absent"),
            Path("absent"),
            flat_spatial=spatial,
            virtual_padding=padding,
            compact_resident_commands=True,
        )
