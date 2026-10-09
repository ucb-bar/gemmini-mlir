"""Prove tail placement from decoded source cells and exact command operands."""

from collections import Counter

import pytest
from resident_weight_packet_trace_probe import prove
from test_resident_conv_command_loops import executed_commands

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.tables import isa


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        ConvShape(5, 7, 32, 67, bn=4, output_dtype="i32"),
        ConvShape(5, 7, 48, 67, bn=4, output_dtype="i32"),
        ConvShape(7, 7, 48, 19, bn=2, output_dtype="i8", scale=0.001, relu=True),
    ],
)
def test_short_tile_reuses_B_before_final_full_tile(shape, prefetch):
    options = {
        "flat_spatial_planes": True,
        "compact_commands": True,
        "prefetch_b": prefetch,
        "weight_issue_tiles": 2 if prefetch else None,
    }
    control = GoldenResidentConv(shape, **options)
    candidate = GoldenResidentConv(shape, **options, tail_before_last_full=True)
    control_module, candidate_module = control.build(), candidate.build()
    assert prove(control, control_module) == prove(candidate, candidate_module)
    a, b = executed_commands(control_module), executed_commands(candidate_module)
    assert Counter(a) == Counter(b)
    assert a != b
    # DMA/config/fences and complete stores retain exact relative order and
    # operands. Only independent spatial PRELOAD/COMPUTE pairs move.
    assert [
        row for row in a if row[0] not in ("gemmini.preload", "gemmini.compute")
    ] == [row for row in b if row[0] not in ("gemmini.preload", "gemmini.compute")]
    # Every short COMPUTE has an immediately following garbage PRELOAD. Its
    # stationary matrix stays live; a real-B PRELOAD never follows that tail.
    relevant = [row for row in b if row[0] in ("gemmini.preload", "gemmini.compute")]
    short = 0
    for index, (name, encoded, _) in enumerate(relevant):
        if name == "gemmini.compute" and encoded[1] >> 48 & 0xFFFF < isa.DIM:
            short += 1
            assert relevant[index + 1][0] == "gemmini.preload"
            assert relevant[index + 1][1][1] & 0xFFFFFFFF == isa.GARBAGE_ADDR
    assert short > 0


@pytest.mark.parametrize(
    "shape,options",
    [
        (
            ConvShape(3, 3, 16, 16),
            {"flat_spatial_planes": True, "compact_commands": True},
        ),
        (
            ConvShape(4, 8, 16, 16),
            {"flat_spatial_planes": True, "compact_commands": True},
        ),
        (ConvShape(5, 7, 16, 16), {"flat_spatial_planes": True}),
        (ConvShape(5, 7, 16, 16), {"compact_commands": True}),
    ],
)
def test_inapplicable_tail_placement_refuses(shape, options):
    with pytest.raises(ValueError, match="tail placement requires"):
        GoldenResidentConv(shape, **options, tail_before_last_full=True)


def test_nonboolean_tail_selection_refuses():
    with pytest.raises(ValueError, match="boolean"):
        GoldenResidentConv(ConvShape(5, 7, 16, 16), tail_before_last_full=1)


def test_explicit_false_keeps_default_IR_bytes():
    from merlin.xdsl_dialects._common import text

    shape = ConvShape(5, 7, 32, 67, bn=4)
    options = {"flat_spatial_planes": True, "compact_commands": True}
    assert text(GoldenResidentConv(shape, **options).build(), generic=True) == text(
        GoldenResidentConv(shape, **options, tail_before_last_full=False).build(),
        generic=True,
    )
