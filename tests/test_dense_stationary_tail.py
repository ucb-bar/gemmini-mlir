"""Exact dense tail placement from executed source/ACC/command witnesses."""

from collections import Counter
from dataclasses import replace

import pytest
from cached_b_resource_trace_probe import prove
from merlin.xdsl_dialects._common import text
from test_resident_conv_command_loops import executed_commands

from mlir_oot.dense_schedule import select_stationary_b_spatial_tail
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.tables import isa


def shape(**kw):
    return replace(
        Shape(37, 35, 33, "i32", bm=3, bn=2, cache_a=True, reuse_b=True, wide_b=True),
        **kw,
    )


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("dma_group", [1, 4])
@pytest.mark.parametrize(
    "s",
    [
        shape(),
        shape(m=49, k=48, n=19, bm=4, bn=2),
        shape(output_dtype="i8", scale=0.001, relu=True, wide_store=True),
    ],
)
def test_actual_source_K_resources_and_command_order(s, prefetch, dma_group):
    s = replace(s, prefetch_b=prefetch)
    control = GoldenGemm(s, resident_a_load_tiles=dma_group)
    candidate, decision = select_stationary_b_spatial_tail(control)
    assert decision["applied"] and candidate.shape is control.shape
    assert candidate.resident_a_load_tiles == control.resident_a_load_tiles
    # Rebuild generators because FnBuilder owns a single emitted function.
    assert prove(control) == prove(candidate)
    a = executed_commands(GoldenGemm(s, resident_a_load_tiles=dma_group).build())
    b = executed_commands(
        GoldenGemm(
            s, resident_a_load_tiles=dma_group, stationary_b_tail_before_last_full=True
        ).build()
    )
    assert Counter(a) == Counter(b) and a != b
    assert [r for r in a if r[0] not in ("gemmini.preload", "gemmini.compute")] == [
        r for r in b if r[0] not in ("gemmini.preload", "gemmini.compute")
    ]
    pairs = [r for r in b if r[0] in ("gemmini.preload", "gemmini.compute")]
    short = 0
    for i, (name, encoded, _) in enumerate(pairs):
        if name == "gemmini.compute" and ((encoded[1] >> 48) & 0xFFFF) < isa.DIM:
            short += 1
            assert pairs[i + 1][0] == "gemmini.preload"
            assert pairs[i + 1][1][1] & 0xFFFFFFFF == isa.GARBAGE_ADDR
    assert short > 0


@pytest.mark.parametrize(
    "s",
    [
        shape(m=9, bm=1),
        shape(m=32, bm=2),
        shape(cache_a=False),
        shape(reuse_b=False),
        shape(bias=True),
    ],
)
def test_inapplicable_choice_retains_control(s):
    control = GoldenGemm(s)
    selected, d = select_stationary_b_spatial_tail(control)
    assert selected is control and not d["applied"] and d["refusal"]
    with pytest.raises(ValueError, match="tail placement requires"):
        GoldenGemm(s, stationary_b_tail_before_last_full=True)


def test_no_new_default_bytes_and_no_truthy_or_subclass_admission():
    s = shape()
    assert text(GoldenGemm(s).build(), generic=True) == text(
        GoldenGemm(s, stationary_b_tail_before_last_full=False).build(), generic=True
    )
    with pytest.raises(ValueError, match="boolean"):
        GoldenGemm(s, stationary_b_tail_before_last_full=1)

    class Other(GoldenGemm):
        pass

    other = Other(s)
    selected, d = select_stationary_b_spatial_tail(other)
    assert selected is other and not d["applied"]


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("dma_group", [1, 4])
def test_borrowed_segmented_owner_keeps_every_source_cell(prefetch, dma_group):
    from merlin.llvmlower.segmented_matrix_view import SegmentedRows

    s = shape(prefetch_b=prefetch)
    view = SegmentedRows(s.m, s.k, 7, 66, 1000, 6000, 1, "i8", origin=3)
    control = GoldenGemm(s, input_view=view, resident_a_load_tiles=dma_group)
    candidate, d = select_stationary_b_spatial_tail(control)
    assert d["applied"] and candidate.input_view is view
    assert prove(control) == prove(candidate)


def test_existing_load_modifier_preserves_typed_tail_order():
    from mlir_oot.dense_schedule import select_coalesced_resident_a

    control = GoldenGemm(shape(), stationary_b_tail_before_last_full=True)
    selected, d = select_coalesced_resident_a(control)
    assert d["applied"] and selected.stationary_b_tail_before_last_full
    assert "gemmini.stationary_b_tail_before_last_full" in selected.build().attributes
