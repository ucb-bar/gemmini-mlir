"""Decoded segmented input packets preserve all resident cells and K work."""

import pytest
from merlin.llvmlower.segmented_matrix_view import SegmentedRows
from test_resident_a_dma_coalescing import commands, fingerprint

from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import rtl_facts as F


@pytest.mark.parametrize(
    "m,n,k,segment,row_stride,segment_stride,origin,group",
    [
        (33, 73, 65, 7, 131, 2000, 5, 4),
        (17, 17, 48, 3, 67, 251, 11, 1),
        (1, 5, 17, 1, 19, 21, 3, 4),
        (49, 128, 1024, 7, 2048, 28672, 0, 4),
    ],
)
def test_actual_decoded_addresses_cover_exact_source_view_without_changing_compute(
    m, n, k, segment, row_stride, segment_stride, origin, group
):
    view = SegmentedRows(
        m,
        k,
        segment,
        row_stride,
        segment_stride,
        origin + ((m - 1) // segment + 1) * segment_stride,
        1,
        "i8",
        origin,
    )
    shape = Shape(
        m,
        n,
        k,
        output_dtype="i32",
        bm=(m + 15) // 16,
        bn=min((n + 15) // 16, 4),
        cache_a=True,
        wide_b=True,
        reuse_b=True,
        prefetch_b=m < 49,
    )
    control = commands(GoldenGemm(shape, resident_a_load_tiles=group))
    candidate = commands(
        GoldenGemm(shape, resident_a_load_tiles=group, input_view=view)
    )
    changed = lambda step: (
        isinstance(step.operation, (G.ConfigLdOp, G.MvinOp))
        and step.operation.a("load_id") == 0
    )
    assert [fingerprint(s) for s in control if not changed(s)] == [
        fingerprint(s) for s in candidate if not changed(s)
    ]
    kt = (k + 15) // 16
    cells = {}
    for step in candidate:
        op = step.operation
        if isinstance(op, G.ConfigLdOp) and op.a("load_id") == 0:
            assert op.a("stride") == row_stride
        if not isinstance(op, G.MvinOp) or op.a("load_id") != 0:
            continue
        ptr = step.inputs[0]
        assert ptr.base == 0
        assert 0 < op.a("rows") <= F.DIM and 0 < op.a("cols") <= 4 * F.DIM
        for b in range((op.a("cols") + 15) // 16):
            width = min(16, op.a("cols") - b * 16)
            for r in range(op.a("rows")):
                local = op.a("local") + b * 16 + r
                assert local not in cells
                mt, remainder = divmod(local, kt * 16)
                ki, lane = divmod(remainder, 16)
                logical_row = mt * 16 + lane
                source = ptr.offset.value + r * row_stride + b * 16
                assert source == view.offset(logical_row, ki * 16)
                assert source + width <= view.source_elements
                assert ki * 16 + width <= k
                cells[local] = tuple((logical_row, ki * 16 + c) for c in range(width))
    assert len(cells) == m * kt
    assert {v for row in cells.values() for v in row} == {
        (row, col) for row in range(m) for col in range(k)
    }


def test_explicit_none_is_byte_identical_and_unknown_storage_refuses():
    shape = Shape(17, 17, 48, bm=2, bn=2, cache_a=True)
    assert str(GoldenGemm(shape).build()) == str(
        GoldenGemm(shape, input_view=None).build()
    )
    view = SegmentedRows(17, 48, 3, 67, 251, 2000, 1, "i8")
    with pytest.raises(ValueError, match="cached A"):
        GoldenGemm(Shape(17, 17, 48), input_view=view)
    with pytest.raises(ValueError, match="logical shape"):
        GoldenGemm(shape, input_view=SegmentedRows(16, 48, 3, 67, 251, 2000, 1, "i8"))
    with pytest.raises(ValueError, match="physical i8"):
        GoldenGemm(shape, input_view=SegmentedRows(17, 48, 3, 67, 251, 2000, 4, "f32"))
    with pytest.raises(ValueError, match="ABI contract"):
        GoldenGemm(shape, input_view=view).build_batched(1)


def test_consumer_resource_bound_is_unchanged():
    view = SegmentedRows(256, 2048, 16, 4096, 131072, 2097152, 1, "i8")
    with pytest.raises(ValueError, match="scratchpad"):
        GoldenGemm(Shape(256, 16, 2048, bm=16, bn=1, cache_a=True), input_view=view)


def test_existing_residency_and_slot_transforms_preserve_view_contract():
    from mlir_oot.b_slot_placement import select_remaining_b_slots
    from mlir_oot.dense_schedule import select_coalesced_resident_a

    view = SegmentedRows(17, 48, 3, 67, 251, 2000, 1, "i8")
    original = GoldenGemm(Shape(17, 17, 48, bm=2, bn=2, cache_a=True), input_view=view)
    coalesced, decision = select_coalesced_resident_a(original)
    assert decision["applied"] and coalesced.input_view is view
    placed, decision = select_remaining_b_slots(coalesced)
    assert decision["applied"] and placed.input_view is view
    placed.build().verify()
