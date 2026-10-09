"""Retained operand extents govern independently chosen output/prefetch panels."""
from dataclasses import replace

import pytest
from merlin.llvmlower.segmented_matrix_view import SegmentedRows
from mlir_oot.b_slot_placement import select_remaining_b_slots
from mlir_oot.dense_schedule import select_resident_a_output_blocks
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.tables import rtl_facts as F
from test_resident_a_output_blocks import source_tiles


@pytest.mark.parametrize('channels', [2, 4])
@pytest.mark.parametrize('m', [289, 529])
def test_all_source_tiles_and_true_A_allocation_survive_panel_prefetch(m, channels):
    shape = Shape(m, 67, 257, bm=(m + 15) // 16, bn=1,
                  output_dtype='i32', cache_a=True, reuse_b=True, wide_b=True)
    original = GoldenGemm(shape, resident_a_load_tiles=4)
    blocked, blocked_decision = select_resident_a_output_blocks(
        original, output_channel_tiles=channels)
    assert blocked_decision['applied'] and blocked.cached_a_output_blocks
    candidate, decision = select_remaining_b_slots(blocked)
    assert decision['applied'] and not decision['timing_claim']
    actual_A_end = ((m + 15) // 16) * ((shape.k + 15) // 16) * 16
    assert decision['reserved_a_rows'] == actual_A_end
    assert decision['reserved_b_panel_rows'] == candidate.shape.bn * 16
    for base in candidate.prefetch_b_rows:
        assert base >= actual_A_end and base + candidate.shape.bn * 16 <= F.SPAD_ROWS
        assert base // F.SPAD_BANK_ROWS == (base + candidate.shape.bn * 16 - 1) // F.SPAD_BANK_ROWS
    assert candidate.emission_options == replace(blocked.emission_options,
                                                prefetch_b_rows=candidate.prefetch_b_rows)
    old, old_counts, old_bytes, old_config = source_tiles(blocked.with_emission_options())
    new, new_counts, new_bytes, new_config = source_tiles(candidate.with_emission_options())
    assert old == new and old_counts == new_counts and old_bytes == new_bytes and old_config == new_config
    assert old_bytes[0] == m * shape.k
    kept, again = select_remaining_b_slots(candidate)
    assert kept is candidate and not again['applied']


def test_widening_output_panel_after_minimal_slots_is_a_decline_not_an_unsafe_relayout():
    shape = Shape(529, 67, 257, bm=34, bn=1, output_dtype='i32',
                  cache_a=True, reuse_b=True, wide_b=True)
    original = GoldenGemm(shape)
    slotted, slots = select_remaining_b_slots(original)
    assert slots['applied'] and slotted.prefetch_b_rows == (9248, 9264)
    declined, decision = select_resident_a_output_blocks(slotted, output_channel_tiles=4)
    assert declined is slotted and not decision['applied']
    assert 'overlap' in decision['refusal']
    # The final output-panel span must be selected before new slots are derived.
    blocked, _ = select_resident_a_output_blocks(original, output_channel_tiles=4)
    candidate, final = select_remaining_b_slots(blocked)
    assert final['applied'] and candidate.prefetch_b_rows == (9248, 9312)


def test_bank_boundary_is_derived_from_full_storage_and_current_panel_span():
    shape = Shape(17, 67, 6128, bm=2, bn=1, output_dtype='i32',
                  cache_a=True, reuse_b=True, wide_b=True)
    blocked, _ = select_resident_a_output_blocks(GoldenGemm(shape))
    candidate, decision = select_remaining_b_slots(blocked)
    assert decision['applied'] and decision['reserved_a_rows'] == 12256
    assert candidate.prefetch_b_rows == (12288, 12352)
    candidate.build().verify()


def test_real_full_A_capacity_refusal_retains_original_input_allocation():
    shape = Shape(513, 16, 496, bm=33, bn=1, output_dtype='i32', cache_a=True)
    original = GoldenGemm(shape)
    old = str(original.with_emission_options().build())
    declined, decision = select_remaining_b_slots(original)
    assert declined is original and not decision['applied']
    assert decision['reserved_a_rows'] == 16368
    assert 'scratchpad' in decision['refusal']
    assert str(original.with_emission_options().build()) == old


def test_unrelated_capacity_family_and_immutable_segmented_owner_are_preserved():
    capacity = GoldenGemm(Shape(3, 128, 512, bm=1, bn=8, cache_b=True),
                          cached_b_resource_capacity=True)
    declined, decision = select_remaining_b_slots(capacity)
    assert declined is capacity and not decision['applied']
    shape = Shape(529, 67, 257, bm=34, bn=1, output_dtype='i32',
                  cache_a=True, reuse_b=True)
    view = SegmentedRows(rows=529, cols=257, segment_rows=7, row_stride=514,
                         segment_stride=7000, source_elements=600000, origin=3,
                         element_bytes=1, dtype='i8')
    blocked, _ = select_resident_a_output_blocks(GoldenGemm(shape, input_view=view))
    candidate, decision = select_remaining_b_slots(blocked)
    assert decision['applied'] and candidate.input_view is view
    assert candidate.cached_a_output_blocks
    candidate.build().verify()


def test_segmented_adapter_and_native_oracle_validate_the_same_explicit_slots():
    from mlir_oot.captured_requant_bundle import scalar_oracle
    from mlir_oot.segmented_input_binding import adapter_source
    shape = Shape(529, 67, 257, bm=34, bn=1, scale=.001953125,
                  cache_a=True, reuse_b=True, wide_store=True, wide_b=True)
    view = SegmentedRows(rows=529, cols=257, segment_rows=7, row_stride=514,
                         segment_stride=7000, source_elements=600000, origin=3,
                         element_bytes=1, dtype='i8')
    blocked, _ = select_resident_a_output_blocks(GoldenGemm(shape, input_view=view))
    candidate, decision = select_remaining_b_slots(blocked)
    assert decision['applied']
    common = dict(cached_a_output_blocks=True)
    old_adapter = adapter_source(blocked.shape, 'consumer', 'kernel', view,
                                 (2000, 300), **common)
    old_native = scalar_oracle(blocked.shape, 'kernel', False, input_view=view,
                               **common)
    actual = dict(common, prefetch_b_rows=candidate.prefetch_b_rows)
    assert adapter_source(candidate.shape, 'consumer', 'kernel', view,
                          (2000, 300), **actual) == old_adapter
    assert scalar_oracle(candidate.shape, 'kernel', False, input_view=view,
                         **actual) == old_native
    with pytest.raises(ValueError, match='lower two banks'):
        adapter_source(candidate.shape, 'consumer', 'kernel', view, (2000, 300),
                       **common)
    with pytest.raises(ValueError, match='overlaps reserved A'):
        scalar_oracle(candidate.shape, 'kernel', False, input_view=view,
                      cached_a_output_blocks=True, prefetch_b_rows=(8192, 8256))
