"""Actual CFG addresses close complete operand/output tile ownership and K."""
from collections import Counter
from dataclasses import replace

import pytest

from mlir_oot.dense_schedule import select_resident_a_output_blocks
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.tables import isa
from test_resident_conv_command_loops import executed_commands


def source_tiles(emitter):
    """Recover source contraction tiles through every DMA/local/store address.

    This deliberately narrow WS oracle refuses seeds and segmented sources.
    It proves logical work, not accelerator timing or physical overlap.
    """
    shape = emitter.shape
    assert not shape.bias and emitter.input_view is None
    storage, accumulator, outputs = {}, {}, {}
    pending = active = None
    c_address = None
    counts = Counter()
    dma_bytes = Counter()
    config = []

    def packet(value):
        return value & 0xffffffff, value >> 48, (value >> 32) & 0xffff

    for name, encoded, pointers in executed_commands(emitter.build()):
        counts[name] += 1
        if encoded is None:
            continue
        funct, left, right = encoded
        if name.startswith('gemmini.config_'):
            config.append((name, left, right))
        elif name == 'gemmini.mvin':
            local, rows, cols = packet(right)
            pointer, = pointers
            assert pointer.base in (0, 1)
            pitch = shape.k if pointer.base == 0 else shape.n
            source_row, source_col = divmod(pointer.offset.value, pitch)
            dma_bytes[pointer.base] += rows * cols
            for col in range(0, cols, 16):
                storage[local + col] = (pointer.base, source_row, source_col + col,
                                        rows, min(16, cols - col))
        elif name == 'gemmini.preload':
            b, b_rows, b_cols = packet(left)
            c, c_rows, c_cols = packet(right)
            assert c & isa.ACC_ADDR_BIT and c_rows <= 16 and c_cols <= 16
            c_address = c & 0x3fff
            if not c & isa.ACC_ACCUMULATE_BIT:
                accumulator[c_address] = []
            else:
                assert c_address in accumulator
            pending = None if b == isa.GARBAGE_ADDR else storage[b]
            if pending is not None:
                assert pending[0] == 1 and pending[3:] == (b_rows, b_cols)
        elif name == 'gemmini.compute':
            a, a_rows, a_cols = packet(left)
            source_a = storage[a]
            if funct == isa.K_COMPUTE_PRELOADED:
                active = pending
            assert active is not None and source_a[0] == 0
            assert source_a[2] == active[1] and a_cols == active[3]
            assert source_a[3] == a_rows and source_a[4] == a_cols
            accumulator[c_address].append((source_a[1], active[2], source_a[2],
                                           a_rows, active[4], a_cols))
        elif name == 'gemmini.mvout':
            local, rows, cols = packet(right)
            pointer, = pointers
            assert pointer.base == 2 and local & isa.ACC_ADDR_BIT
            full = bool(local & isa.ACC_FULL_ROW_BIT)
            assert full == (shape.output_dtype == 'i32')
            output_row, output_col = divmod(pointer.offset.value // (4 if full else 1), shape.n)
            for col in range(0, cols, 16):
                logical = (output_row, output_col + col, rows, min(16, cols - col))
                assert logical not in outputs
                work = tuple(accumulator[(local & 0x3fff) + col])
                assert all(item[:2] == logical[:2] and item[3:5] == logical[2:] for item in work)
                assert [item[2] for item in work] == list(range(0, shape.k, 16))
                assert sum(item[5] for item in work) == shape.k
                outputs[logical] = work
    assert len(outputs) == ((shape.m + 15) // 16) * ((shape.n + 15) // 16)
    return outputs, counts, dma_bytes, config


@pytest.mark.parametrize('channels', [2, 4])
@pytest.mark.parametrize('prefetch', [False, True])
@pytest.mark.parametrize('shape', [
    Shape(785, 73, 20, bm=50, bn=1, output_dtype='i32', wide_b=True),
    Shape(529, 67, 32, bm=34, bn=1, wide_b=True, wide_store=True, scale=.001953125, relu=True),
    Shape(784, 512, 128, bm=49, bn=1, wide_b=True, wide_store=True, scale=.0040965937077999115),
])
def test_every_output_retains_exact_source_k_seed_and_original_a_storage(shape, prefetch, channels):
    shape = replace(shape, cache_a=True, reuse_b=True, prefetch_b=prefetch)
    control = GoldenGemm(shape)
    candidate, decision = select_resident_a_output_blocks(control, output_channel_tiles=channels)
    assert decision['applied'] and decision['performance'] == 'UNKNOWN'
    left, right = source_tiles(control), source_tiles(candidate)
    assert left[0] == right[0] and left[3] == right[3]
    assert left[2][0] == right[2][0] == shape.m * shape.k
    assert left[1]['gemmini.compute'] == right[1]['gemmini.compute']
    assert left[1]['gemmini.preload'] == right[1]['gemmini.preload']


def test_default_false_keeps_module_bytes_and_complete_options_clone():
    shape = Shape(49, 35, 20, bm=4, bn=2, cache_a=True, reuse_b=True)
    control = GoldenGemm(shape)
    assert str(control.build()) == str(GoldenGemm(shape, cached_a_output_blocks=False).build())
    candidate, _ = select_resident_a_output_blocks(control)
    assert str(candidate.build()) == str(candidate.with_emission_options().build())


@pytest.mark.parametrize('option', [1, None, 'yes'])
def test_storage_selection_requires_boolean(option):
    with pytest.raises(ValueError, match='boolean'):
        GoldenGemm(Shape(49, 35, 20, bm=4, bn=2, cache_a=True), cached_a_output_blocks=option)


def test_independent_output_blocks_do_not_reduce_complete_a_capacity_checks():
    with pytest.raises(ValueError, match='scratchpad'):
        GoldenGemm(Shape(4097, 16, 128, bm=1, bn=1, cache_a=True), cached_a_output_blocks=True)
    with pytest.raises(ValueError, match='accumulator'):
        GoldenGemm(Shape(785, 73, 20, bm=32, bn=4, cache_a=True), cached_a_output_blocks=True)
    with pytest.raises(ValueError, match='lower two banks'):
        GoldenGemm(Shape(1025, 73, 128, bm=16, bn=4, cache_a=True, prefetch_b=True), cached_a_output_blocks=True)


def test_existing_tail_order_and_unproved_lifetimes_refuse_without_mutation():
    shape = Shape(785, 73, 20, bm=50, bn=1, cache_a=True, reuse_b=True)
    control = GoldenGemm(shape, stationary_b_tail_before_last_full=True)
    candidate, decision = select_resident_a_output_blocks(control)
    assert candidate is control and not decision['applied']
    with pytest.raises(ValueError, match='complete A residency'):
        GoldenGemm(replace(shape, cache_a=False), cached_a_output_blocks=True)
