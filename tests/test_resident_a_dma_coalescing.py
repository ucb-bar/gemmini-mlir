"""Decode cached input DMAs and their complete subsequent compute stream."""

from collections import defaultdict

import numpy as np
import pytest
from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def commands(generator):
    module = generator.build()
    module.verify()
    lower(module.clone()).verify()
    return list(
        trace_static_function(
            module.body.block.first_op,
            [StaticPointer(i, StaticInt(0, 64)) for i in range(3)],
            observe=lambda op: isinstance(op, G._GemminiOp),
            pointer_index_bits=64,
        )
    )


def fingerprint(step):
    return step.operation.name, step.operation.attributes, step.inputs


@pytest.mark.parametrize(
    "shape,slots",
    [
        (
            Shape(
                33,
                73,
                65,
                output_dtype="i32",
                bm=3,
                bn=4,
                cache_a=True,
                prefetch_b=True,
                wide_b=True,
                reuse_b=True,
                wide_store=True,
            ),
            None,
        ),
        (
            Shape(
                17,
                17,
                48,
                output_dtype="i32",
                bm=2,
                bn=2,
                cache_a=True,
                prefetch_b=True,
                wide_b=True,
                reuse_b=True,
                wide_store=True,
            ),
            (3072, 4096),
        ),
        (
            Shape(1, 5, 17, output_dtype="i32", bm=1, bn=1, cache_a=True, reuse_b=True),
            None,
        ),
    ],
)
def test_decoded_wide_input_partition_and_complete_arithmetic(shape, slots):
    control = commands(GoldenGemm(shape, prefetch_b_rows=slots))
    candidate = commands(
        GoldenGemm(shape, prefetch_b_rows=slots, resident_a_load_tiles=4)
    )
    # Every command after residency, including B lookahead and store semantics,
    # has the same operand and pointer. Only the adjacent A DMA packet changes.
    is_a = lambda s: isinstance(s.operation, G.MvinOp) and s.operation.a("load_id") == 0
    assert [fingerprint(s) for s in control if not is_a(s)] == [
        fingerprint(s) for s in candidate if not is_a(s)
    ]
    kt = (shape.k + F.DIM - 1) // F.DIM
    mt = (shape.m + F.DIM - 1) // F.DIM
    assert sum(is_a(s) for s in control) == mt * kt
    assert sum(is_a(s) for s in candidate) == mt * ((kt + 3) // 4)
    a = ((np.arange(shape.m * shape.k) * 37) % 256 - 128).astype(np.int64)
    b = ((np.arange(shape.k * shape.n) * 53) % 256 - 128).astype(np.int64)
    scratch, acc, order, loads = {}, {}, defaultdict(list), {}
    loaded_a = set()
    written = set()
    output = np.empty(shape.m * shape.n, dtype=np.int64)
    for step in candidate:
        op = step.operation
        if isinstance(op, G.ConfigLdOp):
            loads[op.a("load_id")] = (op.a("stride"), op.a("block_stride", F.DIM))
        elif isinstance(op, G.MvinOp):
            ptr = step.inputs[0]
            ident = op.a("load_id")
            stride, block_stride = loads[ident]
            for block in range((op.a("cols") + F.DIM - 1) // F.DIM):
                width = min(F.DIM, op.a("cols") - block * F.DIM)
                for row in range(op.a("rows")):
                    address = op.a("local") + block * block_stride + row
                    offset = ptr.offset.value + row * stride + block * F.DIM
                    values = np.zeros(F.DIM, dtype=np.int64)
                    if ident == 0:
                        assert ptr.base == 0
                        assert address not in loaded_a
                        loaded_a.add(address)
                        m, k = divmod(offset, shape.k)
                        assert m < shape.m and k + width <= shape.k
                        expected_address = (
                            (m // F.DIM) * kt + k // F.DIM
                        ) * F.DIM + m % F.DIM
                        assert address == expected_address
                        values[:width] = a[offset : offset + width]
                        scratch[address] = (values, None)
                    else:
                        assert ptr.base == 1 and ident == 1
                        assert address >= shape.bm * kt * F.DIM
                        assert offset + width <= len(b)
                        values[:width] = b[offset : offset + width]
                        scratch[address] = (values, ptr.offset.value // shape.n)
        elif isinstance(op, G.PreloadOp):
            if op.a("bd") != isa.GARBAGE_ADDR:
                active_b = np.stack(
                    [scratch[op.a("bd") + k][0] for k in range(op.a("bd_rows"))]
                )
                active_k = scratch[op.a("bd")][1]
            destination = op
        elif isinstance(op, G.ComputeOp):
            start = op.a("a") if not step.inputs else step.inputs[0].value
            for lane in range(op.a("a_rows")):
                assert start + lane in loaded_a
                left = scratch[start + lane][0][: op.a("a_cols")]
                address = (destination.a("c") & 0x3FFF) + lane
                if not destination.a("c") & isa.ACC_ACCUMULATE_BIT:
                    acc[address] = np.zeros(F.DIM, dtype=np.int64)
                    order[address] = []
                acc[address] += left @ active_b[: op.a("a_cols")]
                order[address].append(active_k)
        elif isinstance(op, G.MvoutOp):
            ptr = step.inputs[0]
            assert ptr.base == 2
            for block in range((op.a("cols") + F.DIM - 1) // F.DIM):
                width = min(F.DIM, op.a("cols") - block * F.DIM)
                for row in range(op.a("rows")):
                    address = (op.a("local") & 0x3FFF) + block * F.DIM + row
                    assert order[address] == list(range(0, shape.k, F.DIM))
                    offset = ptr.offset.value // 4 + row * shape.n + block * F.DIM
                    assert not written.intersection(range(offset, offset + width))
                    written.update(range(offset, offset + width))
                    output[offset : offset + width] = acc[address][:width]
    assert len(loaded_a) == shape.m * kt
    assert written == set(range(output.size))
    np.testing.assert_array_equal(
        output.reshape(shape.m, shape.n),
        a.reshape(shape.m, shape.k) @ b.reshape(shape.k, shape.n),
    )


@pytest.mark.parametrize("value", [0, 5, -1, True, 1.5, "4"])
def test_grouping_field_requires_supported_integer(value):
    with pytest.raises(ValueError, match="DMA grouping"):
        GoldenGemm(Shape(1, 1, 17, cache_a=True), resident_a_load_tiles=value)


def test_grouping_requires_proven_complete_residency():
    with pytest.raises(ValueError, match="complete cached A"):
        GoldenGemm(Shape(1, 1, 17), resident_a_load_tiles=4)
    with pytest.raises(ValueError, match="scratchpad"):
        GoldenGemm(
            Shape(256, 16, 2048, cache_a=True, bm=16, bn=1), resident_a_load_tiles=4
        )


def test_default_is_explicit_single_tile_stream():
    shape = Shape(17, 17, 48, cache_a=True, bm=2, bn=2)
    assert [fingerprint(s) for s in commands(GoldenGemm(shape))] == [
        fingerprint(s) for s in commands(GoldenGemm(shape, resident_a_load_tiles=1))
    ]


def test_general_selector_preserves_prefetch_slots_and_source_fields():
    from mlir_oot.dense_schedule import select_coalesced_resident_a, select_kernel

    shape = Shape(
        17,
        73,
        65,
        output_dtype="i32",
        bm=2,
        bn=4,
        cache_a=True,
        prefetch_b=True,
        wide_b=True,
        reuse_b=True,
    )
    original = GoldenGemm(shape, prefetch_b_rows=(3072, 4096))
    candidate, decision = select_coalesced_resident_a(original)
    assert candidate.shape == original.shape
    assert candidate.prefetch_b_rows == original.prefetch_b_rows
    assert candidate.resident_a_load_tiles == 4
    assert decision["applied"] and not decision["timing_claim"]
    assert decision["control_input_dma_commands"] == 10
    assert decision["candidate_input_dma_commands"] == 4
    assert decision["input_requested_bytes"] == 17 * 65
    kept, again = select_coalesced_resident_a(candidate)
    assert kept is candidate and not again["applied"]
    plain, _ = select_kernel(shape)
    enabled, kind = select_kernel(shape, resident_a_load_coalescing=True)
    assert plain.resident_a_load_tiles == 1
    assert enabled.shape == plain.shape and enabled.resident_a_load_tiles == 4
    assert "resident_a_load_coalescing" in kind


def test_selector_keeps_missing_residency_single_tile_and_existing_b_rule():
    from mlir_oot.b_slot_placement import select_remaining_b_slots
    from mlir_oot.dense_schedule import select_coalesced_resident_a

    for shape in (Shape(17, 17, 65), Shape(1, 5, 13, cache_a=True)):
        original = GoldenGemm(shape)
        result, decision = select_coalesced_resident_a(original)
        assert result is original and not decision["applied"]
    original = GoldenGemm(
        Shape(17, 17, 65, cache_a=True, bm=2, bn=2), resident_a_load_tiles=4
    )
    result, decision = select_remaining_b_slots(original)
    assert decision["applied"] and result.resident_a_load_tiles == 4
    result.build().verify()


def test_named_capture_option_requires_boolean_before_io(tmp_path):
    from mlir_oot.captured_requant_bundle import build
    from mlir_oot.dense_schedule import select_kernel

    output = tmp_path / "result"
    with pytest.raises(ValueError, match="boolean selection"):
        build(
            tmp_path / "absent", tmp_path / "llvm", output, resident_a_load_coalescing=1
        )
    assert not output.exists()
    with pytest.raises(ValueError, match="boolean selection"):
        select_kernel(Shape(1, 1, 16), resident_a_load_coalescing=1)
