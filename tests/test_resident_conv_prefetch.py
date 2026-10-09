"""Independent static resource/lifetime checks for optional weight lookahead."""

from collections import Counter

import pytest
from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_resident_conv import GoldenResidentConv, choose_compact_resident
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def _logical_operations(generator):
    module = generator.build()
    module.verify()
    lower(module.clone()).verify()
    fn = module.body.block.first_op
    args = [StaticPointer(i, StaticInt(0, 64)) for i in range(3)]
    slots = {}
    last_preload = None
    counts = Counter()
    computes, outputs, weight_reads = [], [], []
    for step in trace_static_function(
        fn,
        args,
        observe=lambda op: isinstance(op, G._GemminiOp),
        pointer_index_bits=64,
    ):
        op = step.operation
        counts[op.name] += 1
        if isinstance(op, G.MvinOp) and op.a("load_id", 0) == 1:
            ptr = step.inputs[0]
            assert isinstance(ptr, StaticPointer) and ptr.base == 1
            for block in range((op.a("cols") + F.DIM - 1) // F.DIM):
                address = op.a("local") + block * F.DIM
                assert address >= 2 * F.SPAD_BANK_ROWS
                assert address + F.DIM <= F.SPAD_ROWS
                token = (ptr.offset.value + block * F.DIM, op.a("rows"))
                slots[address] = token
                weight_reads.append(token)
        elif isinstance(op, G.PreloadOp):
            address = op.a("bd")
            token = None if address == isa.GARBAGE_ADDR else slots[address]
            last_preload = (
                token,
                op.a("c"),
                op.a("c_rows"),
                op.a("c_cols"),
                op.a("bd_rows"),
                op.a("bd_cols"),
            )
        elif isinstance(op, G.ComputeOp):
            assert not step.inputs
            computes.append(
                (
                    op.a("a"),
                    op.a("a_rows"),
                    op.a("a_cols"),
                    op.a("accumulate", False),
                    last_preload,
                )
            )
        elif isinstance(op, G.MvoutOp):
            ptr = step.inputs[0]
            outputs.append(
                (ptr.base, ptr.offset.value, op.a("local"), op.a("rows"), op.a("cols"))
            )
    return counts, computes, outputs, weight_reads


@pytest.mark.parametrize(
    "shape,rows",
    [
        (ConvShape(3, 1, 16, 17, bn=2, output_dtype="i32"), 3),
        (ConvShape(5, 5, 32, 19, bn=2, output_dtype="i32"), 2),
        (ConvShape(7, 7, 64, 67, bn=4, output_dtype="i8", scale=0.01), 2),
    ],
)
def test_lookahead_preserves_actual_k_order_and_source_storage(shape, rows):
    control = GoldenResidentConv(shape, rows_per_tile=rows)
    candidate = GoldenResidentConv(shape, rows_per_tile=rows, prefetch_b=True)
    # Each full reserved slot is disjoint from A and from the other weight slot.
    width = shape.bn * F.DIM
    assert shape.cin // F.DIM * candidate.plane <= candidate.bases[0]
    assert candidate.bases[0] + width <= candidate.bases[1]
    assert candidate.bases[1] + width <= F.SPAD_ROWS
    assert _logical_operations(control) == _logical_operations(candidate)


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_lookahead_refuses_untyped_options(value):
    with pytest.raises(ValueError, match="prefetch selection must be boolean"):
        GoldenResidentConv(ConvShape(3, 3, 16, 16), prefetch_b=value)


def test_lookahead_refuses_unsupported_channel_loop_before_emission():
    with pytest.raises(ValueError, match="static reduction"):
        GoldenResidentConv(ConvShape(3, 3, 16, 16), prefetch_b=True, loop_channels=True)


def test_compiler_lookahead_option_derives_rows_and_refuses_missing_source_proof(
    tmp_path,
):
    from mlir_oot.captured_requant_bundle import build

    generator, options = choose_compact_resident(
        ConvShape(5, 5, 32, 19, bn=2), prefetch_b=True
    )
    assert options.rows_per_tile == 2 and options.prefetch_b
    assert not options.loop_channels and generator.prefetch_b
    generator.build().verify()
    with pytest.raises(ValueError, match="proved virtual padding"):
        build(
            tmp_path / "missing",
            tmp_path / "missing_tools",
            tmp_path / "output",
            resident_input_policy="compact_channel_planes_prefetch_b",
        )
    assert not (tmp_path / "output").exists()
