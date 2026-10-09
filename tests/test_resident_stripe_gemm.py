import pytest

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_resident_stripe_gemm import (
    GoldenResidentStripeGemm,
    command_counts,
)
from mlir_oot.ir.gemmini_dialect import ComputeOp, MvinOp
from mlir_oot.tables import rtl_facts as F


@pytest.mark.parametrize(
    "shape",
    [
        Shape(521, 73, 65, output_dtype="i32", bn=4),
        Shape(17, 19, 1, output_dtype="i32", bn=2),
        Shape(3136, 256, 64, bm=16, bn=4, scale=0.002, relu=True),
        Shape(784, 512, 128, bm=16, bn=4, scale=0.003),
    ],
)
def test_complete_input_partition_and_compute_extent(shape):
    g = GoldenResidentStripeGemm(shape)
    m = g.build()
    m.verify()
    covered = set()
    for op in m.walk():
        if isinstance(op, MvinOp) and op.a("load_id") == 0:
            for panel in range((op.a("cols") + F.DIM - 1) // F.DIM):
                base = op.a("local") + panel * g.plane
                rows = set(range(base, base + op.a("rows")))
                assert covered.isdisjoint(rows)
                assert max(rows) < g.input_rows
                covered.update(rows)
        if isinstance(op, ComputeOp):
            assert 0 <= op.a("a_max")
            assert (
                op.a("a_max") + op.a("a_rows")
                <= op.a("a_reserved_rows")
                == g.input_rows
            )
    expected = {ki * g.plane + r for ki in range(g.kt) for r in range(shape.m)}
    assert covered == expected
    assert g.input_rows <= g.bbase
    assert g.bbase + g.weight_rows == F.SPAD_ROWS
    lower(m).verify()


@pytest.mark.parametrize(
    "bad",
    [
        Shape(3136, 256, 128, bm=16, bn=4),
        Shape(17, 19, 65, output_dtype="i32", bias=True),
    ],
)
def test_invalid_resource_or_seed_refuses(bad):
    with pytest.raises(ValueError):
        GoldenResidentStripeGemm(bad)


@pytest.mark.parametrize("stripe", [0, 17, True, 1.5])
def test_invalid_output_stripe_refuses(stripe):
    with pytest.raises(ValueError):
        GoldenResidentStripeGemm(Shape(521, 73, 65, bn=4), stripe_tiles=stripe)


def test_full_input_remains_once_across_n_blocks():
    counts = command_counts(Shape(3136, 256, 64, bm=16, bn=4))
    assert counts["resident_input_rows"] == 12544
    assert counts["resident_weight_rows"] == 256
    assert counts["stripe_tiles"] == 16
    assert counts["mvin_a"] == 196
    assert counts["mvin_b"] == 16
    assert counts["compute"] == 196 * 16 * 4


def test_original_geometry_reduces_stores_with_exact_reserved_extents():
    from collections import Counter

    from test_resident_conv_command_loops import executed_commands

    from mlir_oot.golden_gemm import GoldenGemm

    source = Shape(
        784,
        512,
        128,
        bm=49,
        bn=1,
        cache_a=True,
        prefetch_b=True,
        wide_b=True,
        reuse_b=True,
        wide_store=True,
        scale=0.0040965937077999115,
    )
    selected = Shape(784, 512, 128, bm=16, bn=4, scale=source.scale)
    g = GoldenResidentStripeGemm(selected, stripe_tiles=16)
    assert g.input_rows == 6272 and g.weight_rows == 512
    assert g.bbase == 15872 and g.input_rows <= 2 * F.SPAD_BANK_ROWS
    assert g.bbase // F.SPAD_BANK_ROWS == 3
    assert g.stripe_tiles * g.shape.bn * F.DIM == F.ACC_ROWS == 1024
    counts = Counter(x[0] for x in executed_commands(g.build()))
    old = Counter(x[0] for x in executed_commands(GoldenGemm(source).build()))
    assert old["gemmini.mvout"] == 1568 and counts["gemmini.mvout"] == 392
    assert old["gemmini.compute"] == counts["gemmini.compute"] == 12544
    assert old["gemmini.preload"] == counts["gemmini.preload"] == 12544
    assert counts["gemmini.mvin"] == 162


@pytest.mark.parametrize(
    "shape",
    [
        Shape(521, 73, 65, bm=16, bn=4, output_dtype="i32"),
        Shape(37, 69, 33, bm=8, bn=3, scale=0.005),
        Shape(784, 512, 128, bm=16, bn=4, scale=0.0040965937077999115),
    ],
)
def test_actual_cfg_preserves_every_output_reduction_sequence(shape):
    from merlin.llvmlower.static_llvm_cfg import (
        StaticInt,
        StaticPointer,
        trace_static_function,
    )

    from mlir_oot.ir import gemmini_dialect as G
    from mlir_oot.tables import isa

    g = GoldenResidentStripeGemm(
        shape, stripe_tiles=min(shape.bm, F.ACC_ROWS // (shape.bn * F.DIM))
    )
    module = g.build()
    module.verify()
    fn = module.body.block.first_op
    sequence = {}
    column_base = 0
    active_k = None
    active_d = None
    args = [StaticPointer(i, StaticInt(0, 64)) for i in range(3)]
    for step in trace_static_function(
        fn, args, observe=lambda op: isinstance(op, G._GemminiOp), pointer_index_bits=64
    ):
        op = step.operation
        if isinstance(op, G.MvinOp) and op.a("load_id") == 1:
            pointer = next(x for x in step.inputs if isinstance(x, StaticPointer))
            # Full-K B panels are loaded in increasing K for one N group.
            local_k = (op.a("local") - g.bbase) // (g.shape.bn * F.DIM)
            assert pointer.offset.value // shape.n == local_k * F.DIM
            column_base = pointer.offset.value % shape.n - (
                (op.a("local") - g.bbase) % (g.shape.bn * F.DIM)
            )
        elif isinstance(op, G.PreloadOp):
            if op.a("bd") != isa.GARBAGE_ADDR:
                active_k = (op.a("bd") - g.bbase) // (g.shape.bn * F.DIM)
                active_d = ((op.a("bd") - g.bbase) // F.DIM) % g.shape.bn
        elif isinstance(op, G.ComputeOp):
            address = step.inputs[0].value
            assert address // g.plane == active_k
            row_tile = (address % g.plane) // F.DIM
            key = (row_tile, column_base // F.DIM + active_d)
            sequence.setdefault(key, []).append(active_k)
    assert len(sequence) == ((shape.m + 15) // 16) * ((shape.n + 15) // 16)
    assert all(ks == list(range(g.kt)) for ks in sequence.values())
