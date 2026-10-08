"""Public source/CFG gates, independent of any captured-model experiment."""

from collections import Counter

import pytest
from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)
from xdsl.dialects import llvm

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_product_sum import GoldenProductSum, operand_residency_plan
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F

PAIRS = (
    ((0, 0),),
    ((0, 1), (1, 0)),
    ((0, 2), (1, 1), (2, 0)),
    ((1, 2), (2, 1)),
    ((2, 2),),
)


def generator(shape, pairs=PAIRS[2], selected=True):
    return GoldenProductSum(
        shape,
        pairs=pairs,
        lhs_planes=3,
        rhs_planes=3,
        absolute_bound=len(pairs) * shape.k * 128**2,
        resident_operands=selected,
    )


def commands(module):
    fn = next(op for op in module.walk() if isinstance(op, llvm.FuncOp))
    return trace_static_function(
        fn,
        [StaticPointer(i, StaticInt(0, 64)) for i in range(3)],
        observe=lambda op: isinstance(op, G._GemminiOp),
        pointer_index_bits=64,
    )


def counts(module):
    result = Counter()
    for step in commands(module):
        op = step.operation
        result[op.name] += 1
        if isinstance(op, G.MvinOp):
            result["A_bytes" if op.a("load_id") == 0 else "B_bytes"] += op.a(
                "rows"
            ) * op.a("cols")
        elif isinstance(op, G.MvoutOp):
            result["C_bytes"] += 4 * op.a("rows") * op.a("cols")
    return result


def execute(module, shape):
    """Independent primitive interpreter; poison unwritten rows and DMA tails."""
    m, n, k = shape.m, shape.n, shape.k
    inputs = [
        [
            ((i * 29 + base * 17) % 256) - 128
            for i in range(3 * (m * k if base == 0 else k * n))
        ]
        for base in (0, 1)
    ]
    output = [None] * (m * n)
    spad, acc, strides, pending, stationary = {}, {}, {}, None, None
    for step in commands(module):
        op = step.operation
        if isinstance(op, G.ConfigLdOp):
            strides[op.a("load_id")] = op.a("stride")
        elif isinstance(op, G.MvinOp):
            pointer = step.inputs[0]
            assert isinstance(pointer, StaticPointer) and pointer.base in (0, 1)
            for row in range(op.a("rows")):
                for col in range(op.a("cols")):
                    source = pointer.offset.value + row * strides[op.a("load_id")] + col
                    assert 0 <= source < len(inputs[pointer.base])
                    local = op.a("local") + col // F.DIM * F.DIM + row
                    assert 0 <= local < F.SPAD_ROWS
                    spad[local, col % F.DIM] = inputs[pointer.base][source]
        elif isinstance(op, G.PreloadOp):
            dynamic = dict(zip(op.dynamic_rows(), step.inputs, strict=True))
            bd = dynamic["bd"].value if "bd" in dynamic else op.a("bd")
            if bd != isa.GARBAGE_ADDR:
                stationary = [
                    [spad[bd + row, col] for col in range(op.a("bd_cols"))]
                    for row in range(op.a("bd_rows"))
                ]
            assert stationary is not None
            pending = op
        elif isinstance(op, G.ComputeOp):
            assert pending is not None
            a = step.inputs[0].value if op.operands_ else op.a("a")
            c = pending.a("c")
            accumulate = bool(c & (1 << 30))
            base = c & ((1 << 29) - 1)
            for row in range(op.a("a_rows")):
                for col in range(pending.a("c_cols")):
                    value = sum(
                        spad[a + row, ki] * stationary[ki][col]
                        for ki in range(op.a("a_cols"))
                    )
                    acc[base + row, col] = value + (
                        acc[base + row, col] if accumulate else 0
                    )
        elif isinstance(op, G.MvoutOp):
            pointer = step.inputs[0]
            assert isinstance(pointer, StaticPointer) and pointer.base == 2
            base = op.a("local") & ((1 << 29) - 1)
            for row in range(op.a("rows")):
                for col in range(op.a("cols")):
                    destination = pointer.offset.value // 4 + row * n + col
                    assert (
                        0 <= destination < len(output) and output[destination] is None
                    )
                    output[destination] = acc[base + row, col]
    return inputs, output


@pytest.mark.parametrize(
    "m,n,k,bm,bn,reuse",
    [(17, 67, 19, 2, 3, True), (5, 7, 3, 1, 1, False), (33, 21, 37, 2, 2, True)],
)
@pytest.mark.parametrize("pairs", [PAIRS[2], ((2, 2), (2, 2)), ((1, 0),)])
def test_independent_exact_tails_sparse_and_repeated_pairs(
    m, n, k, bm, bn, reuse, pairs
):
    shape = Shape(m, n, k, "i32", bm=bm, bn=bn, reuse_b=reuse, wide_b=True)
    inputs, output = execute(generator(shape, pairs).build(), shape)
    expected = [
        sum(
            inputs[0][ap * m * k + row * k + ki] * inputs[1][bp * k * n + ki * n + col]
            for ap, bp in pairs
            for ki in range(k)
        )
        for row in range(m)
        for col in range(n)
    ]
    assert output == expected


@pytest.mark.parametrize(
    "m,n,k,bm,bn", [(256, 512, 64, 4, 16), (256, 64, 192, 4, 4), (256, 64, 128, 4, 4)]
)
def test_all_degree_complete_request_counts_and_unchanged_work(m, n, k, bm, bn):
    shape = Shape(m, n, k, "i32", bm=bm, bn=bn, reuse_b=True, wide_b=True)
    cold, resident = Counter(), Counter()
    for pairs in PAIRS:
        cold.update(counts(generator(shape, pairs, False).build()))
        resident.update(counts(generator(shape, pairs).build()))
    assert resident["A_bytes"] == 9 * m * k
    assert resident["B_bytes"] == 9 * k * n
    assert resident["C_bytes"] == cold["C_bytes"] == 5 * 4 * m * n
    for name in (
        "gemmini.compute",
        "gemmini.preload",
        "gemmini.mvout",
        "gemmini.fence",
        "gemmini.flush",
        "gemmini.config_ex",
        "gemmini.config_ld",
        "gemmini.config_st",
    ):
        assert resident[name] == cold[name]
    assert resident["B_bytes"] < cold["B_bytes"]
    assert resident["A_bytes"] <= cold["A_bytes"]


def test_default_exact_and_oversize_explicit_fallback():
    shape = Shape(8, 64, 2048, "i32", reuse_b=True, wide_b=True)
    plan = operand_residency_plan(shape, PAIRS[2])
    assert not plan.admitted
    with pytest.raises(ValueError, match="exceed target storage"):
        generator(shape).build()
    a = generator(shape, selected=False).build()
    b = GoldenProductSum(
        shape,
        pairs=PAIRS[2],
        lhs_planes=3,
        rhs_planes=3,
        absolute_bound=3 * 2048 * 128**2,
    ).build()
    assert str(a) == str(b)
    lower(a)


def test_lowered_cfg_bounds_and_no_target_ops():
    module = generator(Shape(17, 67, 19, "i32", reuse_b=True, wide_b=True)).build()
    lower(module)
    assert not any(isinstance(op, G._GemminiOp) for op in module.walk())


def test_only_referenced_planes_reserve_storage():
    shape = Shape(8, 64, 2048, "i32", reuse_b=True, wide_b=True)
    plan = operand_residency_plan(shape, ((2, 2),))
    assert plan.admitted and plan.lhs_indices == plan.rhs_indices == (2,)
    lower(generator(shape, ((2, 2),)).build())


def test_declared_dynamic_bound_cannot_hide_actual_row():
    from xdsl.dialects.builtin import IntegerAttr, i64

    module = generator(Shape(17, 67, 19, "i32", reuse_b=True, wide_b=True)).build()
    op = next(
        op for op in module.walk() if isinstance(op, G.PreloadOp) and op.operands_
    )
    op.attributes["bd_min"] = IntegerAttr(op.a("bd_min") + F.DIM, i64)
    with pytest.raises(ValueError, match="executed dynamic B"):
        lower(module)


def test_batched_plane_abi_is_explicitly_refused():
    with pytest.raises(ValueError, match="batched ABI"):
        generator(Shape(1, 1, 1, "i32")).build_batched(2)


def test_incompatible_bank_placement_refuses_without_changing_default():
    shape = Shape(17, 67, 19, "i32", separate_b_bank=True)
    with pytest.raises(ValueError, match="bank placement"):
        generator(shape).build()
    generator(shape, selected=False).build().verify()


@pytest.mark.parametrize("value", [1, "true", None])
def test_selection_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        generator(Shape(1, 1, 1, "i32"), selected=value)


def test_zero_metadata_provider_refuses_competing_panel_placement():
    from mlir_oot.golden_zero_product_sum import GoldenZeroProductSum

    shape = Shape(33, 21, 37, "i32", bm=2, bn=2, reuse_b=True, wide_b=True)
    options = {
        "pairs": PAIRS[2],
        "lhs_planes": 3,
        "rhs_planes": 3,
        "absolute_bound": 3 * shape.k * 128**2,
    }
    with pytest.raises(ValueError, match="distinct panel placement"):
        GoldenZeroProductSum(shape, **options, resident_operands=True)
    original = GoldenZeroProductSum(shape, **options).build()
    explicit_default = GoldenZeroProductSum(
        shape, **options, resident_operands=False
    ).build()
    assert str(original) == str(explicit_default)
    lower(original)
