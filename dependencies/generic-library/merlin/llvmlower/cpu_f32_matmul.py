"""Explicit source-compatible f32 matmul with ordered fused accumulation.

This policy is opt-in: callers establish the source's increasing-K FMA order,
zero initialization, finite intermediates and RNE. It does not authorize a
global contraction or reassociation change, or imply every CPU GEMM uses it.
"""

from __future__ import annotations

CPU_F32_ORDERED_FMA_POLICY = "torch_2_10_cpu_sgemm_ordered_fma"


def build_cpu_f32_matmul_ordered_fma(
    lhs, rhs, *, backend_policy: str, allow_contraction: bool = False, assume_finite_intermediates: bool = False
):
    """Return detached ops and a zero-seeded static f32 matrix product.

    Leading batch dimensions must match exactly; no implicit broadcasting or
    transpose is supported. An explicit scf loop preserves increasing K order.
    """
    from xdsl.dialects import arith, math, scf, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IndexType, TensorType, f32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    if backend_policy != CPU_F32_ORDERED_FMA_POLICY:
        raise ValueError("explicit source ordered-FMA policy required")
    if not allow_contraction or not assume_finite_intermediates:
        raise ValueError("finite intermediates and source contraction contract required")
    if any(not isinstance(v.type, TensorType) or v.type.element_type != f32 for v in (lhs, rhs)):
        raise TypeError("f32 tensor operands required")
    ls, rs = lhs.type.get_shape(), rhs.type.get_shape()
    if len(ls) < 2 or len(ls) != len(rs) or any(n <= 0 for n in (*ls, *rs)):
        raise ValueError("equal static positive operand ranks at least two required")
    if ls[:-2] != rs[:-2] or ls[-1] != rs[-2]:
        raise ValueError("matching batch and contraction dimensions required")
    shape = (*ls[:-2], ls[-2], rs[-1])
    output_type = TensorType(f32, shape)
    empty = tensor.EmptyOp((), output_type)
    body = Block(arg_types=[f32])
    indices = [L.IndexOp(i) for i in range(len(shape))]
    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    start = arith.ConstantOp.from_int_and_width(0, IndexType())
    end = arith.ConstantOp.from_int_and_width(ls[-1], IndexType())
    step = arith.ConstantOp.from_int_and_width(1, IndexType())
    body.add_ops([*indices, zero, start, end, step])
    batch = [x.result for x in indices[:-2]]
    inner = Block(arg_types=[IndexType(), f32])
    a = tensor.ExtractOp(lhs, [*batch, indices[-2].result, inner.args[0]], f32)
    b = tensor.ExtractOp(rhs, [*batch, inner.args[0], indices[-1].result], f32)
    fused = math.FmaOp(a.result, b.result, inner.args[1])
    inner.add_ops([a, b, fused, scf.YieldOp(fused.result)])
    loop = scf.ForOp(start.result, end.result, step.result, [zero.result], Region(inner))
    body.add_ops([loop, L.YieldOp(loop.results[0])])
    result = L.GenericOp(
        inputs=(),
        outputs=(empty.tensor,),
        body=Region(body),
        indexing_maps=ArrayAttr([AffineMapAttr(AffineMap.identity(len(shape)))]),
        iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(shape)),
        result_types=(output_type,),
    )
    return [empty, result], result.results[0]
