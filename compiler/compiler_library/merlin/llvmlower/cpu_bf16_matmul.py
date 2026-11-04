"""Explicit source CPUBlas BF16 dot-product schedule, including reduction tails.

PyTorch 2.10 BlasKernel.cpp gemm_notrans_ uses four f32 partial accumulators,
assigns all trailing products to partial zero, then left-folds the partials.
The caller establishes actual CPUBlas fallback dispatch and finite/RNE inputs;
this policy is never selected automatically for generic mathematical matmul.
Separate strict multiply/add operations preserve the selected source order.
"""

from __future__ import annotations

from .cpu_bf16_conv import CPU_BF16_BLAS_ILP4_POLICY


def build_cpu_bf16_matmul_ilp4(
    lhs, rhs, *, backend_policy: str, allow_reassociation: bool = False, assume_finite_intermediates: bool = False
):
    """Build a static BF16 matmul with identical leading batch dimensions.

    Returns detached operations and the BF16 output. Broadcasting, transpose
    semantics and source dispatch must be resolved by the caller before use.
    """
    from xdsl.dialects import arith, scf, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IndexType, TensorType, bf16, f32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    if backend_policy != CPU_BF16_BLAS_ILP4_POLICY:
        raise ValueError("explicit pinned CPUBlas BF16 policy required")
    if not allow_reassociation or not assume_finite_intermediates:
        raise ValueError("finite intermediates and source reassociation contract required")
    if any(not isinstance(v.type, TensorType) or v.type.element_type != bf16 for v in (lhs, rhs)):
        raise TypeError("BF16 tensor operands required")
    ls, rs = lhs.type.get_shape(), rhs.type.get_shape()
    if len(ls) < 2 or len(ls) != len(rs) or any(n <= 0 for n in (*ls, *rs)):
        raise ValueError("equal static positive operand ranks at least two required")
    if ls[:-2] != rs[:-2] or ls[-1] != rs[-2]:
        raise ValueError("matching batch and contraction dimensions required")
    shape = (*ls[:-2], ls[-2], rs[-1])
    rank, k = len(shape), ls[-1]
    output_type = TensorType(bf16, shape)
    empty = tensor.EmptyOp((), output_type)
    body = Block(arg_types=[bf16])
    indices = [L.IndexOp(i) for i in range(rank)]
    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    start = arith.ConstantOp.from_int_and_width(0, IndexType())
    end = arith.ConstantOp.from_int_and_width(k - k % 4, IndexType())
    step = arith.ConstantOp.from_int_and_width(4, IndexType())
    body.add_ops([*indices, zero, start, end, step])
    batch = [x.result for x in indices[:-2]]

    def product(block, position):
        a = tensor.ExtractOp(lhs, [*batch, indices[-2].result, position], bf16)
        b = tensor.ExtractOp(rhs, [*batch, position, indices[-1].result], bf16)
        af, bf = arith.ExtFOp(a.result, f32), arith.ExtFOp(b.result, f32)
        p = arith.MulfOp(af.result, bf.result)
        block.add_ops([a, b, af, bf, p])
        return p.result

    loop_body = Block(arg_types=[IndexType(), f32, f32, f32, f32])
    sums = []
    for lane in range(4):
        offset = arith.ConstantOp.from_int_and_width(lane, IndexType())
        position = arith.AddiOp(loop_body.args[0], offset.result)
        loop_body.add_ops([offset, position])
        p = product(loop_body, position.result)
        add = arith.AddfOp(loop_body.args[lane + 1], p)
        loop_body.add_op(add)
        sums.append(add.result)
    loop_body.add_op(scf.YieldOp(*sums))
    loop = scf.ForOp(start.result, end.result, step.result, [zero.result] * 4, Region(loop_body))
    body.add_op(loop)
    sums = list(loop.results)
    for tail in range(k - k % 4, k):
        position = arith.ConstantOp.from_int_and_width(tail, IndexType())
        body.add_op(position)
        p = product(body, position.result)
        add = arith.AddfOp(sums[0], p)
        body.add_op(add)
        sums[0] = add.result
    total = sums[0]
    for item in sums[1:]:
        add = arith.AddfOp(total, item)
        body.add_op(add)
        total = add.result
    narrow = arith.TruncFOp(total, bf16)
    body.add_ops([narrow, L.YieldOp(narrow.result)])
    result = L.GenericOp(
        inputs=(),
        outputs=(empty.tensor,),
        body=Region(body),
        indexing_maps=ArrayAttr([AffineMapAttr(AffineMap.identity(rank))]),
        iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * rank),
        result_types=(output_type,),
    )
    return [empty, result], result.results[0]
