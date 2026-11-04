"""Explicit PyTorch 2.10 AVX2 BF16 LayerNorm arithmetic for a pinned width.

Default mathematical LayerNorm is unchanged. The policy retains Welford chunk
and lane order, scalar moment merges, and the two fused affine updates. It
requires finite intermediate arithmetic and round-to-nearest-even; exception
flags and arbitrary backend/ISA equivalence are not promised.

Source: PyTorch v2.10.0 aten/src/ATen/native/cpu/{moments_utils.h,
layer_norm_kernel.cpp}. Initial admission is width768; other widths are refused
until their chunk/tail paths have executed numerical coverage.
"""

from __future__ import annotations

import math as pymath
import struct

CPU_LAYER_NORM_BF16_AVX2_POLICY = "torch_2_10_cpu_layer_norm_bf16_avx2"


def build_cpu_layer_norm(
    x,
    gamma,
    beta,
    *,
    epsilon: float,
    backend_policy: str,
    assume_finite_intermediates: bool = False,
    allow_reassociation: bool = False,
):
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        BFloat16Type,
        DenseIntOrFPElementsAttr,
        FloatAttr,
        IntegerAttr,
        StringAttr,
        TensorType,
        f32,
        i64,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

    if backend_policy != CPU_LAYER_NORM_BF16_AVX2_POLICY:
        raise ValueError("explicit pinned BF16 LayerNorm policy required")
    if not assume_finite_intermediates or not allow_reassociation:
        raise ValueError("finite-intermediate and reassociation contracts required")
    if not pymath.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite positive epsilon required")
    try:
        epsilon_f32 = struct.unpack("f", struct.pack("f", epsilon))[0]
    except OverflowError as error:
        raise ValueError("epsilon must be representable as finite positive f32") from error
    if not pymath.isfinite(epsilon_f32) or epsilon_f32 <= 0:
        raise ValueError("epsilon must be representable as finite positive f32")
    if not isinstance(x.type, TensorType) or not isinstance(x.type.element_type, BFloat16Type):
        raise TypeError("BF16 tensor input required")
    shape = list(x.type.get_shape())
    if len(shape) < 2 or any(d <= 0 for d in shape) or shape[-1] != 768:
        raise ValueError("audited static width768 with at least one outer dimension required")
    for p in (gamma, beta):
        if not isinstance(p.type, TensorType) or list(p.type.get_shape()) != [768]:
            raise ValueError("width768 affine vectors required")
        if p.type.element_type not in (x.type.element_type, f32):
            raise TypeError("BF16 or f32 affine parameters required")
    if gamma.type.element_type != beta.type.element_type:
        raise TypeError("affine parameter dtypes must agree")
    bf16 = x.type.element_type
    rows = pymath.prod(shape[:-1])
    ops = []

    def emit(o):
        ops.append(o)
        return o.results[0]

    def amap(rank, axes):
        return AffineMapAttr(AffineMap(rank, 0, tuple(AffineDimExpr(a) for a in axes)))

    def constant(v):
        return arith.ConstantOp(FloatAttr(v, f32))

    def splat(s):
        z = emit(constant(0.0))
        return emit(tensor.SplatOp(z, [], TensorType(f32, s)))

    groups = ArrayAttr(
        [ArrayAttr([IntegerAttr(i, i64) for i in range(len(shape) - 1)]), ArrayAttr([IntegerAttr(len(shape) - 1, i64)])]
    )
    flat = (
        x
        if len(shape) == 2
        else emit(
            tensor.CollapseShapeOp.build(
                operands=[x], properties={"reassociation": groups}, result_types=[TensorType(bf16, [rows, 768])]
            )
        )
    )

    def point(inputs, s, fn, maps=None, outputs=2):
        empties = [emit(tensor.EmptyOp((), TensorType(f32, s))) for _ in range(outputs)]
        block = Block(arg_types=[a.type.element_type for a in inputs] + [f32] * outputs)
        body, values = fn(list(block.args[: len(inputs)]))
        block.add_ops([*body, L.YieldOp(*values)])
        maps = maps or [amap(len(s), range(len(s)))] * len(inputs)
        o = L.GenericOp(
            inputs=inputs,
            outputs=empties,
            body=Region(block),
            indexing_maps=ArrayAttr([*maps, *([amap(len(s), range(len(s)))] * outputs)]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(s)),
            result_types=[a.type for a in empties],
        )
        ops.append(o)
        return list(o.results)

    coeff = emit(
        arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(f32, [16]), [1.0 / (j + 1) for j in range(16)]))
    )

    def partial(chunk):
        tile = emit(tensor.ExtractSliceOp.from_static_parameters(flat, [0, chunk * 256], [rows, 256]))
        reassoc = ArrayAttr([ArrayAttr([IntegerAttr(0, i64)]), ArrayAttr([IntegerAttr(i, i64) for i in [1, 2, 3]])])
        tile = emit(tensor.ExpandShapeOp(tile, (), reassoc, [rows, 16, 2, 8], TensorType(bf16, [rows, 16, 2, 8])))
        means, vars = splat([rows, 2, 8]), splat([rows, 2, 8])
        block = Block(arg_types=[bf16, f32, f32, f32])
        wide = arith.ExtFOp(block.args[0], f32)
        delta = arith.SubfOp(wide.result, block.args[2])
        mean = math.FmaOp(delta.result, block.args[1], block.args[2])
        delta2 = arith.SubfOp(wide.result, mean.result)
        var = math.FmaOp(delta.result, delta2.result, block.args[3])
        block.add_ops([wide, delta, mean, delta2, var, L.YieldOp(mean.result, var.result)])
        o = L.GenericOp(
            inputs=(tile, coeff),
            outputs=(means, vars),
            body=Region(block),
            indexing_maps=ArrayAttr([amap(4, [0, 3, 1, 2]), amap(4, [3]), amap(4, [0, 1, 2]), amap(4, [0, 1, 2])]),
            iterator_types=ArrayAttr(
                [L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 3 + [L.IteratorTypeAttr(L.IteratorType.REDUCTION)]
            ),
            result_types=(means.type, vars.type),
        )
        ops.append(o)
        return o.results

    def merge(nadd, source, state, source_map=None):
        n, oldm, oldv = state
        count = n + nadd

        def body(a):
            oldm, oldv, newm, newv = a
            delta = arith.SubfOp(newm, oldm)
            c = constant(nadd / count)
            nn = constant(n)
            cd = arith.MulfOp(c.result, delta.result)
            nd = arith.MulfOp(delta.result, nn.result)
            tmp = arith.AddfOp(oldv, newv)
            m = arith.AddfOp(oldm, cd.result)
            v = math.FmaOp(nd.result, cd.result, tmp.result)
            return [delta, c, nn, cd, nd, tmp, m, v], [m.result, v.result]

        maps = [amap(2, [0, 1])] * 2 + [source_map or amap(2, [0, 1])] * 2
        m, v = point([oldm, oldv, *source], [rows, 8], body, maps)
        return count, m, v

    stacks = [(0, splat([rows, 8]), splat([rows, 8])) for _ in range(2)]
    for chunk in range(3):
        p = partial(chunk)
        for half in range(2):
            mapping = AffineMapAttr(AffineMap(2, 0, (AffineDimExpr(0), AffineConstantExpr(half), AffineDimExpr(1))))
            stacks[0] = merge(16, p, stacks[0], mapping)
        if chunk == 1:
            stacks[1] = merge(stacks[0][0], stacks[0][1:], stacks[1])
            stacks[0] = (0, splat([rows, 8]), splat([rows, 8]))
    stacks[0] = merge(stacks[1][0], stacks[1][1:], stacks[0])
    mean, var = splat([rows]), splat([rows])
    count = 0
    for lane in range(8):

        def body(a, count=count):
            oldm, oldv, newm, newv = a
            d = arith.SubfOp(newm, oldm)
            c = constant(96 / (count + 96))
            nn = constant(count)
            m = math.FmaOp(c.result, d.result, oldm)
            sq = arith.MulfOp(d.result, d.result)
            frac = arith.MulfOp(sq.result, c.result)
            weighted = arith.MulfOp(frac.result, nn.result)
            added = arith.AddfOp(newv, weighted.result)
            v = arith.AddfOp(oldv, added.result)
            return [d, c, nn, m, sq, frac, weighted, added, v], [m.result, v.result]

        mapping = AffineMapAttr(AffineMap(1, 0, (AffineDimExpr(0), AffineConstantExpr(lane))))
        mean, var = point([mean, var, *stacks[0][1:]], [rows], body, [amap(1, [0])] * 2 + [mapping] * 2)
        count += 96

    def stats(a):
        mean, var = a
        den = constant(768)
        eps = constant(epsilon)
        one = constant(1)
        variance = arith.DivfOp(var, den.result)
        shift = arith.AddfOp(variance.result, eps.result)
        sqrt = math.SqrtOp(shift.result)
        scale = arith.DivfOp(one.result, sqrt.result)
        negative = arith.NegfOp(scale.result)
        bias = arith.MulfOp(negative.result, mean)
        return [den, eps, one, variance, shift, sqrt, scale, negative, bias], [scale.result, bias.result]

    scale, bias = point([mean, var], [rows], stats)
    empty = emit(tensor.EmptyOp((), TensorType(bf16, [rows, 768])))
    block = Block(arg_types=[bf16, gamma.type.element_type, beta.type.element_type, f32, f32, bf16])
    body = []

    def widen(v):
        if v.type == f32:
            return v
        o = arith.ExtFOp(v, f32)
        body.append(o)
        return o.result

    xx, gg, bb = [widen(a) for a in block.args[:3]]
    normalized = math.FmaOp(xx, block.args[3], block.args[4])
    affine = math.FmaOp(normalized.result, gg, bb)
    out = arith.TruncFOp(affine.result, bf16)
    block.add_ops([*body, normalized, affine, out, L.YieldOp(out.result)])
    result = emit(
        L.GenericOp(
            inputs=(flat, gamma, beta, scale, bias),
            outputs=(empty,),
            body=Region(block),
            indexing_maps=ArrayAttr(
                [amap(2, [0, 1]), amap(2, [1]), amap(2, [1]), amap(2, [0]), amap(2, [0]), amap(2, [0, 1])]
            ),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 2),
            result_types=(empty.type,),
        )
    )
    if len(shape) != 2:
        result = emit(tensor.ExpandShapeOp(result, (), groups, shape, x.type))
    for o in ops:
        o.attributes["merlin.layer_norm_backend_policy"] = StringAttr(backend_policy)
    return ops, result
