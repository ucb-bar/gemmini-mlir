"""Opt-in reproduction of a pinned CPU bf16 flash-attention numerical policy.

This is an alternative schedule, not the default mathematical SDPA lowering.
The currently admitted domain admits broadcast boolean masks and excludes dropout, GQA and
tile remainders. Source policy: PyTorch v2.10.0 CPU bf16 flash, AVX2 arithmetic.
The emitted IR is scalar-target independent and does not contain vector or
accelerator instructions. Finite intermediates and f32 RNE are explicit caller
obligations; matching an arbitrary BLAS reduction implementation is not claimed.
The optional MKL DEF Zen policy reproduces the observed 192-element PV panel
schedule of oneMKL 2024.0 Update2 on the audited AMD host. It is independent of
ATen's AVX2 exponential policy; other MKL versions or dispatches are not implied.
Floating-point status flags are not part of this unconstrained LLVM FP policy.
"""

from __future__ import annotations

import math as python_math
import struct

from .cpu_flash_exp import CPU_FLASH_BF16_AVX2_POLICY, build_cpu_flash_exp_nonpositive

CPU_FLASH_PV_MKL_DEF_ZEN_POLICY = "mkl_2024_0_u2_def_zen_k192"


def build_cpu_flash_sdpa(
    query,
    key,
    value,
    *,
    backend_policy: str,
    scale: float,
    assume_finite_intermediates: bool = False,
    mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    enable_gqa: bool = False,
    pv_backend_policy: str | None = None,
):
    """Return detached tensor operations and a bf16 attention result.

    Refuse unimplemented source contracts before constructing any operations.
    Initial support is rank-four equal-batch/head bf16, head dimension 64,
    query length >=768 divisible by256, key length divisible by512.
    """
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        BFloat16Type,
        DenseArrayBase,
        FloatAttr,
        IntegerAttr,
        StringAttr,
        TensorType,
        f32,
        i1,
        i64,
    )
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

    if backend_policy != CPU_FLASH_BF16_AVX2_POLICY:
        raise ValueError("explicit pinned CPU bf16 flash policy required")
    if pv_backend_policy not in (None, CPU_FLASH_PV_MKL_DEF_ZEN_POLICY):
        raise ValueError("unsupported pinned PV GEMM policy")
    if not assume_finite_intermediates:
        raise ValueError("finite-intermediate contract required")
    if dropout_p != 0.0 or is_causal or enable_gqa:
        raise ValueError("causal, dropout and GQA CPU flash contracts are unsupported")
    if not python_math.isfinite(scale) or scale <= 0:
        raise ValueError("finite positive scale required")
    try:
        scale_f32 = struct.unpack("f", struct.pack("f", scale))[0]
    except OverflowError as error:
        raise ValueError("scale must be representable as finite positive f32") from error
    if not python_math.isfinite(scale_f32) or scale_f32 <= 0:
        raise ValueError("scale must be representable as finite positive f32")
    types = [operand.type for operand in (query, key, value)]
    if any(not isinstance(t, TensorType) or not isinstance(t.element_type, BFloat16Type) for t in types):
        raise TypeError("CPU flash policy requires bf16 tensor Q/K/V")
    shapes = [list(t.get_shape()) for t in types]
    if any(len(s) != 4 or any(n <= 0 for n in s) for s in shapes):
        raise ValueError("static positive rank-four Q/K/V required")
    sq, sk, sv = shapes
    if sq[:2] != sk[:2] or sk[:2] != sv[:2] or sk[2] != sv[2]:
        raise ValueError("batch, heads and K/V sequence lengths must match")
    if sq[3] != 64 or sk[3] != 64 or sv[3] != 64:
        raise ValueError("only the audited head dimension64 is admitted")
    if sq[2] < 768 or sq[2] % 256 or sk[2] % 512:
        raise ValueError("query256/key512 tile remainders are unsupported")

    mask_shape = None
    if mask is not None:
        mask_type = getattr(mask, "type", None)
        if not isinstance(mask_type, TensorType) or mask_type.element_type != i1:
            raise TypeError("CPU flash mask must be a rank-four i1 tensor")
        mask_shape = list(mask_type.get_shape())
        target_shape = [*sq[:3], sk[2]]
        if len(mask_shape) != 4 or any(m not in (1, n) for m, n in zip(mask_shape, target_shape)):
            raise ValueError("CPU flash mask dimensions must be one or match batch/head/query/key")

    ops = []
    bf16 = types[0].element_type
    B, H, length, depth = sq
    dims = [AffineDimExpr(i) for i in range(5)]

    def emit(op):
        ops.append(op)
        return op.results[0]

    def amap(rank, positions):
        return AffineMapAttr(AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in positions)))

    def splat(shape, number):
        constant = emit(arith.ConstantOp(FloatAttr(number, f32)))
        return emit(tensor.SplatOp(constant, [], TensorType(f32, shape)))

    def pointwise(inputs, shape, callback, *, maps=None, element=f32):
        output_type = TensorType(element, shape)
        empty = emit(tensor.EmptyOp((), output_type))
        body = Block(arg_types=[x.type.element_type for x in inputs] + [element])
        body_ops, result = callback(list(body.args[:-1]))
        body.add_ops([*body_ops, linalg.YieldOp(result)])
        rank = len(shape)
        indexing = maps or [amap(rank, range(rank))] * len(inputs)
        return emit(
            linalg.GenericOp(
                inputs=inputs,
                outputs=(empty,),
                body=Region(body),
                indexing_maps=ArrayAttr([*indexing, amap(rank, range(rank))]),
                iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * rank),
                result_types=(output_type,),
            )
        )

    def binary(cls):
        def body(args):
            op = cls(*args)
            return [op], op.results[0]

        return body

    def reduce_last(source, number, cls):
        shape = list(source.type.get_shape())
        init = splat(shape[:-1], number)
        body = Block(arg_types=[f32, f32])
        op = cls(*body.args)
        body.add_ops([op, linalg.YieldOp(op.results[0])])
        return emit(linalg.ReduceOp(source, init, DenseArrayBase.from_list(i64, [len(shape) - 1]), Region(body)))

    def contraction(lhs, rhs, init, *, transposed_rhs):
        # Caller chooses the reduction seed; backend panel policies below may
        # instead require a zero-seeded product and a separate destination add.
        body = Block(arg_types=[bf16, bf16, f32])
        left = arith.ExtFOp(body.args[0], f32)
        right = arith.ExtFOp(body.args[1], f32)
        fused = math.FmaOp(left.result, right.result, body.args[2])
        body.add_ops([left, right, fused, linalg.YieldOp(fused.result)])
        lhs_map = amap(5, [0, 1, 2, 4])
        rhs_map = amap(5, [0, 1, 3, 4] if transposed_rhs else [0, 1, 4, 3])
        return emit(
            linalg.GenericOp(
                inputs=(lhs, rhs),
                outputs=(init,),
                body=Region(body),
                indexing_maps=ArrayAttr([lhs_map, rhs_map, amap(5, [0, 1, 2, 3])]),
                iterator_types=ArrayAttr(
                    [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 4
                    + [linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION)]
                ),
                result_types=(init.type,),
            )
        )

    def lane_sum(exponentials):
        # AVX2: eight sequential lane sums over64 groups, followed by the
        # source horizontal reduction tree (offsets4,2,1), not a serial sum512.
        shape = list(exponentials.type.get_shape())
        expanded_shape = [*shape[:-1], 64, 8]
        reassociation = ArrayAttr(
            [ArrayAttr([IntegerAttr(i, i64)]) for i in range(3)]
            + [ArrayAttr([IntegerAttr(3, i64), IntegerAttr(4, i64)])]
        )
        expanded = emit(
            tensor.ExpandShapeOp(exponentials, (), reassociation, expanded_shape, TensorType(f32, expanded_shape))
        )
        init = splat([*shape[:-1], 8], 0.0)
        body = Block(arg_types=[f32, f32])
        add = arith.AddfOp(*body.args)
        body.add_ops([add, linalg.YieldOp(add.result)])
        lanes = emit(linalg.ReduceOp(expanded, init, DenseArrayBase.from_list(i64, [3]), Region(body)))
        maps = [AffineMapAttr(AffineMap(3, 0, (*dims[:3], AffineConstantExpr(lane)))) for lane in range(8)]

        def horizontal(args):
            a = [arith.AddfOp(args[i], args[i + 4]) for i in range(4)]
            b0 = arith.AddfOp(a[0].result, a[2].result)
            b1 = arith.AddfOp(a[1].result, a[3].result)
            final = arith.AddfOp(b0.result, b1.result)
            return [*a, b0, b1, final], final.result

        return pointwise([lanes] * 8, shape[:-1], horizontal, maps=maps)

    result = emit(tensor.EmptyOp((), TensorType(bf16, sq)))
    for q_start in range(0, length, 256):
        q = emit(tensor.ExtractSliceOp.from_static_parameters(query, [0, 0, q_start, 0], [B, H, 256, depth]))
        row_shape = [B, H, 256]
        output_shape = [B, H, 256, depth]
        running_max = splat(row_shape, -float("inf"))
        running_sum = splat(row_shape, 0.0)
        destination = splat(output_shape, 0.0)
        for k_start in range(0, sk[2], 512):
            k = emit(tensor.ExtractSliceOp.from_static_parameters(key, [0, 0, k_start, 0], [B, H, 512, depth]))
            v = emit(tensor.ExtractSliceOp.from_static_parameters(value, [0, 0, k_start, 0], [B, H, 512, depth]))
            score_shape = [B, H, 256, 512]
            dot = contraction(q, k, splat(score_shape, 0.0), transposed_rhs=True)

            def scale_body(args):
                coefficient = arith.ConstantOp(FloatAttr(scale, f32))
                scaled = arith.MulfOp(args[0], coefficient.result)
                return [coefficient, scaled], scaled.result

            scores = pointwise([dot], score_shape, scale_body)
            if mask_shape is not None:
                mask_offsets = [0, 0, 0 if mask_shape[2] == 1 else q_start, 0 if mask_shape[3] == 1 else k_start]
                mask_sizes = [
                    mask_shape[0],
                    mask_shape[1],
                    1 if mask_shape[2] == 1 else 256,
                    1 if mask_shape[3] == 1 else 512,
                ]
                tile_mask = emit(tensor.ExtractSliceOp.from_static_parameters(mask, mask_offsets, mask_sizes))
                mask_map = AffineMapAttr(
                    AffineMap(
                        4,
                        0,
                        tuple(
                            AffineConstantExpr(0) if size == 1 else AffineDimExpr(i)
                            for i, size in enumerate(mask_shape)
                        ),
                    )
                )

                def apply_mask(args):
                    negative_infinity = arith.ConstantOp(FloatAttr(-float("inf"), f32))
                    selected = arith.SelectOp(args[1], args[0], negative_infinity.result)
                    return [negative_infinity, selected], selected.result

                scores = pointwise([scores, tile_mask], score_shape, apply_mask, maps=[amap(4, [0, 1, 2, 3]), mask_map])
            block_max = reduce_last(scores, -float("inf"), arith.MaximumfOp)
            new_max = pointwise([running_max, block_max], row_shape, binary(arith.MaximumfOp))
            center_max = new_max
            if mask_shape is not None:
                # A wholly masked row has max=-inf. Subtract zero so every
                # centered masked logit stays -inf, never -inf - -inf = NaN.
                def safe_max(args):
                    negative_infinity = arith.ConstantOp(FloatAttr(-float("inf"), f32))
                    zero = arith.ConstantOp(FloatAttr(0.0, f32))
                    masked = arith.CmpfOp(args[0], negative_infinity.result, "oeq")
                    selected = arith.SelectOp(masked.result, zero.result, args[0])
                    return [negative_infinity, zero, masked, selected], selected.result

                center_max = pointwise([new_max], row_shape, safe_max)
            centered = pointwise(
                [scores, center_max],
                score_shape,
                binary(arith.SubfOp),
                maps=[amap(4, [0, 1, 2, 3]), amap(4, [0, 1, 2])],
            )
            exponentials = pointwise(
                [centered],
                score_shape,
                lambda args: build_cpu_flash_exp_nonpositive(
                    args[0], backend_policy=backend_policy, domain="nonpositive"
                ),
            )
            block_sum = lane_sum(exponentials)

            def alpha_body(args):
                alpha_ops = []
                old_max, current_max = args
                if mask_shape is not None:
                    # Select safe operands before subtraction. Selecting only
                    # its result would still raise invalid on -inf - -inf.
                    negative_infinity = arith.ConstantOp(FloatAttr(-float("inf"), f32))
                    zero = arith.ConstantOp(FloatAttr(0.0, f32))
                    masked = arith.CmpfOp(current_max, negative_infinity.result, "oeq")
                    safe_old = arith.SelectOp(masked.result, zero.result, old_max)
                    safe_current = arith.SelectOp(masked.result, zero.result, current_max)
                    alpha_ops += [negative_infinity, zero, masked, safe_old, safe_current]
                    old_max, current_max = safe_old.result, safe_current.result
                difference = arith.SubfOp(old_max, current_max)
                exponential = math.ExpOp(difference.result)
                return [*alpha_ops, difference, exponential], exponential.result

            alpha = pointwise([running_max, new_max], row_shape, alpha_body)

            def update_sum(args):
                fused = math.FmaOp(args[0], args[1], args[2])
                return [fused], fused.result

            running_sum = pointwise([alpha, running_sum, block_sum], row_shape, update_sum)

            def narrow(args):
                cast = arith.TruncFOp(args[0], bf16)
                return [cast], cast.result

            packed_exp = pointwise([exponentials], score_shape, narrow, element=bf16)
            scaled_destination = pointwise(
                [destination, alpha],
                output_shape,
                binary(arith.MulfOp),
                maps=[amap(4, [0, 1, 2, 3]), amap(4, [0, 1, 2])],
            )
            if pv_backend_policy is None:
                destination = contraction(packed_exp, v, scaled_destination, transposed_rhs=False)
            else:
                # Actual oneMKL 2024.0 Update2 DEF Zen dispatch for M64/N256:
                # K512 is packed as 192+192+128. Each microkernel starts zero,
                # performs serial FMA, then adds the existing destination.
                # This is independent of ATen's AVX2 softmax dispatch.
                destination = scaled_destination
                for panel_start, panel_size in ((0, 192), (192, 192), (384, 128)):
                    p_panel = emit(
                        tensor.ExtractSliceOp.from_static_parameters(
                            packed_exp, [0, 0, 0, panel_start], [B, H, 256, panel_size]
                        )
                    )
                    v_panel = emit(
                        tensor.ExtractSliceOp.from_static_parameters(
                            v, [0, 0, panel_start, 0], [B, H, panel_size, depth]
                        )
                    )
                    partial = contraction(p_panel, v_panel, splat(output_shape, 0.0), transposed_rhs=False)
                    destination = pointwise([partial, destination], output_shape, binary(arith.AddfOp))
            running_max = new_max

        def reciprocal(args):
            one = arith.ConstantOp(FloatAttr(1.0, f32))
            reciprocal_ops = [one]
            denominator = args[0]
            if mask_shape is not None:
                zero = arith.ConstantOp(FloatAttr(0.0, f32))
                masked = arith.CmpfOp(denominator, zero.result, "oeq")
                selected = arith.SelectOp(masked.result, one.result, denominator)
                reciprocal_ops += [zero, masked, selected]
                denominator = selected.result
            quotient = arith.DivfOp(one.result, denominator)
            return [*reciprocal_ops, quotient], quotient.result

        inverse = pointwise([running_sum], row_shape, reciprocal)

        def normalize(args):
            product = arith.MulfOp(*args)
            narrowed = arith.TruncFOp(product.result, bf16)
            return [product, narrowed], narrowed.result

        output = pointwise(
            [destination, inverse],
            output_shape,
            normalize,
            maps=[amap(4, [0, 1, 2, 3]), amap(4, [0, 1, 2])],
            element=bf16,
        )
        result = emit(tensor.InsertSliceOp.from_static_parameters(output, result, [0, 0, q_start, 0], output_shape))
    for op in ops:
        op.attributes["merlin.sdpa_backend_policy"] = StringAttr(backend_policy)
        if pv_backend_policy is not None:
            op.attributes["merlin.sdpa_pv_backend_policy"] = StringAttr(pv_backend_policy)
    return ops, result
