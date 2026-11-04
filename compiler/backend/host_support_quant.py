"""Expose captured per-tensor quantization as ordinary typed scalar MLIR.

The decomposition retains the captured f32 scale, i64 zero point and explicit
integer limits. It uses round-to-nearest-even, never an integer truncation as
rounding. The generic host compiler remains responsible for code generation.
"""

def expand_quantization(module):
    from xdsl.dialects import builtin, arith, math
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    for op in list(module.walk()):
        name = op.op_name.data if op.name == 'builtin.unregistered' else op.name
        if name not in ('quant_ext.quantize_per_tensor', 'quant_ext.dequantize_per_tensor'):
            continue
        if len(op.operands) != 3 or len(op.results) != 1:
            raise ValueError('quant_ext requires value, scale and zero-point tensors')
        x, scale, zero = op.operands
        out_type = op.results[0].type
        if (str(scale.type) != 'tensor<f32>' or str(zero.type) != 'tensor<i64>'
                or x.type.get_shape() != out_type.get_shape()):
            raise ValueError('quant_ext requires same-shape values and per-tensor f32/i64 parameters')
        quantize = name == 'quant_ext.quantize_per_tensor'
        if (str(x.type.element_type), str(out_type.element_type)) != (('f32', 'i8') if quantize else ('i8', 'f32')):
            raise ValueError('quant_ext typed scalar lowering currently admits f32/i8 only')
        attributes = {**op.attributes, **op.properties}
        qmin = attributes.get('quant_min')
        qmax = attributes.get('quant_max')
        if qmin is None or qmax is None or not (-128 <= qmin.value.data <= qmax.value.data <= 127):
            raise ValueError('quant_ext requires explicit signed-byte clamp limits')
        empty = EmptyOp([], out_type)
        body = Block(arg_types=[v.type.element_type for v in (x, scale, zero)] + [out_type.element_type])
        a, s, z, _ = body.args
        def add(node):
            body.add_op(node)
            return node.results[0]
        zf = add(arith.SIToFPOp(z, builtin.f32))
        if quantize:
            one = add(arith.ConstantOp(builtin.FloatAttr(1.0, builtin.f32)))
            inverse = add(arith.DivfOp(one, s))
            scaled = add(arith.MulfOp(a, inverse))
            rounded = add(math.RoundEvenOp(scaled))
            shifted = add(arith.AddfOp(rounded, zf))
            lo = add(arith.ConstantOp(builtin.FloatAttr(float(qmin.value.data), builtin.f32)))
            hi = add(arith.ConstantOp(builtin.FloatAttr(float(qmax.value.data), builtin.f32)))
            clamped_lo = add(arith.MaximumfOp(shifted, lo))
            clamped = add(arith.MinimumfOp(clamped_lo, hi))
            value = add(arith.FPToSIOp(clamped, builtin.i8))
        else:
            af = add(arith.SIToFPOp(a, builtin.f32))
            centered = add(arith.SubfOp(af, zf))
            value = add(arith.MulfOp(centered, s))
        body.add_op(YieldOp(value))
        rank = len(out_type.get_shape())
        identity = builtin.AffineMapAttr(AffineMap.identity(rank))
        scalar = builtin.AffineMapAttr(AffineMap(rank, 0, ()))
        generic = GenericOp([x, scale, zero], [empty.results[0]], Region(body),
                            [identity, scalar, scalar, identity],
                            [IteratorTypeAttr.parallel()] * rank, [out_type])
        generic.attributes.update(op.attributes)
        op.parent.insert_op_before(empty, op)
        op.parent.insert_op_before(generic, op)
        op.results[0].replace_all_uses_with(generic.results[0])
        op.parent.erase_op(op)
