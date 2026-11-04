"""Exact captured quant_ext boundaries for the mixed source graph compiler.

Only floating host seams are implemented here. An integer accumulator must use
an admitted accelerator readout path; it is never silently converted on the CPU.
"""
from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import TensorType, IntegerType, FloatAttr, f32, i32


def _name(op):
    return getattr(getattr(op, 'op_name', None), 'data', op.name)


def _attribute(op, key, default=None):
    return op.properties.get(key, op.attributes.get(key, default))


def _integer(op, key, default=None):
    attr = _attribute(op, key)
    if attr is None:
        if default is None:
            raise ValueError(f'{_name(op)} requires {key}')
        return default
    return int(attr.value.data)


def lower_quant(graph, op) -> bool:
    """Emit one captured quantize/dequantize into GraphLowering, or return False.

    Supports rank-zero per-tensor parameters and axis-projected per-channel
    parameters. LLVM floating operations retain f32 reciprocal/multiply order,
    half-even rounding, zero point addition and explicit saturation.
    """
    name = _name(op)
    kinds = {'quant_ext.quantize_per_tensor': (True, False),
             'quant_ext.quantize_per_channel': (True, True),
             'quant_ext.dequantize_per_tensor': (False, False),
             'quant_ext.dequantize_per_channel': (False, True)}
    if name not in kinds:
        return False
    quantize, channel = kinds[name]
    if len(op.operands) != 3 or len(op.results) != 1:
        raise ValueError(f'{name} requires three operands and one result')
    value, scale, zero = op.operands
    result = op.results[0]
    if any(not isinstance(v.type, TensorType) for v in (*op.operands, result)):
        raise ValueError(f'{name} requires ranked tensor operands')
    shape = tuple(value.type.get_shape())
    if tuple(result.type.get_shape()) != shape or any(n < 0 for n in shape):
        raise ValueError(f'{name} requires equal static source and result shapes')
    floating = value if quantize else result
    integer = result if quantize else value
    if floating.type.element_type != f32 or scale.type.element_type != f32:
        raise ValueError(f'{name} currently requires captured f32 input/output and scale')
    if not isinstance(integer.type.element_type, IntegerType) or not isinstance(zero.type.element_type, IntegerType):
        raise ValueError(f'{name} requires signed integral values and zero points')
    if integer.type.element_type.width.data != 8:
        raise ValueError(f'{name}: eligible integer accumulator paths require accelerator readout')
    axis = None
    if channel:
        axis = _integer(op, 'axis')
        if axis < 0:
            axis += len(shape)
        if not 0 <= axis < len(shape):
            raise ValueError(f'{name} has invalid channel axis')
        expected = (shape[axis],)
    else:
        expected = ()
    if tuple(scale.type.get_shape()) != expected or tuple(zero.type.get_shape()) != expected:
        raise ValueError(f'{name} scale/zero-point shape differs from its granularity')
    if any(v not in graph.ptrs for v in (*op.operands, result)):
        raise ValueError(f'{name} has an unbound tensor pointer')
    qmin = _integer(op, 'quant_min') if quantize else _integer(op, 'quant_min', -128)
    qmax = _integer(op, 'quant_max') if quantize else _integer(op, 'quant_max', 127)
    if not -128 <= qmin <= qmax <= 127:
        raise ValueError(f'{name} quantization range is outside signed i8')
    b = graph.b
    def add(operation):
        return b.add(operation).results[0]
    def constant(number):
        return add(llvm.ConstantOp(FloatAttr(float(number), f32), f32))
    def intrinsic(kind, operands):
        return add(llvm.CallIntrinsicOp(f'llvm.{kind}.f32', operands, [f32]))
    # The reciprocal is computed once for each scale, at the same precision as
    # the source. Hoisting it changes no arithmetic or rounding operation.
    one = constant(1.) if quantize else None
    lower = constant(qmin) if quantize else None
    upper = constant(qmax) if quantize else None
    scalar_scale = graph.load(scale, []) if not channel else None
    scalar_zero = graph.load(zero, []) if not channel else None
    reciprocal = add(llvm.FDivOp(one, scalar_scale)) if quantize and not channel else None
    def body(indices):
        sv = graph.load(scale, [indices[axis]]) if channel else scalar_scale
        zv = graph.load(zero, [indices[axis]]) if channel else scalar_zero
        xv = graph.load(value, indices)
        if quantize:
            inv = add(llvm.FDivOp(one, sv)) if channel else reciprocal
            scaled = add(llvm.FMulOp(xv, inv))
            rounded = intrinsic('roundeven', [scaled])
            zf = add(llvm.SIToFPOp(zv, f32))
            shifted = add(llvm.FAddOp(rounded, zf))
            low = intrinsic('maximum', [shifted, lower])
            high = intrinsic('minimum', [low, upper])
            converted = graph.emit('llvm.fptosi', [high], [result.type.element_type])
        else:
            # The source per-channel i8/i32 contract subtracts after integer
            # promotion; large zero points distinguish this from float subtraction.
            if channel and zero.type.element_type == i32:
                extended = add(llvm.SExtOp(xv, i32))
                difference = add(llvm.SubOp(extended, zv))
                unscaled = add(llvm.SIToFPOp(difference, f32))
            else:
                xf = add(llvm.SIToFPOp(xv, f32))
                zf = add(llvm.SIToFPOp(zv, f32))
                unscaled = add(llvm.FSubOp(xf, zf))
            converted = add(llvm.FMulOp(unscaled, sv))
        graph.store(result, indices, converted)
    graph.loops(shape, body)
    return True
