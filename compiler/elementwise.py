"""Accumulator DMA paths for typed movement, column bias and quantized sums.

All tensor arithmetic executes in Gemmini. Host integer arithmetic only computes
addresses and counted loop bounds. Residual admission checks the entire i8 domain
against the source's declared bound before choosing the load/readout factorization.
"""
from __future__ import annotations
from functools import lru_cache
from math import prod, isfinite
import struct
try:
    from . import target
except ImportError:
    import target


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def _narrow(x, relu=False):
    return max(0 if relu else -128, min(127, round(x)))


@lru_cache(maxsize=256)
def residual_parameters(lhs_scale, rhs_scale, bound_lsb, relu=False):
    """Prove the rounded operand path against every pair of signed i8 values."""
    if type(bound_lsb) is not int or bound_lsb < 0:
        raise ValueError('residual add requires its declared nonnegative bound_lsb')
    ls, rs = _f32(lhs_scale), _f32(rhs_scale)
    if not all(isfinite(s) and s >= 0 for s in (ls, rs)):
        raise ValueError('residual load multipliers must be finite and nonnegative')
    common = _f32(max(1., ls, rs))
    ll, rl = _f32(ls / common), _f32(rs / common)
    vals = range(-128, 128)
    left = [_narrow(_f32(x * ll)) for x in vals]
    right = [_narrow(_f32(x * rl)) for x in vals]
    ideal_l = [_f32(x * ls) for x in vals]
    ideal_r = [_f32(x * rs) for x in vals]
    worst = 0
    for i in range(256):
        for j in range(256):
            reference = _narrow(_f32(ideal_l[i] + ideal_r[j]), relu)
            actual = _narrow(_f32((left[i] + right[j]) * common), relu)
            worst = max(worst, abs(reference - actual))
            if worst > bound_lsb:
                raise ValueError(f'residual DMA path exceeds declared bound {bound_lsb}: '
                                 f'pair {(i-128,j-128)}, difference {worst}')
    return ll, rl, common, worst


def _geometry(shape):
    if not shape or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('elementwise requires a positive static row-major shape')
    return prod(shape[:-1]), shape[-1]


def _tiles(b, rows, cols, emit):
    """One counted loop per full region; scalar tails keep command shapes static."""
    edge = target.DIM
    def spans(extent, callback):
        full = extent // edge * edge
        if full:
            b.for_loop(0, full, edge, lambda i: callback(i, edge))
        if extent != full:
            callback(b.const(full), extent-full)
    spans(rows, lambda row, nr: spans(cols, lambda col, nc: emit(row, col, nr, nc)))


def _address(b, pointer, row, col, width, item_bytes):
    element = b.add_i(b.mul_i(row, b.const(width)), col)
    return b.ptr(pointer, b.mul_i(element, b.const(item_bytes)))


def lower_movement(b, src, out, shape, input_dtype='i8', output_dtype='i32'):
    if input_dtype not in ('i8', 'i32') or output_dtype not in ('i8', 'i32'):
        raise ValueError('movement accumulator path supports i8/i32 tensors')
    if input_dtype == 'i32' and output_dtype == 'i8':
        raise ValueError('movement cannot silently saturate an i32 source')
    rows, cols = _geometry(shape)
    ib, ob = (1 if input_dtype == 'i8' else 4), (1 if output_dtype == 'i8' else 4)
    target.cfg_ld(b, cols * ib, shrunk=input_dtype == 'i8')
    target.cfg_st(b, cols * ob, scale=1., activation=0)
    def emit(row, col, nr, nc):
        target.mvin(b, _address(b, src, row, col, cols, ib), target.ACC, nr, nc)
        b.fence()
        target.mvout(b, _address(b, out, row, col, cols, ob),
                     target.ACC | (target.FULL if ob == 4 else 0), nr, nc)
        b.fence()
    _tiles(b, rows, cols, emit)


def lower_residual_add(b, lhs, rhs, out, shape, attrs):
    stages = tuple(attrs.get('epilogue', ()))
    if any(s != 'relu' for s in stages):
        raise ValueError('residual add supports only declared ReLU epilogue')
    relu = 'relu' in stages
    ll, rl, readout, worst = residual_parameters(attrs['lhs_scale'], attrs['rhs_scale'],
                                                attrs['bound_lsb'], relu)
    rows, cols = _geometry(shape)
    target.cfg_ld(b, cols, stream=0, scale=ll, shrunk=True)
    target.cfg_ld(b, cols, stream=1, scale=rl, shrunk=True)
    target.cfg_st(b, cols, scale=readout, activation=int(relu))
    def emit(row, col, nr, nc):
        target.mvin(b, _address(b, lhs, row, col, cols, 1), target.ACC, nr, nc, stream=0)
        b.fence()
        target.mvin(b, _address(b, rhs, row, col, cols, 1),
                    target.ACC | target.ACCUMULATE, nr, nc, stream=1)
        b.fence()
        target.mvout(b, _address(b, out, row, col, cols, 1), target.ACC, nr, nc)
        b.fence()
    _tiles(b, rows, cols, emit)
    return {'operand_scales': [ll, rl], 'readout_scale': readout,
            'domain_pairs': 65536, 'max_error_lsb': worst}


def lower_bias_add(b, src, bias, out, shape, attrs=None, output_dtype='i32'):
    """Exact i32 source plus i32 per-column bias, optionally narrowed on readout."""
    attrs = attrs or {}
    stages = tuple(attrs.get('epilogue', ()))
    if any(s not in ('acc_scale', 'relu', 'bias', 'bias_add') for s in stages):
        raise ValueError('unsupported bias accumulator readout epilogue')
    if output_dtype not in ('i8', 'i32'):
        raise ValueError('bias output must be i8 or i32')
    if output_dtype == 'i32' and any(s in ('acc_scale', 'relu') for s in stages):
        raise ValueError('full accumulator readout does not apply scale or activation')
    scale = attrs['acc_scale'] if 'acc_scale' in stages else 1.
    if isinstance(scale, (list, tuple)):
        raise ValueError('column scale vector requires partitioned readout')
    rows, cols = _geometry(shape)
    ob = 1 if output_dtype == 'i8' else 4
    target.cfg_ld(b, cols * 4, stream=0, shrunk=False)
    target.cfg_ld(b, 0, stream=1, shrunk=False)
    target.cfg_st(b, cols * ob, scale=scale, activation=int('relu' in stages))
    def emit(row, col, nr, nc):
        target.mvin(b, _address(b, src, row, col, cols, 4), target.ACC, nr, nc, stream=0)
        b.fence()
        target.mvin(b, b.ptr(bias, b.mul_i(col, b.const(4))),
                    target.ACC | target.ACCUMULATE, nr, nc, stream=1)
        b.fence()
        target.mvout(b, _address(b, out, row, col, cols, ob),
                     target.ACC | (target.FULL if ob == 4 else 0), nr, nc)
        b.fence()
    _tiles(b, rows, cols, emit)


def lower_elementwise(b, cb, command, pointers):
    op, a = command['opcode'], command.get('attributes', {})
    operands = command['operands']
    dst = operands.get('dst', operands.get('out'))
    if dst is None:
        raise ValueError(f'{op} has no declared destination')
    tensors = cb['tensors']
    shape, dtype = tensors[dst]['shape'], tensors[dst]['dtype']
    if op == 'MOVEMENT':
        src = operands['src']
        if tensors[src]['shape'] != shape:
            raise ValueError('movement changes tensor shape')
        return lower_movement(b, pointers[src], pointers[dst], shape, tensors[src]['dtype'], dtype)
    if op == 'RESIDUAL_ADD':
        lhs, rhs = operands['lhs'], operands['rhs']
        if any(tensors[name]['shape'] != shape or tensors[name]['dtype'] != 'i8'
               for name in (lhs, rhs)) or dtype != 'i8':
            raise ValueError('residual path requires equal-shape signed i8 tensors')
        return lower_residual_add(b, pointers[lhs], pointers[rhs], pointers[dst], shape, a)
    if op == 'BIAS_ADD':
        src, bias = operands['src'], operands.get('bias', a.get('bias'))
        if tensors[src]['dtype'] != 'i32' or tensors[src]['shape'] != shape:
            raise ValueError('bias add requires an i32 tensor source')
        if tensors[bias]['dtype'] != 'i32' or tensors[bias]['shape'] != [shape[-1]]:
            raise ValueError('bias add requires an i32 per-column bias')
        return lower_bias_add(b, pointers[src], pointers[bias], pointers[dst], shape, a, dtype)
    raise ValueError(f'unhandled elementwise operation {op}')
