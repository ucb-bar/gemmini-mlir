"""Standalone pointwise transfers through Gemmini's accumulator datapath.

Only address arithmetic and counted command loops run on the host. Tensor data
is loaded, scaled, accumulated and narrowed by the accelerator. The serial path
borrows private rows 0..15. Optional checked residual epochs borrow disjoint
windows bounded by selected accumulator capacity, with full drains before reuse.
"""
from __future__ import annotations

from functools import lru_cache
from math import isfinite, prod
import struct

import gemmini_encode as enc


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def _sat(x):
    return max(-128, min(127, int(x)))


@lru_cache(maxsize=128)
def residual_numeric_plan(lhs_scale, rhs_scale, bound_lsb, relu=False):
    """Exhaustively check the selected load/readout arithmetic on every i8 pair.

    Gemmini scales and rounds each input into signed i8 before summation. A
    common factor on readout prevents input saturation for multipliers > 1.
    This approximation is admitted only within the original source's bound.
    """
    lhs, rhs = _f32(lhs_scale), _f32(rhs_scale)
    if not all(isfinite(x) and x >= 0 for x in (lhs, rhs)):
        raise ValueError('residual scales must be finite nonnegative f32')
    if type(bound_lsb) is not int or bound_lsb < 0:
        raise ValueError('residual bound_lsb must be a nonnegative integer')
    common = _f32(max(1.0, lhs, rhs))
    left, right = _f32(lhs / common), _f32(rhs / common)
    scaled_l = [_sat(round(_f32(x * left))) for x in range(-128, 128)]
    scaled_r = [_sat(round(_f32(x * right))) for x in range(-128, 128)]
    target_l = [_f32(x * lhs) for x in range(-128, 128)]
    target_r = [_f32(x * rhs) for x in range(-128, 128)]
    worst = 0
    for i in range(256):
        for j in range(256):
            desired = round(_f32(target_l[i] + target_r[j]))
            actual = round(_f32((scaled_l[i] + scaled_r[j]) * common))
            if relu:
                desired, actual = max(0, desired), max(0, actual)
            worst = max(worst, abs(_sat(actual) - _sat(desired)))
    if worst > bound_lsb:
        raise ValueError(f'Gemmini residual load/readout error {worst} exceeds declared bound_lsb {bound_lsb}')
    return {'lhs_load_scale': left, 'rhs_load_scale': right,
            'store_scale': common, 'max_error_lsb': worst,
            'checked_pairs': 65536}


def _tiles(fn, count, body):
    """Cover a flat contiguous tensor with one retained full-tile command loop."""
    if type(count) is not int or count < 0:
        raise ValueError('pointwise element count must be a nonnegative integer')
    full = count // 256 * 256
    if full:
        fn.for_loop(0, full, 256, lambda offset: body(offset, 16, 16), retain_loop=True)
    offset = full
    rows, tail = divmod(count - full, 16)
    if rows:
        body(offset, rows, 16)
        offset += rows * 16
    if tail:
        body(offset, 1, tail)


def _ptr(fn, base, element_offset, width=1):
    if isinstance(element_offset, int):
        offset = element_offset * width
    else:
        offset = element_offset if width == 1 else fn.mul_i(element_offset, fn.const(width))
    return enc.pointer_offset(fn, base, offset)


def emit_residual(fn, a, b, dst, m, n, *, attrs=None, output_dtype='i8',
                  external_buffers_disjoint=False):
    """Sum flattened i8 tensors through load-accumulate and device readout.

    ``attrs`` accepts source lhs_scale/rhs_scale, bound_lsb and []/[relu]
    epilogue. Missing scales are unit scales and require an exact result.
    ``output_dtype='i32'`` supports only unit scales without an epilogue,
    because full-width accumulator readout bypasses scale/activation.
    Returns the numerical admission record; emits no host tensor arithmetic.
    Epoch scheduling requires the caller's original distinct tensor binding
    check through ``external_buffers_disjoint``; otherwise it retains serial.
    """
    attrs = dict(attrs or {})
    output_dtype = attrs.get('output_dtype', output_dtype)
    epilogue = attrs.get('epilogue', []) or []
    if isinstance(epilogue, str):
        epilogue = [epilogue]
    if list(epilogue) not in ([], ['relu']):
        raise ValueError(f'residual device readout does not support epilogue {epilogue!r}')
    if type(m) is not int or type(n) is not int or min(m, n) < 0:
        raise ValueError('residual extents must be nonnegative integers')
    lhs, rhs = float(attrs.get('lhs_scale', 1.0)), float(attrs.get('rhs_scale', 1.0))
    if output_dtype == 'i32':
        if lhs != 1.0 or rhs != 1.0 or epilogue:
            raise ValueError('i32 readout requires unit residual scales and no activation')
        plan = {'lhs_load_scale': 1.0, 'rhs_load_scale': 1.0,
                'store_scale': 1.0, 'max_error_lsb': 0, 'checked_pairs': 65536}
    elif output_dtype == 'i8':
        plan = residual_numeric_plan(lhs, rhs, int(attrs.get('bound_lsb', 0)), bool(epilogue))
    else:
        raise ValueError(f'unsupported residual output dtype {output_dtype!r}')
    from merlin.compile.tuning_knobs import knob
    from pathlib import Path
    schedule = knob('residual_schedule', default='serial', values=('serial', 'epochs', 'auto'),
                    root=Path(__file__).resolve().parent)
    epoch_plan, admission, refusal, policy = None, None, None, None
    if schedule in ('epochs', 'auto'):
        from gemmini_residual_schedule import residual_epoch_plan
        try:
            epoch_plan, admission = residual_epoch_plan(m * n, plan,
                output_dtype=output_dtype, relu=bool(epilogue),
                external_buffers_disjoint=external_buffers_disjoint,
                source_numeric={'lhs_scale_f32': _f32(lhs), 'rhs_scale_f32': _f32(rhs),
                                'bound_lsb': int(attrs.get('bound_lsb', 0))})
            # Batching one capacity group only adds counted control and an
            # acquisition drain. Require two full groups from current source
            # geometry; no per-case identity or runtime timing is consulted.
            if epoch_plan.grouped_extent // epoch_plan.live_tiles < 2:
                epoch_plan, admission = None, None
                refusal = 'fewer than two complete capacity-bounded epoch groups'
            elif schedule == 'auto':
                from gemmini_residual_policy import select_residual_schedule
                selected, policy = select_residual_schedule(epoch_plan, admission)
                if selected != 'epochs':
                    epoch_plan, admission = None, None
                    refusal = 'outside frozen residual auto policy region'
        except ValueError as exc:
            refusal = str(exc)
    enc.fence(fn)
    enc.config_load(fn, 16, stream=0, scale=plan['lhs_load_scale'], shrunk=True)
    enc.config_load(fn, 16, stream=1, scale=plan['rhs_load_scale'], shrunk=True)
    width = 4 if output_dtype == 'i32' else 1
    enc.config_store(fn, 16 * width, scale=plan['store_scale'], activation=int(bool(epilogue)))

    def tile(offset, rows, cols):
        enc.mvin(fn, _ptr(fn, a, offset), enc.ACC, rows, cols, stream=0)
        enc.fence(fn)
        enc.mvin(fn, _ptr(fn, b, offset), enc.ACC_ADD, rows, cols, stream=1)
        enc.fence(fn)
        enc.mvout(fn, _ptr(fn, dst, offset, width),
                  enc.ACC_FULL if output_dtype == 'i32' else enc.ACC, rows, cols)
        enc.fence(fn)

    if epoch_plan is not None:
        from gemmini_residual_schedule import emit_residual_epochs
        receipt = emit_residual_epochs(fn, a, b, dst, epoch_plan, admission,
                                       output_dtype=output_dtype)
        prefix = admission['prefix_elements']
        _tiles(fn, m * n - prefix, lambda offset, rows, cols: tile(prefix + offset, rows, cols))
        return {**plan, 'residual_schedule': 'epochs', 'epoch_receipt': receipt,
                **({'residual_policy': policy} if policy is not None else {})}
    _tiles(fn, m * n, tile)
    if schedule in ('epochs', 'auto'):
        return {**plan, 'residual_schedule': 'serial', 'epoch_refusal': refusal,
                **({'residual_policy': policy} if policy is not None else {})}
    return plan


def emit_movement(fn, src, dst, shape, *, dtype='i8', attrs=None):
    """Roundtrip contiguous data through device storage.

    Public MOVEMENT semantics default to i8->i32 sign extension. An explicit
    attrs.output_dtype='i8' requests exact i8 copying. Input i32 supports i32
    copying or scaled/ReLU i8 narrowing with the device's readout semantics.
    """
    attrs = dict(attrs or {})
    shape = tuple(shape)
    if any(type(x) is not int or x < 0 for x in shape):
        raise ValueError('movement needs static nonnegative shape extents')
    output_dtype = attrs.get('output_dtype', 'i32')
    if dtype not in ('i8', 'i32') or output_dtype not in ('i8', 'i32'):
        raise ValueError(f'unsupported device movement {dtype}->{output_dtype}')
    epilogue = attrs.get('epilogue', []) or []
    if isinstance(epilogue, str):
        epilogue = [epilogue]
    if list(epilogue) not in ([], ['relu']):
        raise ValueError(f'unsupported movement epilogue {epilogue!r}')
    scale = float(attrs.get('acc_scale', attrs.get('scale', 1.0)))
    if not isfinite(scale):
        raise ValueError('movement scale must be finite')
    if output_dtype == 'i32' and (scale != 1.0 or epilogue):
        raise ValueError('full-width movement bypasses readout scale/activation')
    source_width, target_width = (1 if dtype == 'i8' else 4), (1 if output_dtype == 'i8' else 4)
    enc.fence(fn)
    enc.config_load(fn, 16 * source_width, shrunk=dtype == 'i8')
    enc.config_store(fn, 16 * target_width, scale=scale, activation=int(bool(epilogue)))

    def tile(offset, rows, cols):
        enc.mvin(fn, _ptr(fn, src, offset, source_width), enc.ACC, rows, cols)
        enc.fence(fn)
        enc.mvout(fn, _ptr(fn, dst, offset, target_width),
                  enc.ACC_FULL if output_dtype == 'i32' else enc.ACC, rows, cols)
        enc.fence(fn)

    _tiles(fn, prod(shape), tile)
    return {'input_dtype': dtype, 'output_dtype': output_dtype,
            'elements': prod(shape), 'route': 'accumulator_load_readout'}


def emit_bias_add(fn, src, bias, dst, m, n, *, attrs=None, output_dtype='i32'):
    """Add an i32 column bias using full-width accumulator loads.

    Source and output are disjoint row-major buffers; bias is a contiguous n
    vector. Zero load stride broadcasts the bias row inside the accelerator.
    The i32 result uses the accumulator's native two's-complement arithmetic.
    """
    attrs = dict(attrs or {})
    output_dtype = attrs.get('output_dtype', output_dtype)
    epilogue = attrs.get('epilogue', []) or []
    if isinstance(epilogue, str):
        epilogue = [epilogue]
    if list(epilogue) not in ([], ['relu']):
        raise ValueError(f'unsupported standalone bias epilogue {epilogue!r}')
    scale = float(attrs.get('acc_scale', attrs.get('scale', 1.0)))
    if output_dtype not in ('i32', 'i8') or not isfinite(scale):
        raise ValueError('bias readout requires finite scale and i8/i32 output')
    if output_dtype == 'i32' and (scale != 1.0 or epilogue):
        raise ValueError('full-width bias readout bypasses scale/activation')
    if type(m) is not int or type(n) is not int or min(m, n) < 0:
        raise ValueError('bias extents must be nonnegative integers')
    width = 4 if output_dtype == 'i32' else 1
    enc.fence(fn)
    enc.config_load(fn, n * 4, stream=0, shrunk=False)
    enc.config_load(fn, 0, stream=1, shrunk=False)
    enc.config_store(fn, n * width, scale=scale, activation=int(bool(epilogue)))

    def row_block(row, rows):
        def column(col, cols):
            row_offset = fn.mul_i(row, fn.const(n)) if not isinstance(row, int) else fn.const(row * n)
            col_value = fn.const(col) if isinstance(col, int) else col
            offset = fn.add_i(row_offset, col_value)
            enc.mvin(fn, _ptr(fn, src, offset, 4), enc.ACC, rows, cols, stream=0)
            enc.fence(fn)
            enc.mvin(fn, _ptr(fn, bias, col, 4), enc.ACC_ADD, rows, cols, stream=1)
            enc.fence(fn)
            enc.mvout(fn, _ptr(fn, dst, offset, width),
                      enc.ACC_FULL if output_dtype == 'i32' else enc.ACC, rows, cols)
            enc.fence(fn)
        end = n // 16 * 16
        if end:
            fn.for_loop(0, end, 16, lambda col: column(col, 16), retain_loop=True)
        if n != end:
            column(end, n - end)

    end = m // 16 * 16
    if end:
        fn.for_loop(0, end, 16, lambda row: row_block(row, 16), retain_loop=True)
    if end != m:
        row_block(end, m - end)
    return {'input_dtype': 'i32', 'bias_dtype': 'i32', 'output_dtype': output_dtype,
            'route': 'accumulator_load_accumulate_readout'}
