"""Exact scale-palette producers and device-only column permutation.

Columns with identical binary32 readout words share a wide contraction. Weight
and bias permutations are DMA copies; each product uses the ordinary owner and
encoder. The final i8 scatter applies identity readout, after the original
rounding, activation and saturation have already happened exactly once.
"""
from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from . import isa
from .column_batches import _admission, _column_at


def selected(attributes):
    route = attributes.get('column_readout_route', knob('COLUMN_READOUT_ROUTE',
        default='raw_reload', values=('raw_reload', 'palette_producer', 'raw_pipeline'),
        root=Path(__file__).resolve().parents[1]))
    if route not in ('raw_reload', 'palette_producer', 'raw_pipeline'):
        raise ValueError('COLUMN_READOUT_ROUTE must be raw_reload, palette_producer or raw_pipeline')
    return route == 'palette_producer'


def _plan(ctx, lhs, rhs, dst, m, k, n, attrs):
    if getattr(ctx, '_is_conv_channel_view', False):
        return None
    scales = attrs.get('acc_scale_by_column')
    inputs = (lhs, rhs, attrs.get('bias'))
    if dst in inputs:
        return None
    buffers = getattr(ctx, 'buffers', {})
    output = buffers.get(dst)
    if output is not None and any(output is buffers.get(name) for name in inputs if name):
        return None
    if (scales is None or len(scales) != n or ctx.spec(rhs)['dtype'] != 'i8'
            or ctx.spec(dst)['dtype'] != 'i8' or attrs.get('accumulator_init')
            or any('pool' in str(stage).lower() for stage in attrs.get('epilogue', ()))
            or attrs.get('requant_shift') is not None):
        return None
    groups = {}
    for c, scale in enumerate(scales):
        groups.setdefault(isa.f32_bits(scale), []).append(c)
    # Wide groups are the opportunity. Distinct individual gains retain the
    # existing exact raw-reload implementation rather than expanding N owners.
    if not 1 < len(groups) <= 16 or max(map(len, groups.values())) < 2:
        return None
    if prod(ctx.spec(rhs)['shape']) != k*n or prod(ctx.spec(dst)['shape']) != m*n:
        return None
    bias = attrs.get('bias')
    if bias and (ctx.spec(bias)['dtype'] != 'i32' or prod(ctx.spec(bias)['shape']) != n):
        return None
    if 'bias_add' in attrs.get('epilogue', ()) and not bias:
        return None
    if (m*n + k*n + (4*n if bias else 0)) > 256*1024*1024:
        return None
    columns = [c for group in groups.values() for c in group]
    if sorted(columns) != list(range(n)):
        raise ValueError('readout palette does not partition original columns')
    return tuple((bits, tuple(cs)) for bits, cs in groups.items())


def _copy_columns(ctx, src, dst, rows, source_pitch, dest_pitch, columns, *, gather):
    """Copy each original column once through disjoint lane-zero ACC windows."""
    dtype = ctx.spec(src)['dtype']
    if dtype != ctx.spec(dst)['dtype'] or dtype not in ('i8', 'i32'):
        raise ValueError('palette permutation requires equal integer element types')
    full = dtype == 'i32'
    size = 4 if full else 1
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(source_pitch*size, channel=2, shrink=not full))
    ctx.emit(*isa.config_st(dest_pitch*size, scale=1.0))

    def tile(row, extent):
        capacity = isa.ACC_ROWS // extent
        def batch(first, count):
            def transfer(slot, load):
                index = fb.add_i(first, slot)
                original = _column_at(fb, index, columns)
                column = original if (gather == load) else index
                pitch = source_pitch if load else dest_pitch
                offset = fb.add_i(fb.mul_i(row, fb.const(pitch)), column)
                local = isa.acc_addr(fb.mul_i(slot, fb.const(extent)),
                                     full=full and not load, fb=fb)
                if load:
                    ctx.emit(*isa.mvin(ctx.ptr(src, offset), local, extent, 1, channel=2, fb=fb))
                else:
                    ctx.emit(*isa.mvout(ctx.ptr(dst, offset), local, extent, 1, fb=fb))
            fb.for_loop(0, count, 1, lambda slot: transfer(slot, True), retain_loop=True)
            ctx.fence()
            fb.for_loop(0, count, 1, lambda slot: transfer(slot, False), retain_loop=True)
            ctx.fence()
        complete = len(columns)//capacity*capacity
        if complete:
            fb.for_loop(0, complete, capacity, lambda first: batch(first, capacity), retain_loop=True)
        if complete < len(columns):
            batch(fb.const(complete), len(columns)-complete)
    complete = rows//isa.DIM*isa.DIM
    if complete:
        fb.for_loop(0, complete, isa.DIM, lambda row: tile(row, isa.DIM), retain_loop=True)
    if complete < rows:
        tile(fb.const(complete), rows-complete)
    ctx.fence()


def lower(ctx, lhs, rhs, dst, *, m, k, n, attributes, convolution=False):
    """Return None before mutation on refusal; otherwise emit complete products."""
    attrs = dict(attributes)
    groups = _plan(ctx, lhs, rhs, dst, m, k, n, attrs)
    if groups is None:
        return None
    admission = _admission()
    serial = getattr(ctx, '_readout_palette_serial', 0)
    ctx._readout_palette_serial = serial+1
    receipts = []
    for index, (bits, columns) in enumerate(groups):
        count = len(columns)
        prefix = '_readout_palette_%d_%d_' % (serial, index)
        weight, output = prefix+'weight', prefix+'output'
        ctx.alloc(weight, (k, count), 'i8')
        ctx.alloc(output, (m, count), 'i8')
        _copy_columns(ctx, rhs, weight, k, n, count, columns, gather=True)
        group_attrs = dict(attrs)
        group_attrs.pop('acc_scale_by_column', None)
        group_attrs['acc_scale'] = attrs['acc_scale_by_column'][columns[0]]
        group_attrs['column_readout_route'] = 'raw_reload'
        if attrs.get('bias'):
            bias = prefix+'bias'
            ctx.alloc(bias, (count,), 'i32')
            _copy_columns(ctx, attrs['bias'], bias, 1, n, count, columns, gather=True)
            group_attrs['bias'] = bias
        if convolution:
            from .conv import lower_conv
            group_attrs['kernel'] = [*attrs['kernel'][:-1], count]
            receipt = lower_conv(ctx, lhs, weight, output, group_attrs)
        else:
            from .dense import lower_dense
            receipt = lower_dense(ctx, lhs, weight, output, m=m, k=k, n=count, attributes=group_attrs)
        _copy_columns(ctx, output, dst, m, count, n, columns, gather=False)
        receipts.append(dict(columns=list(columns), scale_bits=bits, producer=receipt))
    identity = hashlib.sha256(json.dumps(dict(m=m,k=k,n=n,groups=groups), sort_keys=True).encode()).hexdigest()
    receipt = dict(column_readout=dict(route='palette_producer', source_partition_sha256=identity,
        groups=receipts, product_columns=sum(len(cs) for _,cs in groups),
        contraction='each original output column computed once',
        readout='original binary32 scale, activation and saturation in uniform producer',
        permutation='exact weight/bias gather and already-i8 output scatter',
        host_payload_arithmetic=False, raw_i32_output_bytes=0,
        private_storage_bytes=m*n+k*n+(4*n if attrs.get('bias') else 0),
        admission=admission, target_encoder='backend.isa', timing='UNKNOWN'))
    ctx.metadata.setdefault('palette_producers', []).append(receipt)
    return receipt
