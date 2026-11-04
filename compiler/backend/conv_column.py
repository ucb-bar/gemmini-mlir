"""Optional full-width NHWC convolution with exact device column readout.

The common convolution producer writes every biased i32 accumulator bit once.
Strided MVIN3 then places one output column in accumulator lane zero; the shared
readout applies its source f32 scale, rectifier and saturation on the device.
No tensor arithmetic or configuration-table loads execute on the host.
"""
from __future__ import annotations

from math import prod
from merlin.compile.scheduling.window_transfer import Window2D
from . import isa
from .dense_readout import store_tile
from .column_batches import emit_batched_column_readout


def lower_conv_column_readout(ctx, ifm, rhs, dst, attributes, *, producer=None):
    """Lower original convolution attributes through raw i32 private storage.

    Pooling must already have been separated by the caller. The producer's
    ``conv_column_readout='recompute'`` override prevents recursive selection.
    This function makes no global tuning choice and retains completion edges.
    """
    from .conv import lower_conv
    attrs = dict(attributes or {})
    ep = attrs.get('epilogue', ())
    if any('pool' in str(stage).lower() for stage in ep):
        raise ValueError('column reload readout requires separate spatial pooling')
    if attrs.get('output_dtype', ctx.spec(dst)['dtype']) != 'i8':
        raise ValueError('convolution column reload requires an i8 destination')
    if attrs.get('requant_shift') is not None and 'requant' in ep:
        raise ValueError('half-up requantization has no exact Gemmini readout contract')
    shape = tuple(map(int, ctx.spec(ifm)['shape']))
    if len(shape) != 4 or attrs.get('layout', 'nhwc').lower() != 'nhwc':
        raise ValueError('convolution requires explicit rank-four NHWC input')
    kernel = tuple(map(int, attrs['kernel']))
    if len(kernel) != 4 or kernel[2] != shape[-1]:
        raise ValueError('convolution kernel/input channel geometry differs')
    kh, kw, ci, co = kernel
    padding = tuple(map(int, attrs.get('padding', (0, 0, 0, 0))))
    if len(padding) == 2:
        padding = (*padding, *padding)
    window = Window2D(shape, (kh, kw), tuple(attrs.get('stride', (1, 1))),
                      tuple(attrs.get('dilation', (1, 1))), padding)
    oh, ow = window.output_spatial
    m = shape[0] * oh * ow
    if prod(ctx.spec(dst)['shape']) != m * co:
        raise ValueError('convolution output shape differs from original source window')
    scales = attrs.get('acc_scale_by_column')
    if scales is not None and len(scales) != co:
        raise ValueError('convolution column scales differ from output channels')

    index = getattr(ctx, '_conv_column_scratch_count', 0)
    ctx._conv_column_scratch_count = index + 1
    raw = '_conv_column_raw_' + str(index)
    ctx.alloc(raw, (m, co), 'i32')
    product_attrs = dict(attrs)
    for name in ('acc_scale_by_column', 'acc_scale', 'relu', 'requant_shift'):
        product_attrs.pop(name, None)
    product_attrs['epilogue'] = [stage for stage in ep if stage == 'bias_add']
    product_attrs['output_dtype'] = 'i32'
    product_attrs['conv_column_readout'] = 'recompute'
    receipt = (producer or lower_conv)(ctx, ifm, rhs, raw, product_attrs)
    batching = emit_batched_column_readout(ctx, raw, dst, m, co, attrs)
    if batching is not None:
        receipt = dict(receipt)
        receipt['column_readout'] = dict(
            route='raw_reload', raw_storage_bytes=m * co * 4,
            raw_reload_bytes=m * co * 4, output_bytes=m * co,
            product_tile_columns=min(isa.DIM, co), output_rows=m,
            output_channels=co, bias='producer only', batching=batching)
        return receipt

    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(co * 4, channel=2, shrink=False))

    def column(col):
        def tile(row, rows):
            offset = fb.add_i(fb.mul_i(row, fb.const(co)), col)
            ctx.emit(*isa.mvin(ctx.ptr(raw, offset), isa.acc_addr(0),
                              rows, 1, channel=2, fb=fb))
            ctx.fence()
            store_tile(ctx, dst, 0, rows, 1, co, row, col, attrs)
        complete = m // isa.DIM * isa.DIM
        if complete:
            fb.for_loop(0, complete, isa.DIM,
                        lambda row: tile(row, isa.DIM), retain_loop=True)
        if complete < m:
            tile(fb.const(complete), m - complete)
    fb.for_loop(0, co, 1, column, retain_loop=True)
    ctx.fence()
    receipt = dict(receipt)
    receipt['column_readout'] = dict(
        route='raw_reload', raw_storage_bytes=m * co * 4,
        raw_reload_bytes=m * co * 4, output_bytes=m * co,
        product_tile_columns=min(isa.DIM, co), output_rows=m,
        output_channels=co, bias='producer only',
        completion='synchronous producer and column readout fences retained')
    return receipt
