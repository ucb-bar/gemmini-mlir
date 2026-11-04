"""Checked compute epochs for original tiles with projected output/ACC rows.

The column-block emitter intentionally accepts only static disjoint descriptors.
This single-output adapter needs no cross-output retention premise: the original
source tile owns its existing private ACC interval until its final full drain.
"""
from dataclasses import asdict
from .window_transfer import (
    CountedConvolutionPlan, ConvolutionComputeEpochLaw,
    _validate_convolution_load_issue, _emit_convolution_tile_effects,
)


def emit_convolution_compute_epoch_tile(plan, builder, issue, complete, *,
        output_row, output_column, rows, columns, accumulator_row,
        law, epoch_law, compute_pairs):
    if type(plan) is not CountedConvolutionPlan or type(epoch_law) is not ConvolutionComputeEpochLaw:
        raise ValueError('tile epochs require original counted source and stronger selected law')
    before = _validate_convolution_load_issue(plan, builder, issue, complete, rows, law, int(plan.bias))
    epoch_law.require(compute_pairs)
    if any(getattr(law, key) != getattr(epoch_law, key) for key in (
            'hardware_identity', 'numerical_policy_identity', 'source_identity', 'qualification_identity')):
        raise ValueError('tile epoch and load issue select different source/target contracts')
    if epoch_law.source_identity != plan.identity:
        raise ValueError('tile epoch requires the exact original convolution source')
    before_epoch = asdict(epoch_law)
    ci = plan.window.input_shape[-1]
    chunks = (ci + plan.channel_tile - 1) // plan.channel_tile
    total = plan.window.kernel[0] * plan.window.kernel[1] * chunks
    pending = None
    stores = 0

    def lower(kind, fields):
        nonlocal pending, stores
        if kind == 'load_weight':
            if pending is not None:
                raise ValueError('tile epoch lost original RHS/compute pairing')
            pending = fields['reduction_row']
        issue(kind, fields)
        if kind == 'compute':
            if pending is None:
                raise ValueError('tile epoch compute has no original RHS descriptor')
            # Runtime original kh/kw/ci coordinates, never a Python template counter.
            tap = builder.op('udiv', pending, ci)
            chunk = builder.op('udiv', builder.op('urem', pending, ci), plan.channel_tile)
            pair = builder.op('add', builder.op('add', builder.op('mul', tap, chunks), chunk), 1)
            boundary = builder.cmp('eq', builder.op('urem', pair, compute_pairs), 0)
            unfinished = builder.cmp('ult', pair, total)
            builder.branch(builder.op('and', boundary, unfinished, 'i1'),
                lambda: complete(kind, fields), lambda: None)
            pending = None
        elif kind == 'store':
            if pending is not None or stores:
                raise ValueError('tile epoch changed original final observation')
            stores += 1
            # A retained raw tile's store adapter can be a no-op; this drain is
            # nevertheless real and precedes every subsequent halo/readout use.
            complete(kind, fields)

    _emit_convolution_tile_effects(plan, builder, lower,
        output_row=output_row, output_column=output_column, rows=rows,
        columns=columns, accumulator_row=accumulator_row)
    plan.__post_init__()
    law.__post_init__()
    epoch_law.require(compute_pairs)
    if (pending is not None or stores != 1 or before != (plan.identity, asdict(law))
            or before_epoch != asdict(epoch_law)):
        raise ValueError('tile epoch source, resources or selected laws changed during emission')
    return dict(schema='merlin.convolution_compute_epoch_tile.v1', source_identity=plan.identity,
        issue_law=asdict(law), epoch_law=before_epoch, compute_pairs=compute_pairs,
        total_pairs=total, index='(kh*kw+kx)*ceil(CI/channel_tile)+channel/channel_tile+1',
        completion='real intermediate source-index boundaries; full drain after original raw tile')
