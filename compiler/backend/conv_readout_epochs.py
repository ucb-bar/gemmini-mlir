"""Uniform readout epochs for private, fully reduced convolution panels.

The producer's real readiness drain and the consumer's real last-use drain
surround this store-only region. CONFIG_ST is invariant throughout the region,
so no per-command configuration snapshot assumption is needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import isa
from .column_batches import _admission


@dataclass(frozen=True)
class UniformConvolutionReadout:
    command: tuple
    full: bool
    pitch: int
    admission: dict

    def emit(self, ctx, dst, fields, builder):
        """Project unchanged logical coordinates to original physical output."""
        offset = builder.op('add', builder.op('mul', fields['output_row'], self.pitch),
                            fields['output_column'])
        ctx.emit(*isa.mvout(ctx.ptr(dst, offset),
            isa.acc_addr(fields['accumulator_row'], full=self.full, fb=ctx.fb),
            fields['rows'], fields['columns'], fb=ctx.fb))

    def record(self):
        return dict(asdict(self),
            readiness='existing full reduction completion before CONFIG_ST',
            ownership='existing full output-last-use completion after every panel store',
            configuration='one invariant CONFIG_ST; no snapshots assumed',
            pending_capacity='selected finite store entries; ordinary backpressure')


def prepare_uniform_readout(dtype, pitch, attrs, group):
    """Admit before mutation, using the actual common private panel mapping."""
    group.__post_init__()
    if dtype not in ('i8', 'i32') or type(pitch) is not int or pitch <= 0:
        raise ValueError('uniform convolution readout requires original output type and pitch')
    scales = attrs.get('acc_scale_by_column')
    if scales is not None and (not scales or len(set(map(float, scales))) != 1):
        raise ValueError('uniform readout cannot change per-column scales')
    scale = scales[0] if scales else attrs.get('acc_scale', 1.0)
    activation = bool(attrs.get('relu') or 'relu' in attrs.get('epilogue', ()))
    full = dtype == 'i32'
    if full and (scales is not None or float(scale) != 1.0 or activation):
        raise ValueError('raw i32 readout cannot apply an epilogue')
    # Reopen selected store-lane/configuration ordering, finite capacity and
    # completion order. The original real producer drain supplies visibility.
    admission = _admission()
    admission['store_configuration_snapshot'] = 'not assumed: invariant CONFIG_ST for whole panel'
    spans = tuple((group.accumulator_row(0, n), group.panel.tile)
                  for n in range(group.panel.column_blocks))
    if any(lo < other + size and other < lo + rows
           for i, (lo, rows) in enumerate(spans) for other, size in spans[i+1:]):
        raise ValueError('convolution panel stores overlap private ACC rows')
    admission['actual_private_acc_spans'] = spans
    return UniformConvolutionReadout(
        isa.config_st(pitch * (4 if full else 1), scale=scale, activation=int(activation)),
        full, pitch, admission)
