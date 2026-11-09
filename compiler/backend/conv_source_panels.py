"""Source-axis projection of checked spatial convolution input fragments.

The resident-panel scheduler already retains RHS across the entire batch. This
adapter changes only address formation: consume the shared gather's exact row
partition and project its logical output row directly onto the original NHWC
source. No padded payload, extra transfer, storage or completion is introduced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from merlin.compiler.schedule.affine_gather_runs import (
    ConvolutionGatherRuns, emit_convolution_gather_runs,
)
from merlin.compiler.schedule.window_transfer import Window2D


@dataclass(frozen=True)
class SourceCoordinatePlan:
    source_identity: str
    window: Window2D
    gather: ConvolutionGatherRuns
    lhs_row: int

    def __post_init__(self):
        w, g = self.window, self.gather
        if type(w) is not Window2D or type(g) is not ConvolutionGatherRuns:
            raise ValueError('exact original window and checked gather required')
        g.__post_init__()
        b, h, width, ci = w.input_shape
        pt, pl, pb, pr = w.padding
        if (not self.source_identity or b <= 1
                or g.input_shape != (b, h+pt+pb, width+pl+pr, ci)
                or g.output_spatial != (b, *w.output_spatial)
                or g.kernel != w.kernel or g.stride != w.stride
                or g.dilation != w.dilation
                or type(self.lhs_row) is not int or self.lhs_row < 0):
            raise ValueError('source projection must preserve every original convolution axis')

    @property
    def realization(self):
        record = asdict(self)
        return dict(mechanism='checked_convolution_output_axis_projection',
            plan_sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
            source_identity=self.source_identity, window=asdict(self.window),
            gather=asdict(self.gather), lhs_row=self.lhs_row,
            projection='position=first+local; image=position//(OH*OW); y=(position//OW)%OH; x=position%OW',
            effects='original shared fragment rows and exact zero/valid/zero bands',
            storage='unchanged admitted resident panel and output group',
            completion='all original completion callbacks retained',
            physical_authority='none; source address projection only')


def plan_source_coordinates(prepared):
    """Read-only optional admission; batch one keeps its existing shortcut."""
    try:
        prepared.group.__post_init__()
        return SourceCoordinatePlan(prepared.source.identity,
            prepared.source.plan.window, prepared.gather, prepared.group.lhs_row(0))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def emit_source_input_fragments(plan, prepared, builder, first, count,
                               ky, kx, channel, depth, transfer):
    """Consume actual shared fragments with exactly the original input effects."""
    plan.__post_init__()
    if (prepared.source.identity != plan.source_identity
            or prepared.source.plan.window != plan.window
            or prepared.gather != plan.gather
            or prepared.group.lhs_row(0) != plan.lhs_row):
        raise ValueError('stale convolution source coordinate projection')
    w = plan.window
    _, h, width, ci = w.input_shape
    oh, ow = w.output_spatial
    pt, pl, _, _ = w.padding
    sh, sw = w.stride
    dh, dw = w.dilation
    # Origin remains supplied by the checked shared partition. Its local row
    # identifies the same original output position without decoding an input
    # byte offset back into padded coordinates. Shared coordinate SSA can CSE.
    for _, local, rows in emit_convolution_gather_runs(plan.gather, builder, first, count):
        def active(local=local, rows=rows):
            position = builder.op('add', first, local)
            image = builder.op('udiv', position, oh*ow)
            oy = builder.op('urem', builder.op('udiv', position, ow), oh)
            ox = builder.op('urem', position, ow)
            vy = builder.op('add', builder.op('mul', oy, sh), builder.op('mul', ky, dh))
            vx = builder.op('add', builder.op('mul', ox, sw), builder.op('mul', kx, dw))

            def issue(zero, start, length):
                def nonempty():
                    ix = builder.op('sub', builder.op('add', vx, builder.op('mul', start, sw)), pl)
                    physical = 0 if zero else builder.op('add', builder.op('mul',
                        builder.op('add', builder.op('mul', image, h), builder.op('sub', vy, pt)), width*ci),
                        builder.op('add', builder.op('mul', ix, ci), channel))
                    transfer(dict(zero=zero, first_element=physical,
                        local_row=builder.op('add', plan.lhs_row, builder.op('add', local, start)),
                        rows=length, columns=depth, row_stride_elements=sw*ci))
                builder.branch(builder.cmp('ult', 0, length), nonempty, lambda: None)

            def in_row():
                def before(limit):
                    delta = builder.select(builder.cmp('ult', vx, limit), builder.op('sub', limit, vx), 0)
                    n = builder.op('udiv', builder.op('add', delta, sw-1), sw)
                    return builder.select(builder.cmp('ult', n, rows), n, rows)
                prefix, end = before(pl), before(pl+width)
                issue(True, 0, prefix)
                issue(False, prefix, builder.op('sub', end, prefix))
                issue(True, end, builder.op('sub', rows, end))

            from .conv_spatial_panels import _predicate_and
            valid = _predicate_and(builder, builder.cmp('ule', pt, vy), builder.cmp('ult', vy, pt+h))
            builder.branch(valid, in_row, lambda: issue(True, 0, rows))
        builder.branch(builder.cmp('ult', 0, rows), active, lambda: None)
