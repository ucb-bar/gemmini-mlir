"""Invocation-resident complete RHS for a private convolution/pool producer.

This source transformation owns preparation and remaps canonical producer
weight effects. It grants no asynchronous completion or target encoding.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

from .affine_gather_runs import ConvolutionResidentPanel, plan_convolution_resident_panel
from .block_schedule import Geometry
from .spatial_producer import SpatialProducerPlan


@dataclass(frozen=True)
class SpatialProducerRhsResidency:
    source: SpatialProducerPlan
    panel: ConvolutionResidentPanel

    def __post_init__(self):
        if type(self.source)is not SpatialProducerPlan or type(self.panel)is not ConvolutionResidentPanel:
            raise ValueError('spatial residency requires the checked original producer and resident panel')
        self.source.__post_init__();self.panel.__post_init__()
        p=self.source.producer;kh,kw=p.window.kernel;ci=p.window.input_shape[-1]
        if (p.channel_tile!=min(p.primitive,ci)
                or p.column_tile!=min(p.primitive,p.output_channels)
                or self.panel.tile!=p.primitive
                or self.panel.channel_tiles!=(ci+p.channel_tile-1)//p.channel_tile
                or self.panel.reduction_tiles!=kh*kw*self.panel.channel_tiles
                or self.panel.column_blocks!=(p.output_channels+p.column_tile-1)//p.column_tile
                or self.panel.rhs_base<p.lhs_row+p.row_tile
                or self.panel.operand_capacity_rows!=p.operand_rows):
            raise ValueError('spatial resident panel differs from complete canonical K/N domain')

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def plan_spatial_producer_rhs_residency(source, *, private_rows_owned=False,
                                       immutable_disjoint_buffers=False):
    if private_rows_owned is not True or immutable_disjoint_buffers is not True:
        return None
    if type(source)is not SpatialProducerPlan:return None
    source.__post_init__();p=source.producer;kh,kw=p.window.kernel
    ci=p.window.input_shape[-1]
    panel=plan_convolution_resident_panel((kh,kw,ci,p.output_channels),
        Geometry(p.primitive,p.operand_rows,None,p.accumulator_rows,True),private_rows_owned=True)
    if panel is None or panel.column_blocks!=(p.output_channels+p.column_tile-1)//p.column_tile:
        return None
    return SpatialProducerRhsResidency(source,panel)


def emit_spatial_producer_rhs_residency(owner,builder,issue,complete,emit_source):
    """Prepare once; consume every canonical K/N source through one encoder.

    ``emit_source(effect)`` must emit the original complete SpatialProducerPlan,
    including its real pool/last-use completion. Its per-tile weight transfer is
    replaced with the matching immutable resident view. Pending source weight
    coordinates remain paired with their immediately consuming product.
    """
    if type(owner)is not SpatialProducerRhsResidency or not callable(emit_source):
        raise ValueError('original resident producer and actual source consumer required')
    owner.__post_init__();identity=owner.identity;p=owner.source.producer;panel=owner.panel
    ci=p.window.input_shape[-1];kh,kw=p.window.kernel;op=builder.op
    def row(reduction,column):
        tap=op('udiv',reduction,ci);channel=op('urem',reduction,ci)
        slot=op('add',op('mul',tap,panel.channel_tiles),op('udiv',channel,p.channel_tile))
        return op('add',panel.rhs_base,op('mul',op('add',op('mul',slot,panel.column_blocks),
            op('udiv',column,p.column_tile)),p.primitive))
    def tap(t):
        def channel(c,depth):
            reduction=op('add',op('mul',t,ci),c)
            for column in range(0,p.output_channels,p.column_tile):
                issue('load_weight',dict(reduction_row=reduction,output_column=column,
                    local_row=row(reduction,column),rows=depth,
                    columns=min(p.column_tile,p.output_channels-column)))
        full,tail=divmod(ci,p.channel_tile)
        if full:builder.loop(0,full,1,lambda c:channel(op('mul',c,p.channel_tile),p.channel_tile))
        if tail:channel(full*p.channel_tile,tail)
    builder.loop(0,kh*kw,1,tap)
    complete('immutable_spatial_rhs_ready',{})
    pending=None
    def consume(kind,fields):
        nonlocal pending
        if kind=='load_weight':
            if pending is not None:raise ValueError('resident source weight lacks immediate consumer')
            pending=(fields['reduction_row'],fields['output_column'])
            return
        if kind=='compute':
            if pending is None:raise ValueError('resident source product lacks its canonical K/N weight')
            fields=dict(fields,rhs_row=row(*pending));pending=None
        issue(kind,fields)
    receipt=emit_source(consume)
    if pending is not None:raise ValueError('resident source ended with unconsumed weight')
    complete('immutable_spatial_rhs_last_use',{})
    if owner.identity!=identity:raise ValueError('resident source ownership changed during emission')
    return dict(receipt,rhs_residency=dict(source_sha256=owner.source.identity,
        schedule_sha256=identity,panel=asdict(panel),
        preparation='one complete immutable original K/N panel per invocation',
        consumers='canonical source reduction pairs remapped to prepared private rows',
        completion='real preparation and invocation last-use drains; original pool drains retained'))
