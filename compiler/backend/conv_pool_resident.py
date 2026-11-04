"""Explicit immutable complete-RHS product for the existing pooled ACC owner.

The shared resident-panel artifact supplies the actual injective row layout.
Original canonical producer tiles remain the source of all compute coordinates;
only repeated immutable RHS loads are replaced by one completed preparation.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
import json
from merlin.compile.scheduling.affine_gather_runs import plan_convolution_resident_panel
from merlin.compile.scheduling.block_schedule import Geometry
from . import isa

@dataclass(frozen=True)
class PoolResidentRHS:
    source:object
    panel:object
    source_sha256:str
    panel_sha256:str

    def revalidate(self,source):
        source.__post_init__();self.panel.__post_init__()
        if source is not self.source or source.identity!=self.source_sha256:
            raise ValueError('pool resident RHS changed its complete source/block owner')
        expected=_panel(source)
        if self.panel!=expected or self.panel_sha256!=_digest(self.panel):
            raise ValueError('pool resident RHS changed its shared source/resource layout')

    def row(self,builder,reduction,column):
        p=self.source.source.producer;panel=self.panel
        tap=builder.op('udiv',reduction,p.window.input_shape[-1])
        chunk=builder.op('udiv',builder.op('urem',reduction,p.window.input_shape[-1]),p.channel_tile)
        n=builder.op('udiv',column,p.column_tile)
        slot=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',tap,panel.channel_tiles),chunk),panel.column_blocks),n)
        return builder.op('add',panel.rhs_base,builder.op('mul',slot,panel.tile))

    def prepare(self,builder,lower,complete):
        self.revalidate(self.source)
        p=self.source.source.producer;w=p.window;ci=w.input_shape[-1];kh,kw=w.kernel
        def tap_y(ky):
            def tap_x(kx):
                def channel(chunk,depth):
                    reduction=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',ky,kw),kx),ci),chunk*p.channel_tile if type(chunk)is int else builder.op('mul',chunk,p.channel_tile))
                    def column(n,columns):
                        col=n*p.column_tile if type(n)is int else builder.op('mul',n,p.column_tile)
                        lower('load_weight',dict(reduction_row=reduction,output_column=col,
                            local_row=self.row(builder,reduction,col),rows=depth,columns=columns))
                    full,tail=divmod(p.output_channels,p.column_tile)
                    if full:builder.loop(0,full,1,lambda n:column(n,p.column_tile))
                    if tail:column(full,tail)
                full,tail=divmod(ci,p.channel_tile)
                if full:builder.loop(0,full,1,lambda c:channel(c,p.channel_tile))
                if tail:channel(full,tail)
            builder.loop(0,kw,1,tap_x)
        builder.loop(0,kh,1,tap_y)
        complete('pool_resident_rhs_ready',{})
        self.revalidate(self.source)

    def bind(self,builder,lower):
        """Each canonical descriptor still binds its exact reduction/N tile."""
        self.revalidate(self.source)
        current=[None]
        def consume(kind,fields):
            if kind=='load_weight':
                if current[0] is not None:raise ValueError('resident pool lost original descriptor/product pairing')
                current[0]=self.row(builder,fields['reduction_row'],fields['output_column'])
                return
            if kind=='compute':
                if current[0] is None:raise ValueError('resident pool product lacks original weight coordinate')
                fields=dict(fields,rhs_row=current[0]);current[0]=None
            lower(kind,fields)
        return consume

    def receipt(self):
        return dict(source_block_sha256=self.source_sha256,shared_panel=asdict(self.panel),
            shared_panel_sha256=self.panel_sha256,
            ownership='complete immutable RHS prepared and drained before any producer; LHS disjoint; pool/ACC epoch unchanged',
            mapping='original tap/channel/N -> shared reduction-major resident row; original products and output coordinates')

def _digest(panel):return sha256(json.dumps(asdict(panel),sort_keys=True).encode()).hexdigest()

def _panel(source):
    p=source.source.producer;w=p.window
    panel=plan_convolution_resident_panel((*w.kernel,w.input_shape[-1],p.output_channels),
        Geometry(isa.DIM,p.operand_rows,None,p.accumulator_rows,True),private_rows_owned=True)
    if (panel is None or panel.column_blocks!=(p.output_channels+p.column_tile-1)//p.column_tile
            or p.column_tile!=min(isa.DIM,p.output_channels) or p.channel_tile!=min(isa.DIM,w.input_shape[-1])
            or p.lhs_row+p.row_tile>panel.rhs_base):return None
    return panel

def plan_pool_resident(source):
    from .conv_pool_blocks import SpatialBlockPlan
    if type(source)is not SpatialBlockPlan:return None
    source.__post_init__();panel=_panel(source)
    if panel is None:return None
    result=PoolResidentRHS(source,panel,source.identity,_digest(panel));result.revalidate(source)
    return result
