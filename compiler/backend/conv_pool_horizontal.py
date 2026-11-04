"""Horizontal source-reduction tiles inside the original private pool block.

The shared horizontal plan provides complete K/N coverage and disjoint private
input/RHS layout. Only full horizontal source runs use its collapsed KW*CI
reduction; original border products retain the canonical convolution owner.
"""
from dataclasses import asdict,dataclass,replace
import hashlib,json

from merlin.compile.scheduling.horizontal_convolution import (
    HorizontalConvolutionPanels,plan_horizontal_convolution_panels,
)
from merlin.compile.scheduling.spatial_rhs_residency import (
    SpatialProducerRhsResidency,emit_spatial_producer_rhs_residency,
)


@dataclass(frozen=True)
class HorizontalSpatialTiles:
    block: object
    horizontal: HorizontalConvolutionPanels
    canonical_rhs: SpatialProducerRhsResidency

    def __post_init__(self):
        self.block.__post_init__();self.horizontal.__post_init__();self.canonical_rhs.__post_init__()
        p=self.block.source.producer
        if (self.horizontal.source!=p or self.canonical_rhs.source!=self.block.source
                or self.horizontal.row_blocks!=1
                or self.horizontal.operand_rows>self.canonical_rhs.panel.rhs_base
                or p.window.dilation[1]!=1):
            raise ValueError('horizontal pool source or disjoint live panel layout differs')

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def plan_horizontal_spatial_tiles(block,canonical_rhs):
    if canonical_rhs is None:return None
    p=block.source.producer;ci=p.window.input_shape[-1];kh,kw=p.window.kernel
    if ci>=p.primitive or kw<=1 or p.window.dilation[1]!=1:return None
    horizontal=plan_horizontal_convolution_panels(p,max_row_blocks=1,
        maximum_transfer_columns=4*p.primitive,private_rows_owned=True)
    if (horizontal is None or horizontal.horizontal_chunks>=kw
            or horizontal.operand_rows>canonical_rhs.panel.rhs_base):return None
    return HorizontalSpatialTiles(block,horizontal,canonical_rhs)


def emit_horizontal_spatial_blocks(owner,epochs,builder,lower,complete):
    """Consume horizontal interior tiles and canonical exact source borders."""
    from .conv_pool_blocks import emit_spatial_blocks,_emit_epoch_tile
    owner.__post_init__();identity=owner.identity
    plan=owner.horizontal;p=plan.source;w=p.window;d=p.primitive
    batch,height,width,ci=w.input_shape;kh,kw=w.kernel;oh,ow=w.output_spatial
    sh,sw=w.stride;dh,dw=w.dilation;pt,pl,pb,pr=w.padding;op=builder.op
    # The finite execute queue backpressures ordinary ordered descriptor/pairs.
    # All operands are immutable across this issued-work epoch; no pending
    # queue capacity is inferred from the larger complete-reduction bound.
    epoch=replace(epochs.epoch_law,max_compute_pairs=max(
        epochs.compute_pairs,plan.reduction_tiles))
    epoch.require(plan.reduction_tiles)
    for ky in range(kh):
        for chunk in range(plan.horizontal_chunks):
            slot=ky*plan.horizontal_chunks+chunk
            depth=min(d,plan.horizontal_columns-chunk*d)
            for n,(column,columns)in enumerate(plan.column_intervals):
                lower('load_weight',dict(reduction_row=ky*plan.horizontal_columns+chunk*d,
                    output_column=column,local_row=plan.rhs_base+(n*plan.reduction_tiles+slot)*d,
                    rows=depth,columns=columns))
    complete('horizontal_pool_rhs_ready',{})

    def tile(consumer,**coordinates):
        interval=coordinates.pop('source_x_interval',None)
        row=coordinates['output_row'];rows=coordinates['rows']
        x=op('urem',row,ow);image=op('udiv',row,oh*ow)
        y=op('urem',op('udiv',row,ow),oh)
        ix=op('sub',op('mul',x,sw),pl)
        last=op('add',ix,(rows-1)*sw+kw)
        valid=op('and',builder.cmp('ult',ix,width),builder.cmp('ule',last,width),'i1')
        def horizontal():
            if p.bias:lower('initialize_bias',coordinates)
            # Every input slot is written once, from valid original bytes OR
            # the selected zero source. The complete stage precedes all reads.
            for ky in range(kh):
                iy=op('add',op('sub',op('mul',y,sh),pt),ky*dh)
                in_y=builder.cmp('ult',iy,height)
                def stage(zero,ky=ky,iy=iy):
                    first=0 if zero else op('add',op('mul',op('add',op('mul',image,height),iy),width*ci),op('mul',ix,ci))
                    for chunk in range(plan.horizontal_chunks):
                        slot=ky*plan.horizontal_chunks+chunk
                        lower('load_input',dict(zero=zero,
                            first_element=0 if zero else op('add',first,chunk*d),
                            local_row=slot*plan.input_block_stride,rows=rows,
                            columns=min(d,plan.horizontal_columns-chunk*d),
                            row_stride_elements=sw*ci))
                builder.branch(in_y,lambda:stage(False),lambda:stage(True))
            complete('horizontal_pool_inputs_ready',coordinates)
            column_slot=op('udiv',coordinates['output_column'],p.column_tile)
            for ky in range(kh):
                for chunk in range(plan.horizontal_chunks):
                    slot=ky*plan.horizontal_chunks+chunk
                    lower('compute',dict(coordinates,
                        lhs_row=slot*plan.input_block_stride,
                        rhs_row=op('add',plan.rhs_base,op('mul',
                            op('add',op('mul',column_slot,plan.reduction_tiles),slot),d)),
                        depth=min(d,plan.horizontal_columns-chunk*d),
                        first_reduction=slot==0,bias=p.bias,fresh_weights=True))
            complete('horizontal_pool_raw_tile_complete',coordinates)
        def canonical():
            _emit_epoch_tile(owner.block,epochs,builder,consumer,complete,**coordinates)
        if interval is None:
            builder.branch(valid,horizontal,canonical)
        else:
            # The common block decomposition owns this entire original x
            # interval. Its static endpoints discharge the same byte-range
            # predicate for every runtime tile, including the tail.
            start,stop=interval
            if start*sw-pl>=0 and (stop-1)*sw-pl+kw<=width:
                horizontal()
            else:
                canonical()

    receipt=emit_spatial_producer_rhs_residency(owner.canonical_rhs,builder,lower,complete,
        lambda consumer:emit_spatial_blocks(owner.block,builder,consumer,
            synchronous_completion=False,epochs=epochs,complete=complete,
            tile_emitter=lambda **coords:tile(consumer,**coords),
            producer_x_cuts=(min(ow, (pl+sw-1)//sw),
                max(0, min(ow, (width-kw+pl)//sw+1)))))
    if owner.identity!=identity:raise ValueError('horizontal spatial ownership changed')
    return dict(receipt,horizontal_tiles=dict(schedule_sha256=identity,
        plan=plan.summary(),source_reduction='ky then original q=kx*CI+c; exact contiguous source order',
        borders='canonical resident source products where any horizontal byte is outside original input',
        epoch=asdict(epoch),completion='complete all private input slots; complete raw tile before pool readout'))
