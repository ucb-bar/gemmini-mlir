"""Bounded original-row raw convolution tiles with contiguous horizontal K.

The shared horizontal panel artifact owns complete immutable RHS placement.
Interior rows use flattened K; clipped scalar edge chunks shift the RHS row
to retain the exact original coefficient alignment. Exotic all-padding spans
keep the synchronous window consumer. A caller owns raw ACC rectangles,
source observers and subsequent readout/pooling semantics.
"""
from dataclasses import dataclass
from math import prod

from merlin.compile.scheduling.horizontal_convolution import plan_horizontal_convolution_panels
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan,Window2D,emit_convolution_tile
from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
from . import isa


@dataclass(frozen=True)
class Prepared:
    source: object
    panels: object
    epoch: object
    ordering: dict

    @property
    def identity(self):
        return self.source.identity

    @property
    def interior_columns(self):
        w=self.source.plan.window
        # Half-open original output columns whose full horizontal source
        # footprint belongs to the original input row.
        first=min(w.output_spatial[1],max(0,(w.padding[1]+w.stride[1]-1)//w.stride[1]))
        end=(w.input_shape[2]+w.padding[1]-w.kernel[1])//w.stride[1]+1
        return first,min(w.output_spatial[1],max(first,end))

    def verify(self):
        self.source.__post_init__();self.panels.__post_init__()
        if self.panels.source!=self.source.plan:
            raise ValueError('horizontal raw producer changed its original source')
        self.epoch.require(max(2,self.panels.reduction_tiles))


def prepare(ctx,ifm,rhs,attrs):
    """Read-only geometry/storage admission; readout remains caller-owned."""
    try:
        a,b=ctx.spec(ifm),ctx.spec(rhs)
        shape=tuple(a['shape']);kh,kw,ci,co=tuple(attrs['kernel'])
        if (len(shape)!=4 or a['dtype']!='i8' or b['dtype']!='i8'
                or shape[-1]!=ci or not 0<ci<isa.DIM or kw<=1
                or attrs.get('layout','nhwc').lower()!='nhwc'
                or prod(b['shape'])!=kh*kw*ci*co):return None
        padding=tuple(attrs.get('padding',(0,0,0,0)))
        if len(padding)==2:padding=(*padding,*padding)
        w=Window2D(shape,(kh,kw),tuple(attrs.get('stride',(1,1))),
            tuple(attrs.get('dilation',(1,1))),padding)
        if w.dilation[1]!=1 or w.output_spatial[1]<isa.DIM:return None
        chunks=(kw*ci+isa.DIM-1)//isa.DIM
        if chunks>=kw or kh*chunks>16 or kw*ci>4*isa.DIM:return None
        bias=attrs.get('bias')
        if bias and (ctx.spec(bias)['dtype']!='i32' or prod(ctx.spec(bias)['shape'])!=co):return None
        p=CountedConvolutionPlan(w,co,isa.DIM,isa.DIM,min(isa.DIM,co),min(isa.DIM,ci),
            isa.SPAD_ROWS,isa.ACC_ROWS,0,isa.SPAD_ROWS//2,0,bias=bool(bias))
        intervals=tuple((n,min(isa.DIM,co-n))for n in range(0,co,isa.DIM))
        if len(intervals)>32 or kh*chunks*len(intervals)>128:return None
        panels=plan_horizontal_convolution_panels(p,max_row_blocks=1,
            maximum_transfer_columns=4*isa.DIM,private_rows_owned=True)
        if panels is None or panels.operand_rows>p.rhs_row:return None
        source=ConvolutionExplorationSource(p,intervals,p.identity)
        from .conv_epochs import epoch_admissions
        _,epoch,_,ordering=epoch_admissions(source,'gemmini-fire-aligned-b8ff28edc096392c',
            'signed-i8-products-i32-source-ordered-readout',
            'raw-horizontal-region-private-rhs-and-acc-v1',complete_reduction=True)
        result=Prepared(source,panels,epoch,ordering);result.verify();return result
    except (KeyError,ValueError,TypeError,OverflowError):return None


def emit_prepare(prepared,builder,lower,complete):
    prepared.verify();p=prepared.source.plan;q=prepared.panels;ci=p.window.input_shape[-1]
    for ky in range(p.window.kernel[0]):
        for chunk in range(q.horizontal_chunks):
            slot=ky*q.horizontal_chunks+chunk
            depth=min(isa.DIM,q.horizontal_columns-chunk*isa.DIM)
            for n,(column,columns)in enumerate(q.column_intervals):
                lower('load_weight',dict(reduction_row=ky*q.horizontal_columns+chunk*isa.DIM,
                    output_column=column,local_row=q.rhs_base+(n*q.reduction_tiles+slot)*isa.DIM,
                    rows=depth,columns=columns))
    complete('horizontal_raw_rhs_ready',dict(local_row=q.rhs_base,local_rows=q.rhs_rows))
    return dict(mechanism='horizontal_raw_producer',source_sha256=prepared.source.identity,
        panels=q.summary(),ordering=prepared.ordering,
        completion='complete immutable RHS; selected interior/clipped-border epochs; synchronous all-padding fallback; raw-tile last-use',
        timing='UNKNOWN')


def emit_tile(prepared,builder,lower,complete,*,output_row,rows,output_column,columns,accumulator_row):
    """Emit one caller-owned contiguous ACC tile within one original output row."""
    return emit_tiles(prepared,builder,lower,complete,output_row=output_row,rows=rows,
        outputs=(dict(output_column=output_column,columns=columns,accumulator_row=accumulator_row),))


def emit_tiles(prepared,builder,lower,complete,*,output_row,rows,outputs):
    """Stage each horizontal LHS once for separate caller-owned N rectangles."""
    prepared.verify();p=prepared.source.plan;w=p.window;q=prepared.panels
    _,h,width,ci=w.input_shape;oh,ow=w.output_spatial;kh,kw=w.kernel
    outputs=tuple(dict(item)for item in outputs)
    if (type(rows)is not int or not 0<rows<=isa.DIM or not outputs
            or any(type(item['columns'])is not int or not 0<item['columns']<=isa.DIM for item in outputs)):
        raise ValueError('raw horizontal tile requires original primitive extents')
    for item in outputs:
        if type(item['output_column'])is int and item['output_column']%isa.DIM:
            raise ValueError('raw horizontal output columns must start at primitive boundaries')
    outs=tuple(dict(item,output_row=output_row,rows=rows)for item in outputs)
    y=builder.op('urem',builder.op('udiv',output_row,ow),oh)
    x=builder.op('urem',output_row,ow)
    image=builder.op('udiv',output_row,oh*ow)
    ix=builder.op('sub',builder.op('mul',x,w.stride[1]),w.padding[1])
    end=builder.op('add',ix,(rows-1)*w.stride[1]+kw)
    interior=builder.op('and',builder.cmp('ule',ix,width),builder.cmp('ule',end,width),'i1')
    def horizontal():
        if p.bias:
            for out in outs:lower('initialize_bias',out)
        for ky in range(kh):
            sy=builder.op('sub',builder.op('add',builder.op('mul',y,w.stride[0]),
                ky*w.dilation[0]),w.padding[0])
            valid=builder.cmp('ult',sy,h)
            base=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,h),sy),width*ci),
                builder.op('mul',ix,ci))
            for chunk in range(q.horizontal_chunks):
                depth=min(isa.DIM,q.horizontal_columns-chunk*isa.DIM)
                def stage(zero):
                    lower('load_input',dict(zero=zero,first_element=0 if zero else builder.op('add',base,chunk*isa.DIM),
                        row_stride_elements=w.stride[1]*ci,local_row=p.lhs_row,rows=rows,columns=depth))
                builder.branch(valid,lambda:stage(False),lambda:stage(True))
                slot=ky*q.horizontal_chunks+chunk
                for out in outs:
                    ni=builder.op('udiv',out['output_column'],isa.DIM)
                    rhs_row=builder.op('add',q.rhs_base,
                        builder.op('mul',builder.op('add',builder.op('mul',ni,q.reduction_tiles),slot),isa.DIM))
                    # Original row cache rewrite follows all consuming N
                    # panels; selected RAW/WAR/configuration waits apply.
                    lower('compute',dict(out,lhs_row=p.lhs_row,rhs_row=rhs_row,depth=depth,
                        first_reduction=(slot==0),bias=p.bias,fresh_weights=True))
        complete('horizontal_raw_tile_complete',dict(output_row=output_row,rows=rows,outputs=outs))
        for out in outs:lower('finish_raw_tile',out)
    def border():
        # Clipping a scalar original output window keeps each horizontal K
        # chunk contiguous. Moving its missing prefix to the RHS row address
        # aligns valid input lane zero with the same original coefficients;
        # omitted padding products are exact additive zeros. Every N panel
        # shares that input, including partial first/last chunks.
        if w.padding[1]<kw and w.padding[3]<kw:
            def scalar(r):
                scalar_outs=tuple(dict(out,output_row=builder.op('add',output_row,r),rows=1,
                    accumulator_row=builder.op('add',out['accumulator_row'],r))for out in outs)
                if p.bias:
                    for out in scalar_outs:lower('initialize_bias',out)
                left=builder.op('mul',builder.op('sub',builder.op('mul',builder.op('add',x,r),w.stride[1]),w.padding[1]),ci)
                missing=builder.select(builder.cmp('slt',left,0),builder.op('sub',0,left),0)
                first_chunk=builder.op('udiv',missing,isa.DIM)
                for ky in range(kh):
                    sy=builder.op('sub',builder.op('add',builder.op('mul',y,w.stride[0]),
                        ky*w.dilation[0]),w.padding[0])
                    valid_y=builder.cmp('ult',sy,h)
                    row_base=builder.op('mul',builder.op('add',builder.op('mul',image,h),sy),width*ci)
                    for chunk in range(q.horizontal_chunks):
                        depth=min(isa.DIM,q.horizontal_columns-chunk*isa.DIM)
                        start=builder.op('add',left,chunk*isa.DIM)
                        prefix=builder.select(builder.cmp('slt',start,0),builder.op('sub',0,start),0)
                        available=builder.op('sub',width*ci,start)
                        limit=builder.select(builder.cmp('slt',available,depth),available,depth)
                        live=builder.op('sub',limit,prefix)
                        def product(chunk=chunk,prefix=prefix,live=live,start=start):
                            def stage(zero):
                                lower('load_input',dict(zero=zero,
                                    first_element=0 if zero else builder.op('add',row_base,builder.op('add',start,prefix)),
                                    row_stride_elements=w.stride[1]*ci,local_row=p.lhs_row,rows=1,columns=live))
                            builder.branch(valid_y,lambda:stage(False),lambda:stage(True))
                            slot=ky*q.horizontal_chunks+chunk
                            for out in scalar_outs:
                                ni=builder.op('udiv',out['output_column'],isa.DIM)
                                rhs_row=builder.op('add',builder.op('add',q.rhs_base,prefix),
                                    builder.op('mul',builder.op('add',builder.op('mul',ni,q.reduction_tiles),slot),isa.DIM))
                                first=False if ky else builder.cmp('eq',first_chunk,chunk)
                                lower('compute',dict(out,lhs_row=p.lhs_row,rhs_row=rhs_row,depth=live,
                                    first_reduction=first,bias=p.bias,fresh_weights=True))
                        builder.branch(builder.cmp('slt',0,live),product,lambda:None)
                complete('horizontal_raw_clipped_border_complete',dict(output_row=output_row,rows=1,outputs=scalar_outs))
                for out in scalar_outs:lower('finish_raw_tile',out)
            builder.loop(0,rows,1,scalar)
            return
        def synchronous(kind,fields):
            lower('finish_raw_tile'if kind=='store'else kind,fields)
            complete('horizontal_raw_border_'+kind,fields)
        for out in outs:
            emit_convolution_tile(p,builder,synchronous,**out,synchronous_completion=True)
    builder.branch(interior,horizontal,border)
    prepared.verify()
