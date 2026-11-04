"""Horizontal-K convolution panels without im2col or host payload movement.

The shared horizontal-convolution transform partitions q=kx*CI+c into primitive
chunks. Every chunk is a contiguous original NHWC row span, even when it crosses
an original kx boundary. The RHS row is ky*(KW*CI)+q, exactly the original
KH/KW/CI/CO order. Original output pixels stay M rows; gathers split only at
output-row/image boundaries. Vertical halo rows use the device zero source.

Horizontal boundary columns use affine vertical output-pixel runs, with exact
zero/valid/zero source bands. Only complete horizontal spans enter the flattened
interior. This module does not change a global schedule default.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod

from merlin.compile.scheduling.horizontal_convolution import (
    emit_horizontal_convolution_panels, plan_horizontal_convolution_panels,
)
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.compile.scheduling.affine_gather_runs import (
    ConvolutionGatherRuns, emit_convolution_matrix_gather,
    plan_convolution_resident_panel,
)
from merlin.compile.scheduling.block_schedule import Geometry
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from . import isa
from .dense_readout import store_tile


@dataclass(frozen=True)
class _Prepared:
    panels: object
    original: object
    first_column: int
    input_x: int
    boundary_gather: object
    boundary_panel: object
    attributes: dict
    zero_seed: bool


def _prepare(ctx, ifm, rhs, dst, attributes):
    """Read-only source/capacity admission; None leaves the incumbent untouched."""
    try:
        attrs=dict(attributes or {})
        image,weight,output=(ctx.spec(n) for n in (ifm,rhs,dst))
        shape=tuple(image['shape'])
        kernel=tuple(attrs['kernel'])
        if (len(shape)!=4 or len(kernel)!=4
                or any(type(x)is not int or x<=0 for x in (*shape,*kernel))
                or any(type(x)is not int or x<=0 for spec in (weight,output) for x in spec['shape'])
                or attrs.get('layout','nhwc').lower()!='nhwc'
                or image['dtype']!='i8' or weight['dtype']!='i8'
                or output['dtype'] not in ('i8','i32')):
            return None
        kh,kw,ci,co=kernel
        if ci!=shape[-1] or kw<=1 or ci>=isa.DIM:return None
        padding=tuple(attrs.get('padding',(0,0,0,0)))
        if len(padding)==2:padding=(*padding,*padding)
        window=Window2D(shape,(kh,kw),tuple(attrs.get('stride',(1,1))),
            tuple(attrs.get('dilation',(1,1))),padding)
        oh,ow=window.output_spatial
        if window.dilation[1]!=1:return None
        first_column=min(ow,(padding[1]+window.stride[1]-1)//window.stride[1])
        end_column=max(first_column,min(ow,(shape[2]+padding[1]-kw)//window.stride[1]+1))
        input_x=first_column*window.stride[1]-padding[1]
        if end_column-first_column<isa.DIM:return None
        if (tuple(weight['shape'])!=(kh*kw*ci,co)
                or prod(output['shape'])!=shape[0]*oh*ow*co
                or attrs.get('output_dtype',output['dtype'])!=output['dtype']):return None
        if any(stage not in ('bias_add','acc_scale','relu') for stage in attrs.get('epilogue',())):return None
        if attrs.get('requant_shift') is not None:return None
        scales=attrs.get('acc_scale_by_column')
        if scales is not None and (len(scales)!=co or len(set(map(float,scales)))!=1):return None
        scale=float(scales[0]if scales is not None else attrs.get('acc_scale',1.0))
        relu=bool(attrs.get('relu')or'relu'in attrs.get('epilogue',()))
        if not isfinite(scale)or scale<0:return None
        if output['dtype']=='i32'and(scale!=1 or scales is not None or relu):return None
        bias,init=attrs.get('bias'),attrs.get('accumulator_init')
        if 'bias_add'in attrs.get('epilogue',())and not bias:return None
        if bias and(ctx.spec(bias)['dtype']!='i32'or prod(ctx.spec(bias)['shape'])!=co):return None
        if init and(ctx.spec(init)['dtype']!='i32'or prod(ctx.spec(init)['shape'])!=shape[0]*oh*ow*co):return None
        names={ifm,rhs,dst}|{x for x in (bias,init)if x}
        aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
        sources={a.get('source')for a in aliases.values()if isinstance(a,dict)}
        if dst in {ifm,rhs,bias,init}or names&(set(aliases)|sources):return None
        buffers=getattr(ctx,'buffers',{})
        if dst in buffers and any(buffers.get(x)is buffers[dst]for x in (ifm,rhs,bias,init)if x in buffers):return None
        chunks=(kw*ci+isa.DIM-1)//isa.DIM
        # Explicit authored expansion budgets, not hardware transfer grants.
        # Every physical DMA below remains <=DIM columns and <=DIM rows.
        ntiles=(co+isa.DIM-1)//isa.DIM
        if chunks>=kw or kw*ci>4*isa.DIM or kh*chunks>16 or kh*chunks*ntiles>128:return None
        zero_seed=not(bias or init)and kh*chunks==1
        original=CountedConvolutionPlan(window,co,isa.DIM,min(isa.DIM,ow),min(isa.DIM,co),
            min(isa.DIM,ci),isa.SPAD_ROWS,isa.ACC_ROWS,0,isa.SPAD_ROWS//2,0,
            bias=bool(bias or init or zero_seed))
        view=Window2D((shape[0],shape[1],shape[2]-input_x,ci),(kh,kw),window.stride,
            window.dilation,(padding[0],0,padding[2],0))
        if view.output_spatial!=(oh,end_column-first_column):return None
        source=CountedConvolutionPlan(view,co,isa.DIM,isa.DIM,min(isa.DIM,co),
            min(isa.DIM,ci),isa.SPAD_ROWS,isa.ACC_ROWS,0,isa.DIM,0,
            bias=bool(bias or init or zero_seed))
        panels=plan_horizontal_convolution_panels(source,max_row_blocks=1,
            maximum_transfer_columns=4*isa.DIM,private_rows_owned=True)
        if panels is None:return None
        # A coordinate view, not a materialized padded/transposed tensor. Its
        # valid-window runs advance original output Y at fixed original X.
        boundary_gather=ConvolutionGatherRuns(
            (shape[0],shape[2]+padding[1]+padding[3],shape[1]+padding[0]+padding[2],ci),
            (shape[0],ow,oh),(window.stride[1],window.stride[0]),
            (window.dilation[1],window.dilation[0]),(kw,kh),isa.DIM)
        boundary_panel=plan_convolution_resident_panel(kernel,
            Geometry(isa.DIM,isa.SPAD_ROWS,None,isa.ACC_ROWS,True),
            private_rows_owned=True,max_column_blocks=1)
        if boundary_panel is None:return None
        return _Prepared(panels,original,first_column,input_x,boundary_gather,boundary_panel,attrs,zero_seed)
    except(KeyError,TypeError,ValueError,OverflowError):
        return None


def can_lower(ctx, ifm, rhs, dst, attributes):
    return _prepare(ctx,ifm,rhs,dst,attributes)is not None


def _input_fragments(panels, builder, fields, transfer, *, original_window=None, input_x=0):
    """Lower a checked horizontal span into primitive-width row-run DMA.

    The shared gather splits each fragment at the original output row, so its
    source Y is uniform. Its virtual padded-row coordinate chooses either the
    real original row or an all-zero row, never zero then overwrite. Splitting
    q into chunks places q//DIM at block_stride_rows and q%DIM in its lane.
    """
    p=panels.source;w=p.window
    _,h,width,ci=w.input_shape
    physical_width=(original_window or w).input_shape[2]
    pt,_,pb,_=w.padding
    plane=(h+pt+pb)*width*ci
    first=fields['first_element']
    batch=builder.op('udiv',first,plane)
    rem=builder.op('urem',first,plane)
    y=builder.op('udiv',rem,width*ci)
    x=builder.op('urem',rem,width*ci)
    valid=builder.op('and',builder.cmp('ule',pt,y),builder.cmp('ult',y,pt+h),'i1')
    def row(zero):
        base=0 if zero else builder.op('add',builder.op('mul',
            builder.op('add',builder.op('mul',batch,h),builder.op('sub',y,pt)),physical_width*ci),
            builder.op('add',x,input_x*ci))
        for chunk in range(panels.horizontal_chunks):
            offset=chunk*isa.DIM
            transfer(dict(zero=zero,first_element=0 if zero else builder.op('add',base,offset),
                row_stride_elements=fields['row_stride_elements'],
                local_row=builder.op('add',fields['local_row'],chunk*fields['block_stride_rows']),
                rows=fields['rows'],columns=min(isa.DIM,panels.horizontal_columns-offset)))
    builder.branch(valid,lambda:row(False),lambda:row(True))


def _vertical_fragments(prepared, builder, first, count, reduction_y, reduction_x, transfer):
    """Embed checked virtual Y runs into disjoint original NHWC DMA bands.

    In the coordinate view, axes are (original X, original Y) and padding is
    an index domain only. The gathered tap kx*KH+ky preserves the caller's
    original ky/kx reduction order. Each active virtual fragment is partitioned
    into prefix-zero, in-bounds, suffix-zero rows; every local row is written
    once before consumption, including completely invalid X spans.
    """
    w=prepared.original.window
    _,height,width,ci=w.input_shape;kh,_=w.kernel;sh,_=w.stride
    pt,pl,pb,pr=w.padding;ph=height+pt+pb;pw=width+pl+pr
    tap=builder.op('add',builder.op('mul',reduction_x,kh),reduction_y)
    fragments=emit_convolution_matrix_gather(prepared.boundary_gather,builder,first,count,
        reduction_block=tap,reduction_tile=ci)
    for base,local,rows in fragments:
        def active(base=base,local=local,rows=rows):
            vy=builder.op('udiv',builder.op('urem',base,ph*ci),ci)
            line=builder.op('udiv',base,ph*ci)
            vx=builder.op('urem',line,pw)
            image=builder.op('udiv',line,pw)
            def issue(zero,start,length):
                def nonempty():
                    sy=builder.op('sub',builder.op('add',vy,builder.op('mul',start,sh)),pt)
                    offset=0 if zero else builder.op('add',builder.op('mul',
                        builder.op('add',builder.op('mul',image,height),sy),width*ci),
                        builder.op('mul',builder.op('sub',vx,pl),ci))
                    transfer(dict(zero=zero,first_element=offset,
                        local_row=builder.op('add',local,start),rows=length,columns=ci,
                        row_stride_elements=sh*width*ci))
                builder.branch(builder.cmp('ult',0,length),nonempty,lambda:None)
            def valid_x():
                def clipped_before(limit):
                    distance=builder.select(builder.cmp('ult',vy,limit),builder.op('sub',limit,vy),0)
                    before=builder.op('udiv',builder.op('add',distance,sh-1),sh)
                    return builder.select(builder.cmp('ult',before,rows),before,rows)
                prefix=clipped_before(pt)
                end=clipped_before(pt+height)
                issue(True,0,prefix)
                issue(False,prefix,builder.op('sub',end,prefix))
                issue(True,end,builder.op('sub',rows,end))
            valid=builder.op('and',builder.cmp('ule',pl,vx),builder.cmp('ult',vx,pl+width),'i1')
            builder.branch(valid,valid_x,lambda:issue(True,0,rows))
        builder.branch(builder.cmp('ult',0,rows),active,lambda:None)


def _emit_boundary(prepared,builder,lower,complete):
    """Private resident RHS and M=output-Y tiles for fixed boundary columns."""
    p=prepared.original;w=p.window;panel=prepared.boundary_panel
    batch,_,_,ci=w.input_shape;kh,kw=w.kernel;oh,ow=w.output_spatial
    end=prepared.first_column+prepared.panels.source.window.output_spatial[1]
    if not prepared.first_column and end==ow:return
    # The preceding interior epoch is complete before these RHS rows are
    # rewritten. This panel remains immutable for every boundary consumer of
    # the current output-column interval. No array-resident retention is used.
    complete('before_vertical_boundary',{})
    for column,columns in prepared.panels.column_intervals:
        def weights(ky):
            def tap(kx):
                index=builder.op('add',builder.op('mul',ky,kw),kx)
                lower('load_weight',dict(reduction_row=builder.op('mul',index,ci),
                    output_column=column,local_row=builder.op('add',panel.rhs_base,
                        builder.op('mul',index,isa.DIM)),rows=ci,columns=columns))
            builder.loop(0,kw,1,tap)
        builder.loop(0,kh,1,weights)
        complete('vertical_boundary_rhs_ready',{})
        def image(image_index):
            def x(output_x):
                def tile(y,rows):
                    output=builder.op('add',builder.op('mul',
                        builder.op('add',builder.op('mul',image_index,oh),y),ow),output_x)
                    common=dict(output_row=output,output_column=column,rows=rows,columns=columns,
                        accumulator_row=0,output_row_stride=ow)
                    if p.bias:
                        lower('initialize_bias',common)
                        complete('vertical_boundary_initializer_ready',{})
                    # Shared view M index: ((batch*OW+fixed_x)*OH+y).
                    first=builder.op('add',builder.op('mul',
                        builder.op('add',builder.op('mul',image_index,ow),output_x),oh),y)
                    def kernel_y(ky):
                        def kernel_x(kx):
                            _vertical_fragments(prepared,builder,first,rows,ky,kx,
                                lambda f:lower('load_input',f))
                            complete('vertical_boundary_input_ready',{})
                            index=builder.op('add',builder.op('mul',ky,kw),kx)
                            first_reduction=builder.op('and',builder.cmp('eq',ky,0),builder.cmp('eq',kx,0),'i1')
                            lower('compute',dict(common,lhs_row=0,
                                rhs_row=builder.op('add',panel.rhs_base,builder.op('mul',index,isa.DIM)),
                                depth=ci,bias=p.bias,first_reduction=first_reduction,fresh_weights=True))
                            # Complete every actual consumer before replacing
                            # LHS rows; this route grants no DMA/compute overlap.
                            complete('vertical_boundary_input_last_use',{})
                        builder.loop(0,kw,1,kernel_x)
                    builder.loop(0,kh,1,kernel_y)
                    lower('store',common)
                    complete('vertical_boundary_output_last_use',{})
                full,tail=divmod(oh,isa.DIM)
                if full:builder.loop(0,full,1,lambda i:tile(builder.op('mul',i,isa.DIM),isa.DIM))
                if tail:tile(full*isa.DIM,tail)
            if prepared.first_column:builder.loop(0,prepared.first_column,1,x)
            if end<ow:builder.loop(end,ow,1,x)
        builder.loop(0,batch,1,image)
        complete('vertical_boundary_rhs_last_use',{})


def _emit_complete(prepared, builder, lower, complete):
    """Consume the flattened interior and the original boundary partitions."""
    from merlin.compile.scheduling.affine_gather_runs import (
        ConvolutionGatherRuns,emit_convolution_gather_runs,
    )
    panels=prepared.panels;p=prepared.original;w=p.window;co=p.output_channels
    batch=w.input_shape[0];oh,ow=w.output_spatial;iw=panels.source.window.output_spatial[1]
    # This valid-window gather is a pure address embedding. Each interior M
    # fragment lands at original (batch,y,x+first_column), including short rows.
    exits=ConvolutionGatherRuns((batch,oh,ow,co),(batch,oh,iw),(1,1),(1,1),
        (1,ow-iw+1),isa.DIM)
    def remap_exit(kind,f):
        for base,local,count in emit_convolution_gather_runs(exits,builder,f['output_row'],f['rows']):
            def active(base=base,local=local,count=count):
                mapped=dict(f,rows=count,
                    output_row=builder.op('add',builder.op('udiv',base,co),prepared.first_column),
                    accumulator_row=builder.op('add',f['accumulator_row'],local))
                lower(kind,mapped)
            builder.branch(builder.cmp('ult',0,count),active,lambda:None)
    def interior(kind,f):
        if kind=='load_horizontal_input':
            _input_fragments(panels,builder,f,lambda x:lower('load_input',x),
                original_window=w,input_x=prepared.input_x)
        elif kind in ('initialize_bias','store'):remap_exit(kind,f)
        else:lower(kind,f)
    result=emit_horizontal_convolution_panels(panels,builder,interior,complete)
    end=prepared.first_column+iw
    _emit_boundary(prepared,builder,lower,complete)
    return dict(result,original_source_sha256=p.identity,
        original_output_hw=[oh,ow],interior_output_columns=[prepared.first_column,end],
        original_input_x=prepared.input_x,
        boundary_transport='fixed original output X; M rows advance output Y; original SH*W*CI input stride',
        boundary_rhs_rows=prepared.boundary_panel.rhs_rows,
        coverage='disjoint [0,first), [first,end), [end,OW) for every original batch/output row',
        input_embedding='original_x = interior_x + first*stride_w - pad_left',
        output_embedding='original_row = (interior_row//interior_width)*original_width + first + interior_row%interior_width')


def lower(ctx, ifm, rhs, dst, attributes):
    """Return an emitted receipt, or False before mutation for unsupported input."""
    prepared=_prepare(ctx,ifm,rhs,dst,attributes)
    if prepared is None:return False
    return _lower_prepared(ctx,ifm,rhs,dst,prepared)


def _lower_prepared(ctx,ifm,rhs,dst,prepared,*,schedule_emitter=_emit_complete,
                    metadata_key='convolution_flattened_rows'):
    """Bind checked horizontal source effects to the shared terminal encoder."""
    panels,attrs,zero_seed=prepared.panels,prepared.attributes,prepared.zero_seed
    p=panels.source;co=p.output_channels;fb=ctx.fb;control=FnBuilderControlAdapter(fb)
    bias,init=attrs.get('bias'),attrs.get('accumulator_init')
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(co,channel=1))
    def input_dma(f):
        ctx.emit(*isa.config_ld(f['row_stride_elements'],channel=0,fb=fb))
        pointer=0 if f['zero']else ctx.ptr(ifm,f['first_element'])
        ctx.emit(*isa.mvin(pointer,f['local_row'],f['rows'],f['columns'],fb=fb))
    def emit(kind,f):
        if kind=='load_input':
            input_dma(f)
        elif kind=='load_weight':
            offset=control.op('add',control.op('mul',f['reduction_row'],co),f['output_column'])
            ctx.emit(*isa.mvin(ctx.ptr(rhs,offset),f['local_row'],f['rows'],f['columns'],channel=1,fb=fb))
        elif kind=='initialize_bias':
            acc=isa.acc_addr(f['accumulator_row'],fb=fb)
            if init:
                offset=control.op('add',control.op('mul',f['output_row'],co),f['output_column'])
                ctx.emit(*isa.config_ld(co*4*f.get('output_row_stride',1),channel=2,shrink=False))
                ctx.emit(*isa.mvin(ctx.ptr(init,offset),acc,f['rows'],f['columns'],channel=2,fb=fb))
                # Two writes to the same ACC cells need completion before the
                # optional bias addition; no load-load order is assumed.
                ctx.fence()
            if bias or zero_seed:
                ctx.emit(*isa.config_ld(0,channel=2,shrink=False))
                pointer=0 if zero_seed else ctx.ptr(bias,f['output_column'])
                ctx.emit(*isa.mvin(pointer,isa.acc_addr(f['accumulator_row'],accumulate=bool(init),fb=fb),
                    f['rows'],f['columns'],channel=2,fb=fb))
        elif kind=='compute':
            if not f.get('fresh_weights',True):
                raise ValueError('horizontal adapter has no array stay admission')
            def product(accumulate):
                acc=isa.acc_addr(f['accumulator_row'],accumulate=accumulate,fb=fb)
                ctx.emit(*isa.preload(f['rhs_row'],acc,weight_rows=f['depth'],weight_cols=f['columns'],
                    c_rows=f['rows'],c_cols=f['columns'],fb=fb))
                ctx.emit(*isa.compute_preloaded(f['lhs_row'],a_rows=f['rows'],a_cols=f['depth'],
                    b_rows=f['depth'],b_cols=f['columns'],fb=fb))
            if p.bias:product(True)
            elif type(f['first_reduction'])is bool:product(not f['first_reduction'])
            else:control.branch(f['first_reduction'],lambda:product(False),lambda:product(True))
            # Prepared immutable LHS/RHS and private ACC cells have no DMA
            # overwrite during this feed. Actual fresh descriptor/data pairs
            # enter the selected in-order execute lane; readout drains remain.
        elif kind=='store':
            pitch=f.get('output_row_stride',1)
            if pitch==1:
                row,column=f['output_row'],f['output_column']
            else:
                row=control.op('udiv',f['output_row'],pitch)
                column=control.op('add',control.op('mul',
                    control.op('urem',f['output_row'],pitch),co),f['output_column'])
            # The common encoder applies the exact same requantization/ReLU.
            # Enlarging n expresses DRAM row pitch only; accumulator rows and
            # columns remain the original primitive tile, with uniform scale.
            store_tile(ctx,dst,f['accumulator_row'],f['rows'],f['columns'],co*pitch,
                row,column,attrs)
        else:raise ValueError('unknown horizontal convolution effect '+kind)
    # Reuse the selected execute-pair ordering evidence, bound to this original
    # source. It grants no array retention or DMA issue/completion shortcuts:
    # every shared panel-ready/last-use callback is a real full drain.
    from .conv_epochs import epoch_admissions
    from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
    source=ConvolutionExplorationSource(p,panels.column_intervals,p.identity)
    _,epoch,_,ordering=epoch_admissions(source,'gemmini-fire-aligned-b8ff28edc096392c',
        'signed-i8-products-i32-source-ordered-readout','horizontal-prepared-private-panels-v1',
        complete_reduction=True)
    epoch.require(max(2,panels.reduction_tiles*len(panels.column_intervals)))
    receipt=schedule_emitter(prepared,control,emit,lambda kind,fields:ctx.fence())
    ctx.fence()
    receipt=dict(receipt,mechanism=receipt.get('mechanism','contiguous_horizontal_reduction_panels'),
        physical_input_transfer='primitive-width affine row runs; vertical device zero source',
        kernel_row_order='ky*(KW*CI)+chunk*DIM+lane == (ky*KW+kx)*CI+channel',
        array_weights='fresh reload for every product; one spatial block per group',
        ordering=ordering,zero_initializer=zero_seed,accumulator_init=bool(init),
        original_tap_tiles=p.window.kernel[0]*p.window.kernel[1]*((p.window.input_shape[-1]+isa.DIM-1)//isa.DIM),
        emitted_reduction_tiles=panels.reduction_tiles)
    ctx.metadata.setdefault(metadata_key,[]).append(receipt)
    return receipt


lower_conv_flattened_rows=lower
