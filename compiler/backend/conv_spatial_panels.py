"""Full primitive M tiles across original NHWC output-row boundaries.

The shared affine gather splits logical M at every original output-row/image
boundary. Its padded coordinate view is an index space, never a payload buffer.
Each fragment becomes disjoint zero/valid/zero DMA bands. A capacity-derived
complete-K RHS panel stays immutable across every spatial tile in its N phase.
Only address/control arithmetic runs on the host; the common terminal encoder
performs all contraction and readout arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, prod
from pathlib import Path
from merlin.compile.tuning_knobs import knob

from merlin.compile.scheduling.affine_gather_runs import (
    ConvolutionGatherRuns, emit_convolution_gather_runs,
    plan_convolution_resident_panel,
)
from merlin.compile.scheduling.block_schedule import Geometry
from merlin.compile.scheduling.convolution_output_groups import plan_convolution_output_group
from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D

from . import isa
from .dense_readout import store_tile


@dataclass(frozen=True)
class _Prepared:
    source: object
    gather: object
    group: object
    attributes: dict
    rhs_pitch: int
    output_pitch: int
    zero_seed: bool
    epoch: object
    ordering: dict


def _prepare(ctx, ifm, rhs, dst, attributes, physical_views=None, *, allow_wide=False):
    """Read-only admission: every refusal precedes any target/context mutation."""
    try:
        if type(allow_wide)is not bool:return None
        attrs=dict(attributes or {})
        a,b,c=(ctx.spec(n) for n in (ifm,rhs,dst))
        shape=tuple(a['shape']);kernel=tuple(attrs['kernel'])
        if (len(shape)!=4 or len(kernel)!=4
                or any(type(x)is not int or x<=0 for x in (*shape,*kernel))
                or attrs.get('layout','nhwc').lower()!='nhwc'
                or a['dtype']!='i8' or b['dtype']!='i8' or c['dtype']not in ('i8','i32')):
            return None
        kh,kw,ci,co=kernel
        if shape[-1]!=ci or tuple(b['shape'])!=(kh*kw*ci,co):return None
        padding=tuple(attrs.get('padding',(0,0,0,0)))
        if len(padding)==2:padding=(*padding,*padding)
        w=Window2D(shape,(kh,kw),tuple(attrs.get('stride',(1,1))),
            tuple(attrs.get('dilation',(1,1))),padding)
        oh,ow=w.output_spatial
        if ow<=0 or(not allow_wide and ow>=isa.DIM):return None
        pixels=shape[0]*oh*ow
        if (any(type(x)is not int or x<=0 for x in c['shape'])
                or prod(c['shape'])!=pixels*co
                or attrs.get('output_dtype',c['dtype'])!=c['dtype']):return None
        if attrs.get('accumulator_init') or attrs.get('requant_shift')is not None:return None
        if any(s not in ('bias_add','acc_scale','relu')for s in attrs.get('epilogue',())):return None
        scales=attrs.get('acc_scale_by_column')
        if scales is not None and(len(scales)!=co or len(set(map(float,scales)))!=1):return None
        scale=float(scales[0]if scales is not None else attrs.get('acc_scale',1.0))
        relu=bool(attrs.get('relu')or'relu'in attrs.get('epilogue',()))
        if not isfinite(scale)or scale<0:return None
        if c['dtype']=='i32'and(scale!=1 or scales is not None or relu):return None
        bias=attrs.get('bias')
        if 'bias_add'in attrs.get('epilogue',())and not bias:return None
        if bias and(ctx.spec(bias)['dtype']!='i32'or tuple(ctx.spec(bias)['shape'])!=(co,)):return None
        names={ifm,rhs,dst}|({bias}if bias else set())
        aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
        alias_sources={x.get('source')for x in aliases.values()if isinstance(x,dict)}
        if dst in {ifm,rhs,bias}or names&(set(aliases)|alias_sources):return None
        buffers=getattr(ctx,'buffers',{})
        if dst in buffers and any(buffers.get(n)is buffers[dst]for n in (ifm,rhs,bias)if n in buffers):return None
        rhs_pitch=output_pitch=co
        if physical_views is not None:
            from .conv_channel_phases import ConvPhysicalViews
            if type(physical_views)is not ConvPhysicalViews:return None
            physical_views.__post_init__()
            if physical_views.logical_channels!=co:return None
            rhs_pitch=physical_views.rhs_row_stride
            output_pitch=physical_views.output_row_stride
        elif getattr(ctx,'physical_views',None)is not None:
            return None
        if max(kh*kw*ci*rhs_pitch,pixels*output_pitch)*4>=1<<63:return None
        intervals=tuple((n,min(isa.DIM,co-n))for n in range(0,co,isa.DIM))
        chunks=(ci+isa.DIM-1)//isa.DIM
        zero_seed=not bias and kh*kw*chunks==1
        original=CountedConvolutionPlan(w,co,isa.DIM,min(isa.DIM,ow),min(isa.DIM,co),
            min(isa.DIM,ci),isa.SPAD_ROWS,isa.ACC_ROWS,0,isa.DIM,0,bias=bool(bias or zero_seed))
        source=ConvolutionExplorationSource(original,intervals,original.identity)
        gather=ConvolutionGatherRuns(
            (shape[0],shape[1]+padding[0]+padding[2],shape[2]+padding[1]+padding[3],ci),
            (shape[0],oh,ow),w.stride,w.dilation,w.kernel,isa.DIM)
        # 32 is an explicit compiler expansion budget. Hardware capacities
        # choose the admitted N extent; no new storage or transfer is inferred.
        panel=plan_convolution_resident_panel(kernel,
            Geometry(isa.DIM,isa.SPAD_ROWS,None,isa.ACC_ROWS,True),
            private_rows_owned=True,max_column_blocks=32)
        if panel is None:return None
        group=plan_convolution_output_group(panel,pixels,max_row_blocks=1)
        from .conv_epochs import epoch_admissions
        _,epoch,_,ordering=epoch_admissions(source,'gemmini-fire-aligned-b8ff28edc096392c',
            'signed-i8-products-i32-source-ordered-readout','flattened-spatial-private-panels-v2',
            complete_reduction=True)
        # Consume the same selected full-reduction law as counted convolution.
        # Each K stage writes disjoint gather bands exactly once, then issues
        # every N consumer before replacing LHS. Selected cross-queue RAW/WAR
        # completion waits order those accesses; ordered execute pairs preserve
        # each original ACC reduction. Issued work is bounded independently of
        # the finite pending queue capacity (ready/valid backpressure).
        epoch.require(max(2,panel.reduction_tiles*panel.column_blocks))
        return _Prepared(source,gather,group,attrs,rhs_pitch,output_pitch,zero_seed,epoch,ordering)
    except(KeyError,TypeError,ValueError,OverflowError):
        return None


def can_lower(ctx, ifm, rhs, dst, attributes, *, physical_views=None, allow_wide=False):
    return _prepare(ctx,ifm,rhs,dst,attributes,physical_views,allow_wide=allow_wide)is not None


def _gather_coordinates(prepared,builder,first,count):
    """Consume the checked gather once before any original K traversal.

    Decode the tap-free padded coordinates before adding ky/kx/channel.  The
    gather's exact valid-window contract proves that each tap remains in the
    same padded image and row, so the per-tap embedding needs only additions.
    No input value or target effect moves across the reduction boundary.
    """
    w=prepared.source.plan.window
    _,h,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
    ph=h+pt+pb;pw=width+pl+pr
    return tuple((builder.op('udiv',origin,ph*pw*ci),
                  builder.op('urem',builder.op('udiv',origin,pw*ci),ph),
                  builder.op('urem',builder.op('udiv',origin,ci),pw),local,rows)
                 for origin,local,rows in emit_convolution_gather_runs(
                     prepared.gather,builder,first,count))


def _input_fragments(prepared,builder,first,count,ky,kx,channel,depth,transfer,*,coordinates=None):
    """Consume shared M fragments; one write per destination row and lane."""
    w=prepared.source.plan.window
    _,h,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
    ph=h+pt+pb;pw=width+pl+pr;sh,sw=w.stride;dh,dw=w.dilation
    if coordinates is None:
        tap=builder.op('add',builder.op('mul',ky,dh*pw*ci),
            builder.op('add',builder.op('mul',kx,dw*ci),channel))
        fragments=tuple((origin,local,rows)for origin,local,rows in
            emit_convolution_gather_runs(prepared.gather,builder,first,count))
    else:
        fragments=tuple(((image,
            builder.op('add',y,builder.op('mul',ky,dh)),
            builder.op('add',x,builder.op('mul',kx,dw))),local,rows)
            for image,y,x,local,rows in coordinates)
    for origin,local,rows in fragments:
        def active(origin=origin,local=local,rows=rows):
            if coordinates is None:
                base=builder.op('add',origin,tap)
                image=builder.op('udiv',base,ph*pw*ci)
                vy=builder.op('urem',builder.op('udiv',base,pw*ci),ph)
                vx=builder.op('urem',builder.op('udiv',base,ci),pw)
            else:
                image,vy,vx=origin
            def issue(zero,start,length):
                def nonempty():
                    ix=builder.op('sub',builder.op('add',vx,builder.op('mul',start,sw)),pl)
                    physical=0 if zero else builder.op('add',builder.op('mul',
                        builder.op('add',builder.op('mul',image,h),builder.op('sub',vy,pt)),width*ci),
                        builder.op('add',builder.op('mul',ix,ci),channel))
                    transfer(dict(zero=zero,first_element=physical,
                        local_row=builder.op('add',prepared.group.lhs_row(0),builder.op('add',local,start)),
                        rows=length,columns=depth,row_stride_elements=sw*ci))
                builder.branch(builder.cmp('ult',0,length),nonempty,lambda:None)
            def in_row():
                def before(limit):
                    delta=builder.select(builder.cmp('ult',vx,limit),builder.op('sub',limit,vx),0)
                    n=builder.op('udiv',builder.op('add',delta,sw-1),sw)
                    return builder.select(builder.cmp('ult',n,rows),n,rows)
                prefix=before(pl);end=before(pl+width)
                issue(True,0,prefix)
                issue(False,prefix,builder.op('sub',end,prefix))
                issue(True,end,builder.op('sub',rows,end))
            valid=builder.op('and',builder.cmp('ule',pt,vy),builder.cmp('ult',vy,pt+h),'i1')
            builder.branch(valid,in_row,lambda:issue(True,0,rows))
        builder.branch(builder.cmp('ult',0,rows),active,lambda:None)


def _emit(prepared,builder,lower,complete,*,hoist_gather=False,valid_gather=None,interior_gather=None,channel_unroll=None):
    """Consume source gather, resident-panel, and output-group artifacts."""
    prepared.gather.__post_init__();prepared.group.__post_init__()
    p=prepared.source.plan;w=p.window;panel=prepared.group.panel
    batch,_,_,ci=w.input_shape;kh,kw=w.kernel;oh,ow=w.output_spatial
    pixels=batch*oh*ow;chunks=panel.channel_tiles
    def phase(column,widths):
        active=len(widths)
        # The checked complete-K RHS/output artifact supplies the actual epoch
        # bound, including a peeled N phase. This law changes emitted completion
        # sites, rather than annotating an otherwise synchronous stream.
        prepared.epoch.require(max(2,panel.reduction_tiles*active))
        def reduction(callback,*,products=False):
            if products and channel_unroll is not None:
                return channel_unroll.emit(builder,callback)
            def y(ky):
                def x(kx):
                    full,tail=divmod(ci,isa.DIM)
                    if full:builder.loop(0,full,1,lambda c:callback(ky,kx,c,isa.DIM))
                    if tail:callback(ky,kx,full,tail)
                builder.loop(0,kw,1,x)
            builder.loop(0,kh,1,y)
        def locations(ky,kx,chunk):
            tap=builder.op('add',builder.op('mul',ky,kw),kx)
            slot=builder.op('add',builder.op('mul',tap,chunks),chunk)
            reduction=builder.op('add',builder.op('mul',tap,ci),builder.op('mul',chunk,isa.DIM))
            return slot,reduction
        def weights(ky,kx,chunk,depth):
            slot,start=locations(ky,kx,chunk)
            for n,columns in enumerate(widths):
                lower('load_weight',dict(reduction_row=start,output_column=builder.op('add',column,n*isa.DIM),
                    local_row=builder.op('add',panel.rhs_base,
                        builder.op('mul',builder.op('add',builder.op('mul',slot,active),n),isa.DIM)),
                    rows=depth,columns=columns))
        reduction(weights)
        complete('spatial_phase_rhs_ready',{})
        def spatial(first,rows):
            coordinates=(_gather_coordinates(prepared,builder,first,rows)
                         if hoist_gather or interior_gather is not None else None)
            outputs=tuple(dict(output_row=first,output_column=builder.op('add',column,n*isa.DIM),
                rows=rows,columns=columns,
                accumulator_row=prepared.group.accumulator_row(0,n,active_column_blocks=active))
                for n,columns in enumerate(widths))
            if p.bias:
                for out in outputs:lower('initialize_bias',out)
                complete('spatial_initializer_ready',{})
            def products(ky,kx,chunk,depth,selected_gather=valid_gather):
                channel=builder.op('mul',chunk,isa.DIM)
                if selected_gather is None:
                    _input_fragments(prepared,builder,first,rows,ky,kx,channel,depth,
                        lambda f:lower('load_input',f),coordinates=coordinates)
                else:
                    selected_gather.emit(builder,first,rows,ky,kx,channel,depth,
                        lambda f:lower('load_input',f),coordinates=coordinates)
                slot,_=locations(ky,kx,chunk)
                initial=builder.op('and',builder.cmp('eq',ky,0),builder.cmp('eq',kx,0),'i1')
                initial=builder.op('and',initial,builder.cmp('eq',chunk,0),'i1')
                for n,out in enumerate(outputs):
                    lower('compute',dict(out,lhs_row=prepared.group.lhs_row(0),
                        rhs_row=builder.op('add',panel.rhs_base,
                            builder.op('mul',builder.op('add',builder.op('mul',slot,active),n),isa.DIM)),
                        depth=depth,first_reduction=initial,bias=p.bias,fresh_weights=True))
            if interior_gather is None:
                reduction(products,products=True)
            else:
                interior=interior_gather.predicate(builder,rows,coordinates)
                builder.branch(interior,
                    lambda:reduction(lambda ky,kx,c,d:products(ky,kx,c,d,interior_gather),products=True),
                    lambda:reduction(products,products=True))
            # One real completion closes all K/N consumers before readout and
            # private ACC/LHS reuse. Preparation, bias initialization, terminal
            # store configuration/readout and phase replacement still drain.
            complete('spatial_reduction_epoch_complete',{})
            for out in outputs:lower('store',out)
            complete('spatial_output_last_use',{})
        full,tail=divmod(pixels,isa.DIM)
        if full:builder.loop(0,full,1,lambda m:spatial(builder.op('mul',m,isa.DIM),isa.DIM))
        if tail:spatial(full*isa.DIM,tail)
        complete('spatial_phase_rhs_last_use',{})
    full_columns,last_columns=divmod(p.output_channels,isa.DIM)
    phases,tail=divmod(full_columns,panel.column_blocks)
    if phases:builder.loop(0,phases,1,lambda n:phase(
        builder.op('mul',n,panel.column_blocks*isa.DIM),(isa.DIM,)*panel.column_blocks))
    widths=(isa.DIM,)*tail+((last_columns,)if last_columns else ())
    if widths:phase(phases*panel.column_blocks*isa.DIM,widths)
    return dict(mechanism='flattened_spatial_convolution_panels',source_sha256=prepared.source.identity,
        original_output_hw=[oh,ow],m_rows=isa.DIM,spatial_tiles=(pixels+isa.DIM-1)//isa.DIM,
        original_row_tiles=batch*oh*((ow+isa.DIM-1)//isa.DIM),
        rhs_rows=panel.rhs_rows,rhs_base=panel.rhs_base,n_phase_tiles=panel.column_blocks,
        accumulator_rows=panel.accumulator_rows,
        reduction_order=['kernel_y','kernel_x','input_channel'],
        gather_coordinates=('hoisted tap-free image/y/x' if hoist_gather else 'tap-dependent decoding'),
        gather='shared padded-coordinate row/image fragments; disjoint zero/valid/zero DMA bands',
        completion='RHS preparation; initializer; complete K/N epoch; common readout; phase last-use',
        compute_epoch=asdict(prepared.epoch),
        max_actual_compute_pairs_per_epoch=panel.reduction_tiles*panel.column_blocks,
        max_input_transfers_per_k=isa.DIM,
        epoch_projection='disjoint gather bands; all N consumers precede next K LHS replacement',
        physical_mapping_verified=False,target_execution=False,timing=None)


def lower(ctx, ifm, rhs, dst, attributes, *, physical_views=None, allow_wide=False, hoist_gather=None):
    """Emit the optional route, or return False before mutation when refused.

    ``allow_wide`` admits every positive output width through the same bounded
    shared gather/panel/epoch consumer; the default retains narrow admission.
    ``hoist_gather`` moves tap-free source index expressions before K.  None
    reads the explicit CONV_GATHER_COORDINATES off/on tuning alternative.
    """
    prepared=_prepare(ctx,ifm,rhs,dst,attributes,physical_views,allow_wide=allow_wide)
    if prepared is None:return False
    if hoist_gather is None:
        hoist_gather=knob('CONV_GATHER_COORDINATES',default='off',values=('off','on'),
            root=Path(__file__).resolve().parents[1])=='on'
    if type(hoist_gather)is not bool:
        raise ValueError('convolution gather coordinate selection must be Boolean')
    p=prepared.source.plan;attrs=prepared.attributes;fb=ctx.fb;builder=FnBuilderControlAdapter(fb)
    valid_gather=None
    if knob('CONV_VALID_GATHER',default='off',values=('off','on'),
            root=Path(__file__).resolve().parents[1])=='on':
        from .conv_reduction_peeling import plan_valid_convolution_gather
        valid_gather=plan_valid_convolution_gather(prepared)
    interior_gather=None
    if knob('CONV_INTERIOR_GATHER',default='off',values=('off','on'),
            root=Path(__file__).resolve().parents[1])=='on':
        from .conv_interior_gather import plan_interior_convolution_gather
        interior_gather=plan_interior_convolution_gather(prepared)
    channel_unroll=None
    unroll=knob('CONV_CHANNEL_UNROLL',default='baseline',values=('baseline','factor2','factor4'),
        root=Path(__file__).resolve().parents[1])
    if unroll!='baseline':
        from .conv_reduction_axes import plan_conv_channel_unroll
        channel_unroll=plan_conv_channel_unroll(prepared,int(unroll[-1]))
    bias=attrs.get('bias')
    ctx.fence();ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(prepared.rhs_pitch,channel=1))
    # Original source input row pitch is invariant across all gather bands.
    ctx.emit(*isa.config_ld(p.window.stride[1]*p.window.input_shape[-1],channel=0))
    if p.bias:ctx.emit(*isa.config_ld(0,channel=2,shrink=False))
    def emit(kind,f):
        if kind=='load_input':
            ptr=0 if f['zero']else ctx.ptr(ifm,f['first_element'])
            ctx.emit(*isa.mvin(ptr,f['local_row'],f['rows'],f['columns'],fb=fb))
        elif kind=='load_weight':
            offset=builder.op('add',builder.op('mul',f['reduction_row'],prepared.rhs_pitch),f['output_column'])
            ctx.emit(*isa.mvin(ctx.ptr(rhs,offset),f['local_row'],f['rows'],f['columns'],channel=1,fb=fb))
        elif kind=='initialize_bias':
            ptr=0 if prepared.zero_seed else ctx.ptr(bias,f['output_column'])
            ctx.emit(*isa.mvin(ptr,isa.acc_addr(f['accumulator_row']),f['rows'],f['columns'],channel=2,fb=fb))
        elif kind=='compute':
            def product(accumulate):
                ctx.emit(*isa.preload(f['rhs_row'],isa.acc_addr(f['accumulator_row'],accumulate=accumulate),
                    weight_rows=f['depth'],weight_cols=f['columns'],c_rows=f['rows'],c_cols=f['columns'],fb=fb))
                ctx.emit(*isa.compute_preloaded(f['lhs_row'],a_rows=f['rows'],a_cols=f['depth'],
                    b_rows=f['depth'],b_cols=f['columns'],fb=fb))
            if p.bias:product(True)
            else:builder.branch(f['first_reduction'],lambda:product(False),lambda:product(True))
        elif kind=='store':
            store_tile(ctx,dst,f['accumulator_row'],f['rows'],f['columns'],prepared.output_pitch,
                f['output_row'],f['output_column'],attrs)
        else:raise ValueError('unknown spatial convolution effect '+kind)
    receipt=_emit(prepared,builder,emit,lambda kind,fields:ctx.fence(),
        hoist_gather=hoist_gather,valid_gather=valid_gather,interior_gather=interior_gather,channel_unroll=channel_unroll)
    if valid_gather is not None:receipt['valid_window_gather']=valid_gather.receipt()
    if interior_gather is not None:receipt['interior_panel_gather']=interior_gather.receipt()
    if channel_unroll is not None:receipt['channel_unroll']=channel_unroll.receipt()
    ctx.fence()
    receipt=dict(receipt,rhs_pitch=prepared.rhs_pitch,output_pitch=prepared.output_pitch,
        zero_initializer=prepared.zero_seed,ordering=prepared.ordering)
    ctx.metadata.setdefault('convolution_spatial_panels',[]).append(receipt)
    return receipt


lower_conv_spatial_panels=lower
