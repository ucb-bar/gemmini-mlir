"""Source-qualified contiguous NHWC 1x1 convolution as a shared dense view.

Only indexing changes: input/output batch and spatial axes become matrix rows,
weights remain input-channel/output-channel ordered, and the normal dense
schedule owns storage, completion and device epilogues.
"""
from __future__ import annotations
from dataclasses import asdict
from math import prod
from merlin.compile.scheduling.pointwise_convolution import pointwise_dense_view
from merlin.compile.scheduling.window_transfer import Window2D
from . import isa


def _qualify(ctx,ifm,rhs,dst,attributes):
    attrs=attributes or {}
    source=ctx.spec(ifm)
    weights=ctx.spec(rhs)
    output=ctx.spec(dst)
    shape=tuple(source['shape'])
    if len(shape)!=4 or attrs.get('layout','nhwc').lower()!='nhwc':
        raise ValueError('pointwise dense view requires rank-four contiguous NHWC input')
    if source['dtype']!='i8' or weights['dtype']!='i8':
        raise ValueError('pointwise dense view requires original signed i8 operands')
    if output['dtype'] not in ('i8','i32') or attrs.get('output_dtype',output['dtype'])!=output['dtype']:
        raise ValueError('pointwise dense view requires matching original i8/i32 output dtype')
    kernel=tuple(attrs.get('kernel',()))
    if len(kernel)!=4 or kernel[2]!=shape[-1] or prod(weights['shape'])!=prod(kernel):
        raise ValueError('pointwise kernel differs from original contiguous KH/KW/CI/CO weights')
    padding=tuple(attrs.get('padding',(0,0,0,0)))
    if len(padding)==2:
        padding=(*padding,*padding)
    dilation=tuple(attrs.get('dilation',(1,1)))
    if dilation!=(1,1):
        raise ValueError('pointwise dense adapter currently requires unit dilation')
    window=Window2D(shape,kernel[:2],tuple(attrs.get('stride',(1,1))),dilation,padding)
    view=pointwise_dense_view(window,kernel[3],output_shape=tuple(output['shape']))
    if view is None:
        raise ValueError('pointwise dense view requires unpadded unit-stride 1x1 source')
    m,k,n=view.dimensions
    bias=attrs.get('bias')
    ep=attrs.get('epilogue',())
    if 'bias_add' in ep and not bias:
        raise ValueError('pointwise bias epilogue requires its original bias tensor')
    if bias and (ctx.spec(bias)['dtype']!='i32' or prod(ctx.spec(bias)['shape'])!=n):
        raise ValueError('pointwise bias must contain one original i32 per output channel')
    init=attrs.get('accumulator_init')
    if init and (ctx.spec(init)['dtype']!='i32' or tuple(ctx.spec(init)['shape']) not in (tuple(output['shape']),(m,n))):
        raise ValueError('pointwise initializer must be an original contiguous i32 output matrix')
    scales=attrs.get('acc_scale_by_column')
    if scales is not None and len(scales)!=n:
        raise ValueError('pointwise column scales differ from original output channels')
    if any('pool' in str(stage).lower() for stage in ep):
        raise ValueError('pointwise dense view requires spatial pooling to be lowered separately')
    if attrs.get('requant_shift') is not None and 'requant' in ep:
        raise ValueError('pointwise dense view has no exact integer-shift requant readout')
    if output['dtype']=='i32' and (scales is not None or float(attrs.get('acc_scale',1.0))!=1.0 or attrs.get('relu',False) or 'relu' in ep):
        raise ValueError('pointwise full i32 readout cannot absorb narrowing epilogues')
    # Match the original convolution's explicit additive-identity lifetime.
    # Small unseeded pointwise reductions initialize zero then accumulate;
    # a genuine bias/matrix seed already establishes that continuation.
    zero_seed=not bias and not init and k<=isa.DIM
    return view,zero_seed


def can_lower_pointwise(ctx,ifm,rhs,dst,attributes=None):
    """Read-only admission; unsupported geometry keeps the convolution route."""
    try:
        _qualify(ctx,ifm,rhs,dst,attributes)
        return True
    except (KeyError,TypeError,ValueError):
        return False


def lower_pointwise(ctx,ifm,rhs,dst,attributes=None):
    """Lower an admitted source via the existing dense terminal path.

    Raises before emitting commands for unsupported geometry. Runtime matrix
    seeds, bias, column scales and destination pointers retain their source
    identity. A small source zero initializer is materialized as immutable i32
    zero coefficients and actually consumed by device accumulator mvin.
    """
    from .dense import lower_dense
    attrs=dict(attributes or {})
    view,zero_seed=_qualify(ctx,ifm,rhs,dst,attrs)
    m,k,n=view.dimensions
    index=getattr(ctx,'_pointwise_views_count',0)
    ctx._pointwise_views_count=index+1
    prefix='_pointwise_'+str(index)+'_'
    aliases={}
    for role,name,shape in (('lhs',ifm,(m,k)),('rhs',rhs,(k,n)),('output',dst,(m,n))):
        alias=prefix+role
        ctx.register(alias,ctx.buffer(name),shape,ctx.spec(name)['dtype'])
        aliases[role]=alias
    init=attrs.get('accumulator_init')
    if init:
        alias=prefix+'initializer'
        ctx.register(alias,ctx.buffer(init),(m,n),'i32')
        attrs['accumulator_init']=alias
    if zero_seed:
        name=prefix+'source_zero'
        ctx.constant(name,(n,),'i32',(0,)*n)
        attrs['bias']=name
    dense=lower_dense(ctx,aliases['lhs'],aliases['rhs'],aliases['output'],m=m,k=k,n=n,attributes=attrs)
    receipt={'schema':'merlin.pointwise_dense_lowering.v1','source_view':view.record(),
        'source_buffers':{'input':ifm,'weights':rhs,'output':dst},
        'matrix_aliases':aliases,'dense_schedule':dense,
        'initializer':{'kind':'source_additive_identity_i32_zero' if zero_seed else
                       ('original_matrix' if init else ('original_bias' if attrs.get('bias') else 'dense_first_product_overwrite')),
                       'original_source':init or (attributes or {}).get('bias'),
                       'device_source':attrs.get('accumulator_init') or attrs.get('bias'),
                       'first_product':'accumulate' if zero_seed or init or attrs.get('bias') else 'overwrite'},
        'payload_copies':0}
    ctx.metadata.setdefault('pointwise_dense_views',[]).append(receipt)
    return receipt


def _strided_geometry(ctx,ifm,rhs,dst,attributes):
    """Bind an original stride-two source before deriving a matrix schedule."""
    attrs=dict(attributes or {})
    shape=tuple(ctx.spec(ifm)['shape'])
    kernel=tuple(attrs.get('kernel',()))
    padding=tuple(attrs.get('padding',(0,0,0,0)))
    if len(padding)==2:
        padding=(*padding,*padding)
    if (len(shape)!=4 or len(kernel)!=4 or kernel[:2]!=(1,1)
            or tuple(attrs.get('stride',(1,1)))!=(2,2)
            or tuple(attrs.get('dilation',(1,1)))!=(1,1) or any(padding)):
        raise ValueError('strided pointwise route requires original unpadded unit-dilation stride2x2 1x1 geometry')
    window=Window2D(shape,(1,1),(2,2),(1,1),padding)
    oh,ow=window.output_spatial
    class LogicalRows:
        def spec(self,name):
            spec=ctx.spec(name)
            return {**spec,'shape':(shape[0],oh,ow,shape[-1])} if name==ifm else spec
    # The established pointwise admission checks source dtypes, actual output,
    # seeds and epilogues against the sampled row domain. It changes no buffer.
    _,zero_seed=_qualify(LogicalRows(),ifm,rhs,dst,{**attrs,'stride':(1,1)})
    tile_m=max(d for d in range(1,min(isa.DIM,ow)+1) if ow%d==0)
    return window,kernel[3],tile_m,zero_seed


def can_lower_strided_pointwise(ctx,ifm,rhs,dst,attributes=None):
    """Read-only admission for the separate opt-in stride-two family."""
    try:
        _strided_geometry(ctx,ifm,rhs,dst,attributes)
        return True
    except (KeyError,TypeError,ValueError):
        return False


def _strided_input_fragments(prepared,builder,first,count,ky,kx,channel,depth,transfer):
    """Project the checked gather directly into original stride-two rows.

    With a single unpadded tap every shared fragment is entirely valid. The
    original-row/image split supplies its exact source origin and disjoint
    destination rows; no halo predicates, zero source, or packed tensor exist.
    """
    from merlin.compile.scheduling.affine_gather_runs import emit_convolution_gather_runs
    w=prepared.source.plan.window
    if w.kernel!=(1,1) or w.stride!=(2,2) or any(w.padding):
        raise ValueError('direct pointwise transfer requires its original sampled source')
    for origin,local,rows in emit_convolution_gather_runs(prepared.gather,builder,first,count):
        def active(origin=origin,local=local,rows=rows):
            transfer(dict(zero=False,first_element=builder.op('add',origin,channel),
                local_row=builder.op('add',prepared.group.lhs_row(0),local),
                rows=rows,columns=depth,row_stride_elements=2*w.input_shape[-1]))
        builder.branch(builder.cmp('ult',0,rows),active,lambda:None)


def _lower_strided_resident(ctx,ifm,rhs,dst,attrs,window,n):
    """Use the shared complete-K RHS owner and full primitive spatial rows.

    Small reductions retain their measured incumbent. Capacity/semantic
    refusal is read-only; the caller can then use the established alternative.
    This path's source, placement and ordering proofs require no SMT solver.
    """
    if window.input_shape[-1]<isa.DIM or n<isa.DIM:
        return None
    from .conv_spatial_panels import lower
    result=lower(ctx,ifm,rhs,dst,attrs,allow_wide=True,input_fragments=_strided_input_fragments)
    if not result:return None
    receipt=dict(schema='merlin.strided_pointwise_dense.v2',window=asdict(window),
        source_identity=result['source_sha256'],dimensions=[window.input_shape[0]*prod(window.output_spatial),window.input_shape[-1],n],
        tile_m=isa.DIM,source_row_stride_elements=2*window.input_shape[-1],payload_copies=0,
        selected_pipeline='resident_rhs_affine_dense_rows',
        same_scanline_proof='shared original-row/image fragments; one disjoint transfer per positive fragment',
        dense_schedule=result,optional_solver=False)
    ctx.metadata.setdefault('strided_pointwise_dense',[]).append(receipt)
    return receipt


def _strided_source_element(window,row,column):
    """Exact integer indexing used by projected-call validation and receipts."""
    _,height,width,channels=window.input_shape
    oh,ow=window.output_spatial
    batch,spatial=divmod(row,oh*ow)
    y,x=divmod(spatial,ow)
    return ((batch*height+y*window.stride[0])*width+x*window.stride[1])*channels+column


def _strided_schedule(window,n,tile_m,seeded,mixed_scales):
    from dataclasses import asdict
    import hashlib,json
    from pathlib import Path
    from .dense import _adapter
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,DenseBiasSource,emit_compact_dense,schedule_compact_dense
    from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission,dense_retained_operand_hoists,dense_retained_compute_order
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
    from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
    from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
    oh,ow=window.output_spatial
    m=window.input_shape[0]*oh*ow
    k=window.input_shape[-1]
    p=DenseProblem(m,k,n,tile_m,isa.DIM,1 if mixed_scales else isa.DIM,1,1,4,bias=seeded)
    a=_adapter()
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',isa.DIM,0,isa.SPAD_ROWS//2),
        DenseMemoryView('rhs','spad',isa.DIM,isa.SPAD_ROWS//2,isa.SPAD_ROWS//2),
        DenseMemoryView('acc','acc',4*isa.DIM,0,isa.ACC_ROWS))
    # One M tile is essential: a single transfer never crosses a scanline.
    r=DenseResources(isa.SPAD_ROWS*isa.DIM,isa.ACC_ROWS*4*isa.DIM,max_m_blocks=1,
        max_n_blocks=4,max_k_blocks=p.tiles[1],partial_output_tiles=True,partial_output_blocks=True)
    bias_source=DenseBiasSource(n,'i32') if seeded else None
    legal=[b for b in dense_blocks(p,r,k_limit=lambda bm,bn:s.k_limit(p,r,bm,bn))
           if b.k==p.tiles[1] and s.admits(p,b,r)]
    evidence=json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if not legal or evidence['completion_order']['execute']['order']!='in_order':
        return schedule_compact_dense(p,r,s,a,name='pointwise_stride2_compact',dtypes=('i8','i8','i32'),
            static_budget=128,external_buffers_disjoint=True,bias_source=bias_source),p,False
    block=min(legal,key=lambda b:(b.loads,-b.n))
    raw=emit_compact_dense(p,block,s,a,name='pointwise_stride2_retained',dtypes=('i8','i8','i32'),bias_source=bias_source)
    transforms=dense_retained_operand_hoists(raw,a,role='lhs')
    transforms+=dense_retained_compute_order(propose_loop_transforms(raw,transforms),a,order=('k','n','m'))
    law=DenseRetainedReductionAdmission(a.configuration_identity,a.numerical_policy_identity,evidence['identity'],
        max(1,block.m*block.k*block.n),max(1,(block.m+block.n)*block.k),2,8)
    source=hashlib.sha256(json.dumps([asdict(window),n,seeded,mixed_scales],sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,
        block_tiles=(block.m,block.k,block.n),loop_transforms=transforms,pipeline_kind='retained_full_k',
        pipeline_prefetch=False,bias_source=bias_source)
    schedule=schedule_dense_pipeline(p,r,s,a,name='pointwise_stride2_retained',dtypes=('i8','i8','i32'),
        source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,
        bias_source=bias_source,readout_mode='separate',static_budget=128,timeout_ms=20000)
    return schedule,p,True


def lower_strided_pointwise(ctx,ifm,rhs,dst,attributes=None):
    """Direct strided source DMA into a checked dense schedule; no payload copy."""
    from dataclasses import asdict
    import hashlib,json
    from .dense_readout import store_tile,_pack
    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    attrs=dict(attributes or {})
    window,n,tile_m,zero_seed=_strided_geometry(ctx,ifm,rhs,dst,attrs)
    bias=attrs.get('bias')
    init=attrs.get('accumulator_init')
    mixed=attrs.get('acc_scale_by_column') is not None and len(set(map(float,attrs['acc_scale_by_column'])))>1
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    configured=knob('CONV_COLUMN_READOUT',default='recompute',values=('recompute','raw_reload'),
                    root=Path(__file__).resolve().parents[1])
    column_route=attrs.get('conv_column_readout',configured)
    if column_route not in ('recompute','raw_reload'):
        raise ValueError('CONV_COLUMN_READOUT must be recompute or raw_reload')
    if mixed and column_route=='raw_reload' and not attrs.get('strided_pointwise_raw_producer',False):
        return _lower_strided_column_readout(ctx,ifm,rhs,dst,attrs,window,n)
    resident=_lower_strided_resident(ctx,ifm,rhs,dst,attrs,window,n)
    if resident is not None:
        return resident
    # Prior paired observations reject dense setup for subprimitive K. Keep
    # its original synchronous device convolution, including source readout.
    # Missing optional SMT support also declines before dense emission.
    from importlib.util import find_spec
    if window.input_shape[-1]<isa.DIM or n<isa.DIM or find_spec('z3') is None:
        if init:
            raise ValueError('strided pointwise matrix seed requires the optional checked dense schedule')
        from .conv import lower_conv
        result=lower_conv(ctx,ifm,rhs,dst,attrs)
        receipt=dict(schema='merlin.strided_pointwise_dense.v2',selected_pipeline='incumbent_device_convolution',
            reason='subprimitive reduction/output'if min(window.input_shape[-1],n)<isa.DIM else 'optional symbolic dense checker unavailable',
            source_row_stride_elements=2*window.input_shape[-1],payload_copies=0,dense_schedule=result)
        ctx.metadata.setdefault('strided_pointwise_dense',[]).append(receipt)
        return receipt
    schedule,p,advanced=_strided_schedule(window,n,tile_m,bool(bias or init or zero_seed),mixed)
    fb=ctx.fb
    control=FnBuilderControlAdapter(fb)
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(p.k*2,channel=0))
    ctx.emit(*isa.config_ld(n,channel=1))
    descriptor_columns=[None]
    def emit(instruction,a,metadata=None):
        a={key:fb.const(value) if isinstance(value,int) else value for key,value in a.items()}
        if instruction=='merlin_dense_panel_drain':
            ctx.fence()
        elif instruction=='gemmini_mvin_lhs':
            # tile_m divides output width, so the complete tile is one legal
            # same-scanline affine run. The shared source geometry computes its
            # first element and exact row stride; no host payload is read.
            run=window.emit_row_run(control,a['dram_row'],tile_m,0,0,a['dram_col'],1)
            offset=run['first_element']
            ctx.emit(isa.FUNCT['mvin'],ctx.ptr(ifm,offset),_pack(fb,a['addr'],a['rows'],a['cols']))
            if not advanced:ctx.fence()
        elif instruction=='gemmini_mvin_weight':
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(n)),a['dram_col'])
            ctx.emit(isa.FUNCT['mvin2'],ctx.ptr(rhs,offset),_pack(fb,a['addr'],a['rows'],a['cols']))
            if not advanced:ctx.fence()
        elif instruction=='gemmini_bias_mvin':
            if init:
                row=fb.mul_i(a['block0'],fb.const(tile_m))
                offset=fb.add_i(fb.mul_i(row,fb.const(n)),a['dram_col'])
                ctx.emit(*isa.config_ld(n*4,channel=2,shrink=False))
                ctx.emit(*isa.mvin(ctx.ptr(init,offset),isa.acc_addr(a['addr'],fb=fb),a['rows'],a['cols'],channel=2,fb=fb))
                ctx.fence()
            if bias or zero_seed:
                ctx.emit(*isa.config_ld(0,channel=2,shrink=False))
                pointer=ctx.ptr(bias,a['dram_col']) if bias else 0
                ctx.emit(*isa.mvin(pointer,isa.acc_addr(a['addr'],accumulate=bool(init),fb=fb),a['rows'],a['cols'],channel=2,fb=fb))
                ctx.fence()
        elif instruction=='gemmini_preload':
            descriptor_columns[0]=a['cols']
            address=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,address,a['rows'],a['cols']))
        elif instruction=='gemmini_compute_preloaded':
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),
                     _pack(fb,isa.GARBAGE,a['depth'],descriptor_columns[0]))
            if not advanced:ctx.fence()
        elif instruction=='gemmini_mvout':
            store_tile(ctx,dst,a['acc'],a['rows'],a['cols'],n,a['dram_row'],a['dram_col'],attrs)
        else:
            raise ValueError('unknown strided pointwise terminal instruction '+instruction)
    schedule.emit_llvm(fb,emit)
    source=asdict(window)
    identity=hashlib.sha256(json.dumps([source,n],sort_keys=True).encode()).hexdigest()
    receipt={'schema':'merlin.strided_pointwise_dense.v1','source_identity':identity,'window':source,
        'dimensions':[p.m,p.k,p.n],'tile_m':tile_m,'scanline_tiles':window.output_spatial[1]//tile_m,
        'source_row_stride_elements':2*p.k,'payload_copies':0,
        'same_scanline_proof':{'tile_divides_output_width':True,'max_m_blocks':1,
                              'channel_bounds':'checked original dense K slices'},
        'selected_pipeline':'retained' if advanced else 'compact','dense_schedule':dict(schedule.realization),
        'initializer':'source_zero' if zero_seed else ('matrix' if init else ('bias' if bias else 'first_product_overwrite'))}
    ctx.metadata.setdefault('strided_pointwise_dense',[]).append(receipt)
    return receipt


def _lower_strided_column_readout(ctx,ifm,rhs,dst,attributes,window,n,*,producer=None):
    """One full-N strided product, then exact device-only column readout."""
    from .column_batches import emit_batched_column_readout
    from .dense_readout import store_tile
    attrs=dict(attributes)
    if attrs.get('output_dtype',ctx.spec(dst)['dtype'])!='i8':
        raise ValueError('strided pointwise raw reload requires original i8 output')
    oh,ow=window.output_spatial
    m=window.input_shape[0]*oh*ow
    index=getattr(ctx,'_strided_pointwise_raw_count',0)
    ctx._strided_pointwise_raw_count=index+1
    raw='_strided_pointwise_raw_'+str(index)
    ctx.alloc(raw,(m,n),'i32')
    producer_attrs=dict(attrs)
    for name in ('acc_scale_by_column','acc_scale','relu','requant_shift'):
        producer_attrs.pop(name,None)
    producer_attrs['epilogue']=[stage for stage in attrs.get('epilogue',()) if stage=='bias_add']
    producer_attrs['output_dtype']='i32'
    producer_attrs['conv_column_readout']='recompute'
    producer_attrs['strided_pointwise_raw_producer']=True
    init=producer_attrs.get('accumulator_init')
    if init:
        # The original seed's NHWC storage has the same flattened row order;
        # retain its runtime pointer while adapting only its typed matrix view.
        alias=raw+'_initializer'
        ctx.register(alias,ctx.buffer(init),(m,n),'i32')
        producer_attrs['accumulator_init']=alias
    receipt=(producer or lower_strided_pointwise)(ctx,ifm,rhs,raw,producer_attrs)
    batching=emit_batched_column_readout(ctx,raw,dst,m,n,attrs)
    if batching is None:
        fb=ctx.fb
        ctx.fence()
        ctx.emit(*isa.config_ld(n*4,channel=2,shrink=False))
        def column(col):
            def tile(row,rows):
                offset=fb.add_i(fb.mul_i(row,fb.const(n)),col)
                ctx.emit(*isa.mvin(ctx.ptr(raw,offset),isa.acc_addr(0),rows,1,channel=2,fb=fb))
                ctx.fence()
                store_tile(ctx,dst,0,rows,1,n,row,col,attrs)
            complete=m//isa.DIM*isa.DIM
            if complete:
                fb.for_loop(0,complete,isa.DIM,lambda row:tile(row,isa.DIM),retain_loop=True)
            if complete<m:
                tile(fb.const(complete),m-complete)
        fb.for_loop(0,n,1,column,retain_loop=True)
        ctx.fence()
        batching={'selection':'off','route':'synchronous_original_columns'}
    receipt=dict(receipt)
    receipt['column_readout']={'route':'raw_reload','raw_buffer':raw,'output_buffer':dst,
        'raw_storage_bytes':m*n*4,'raw_reload_bytes':m*n*4,'output_bytes':m*n,
        'product_tile_columns':isa.DIM,'output_rows':m,'output_channels':n,
        'bias':'producer only','accumulator_init':'producer only',
        'zero_initializer':'producer only','batching':batching}
    ctx.metadata.setdefault('strided_pointwise_dense',[]).append(receipt)
    return receipt
