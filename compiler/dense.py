"""Source-general dense contraction scheduling and explicit Gemmini realization.

The shared compact scheduler owns counted axes, ragged extents, local lifetimes,
source reduction order and accumulator continuation.  This adapter realizes its
actual calls with the selected weight-stationary instructions.
"""
from __future__ import annotations
import json
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'compiler_library'))
from merlin.compile.scheduling.block_schedule import Load, Preload, Compute, Store, InitializeAccumulator
from merlin.compile.scheduling.compact_dense import (DenseLoopAdapter, DenseLoopStorage,
    DenseMemoryView, DenseBiasSource, schedule_compact_dense)
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.sched.ir import Call, as_expr
from target_storage import entry_transpose_tile


def _adapter():
    def encode(o):
        if isinstance(o, Load):
            name=o.role+'_load'; keys=('bi','bj','r','c','rows','cols','local','step')
            values=(*o.block,o.dram_row,o.dram_col,o.rows,o.cols,o.row,o.step)
        elif isinstance(o, Preload):
            name='preload'; keys=('weight','ki','nj','acc','continue','rows','cols','depth')
            values=(o.weight_row,*o.weight_block,o.accumulator_row,o.accumulate,o.rows,o.cols,o.weight_rows)
        elif isinstance(o, Compute):
            name='compute'; keys=('lhs','mi','ki','rows','depth'); values=(o.input_row,*o.input_block,o.rows,o.cols)
        elif isinstance(o, InitializeAccumulator):
            name='bias'; keys=('mi','nj','c','rows','cols','local'); values=(*o.block,o.dram_col,o.rows,o.cols,o.row)
        elif isinstance(o, Store):
            name='store'; keys=('r','c','rows','cols','local'); values=(o.dram_row,o.dram_col,o.rows,o.cols,o.accumulator_row)
        else: raise ValueError(type(o))
        return Call(name,tuple(zip(keys,map(as_expr,values))))
    def project(call):
        a=dict(call.args)
        if call.instr in ('lhs_load','weight_load'):
            return Load(call.instr[:-5],(a['bi'],a['bj']),a['r'],a['c'],a['rows'],a['cols'],a['local'],a['step'])
        if call.instr=='preload': return Preload(a['weight'],(a['ki'],a['nj']),a['acc'],a['continue'],a['rows'],a['cols'],a['depth'])
        if call.instr=='compute': return Compute(a['lhs'],(a['mi'],a['ki']),a['rows'],a['depth'],True)
        if call.instr=='bias': return InitializeAccumulator((a['mi'],a['nj']),a['c'],a['rows'],a['cols'],a['local'])
        if call.instr=='store': return Store(a['r'],a['c'],a['rows'],a['cols'],a['local'])
        raise ValueError(call.instr)
    return DenseLoopAdapter(encode,project,'gemmini-ws-explicit-v1','i8xi8-i32-source-order-v1',True,True,64)


class _IndexBridge:
    def __init__(self,b,index_unroll=None):
        from dense_residency import index_unroll_setting
        self.b=b
        self.index_unroll=index_unroll_setting() if index_unroll is None else index_unroll
        if self.index_unroll not in ('off','tiny'): raise ValueError('unknown dense index unroll mode')
        # Each two-trip expansion adds one callback relative to counted code.
        # Charge nested clones individually, so no original effect can acquire
        # more than 1+15 copies. Singleton expansion has no code growth.
        self.remaining_clones=15
        self.expanded_loops=self.literal_iterations=0
    def op(self,op,a,c): return getattr(self.b,op)(a,c)
    def cmp(self,p,a,c):
        if self.index_unroll=='tiny' and p=='eq' and type(a) is int and type(c) is int:
            return a==c
        return self.b.icmp(p,a,c)
    def select(self,c,a,d):
        if self.index_unroll=='tiny' and type(c) is bool: return a if c else d
        return self.b.select(c,a,d)
    def loop(self,start,stop,step,callback):
        if (self.index_unroll=='tiny' and all(type(v) is int for v in (start,stop,step))
                and 0<=start<=stop and step>0):
            count=(stop-start+step-1)//step
            cost=max(0,count-1)
            if count<=2 and cost<=self.remaining_clones:
                self.remaining_clones-=cost
                self.expanded_loops+=1
                self.literal_iterations+=count
                for i in range(start,stop,step): callback(i)
                return
        with self.b.counted_loop(start,stop,step) as i: callback(i)
    def receipt(self):
        return {'mode':self.index_unroll,'expanded_loops':self.expanded_loops,
            'literal_iteration_callbacks':self.literal_iterations,
            'extra_iteration_callbacks':15-self.remaining_clones,
            'per_original_effect_copy_bound':16 if self.index_unroll=='tiny' else 1,
            'order':'ascending original index/effect order; original completions preserved'}


def _geometry():
    # Consume the selected original facts when compiling; packaged compiler has
    # a copied facts.json and never depends on an absolute workspace location.
    paths=(Path(__file__).with_name('facts.json'),Path(__file__).parent.parent/'rtl_facts'/'facts.json')
    facts=json.loads(next(p for p in paths if p.exists()).read_text())['facts']
    edge=facts['arrays'][0]['rows']
    memories={x['name']:x for x in facts['memories']}
    sp=memories['scratchpad']; acc=memories['accumulator']
    return edge,sp['depth'],sp['bytes'],acc['bytes']


def _lower_dense_pooled(b,lhs_ptr,rhs_ptr,out_ptr,m,k,n,attrs,bias_ptr,rhs_transposed,schedule_options):
    """Compose dense raw ACC exit with checked source-window hardware readouts.

    An i32 global is a device spill, never a host arithmetic buffer. The shared
    window planner retains every source observation and duplicates only border
    cells under its minimum-padding proof. Each staged ACC value undergoes the
    original scalar readout before StoreController's signed maximum.
    """
    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.spatial_gather import plan_gathered_spatial_max
    if attrs.get('output_dtype','i32')!='i8':
        raise ValueError('selected pooling path requires narrowing i8 readout')
    stages=attrs.get('epilogue',[])
    if stages[-1:]!=['maxpool'] or stages.count('maxpool')!=1:
        raise ValueError('pooling must follow the original scalar readout exactly once')
    h,w=map(int,attrs['pool_in_dims'])
    if m%(h*w): raise ValueError('pool spatial extent does not divide dense source rows')
    size=tuple(map(int,attrs['pool_size'])); stride=tuple(map(int,attrs['pool_stride']))
    padding=tuple(map(int,attrs.get('pool_padding',(0,0,0,0))))
    if size[0]!=size[1] or stride[0]!=stride[1] or not 0<size[0]<4 or not 0<stride[0]<4:
        raise ValueError('pool controller requires square size and equal strides in 1..3')
    source=SpatialMaxReadout((h,w),size,stride,padding,attrs.get('pool_pad_value'))
    oh,ow=source.output_hw
    if not 0<ow<256: raise ValueError('pool output width exceeds selected controller field')
    edge,_,_,acc_bytes=_geometry()
    if size[0]*size[1]>acc_bytes//(edge*4): raise ValueError('pool window exceeds private accumulator rows')
    scales=attrs.get('acc_scale_by_column') if 'acc_scale' in stages else None
    if scales is not None and len(scales)!=n: raise ValueError('column scale count differs from N')
    plan=plan_gathered_spatial_max(source,batches=m//(h*w),channels=n,
        channel_block=1 if scales is not None else edge,max_window_elements=acc_bytes//(edge*4))
    raw=b.global_buffer(m*n,'i32')
    producer_attrs=dict(attrs,output_dtype='i32',epilogue=['bias_add'] if bias_ptr is not None else [])
    producer=lower_dense(b,lhs_ptr,rhs_ptr,raw,m,k,n,producer_attrs,bias_ptr,rhs_transposed,schedule_options)
    b.config_ld(n*4,id=0)
    activation=1 if 'relu' in stages else 0
    scale=attrs.get('acc_scale',1.0) if 'acc_scale' in stages else 1.0
    scale_words=None if scales is None else [
        struct.unpack('<I',struct.pack('<f',float(value)))[0] for value in scales]
    def lower(name,a):
        col=a['first_column']
        if name=='gather_window_row':
            offset=b.mul(b.add(b.mul(a['source_row'],n),col),4)
            b.mvin(b.ptr_offset(raw,offset),b.or_(0x80000000,a['window_row']),
                cols=a['columns'],rows=1,id=0)
        elif name=='store_window_max':
            pool=dict(pool_stride=stride[0],pool_size=size[0],pool_out_dim=ow,
                porows=1,pocols=1,orows=size[0],ocols=size[1])
            if scale_words is None: b.config_st(n,activation=activation,scale=scale,**pool)
            else:
                b.constant_i32_dispatch(scale_words,col,
                    lambda bits:b.config_st(n,activation=activation,scale_bits=bits,**pool))
            offset=b.add(b.mul(a['output_row'],n),col)
            b.mvout(b.ptr_offset(out_ptr,offset),0x80000000,cols=a['columns'],rows=0)
        else: raise ValueError('unhandled pooled dense effect '+name)
        b.fence()
    index_mode=(schedule_options or {}).get('index_unroll')
    bridge=_IndexBridge(b,index_mode)
    receipt=emit_index_program(plan.kernel,bridge,lower)
    receipt['index_unroll']=bridge.receipt()
    report={'shape':[m,k,n],'producer':producer['emission'],
        'pooling':'checked_gathered_windows_scalar_readout_then_hardware_max',
        'emission':receipt,'pool_output_shape':[m//(h*w)*oh*ow,n],
        'device_spill_bytes':m*n*4,'temporary_accumulator_rows':plan.temporary_rows,
        'dynamic_pool_calls':plan.dynamic_calls,
        'command_scale_encoding':'literal_config_dispatch' if scales is not None else 'scalar_immediate'}
    b.dense_schedule_receipts.append(report)
    return report


def lower_dense(b,lhs_ptr,rhs_ptr,out_ptr,m,k,n,attrs=None,bias_ptr=None,rhs_transposed=False,schedule_options=None,resident_key=None):
    """Emit MxK @ KxN, with optional row-per-key RHS transpose.

    All pointers refer to contiguous *logical*, unpadded tensors. Output dtype,
    scalar/column scale, bias and activation are taken solely from attrs.
    Returns actual shared scheduling realization metadata.
    """
    attrs=attrs or {}; stages=attrs.get('epilogue',[])
    if 'maxpool' in stages:
        return _lower_dense_pooled(b,lhs_ptr,rhs_ptr,out_ptr,m,k,n,attrs,bias_ptr,rhs_transposed,schedule_options)
    dtype=attrs.get('output_dtype','i32')
    if dtype not in ('i8','i32'): raise ValueError('dense output requires i8 or i32')
    unsupported=set(stages)-{'bias_add','acc_scale','relu'}
    if unsupported: raise ValueError('dense readout stages not yet supported: '+str(sorted(unsupported)))
    if dtype=='i32' and ('acc_scale' in stages or 'relu' in stages):
        raise ValueError('full-width accumulator readout cannot apply scale/activation')
    if ('bias_add' in stages)!=(bias_ptr is not None): raise ValueError('dense bias pointer/stage disagree')
    edge,bank_rows,operand_bytes,acc_bytes=_geometry()
    scales=attrs.get('acc_scale_by_column') if 'acc_scale' in stages else None
    if scales is not None and len(scales)!=n: raise ValueError('column scale count differs from N')
    from column_producer import prepare as prepare_column_readout, raw_attributes, record as record_column_producer
    column_readout=prepare_column_readout(attrs,m,n)
    if column_readout is not None:
        raw=b.global_buffer(m*n,'i32')
        producer=lower_dense(b,lhs_ptr,rhs_ptr,raw,m,k,n,raw_attributes(attrs),
            bias_ptr,rhs_transposed,schedule_options,resident_key)
        narrowed=column_readout.emit(b,raw,out_ptr)
        # The readout owns ACC rows but never touches the retained RHS bank.
        return record_column_producer(b,producer,narrowed,'dense')
    tile_n=1 if scales is not None else edge
    p=DenseProblem(m,k,n,edge,edge,tile_n,1,1,4,bias=bias_ptr is not None)
    # Disjoint bank assignments. A bounded primitive streaming block is the
    # executable seed; shared counted IR handles every shape and K continuation.
    from dense_policy import current_options, regional_legality, apply_geometry_schedule
    options=current_options(m=m,k=k,n=n,edge=edge)
    if schedule_options is not None:
        options.update(schedule_options)
        overrides=dict(options.get('axis_overrides',{}))
        for axis in ('m','k','n'):
            if 'max_'+axis+'_blocks' in schedule_options: overrides[axis]=True
        options['axis_overrides']=overrides
    options=apply_geometry_schedule(options,m=m,k=k,n=n,edge=edge,tile_n=tile_n,rhs_transposed=rhs_transposed)
    legality=regional_legality(options,edge=edge,bank_rows=bank_rows,
        operand_bytes=operand_bytes,accumulator_bytes=acc_bytes,tile_n=tile_n)
    storage=DenseLoopStorage(DenseMemoryView('a','spad',edge,0,bank_rows),
        DenseMemoryView('b','spad',edge,bank_rows,bank_rows),
        DenseMemoryView('c','acc',edge*4,0,acc_bytes//(edge*4)))
    resources=DenseResources(operand_bytes,acc_bytes,
        max_m_blocks=int(options.get('max_m_blocks',1)),
        max_n_blocks=int(options.get('max_n_blocks',1)),
        max_k_blocks=int(options.get('max_k_blocks',1)),
        partial_output_tiles=True,partial_output_blocks=True)
    from dense_residency import setting as resident_setting,prepare as prepare_resident,invalidate_residency
    resident_mode=(schedule_options or {}).get('resident_reuse',resident_setting())
    source_contract = attrs.get('resident_source_contract')
    if source_contract is not None:
        from dense_resident_policy import source_resident_admission
        admission = source_resident_admission(p,storage,resident_key=resident_key,
            requested_mode=resident_mode,rhs_transposed=rhs_transposed,
            per_column=scales is not None,options=options,**source_contract['kwargs'])
        resident_mode = admission['selected_mode']
        if not hasattr(b,'dense_resident_source_receipts'):
            b.dense_resident_source_receipts = []
        b.dense_resident_source_receipts.append(dict(source_contract=source_contract,admission=admission))
    resident_result=resident_state=resident_info=None
    if resident_key is not None and resident_mode in ('retained','retained_tiny') and not rhs_transposed and tile_n==edge:
        try:
            if rhs_ptr==out_ptr:
                raise ValueError('resident RHS pointer aliases a member output')
            resident_result,resident_state,resident_info=prepare_resident(b,p,resources,storage,_adapter(),
                key=resident_key,rhs_ptr=rhs_ptr,bias_source=DenseBiasSource(n,'i32') if bias_ptr is not None else None)
        except ValueError as error:
            resident_info={'selected_path':'incumbent','refusal':str(error)}
    if resident_result is None:
        invalidate_residency(b,reason='dense incumbent writes the private RHS window')
        if resident_key is not None and resident_info is None:
            resident_info={'selected_path':'incumbent','refusal':'residency disabled or unsupported transpose/per-column primitive'}
    from attention_retention import setting as attention_setting
    attention_mode=(schedule_options or {}).get('attention_retention',attention_setting())
    if attention_mode not in ('off','full_k_operand','geometry_supported','resource_phased'):
        raise ValueError('unknown attention operand retention schedule')
    attention_result=attention_info=None
    attention_selection=None
    attention_requested=attention_mode
    if attention_mode in ('geometry_supported','resource_phased'):
        from attention_retention import select_retention
        from dense_wide import setting as preferred_wide_setting
        from dense_wide_policy import select_wide_schedule
        preferred_wide=select_wide_schedule(
            (schedule_options or {}).get('wide_schedule',preferred_wide_setting()),
            p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
        attention_selection=select_retention(attention_mode,p,rhs_transposed=rhs_transposed,
            wide_route_selected=preferred_wide['selected']!='off')
        attention_selection['preferred_wide_policy']=preferred_wide
        attention_mode=attention_selection['selected']
        if attention_mode=='off':
            attention_info={'selected_path':'incumbent','geometry_policy':attention_selection}
    if attention_mode!='off' and resident_result is None:
        from attention_retention import build_retained_schedule
        try:
            attention_result,attention_info=build_retained_schedule(p,resources,storage,_adapter(),options,
                attrs=attrs,rhs_transposed=rhs_transposed,transpose_staging_completed=rhs_transposed,
                bias_source=DenseBiasSource(n,'i32') if bias_ptr is not None else None,mode=attention_mode)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            attention_info={'selected_path':'incumbent','requested_path':attention_requested,'refusal':str(error)}
        if attention_selection is not None:
            attention_info['geometry_policy']=attention_selection
    from decode_projection import setting as decode_setting
    decode_mode=(schedule_options or {}).get('decode_projection',decode_setting())
    if decode_mode not in ('off','raw_segmented','scalar_segmented','geometry_supported'):
        raise ValueError('unknown decode projection mode')
    decode_result=decode_info=None
    if decode_mode!='off' and attention_result is None and resident_result is None:
        from decode_projection import build_projection
        try:
            decode_result,decode_info=build_projection(p,resources,storage,_adapter(),options,
                mode=decode_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':decode_mode,'refusal':str(error)}
    from decode_coalesced import setting as coalesced_setting
    coalesced_mode=(schedule_options or {}).get('decode_coalesced',coalesced_setting())
    if coalesced_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from decode_coalesced import build as build_coalesced
        try:
            decode_result,decode_info=build_coalesced(p,resources,storage,_adapter(),options,
                mode=coalesced_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':coalesced_mode,'refusal':str(error)}
    from dense_epoch_coalescing import setting as epoch_coalescing_setting
    epoch_coalescing_mode=(schedule_options or {}).get('dense_epoch_coalescing',epoch_coalescing_setting())
    if epoch_coalescing_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from dense_epoch_coalescing import build as build_epoch_coalescing
        try:
            decode_result,decode_info=build_epoch_coalescing(p,resources,storage,_adapter(),options,
                mode=epoch_coalescing_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':epoch_coalescing_mode,'refusal':str(error)}
    from grouped_tail_bands import setting as tail_bands_setting
    tail_bands_mode=(schedule_options or {}).get('grouped_tail_bands',tail_bands_setting())
    if tail_bands_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from grouped_tail_bands import build as build_tail_bands
        try:
            decode_result,decode_info=build_tail_bands(p,resources,storage,_adapter(),options,
                mode=tail_bands_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':tail_bands_mode,'refusal':str(error)}
    from quantized_geometry import setting as quantized_bands_setting
    quantized_bands_mode=(schedule_options or {}).get('dense_quantized_bands',quantized_bands_setting())
    if quantized_bands_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from quantized_geometry import build as build_quantized_bands
        try:
            decode_result,decode_info=build_quantized_bands(p,resources,storage,_adapter(),options,
                mode=quantized_bands_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':quantized_bands_mode,'refusal':str(error)}
    from quantized_plan_ablation import setting as quantized_plan_ablation_setting
    quantized_plan_ablation_mode=(schedule_options or {}).get('quantized_plan_ablation',quantized_plan_ablation_setting())
    if quantized_plan_ablation_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from quantized_plan_ablation import build as build_quantized_plan_ablation
        try:
            decode_result,decode_info=build_quantized_plan_ablation(p,resources,storage,_adapter(),options,
                mode=quantized_plan_ablation_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':quantized_plan_ablation_mode,'refusal':str(error)}
    from sibling_stream import setting as sibling_stream_setting
    sibling_stream_mode=(schedule_options or {}).get('sibling_stream',sibling_stream_setting())
    if sibling_stream_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from sibling_stream import build as build_sibling_stream
        try:
            decode_result,decode_info=build_sibling_stream(p,resources,storage,_adapter(),options,
                mode=sibling_stream_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':sibling_stream_mode,'refusal':str(error)}
    multirow_lhs_refusal = None
    multirow_lhs_bands_refusal = None
    from multirow_lhs_policy import setting as multirow_lhs_policy_setting, build as build_multirow_lhs_policy
    multirow_lhs_policy_mode=(schedule_options or {}).get('multirow_lhs_policy',multirow_lhs_policy_setting())
    if multirow_lhs_policy_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from multirow_lhs import setting as original_lhs_setting
        from multirow_lhs_bands import setting as original_bands_setting
        from multirow_lhs_geometry import setting as original_geometry_setting
        policy_options=dict(options,**{key:(schedule_options or {}).get(key,setting()) for key,setting in
            [('multirow_lhs',original_lhs_setting),('multirow_lhs_bands',original_bands_setting),('multirow_lhs_geometry',original_geometry_setting)]})
        try:
            decode_result,decode_info=build_multirow_lhs_policy(p,resources,storage,_adapter(),policy_options,
                mode=multirow_lhs_policy_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            multirow_lhs_bands_refusal=str(error)
    from multirow_lhs_geometry import setting as multirow_lhs_geometry_setting, build as build_multirow_lhs_geometry
    multirow_lhs_geometry_mode=(schedule_options or {}).get('multirow_lhs_geometry',multirow_lhs_geometry_setting())
    if multirow_lhs_geometry_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        try:
            decode_result,decode_info=build_multirow_lhs_geometry(p,resources,storage,_adapter(),options,
                mode=multirow_lhs_geometry_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            multirow_lhs_bands_refusal=str(error)
    from multirow_lhs_bands import setting as multirow_lhs_bands_setting, build as build_multirow_lhs_bands
    multirow_lhs_bands_mode=(schedule_options or {}).get('multirow_lhs_bands',multirow_lhs_bands_setting())
    if multirow_lhs_bands_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        try:
            decode_result,decode_info=build_multirow_lhs_bands(p,resources,storage,_adapter(),options,
                mode=multirow_lhs_bands_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            multirow_lhs_bands_refusal=str(error)
    from multirow_lhs import setting as multirow_lhs_setting, build as build_multirow_lhs
    multirow_lhs_mode=(schedule_options or {}).get('multirow_lhs',multirow_lhs_setting())
    if multirow_lhs_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        try:
            decode_result,decode_info=build_multirow_lhs(p,resources,storage,_adapter(),options,
                mode=multirow_lhs_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            multirow_lhs_refusal=str(error)
    from multirow_raw import setting as multirow_raw_setting
    multirow_mode=(schedule_options or {}).get('multirow_raw',multirow_raw_setting())
    if multirow_mode!='off' and decode_result is None and attention_result is None and resident_result is None:
        from multirow_raw import build as build_multirow_raw
        try:
            decode_result,decode_info=build_multirow_raw(p,resources,storage,_adapter(),options,
                mode=multirow_mode,attrs=attrs,rhs_transposed=rhs_transposed)
        except (ValueError,ModuleNotFoundError) as error:
            if isinstance(error,ModuleNotFoundError) and error.name!='z3':raise
            decode_info={'selected_path':'incumbent','requested_path':multirow_mode,'refusal':str(error)}
    epoch_info=None
    epochs=decode_result is None and attention_result is None and resident_result is None and options.get('completion_epochs','off')!='off'
    readout=decode_result is None and attention_result is None and resident_result is None and options.get('readout_schedule','off')!='off'
    if epochs and readout:
        raise ValueError('dense completion epochs and segmented readout are independent alternatives; select one')
    if readout:
        if rhs_transposed or scales is not None:
            epoch_info={'selected_path':'incumbent','requested_path':options['readout_schedule'],
                'refusal':'transpose staging or per-column numerical snapshots are outside segmented ready law'}
        else:
            from dense_epochs import build_readout_schedule
            try:
                from dataclasses import replace
                # Use the same actual adapter and bounded resources throughout
                # original ready planning and final load-band re-admission.
                readout_adapter=_adapter()
                readout_resources=replace(resources,
                    max_k_blocks=min(resources.max_k_blocks,p.tiles[1]-1))
                result,epoch_info=build_readout_schedule(p,readout_resources,storage,readout_adapter,options,
                    bias_source=DenseBiasSource(n,'i32') if bias_ptr is not None else None)
                if options['readout_schedule']=='segmented_ready':
                    from dense_wide_loads import maybe_coalesce
                    result,epoch_info=maybe_coalesce(result,p,readout_resources,storage,readout_adapter,epoch_info)
                epochs=True
            except (ValueError, ModuleNotFoundError) as error:
                if isinstance(error, ModuleNotFoundError) and error.name != 'z3':
                    raise
                epochs=False
                epoch_info={'selected_path':'incumbent','requested_path':options['readout_schedule'],
                    'refusal':str(error)}
    if epochs and rhs_transposed:
        # CPU transpose scratch reuse needs host/DMA ownership beyond the local
        # memory dependency law; keep the incumbent until that route is qualified.
        epochs=False
        epoch_info={'selected_path':'incumbent','refusal':'CPU transpose staging ownership is outside dense epoch law'}
    if epochs and not readout:
        from dense_epochs import build_epoch_schedule
        try:
            result,epoch_info=build_epoch_schedule(p,resources,storage,_adapter(),options,
                bias_source=DenseBiasSource(n,'i32') if bias_ptr is not None else None)
        except (ValueError, ModuleNotFoundError) as error:
            if isinstance(error, ModuleNotFoundError) and error.name != 'z3':
                raise
            if options.get('schedule_policy','off')=='off': raise
            epochs=False
            epoch_info={'selected_path':'incumbent','requested_path':'two_window_epochs','refusal':str(error)}
            options['schedule_selection']['legality_refusal']=str(error)
    from dense_wide import setting as wide_setting
    wide_mode=(schedule_options or {}).get('wide_schedule',wide_setting())
    if wide_mode not in ('off','retained_lhs_wide','supported_geometry','matrix_coalesced_geometry'):
        raise ValueError('unknown wide dense schedule')
    if wide_mode!='off' and decode_result is None and attention_result is None and resident_result is None and not epochs and not readout:
        if rhs_transposed or scales is not None:
            epoch_info={'selected_path':'incumbent','requested_path':wide_mode,
                'refusal':'wide retention requires ordinary scalar readout and source operand views'}
        else:
            from dense_wide import build_wide_schedule
            try:
                if wide_mode in ('supported_geometry','matrix_coalesced_geometry'):
                    from dense_wide_policy import build_selected_wide
                    owner,decision=build_selected_wide(wide_mode,p,resources,storage,_adapter(),options,
                        attrs=attrs,bias_source=DenseBiasSource(n,'i32') if bias_ptr is not None else None)
                    if owner is None:
                        epoch_info={'selected_path':'incumbent','requested_path':wide_mode,'wide_policy':decision}
                    else:
                        result=owner
                        epoch_info={**owner.info,'wide_policy':owner.policy}
                        epochs=True
                else:
                    result,epoch_info=build_wide_schedule(p,resources,storage,_adapter(),options,
                        bias_source=DenseBiasSource(n,'i32') if bias_ptr is not None else None)
                    epochs=True
            except (ValueError, ModuleNotFoundError) as error:
                if isinstance(error, ModuleNotFoundError) and error.name != 'z3':
                    raise
                epoch_info={'selected_path':'incumbent','requested_path':wide_mode,'refusal':str(error)}
    if decode_result is not None:
        result=decode_result
        epoch_info=decode_info
        epochs=True
    if attention_result is not None:
        result=attention_result
        epoch_info=attention_info
        epochs=True
    if resident_result is not None:
        result=resident_result
        selected_block=vars(result.choice.block)
    elif not epochs:
        result=schedule_compact_dense(p,resources,storage,_adapter(),name='gemmini_dense',
            dtypes=('i8','i8','i32'),static_budget=128,external_buffers_disjoint=True,
            bias_source=DenseBiasSource(n,'i32') if bias_ptr is not None else None,
            outer_order=tuple(options.get('outer_order',('m','n'))))
        selected_block=vars(result.choice.block)
    else:
        selected_block=epoch_info['block']
    result.revalidate()
    transpose_tile=entry_transpose_tile(b,edge) if rhs_transposed else None
    scale_words=None if scales is None else [
        struct.unpack('<I',struct.pack('<f',float(s)))[0] for s in scales]
    b.config_ex(dataflow=1)
    b.config_ld(k,id=0)
    b.config_ld(edge if rhs_transposed else n,id=1)
    if bias_ptr is not None: b.config_ld(0,id=2)
    width=4 if dtype=='i32' else 1
    scale=attrs.get('acc_scale',1.0) if 'acc_scale' in stages else 1.0
    activation=1 if 'relu' in stages else 0
    b.config_st(n*width,activation=activation,scale=scale)
    def offset(ptr,row,col,stride,width=1):
        return b.ptr_offset(ptr,b.mul(b.add(b.mul(row,stride),col),width))
    def lower(name,a):
        if name=='lhs_load':
            b.mvin(offset(lhs_ptr,a['r'],a['c'],k),a['local'],cols=a['cols'],rows=a['rows'],id=0)
            if not epochs: b.fence()
        elif name=='weight_load':
            if rhs_transposed:
                # Pure layout transport; no tensor arithmetic executes on CPU.
                with b.counted_loop(0,a['rows']) as rr:
                    with b.counted_loop(0,a['cols']) as cc:
                        src=offset(rhs_ptr,b.add(a['c'],cc),b.add(a['r'],rr),k)
                        dst=offset(transpose_tile,rr,cc,edge)
                        b.store(b.load(src,'i8'),dst,'i8')
                src=transpose_tile
            else: src=offset(rhs_ptr,a['r'],a['c'],n)
            b.mvin(src,a['local'],cols=a['cols'],rows=a['rows'],id=1)
            # Retained commands may reuse physical operands, but the original
            # host transpose tile remains a single DMA source. Complete its
            # read before any following host overwrite; stronger completion
            # does not remove any of the checked owner's original drains.
            if not epochs or (attention_result is not None and rhs_transposed): b.fence()
        elif name=='bias':
            b.mvin(b.ptr_offset(bias_ptr,b.mul(a['c'],4)),b.or_(a['local'],0x80000000),cols=a['cols'],rows=a['rows'],id=2)
            if not epochs: b.fence()
        elif name=='preload':
            acc=b.or_(b.or_(a['acc'],0x80000000),b.shl(a['continue'],30))
            b.preload(a['weight'],acc,b_cols=a['cols'],b_rows=a['depth'],c_cols=a['cols'],c_rows=a['rows'])
        elif name=='compute':
            b.compute(a['lhs'],0xffffffff,a_cols=a['depth'],a_rows=a['rows'])
            if not epochs: b.fence()
        elif name=='store':
            if scale_words is not None:
                b.constant_i32_dispatch(scale_words,a['c'],
                    lambda bits:b.config_st(n*width,activation=activation,scale_bits=bits))
            local=b.or_(a['local'],0xa0000000 if dtype=='i32' else 0x80000000)
            b.mvout(offset(out_ptr,a['r'],a['c'],n,width),local,cols=a['cols'],rows=a['rows'])
            if not epochs: b.fence()
        else: raise ValueError(name)
    index_mode=(schedule_options or {}).get('index_unroll')
    if index_mode is None and resident_result is not None and resident_mode=='retained_tiny':
        index_mode='tiny'
    bridge=_IndexBridge(b,index_mode)
    if epochs:
        receipt=result.emit_index_program(bridge,lower,complete_epoch=b.fence)
    else:
        receipt=emit_index_program(result.kernel,bridge,lower)
    receipt['index_unroll']=bridge.receipt()
    result.revalidate()
    if attention_selection is not None and attention_result is not None:
        attention_selection['actual_schedule_applied']=True
    if resident_state is not None:
        from dense_residency import commit as commit_resident
        commit_resident(b,resident_state)
    if decode_result is not None:
        decode_info['decode_projection']['actual_schedule_applied']=True
    report={'realization':result.realization,'emission':receipt,'shape':[m,k,n],
        'primitive':[edge,edge,tile_n], 'block':selected_block, 'epoch_schedule':epoch_info,
        'policy':options, 'regional_legality':legality,
        'command_scale_encoding':'literal_config_dispatch' if scales is not None else 'scalar_immediate',
        'resident_rhs':resident_info,'attention_retention':attention_info,'decode_projection':decode_info}
    if multirow_lhs_bands_refusal is not None:
        report['multirow_lhs_bands_refusal'] = multirow_lhs_bands_refusal
    if multirow_lhs_refusal is not None:
        report['multirow_lhs_refusal'] = multirow_lhs_refusal
    if not hasattr(b,'dense_schedule_receipts'): b.dense_schedule_receipts=[]
    b.dense_schedule_receipts.append(report)
    return report


def lower_batched_dense(b,lhs_ptr,rhs_ptr,out_ptr,batch,m,k,n,attrs=None,bias_ptr=None,rhs_transposed=False):
    attrs=attrs or {}; width=4 if attrs.get('output_dtype','i32')=='i32' else 1
    output_rows=m
    if 'maxpool' in attrs.get('epilogue',[]):
        from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
        h,w=attrs['pool_in_dims']
        pool=SpatialMaxReadout(tuple(attrs['pool_in_dims']),tuple(attrs['pool_size']),
            tuple(attrs['pool_stride']),tuple(attrs.get('pool_padding',(0,0,0,0))),attrs.get('pool_pad_value'))
        oh,ow=pool.output_hw
        output_rows=m//(h*w)*oh*ow
    with b.counted_loop(0,batch) as index:
        return lower_dense(b,b.ptr_offset(lhs_ptr,b.mul(index,m*k)),
            b.ptr_offset(rhs_ptr,b.mul(index,k*n)),b.ptr_offset(out_ptr,b.mul(index,output_rows*n*width)),
            m,k,n,attrs,bias_ptr,rhs_transposed)
