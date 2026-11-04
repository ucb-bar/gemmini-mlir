"""General bounded dense lowering through the shared compact schedule IR."""
from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.block_schedule import Load,Preload,Compute,Store,InitializeAccumulator
from merlin.compile.scheduling.compact_dense import DenseLoopAdapter,DenseLoopStorage,DenseMemoryView,DenseBiasSource,schedule_compact_dense
from merlin.transforms.dense_planning import DenseProblem,DenseResources
from merlin.sched.ir import Call,as_expr
from . import isa
from .dense_readout import store_tile,_pack


def _adapter():
    def encode(op):
        if isinstance(op,Load):
            instruction='gemmini_mvin_'+op.role
            names=('block0','block1','dram_row','dram_col','rows','cols','addr','step')
            values=(*op.block,op.dram_row,op.dram_col,op.rows,op.cols,op.row,op.step)
        elif isinstance(op,Preload):
            instruction='gemmini_preload'
            names=('weight','block0','block1','acc','accumulate','rows','cols','depth')
            values=(op.weight_row,*op.weight_block,op.accumulator_row,op.accumulate,op.rows,op.cols,op.weight_rows)
        elif isinstance(op,Compute):
            instruction='gemmini_compute_preloaded'
            names=('lhs','block0','block1','rows','depth')
            values=(op.input_row,*op.input_block,op.rows,op.cols)
        elif isinstance(op,InitializeAccumulator):
            instruction='gemmini_bias_mvin'
            names=('block0','block1','dram_col','rows','cols','addr','src_row_stride','src_col_stride')
            values=(*op.block,op.dram_col,op.rows,op.cols,op.row,op.source_row_stride,op.source_col_stride)
        elif isinstance(op,Store):
            instruction='gemmini_mvout'
            names=('dram_row','dram_col','rows','cols','acc')
            values=(op.dram_row,op.dram_col,op.rows,op.cols,op.accumulator_row)
        else:
            raise ValueError(f'unsupported dense operation {op}')
        return Call(instruction,tuple(zip(names,map(as_expr,values))))
    def project(call):
        a=dict(call.args)
        if call.instr.startswith('gemmini_mvin_'):
            return Load(call.instr.removeprefix('gemmini_mvin_'),(a['block0'],a['block1']),a['dram_row'],a['dram_col'],a['rows'],a['cols'],a['addr'],a['step'])
        if call.instr=='gemmini_preload':
            return Preload(a['weight'],(a['block0'],a['block1']),a['acc'],a['accumulate'],a['rows'],a['cols'],a['depth'])
        if call.instr=='gemmini_compute_preloaded':
            return Compute(a['lhs'],(a['block0'],a['block1']),a['rows'],a['depth'],True)
        if call.instr=='gemmini_bias_mvin':
            return InitializeAccumulator((a['block0'],a['block1']),a['dram_col'],a['rows'],a['cols'],a['addr'],a['src_row_stride'],a['src_col_stride'])
        if call.instr=='gemmini_mvout':
            return Store(a['dram_row'],a['dram_col'],a['rows'],a['cols'],a['acc'])
        raise ValueError('unknown dense instruction '+call.instr)
    return DenseLoopAdapter(encode,project,'gemmini-fire-aligned-dim16-ws','i8-i8-i32-modular',True,True,64)


@lru_cache(maxsize=256)
def _schedule(m,k,n,bias,column_scales=False):
    dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,1 if column_scales else dim,1,1,4,bias=bias)
    # Assign disjoint bank windows; the conservative incumbent streams one
    # primitive tile per role and completes before reusing either window.
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,dim),DenseMemoryView('rhs','spad',dim,isa.SPAD_ROWS//2,dim),DenseMemoryView('acc','acc',4*dim,0,dim))
    r=DenseResources(2*dim*dim,4*dim*dim,max_m_blocks=1,max_n_blocks=1,max_k_blocks=1,partial_output_tiles=True,partial_output_blocks=True)
    return schedule_compact_dense(p,r,s,_adapter(),name='dense',dtypes=('i8','i8','i32'),static_budget=128,external_buffers_disjoint=True,verification_timeout_ms=2000,bias_source=DenseBiasSource(n,'i32') if bias else None)


def lower_dense(ctx,lhs,rhs,dst,*,m,k,n,attributes=None):
    attrs=dict(attributes or {})
    fb=ctx.fb
    bias=attrs.get('bias')
    accumulator_init=attrs.get('accumulator_init')
    if accumulator_init is not None:
        seed_spec=ctx.spec(accumulator_init)
        if seed_spec['dtype']!='i32' or tuple(seed_spec['shape'])!=(m,n):
            raise ValueError('dense accumulator_init requires an i32 matrix with shape [m,n]')
    seeded=bool(bias or accumulator_init)
    scales=attrs.get('acc_scale_by_column')
    per_column=scales is not None and len(set(map(float,scales)))>1
    if scales is not None and len(scales)!=n:
        raise ValueError('column scale count must equal N')
    column_default=knob('DENSE_COLUMN_READOUT',default='recompute',values=('recompute','raw_reload'),root=Path(__file__).resolve().parent.parent)
    column_route=attrs.get('dense_column_readout',os.environ.get('DENSE_COLUMN_READOUT',column_default))
    if column_route not in ('recompute','raw_reload'):
        raise ValueError('DENSE_COLUMN_READOUT must be recompute or raw_reload')
    if per_column and column_route=='raw_reload':
        from .dense_column import lower_column_readout
        receipt=lower_column_readout(ctx,lhs,rhs,dst,m=m,k=k,n=n,attributes=attrs)
        ctx.metadata.setdefault('dense_schedules',[]).append(receipt)
        return receipt
    configured=knob('DENSE_PIPELINE',default='compact',values=('compact','retained','segmented','streaming','output_prefetch','segmented_capacity','regional','capacity_stream'),root=Path(__file__).resolve().parent.parent)
    selection=attrs.get('dense_pipeline',os.environ.get('DENSE_PIPELINE',configured))
    if selection not in ('compact','retained','segmented','streaming','output_prefetch','segmented_capacity','regional','capacity_stream'):
        raise ValueError('unknown DENSE_PIPELINE selection')
    schedule=None
    refusal=None
    realized_selection=selection
    fallback=None
    policy_receipt=None
    geometry_receipt=None
    thin_receipt=None
    thin_default=knob('DENSE_THIN_STORAGE',default='off',values=('off','compact_m'),
        root=Path(__file__).resolve().parent.parent)
    thin_selection=os.environ.get('DENSE_THIN_STORAGE',thin_default)
    if thin_selection not in ('off','compact_m'):
        raise ValueError('DENSE_THIN_STORAGE must be off or compact_m')
    if thin_selection=='compact_m' and selection=='regional' and 0<m<isa.DIM and not per_column:
        from .dense_thin_storage import thin_storage_schedule
        schedule,thin_refusal=thin_storage_schedule(int(m),int(k),int(n),seeded,per_column)
        thin_receipt={'requested':thin_selection,'admitted':schedule is not None,
            'refusal':thin_refusal}
        if schedule is not None:
            realized_selection='thin_storage'
    if schedule is not None:
        pass
    elif selection in ('retained','streaming','output_prefetch','segmented_capacity','regional') and not per_column and m<=isa.DIM and k<=isa.DIM and n<=isa.DIM:
        refusal='single primitive product has no retained operand reuse or reduction overlap'
    elif selection=='regional':
        from .dense_pipeline import PANEL_GEOMETRIES, RETAINED_CAPACITY_REFUSAL
        from .dense_pipeline_policy import execute_native_dense_pipeline_policy
        geometry_default=knob('DENSE_PANEL_GEOMETRY',default='balanced',values=tuple(PANEL_GEOMETRIES),
            root=Path(__file__).resolve().parent.parent)
        geometry=os.environ.get('DENSE_PANEL_GEOMETRY',geometry_default)
        storage_split=knob('DENSE_STORAGE_SPLIT',default='half',values=('half','adaptive'),
            root=Path(__file__).resolve().parent.parent)
        retained_planner=None
        geometry_policy=knob('DENSE_GEOMETRY_POLICY',default='off',values=('off','native'),
            root=Path(__file__).resolve().parent.parent)
        if geometry_policy=='native' and geometry=='wide':
            from .dense_geometry_policy import resolve_admitted_geometry
            from .dense_pipeline import _retained_schedule
            geometry,checked_retained,retained_refusal,geometry_receipt=resolve_admitted_geometry(
                int(m),int(k),int(n),seeded,per_column,storage_split=storage_split)
            original=(int(m),int(k),int(n),seeded,per_column,geometry,storage_split)
            def reuse_retained(mm,kk,nn,bb,cc,segmented,gg,split):
                if not segmented and (mm,kk,nn,bb,cc,gg,split)==original:
                    return checked_retained,retained_refusal
                return _retained_schedule(mm,kk,nn,bb,cc,segmented,gg,split)
            retained_planner=reuse_retained
        schedule,refusal,policy_receipt=execute_native_dense_pipeline_policy(
            int(m),int(k),int(n),seeded,per_column,geometry=geometry,storage_split=storage_split,
            retained_planner=retained_planner,
            output_policy=knob('DENSE_OUTPUT_POLICY',default='off',values=('off','wide_prefetch'),
                root=Path(__file__).resolve().parent.parent))
        realized_selection=policy_receipt['policy']['selected_variant'] if schedule is not None else 'retained'
        if schedule is None and refusal==RETAINED_CAPACITY_REFUSAL:
            from .dense_streaming import streaming_schedule
            schedule,streaming_refusal=streaming_schedule(int(m),int(k),int(n),seeded,per_column)
            fallback={'requested':'regional','attempted':'streaming','reason':refusal}
            realized_selection='streaming'
            if schedule is None:
                fallback['refusal']=streaming_refusal
    elif selection=='capacity_stream':
        from .dense_output_pipeline import capacity_stream_schedule
        k_tiles=knob('DENSE_CAPACITY_K_TILES',default=16,values=(1,4,8,16,32),
            root=Path(__file__).resolve().parent.parent)
        schedule,refusal=capacity_stream_schedule(int(m),int(k),int(n),seeded,per_column,k_tiles=k_tiles)
    elif selection=='output_prefetch':
        from .dense_output_pipeline import output_group_schedule
        schedule,refusal=output_group_schedule(int(m),int(k),int(n),seeded,per_column)
        if schedule is None:
            from .dense_pipeline import retained_schedule,RETAINED_CAPACITY_REFUSAL
            schedule,retained_refusal=retained_schedule(int(m),int(k),int(n),seeded,per_column)
            fallback={'requested':'output_prefetch','attempted':'retained','reason':refusal}
            realized_selection='retained'
            if schedule is None and retained_refusal==RETAINED_CAPACITY_REFUSAL:
                from .dense_streaming import streaming_schedule
                schedule,streaming_refusal=streaming_schedule(int(m),int(k),int(n),seeded,per_column)
                fallback['attempted']='streaming'
                realized_selection='streaming'
                if schedule is None:
                    fallback['refusal']=streaming_refusal
    elif selection=='streaming':
        from .dense_streaming import streaming_schedule
        schedule,refusal=streaming_schedule(int(m),int(k),int(n),seeded,per_column)
    elif selection in ('retained','segmented','segmented_capacity'):
        from .dense_pipeline import retained_schedule,RETAINED_CAPACITY_REFUSAL
        schedule,refusal=retained_schedule(int(m),int(k),int(n),seeded,per_column,selection in ('segmented','segmented_capacity'),
            segment_k_tiles='capacity' if selection=='segmented_capacity' else 16)
        if selection=='retained' and schedule is None and refusal==RETAINED_CAPACITY_REFUSAL:
            # The retained planner has checked every legal full-K block under
            # its assigned storage. Preserve all legal retained realizations;
            # only this capacity refusal permits the bounded streaming family.
            from .dense_streaming import streaming_schedule
            schedule,streaming_refusal=streaming_schedule(int(m),int(k),int(n),seeded,per_column)
            fallback={'requested':'retained','attempted':'streaming','reason':refusal}
            if schedule is not None:
                realized_selection='streaming'
            else:
                fallback['refusal']=streaming_refusal
    advanced=schedule is not None
    if schedule is None:
        schedule=_schedule(int(m),int(k),int(n),seeded,per_column)
    from .dense_store_epochs import select_store_epochs,BEGIN,END,STORE
    store_epoch_report=None
    if not per_column:
        schedule,store_epoch_report=select_store_epochs(schedule,attrs,n,
            attrs.get('output_dtype',ctx.spec(dst)['dtype']))
    from .submission_pipeline import select_control
    schedule,control_report=select_control(schedule,int(m),int(k),int(n),attrs)
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(k,channel=0))
    ctx.emit(*isa.config_ld(n,channel=1))
    if bias and not accumulator_init:
        ctx.emit(*isa.config_ld(0,channel=2,shrink=False))
    descriptor_columns=[None]
    def emit(instruction,a,metadata=None):
        a={key:fb.const(value) if isinstance(value,int) else value for key,value in a.items()}
        if instruction=='merlin_dense_panel_drain':
            ctx.fence()
            return
        if instruction==BEGIN:
            ctx.fence()
            ctx.emit(*schedule.command)
            return
        if instruction==END:
            ctx.fence()
            return
        if instruction.startswith('gemmini_mvin_'):
            left=instruction=='gemmini_mvin_lhs'
            stride=k if left else n
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(stride)),a['dram_col'])
            ctx.emit(isa.FUNCT['mvin'] if left else isa.FUNCT['mvin2'],ctx.ptr(lhs if left else rhs,offset),_pack(fb,a['addr'],a['rows'],a['cols']))
        elif instruction=='gemmini_bias_mvin':
            if accumulator_init:
                # The generic schedule owns this tile's explicit initializer
                # and all later continuation updates. Its block coordinates
                # identify the original matrix seed, rather than a bias row.
                row=fb.mul_i(a['block0'],fb.const(isa.DIM))
                offset=fb.add_i(fb.mul_i(row,fb.const(n)),a['dram_col'])
                ctx.emit(*isa.config_ld(n*4,channel=2,shrink=False))
                ctx.emit(*isa.mvin(ctx.ptr(accumulator_init,offset),isa.acc_addr(a['addr'],fb=fb),a['rows'],a['cols'],channel=2,fb=fb))
                ctx.fence()
            if bias:
                if accumulator_init:
                    ctx.emit(*isa.config_ld(0,channel=2,shrink=False))
                ctx.emit(*isa.mvin(ctx.ptr(bias,a['dram_col']),isa.acc_addr(a['addr'],accumulate=bool(accumulator_init),fb=fb),a['rows'],a['cols'],channel=2,fb=fb))
                ctx.fence()
        elif instruction=='gemmini_preload':
            descriptor_columns[0]=a['cols']
            caddr=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,caddr,a['rows'],a['cols']))
        elif instruction=='gemmini_compute_preloaded':
            if descriptor_columns[0] is None:
                raise ValueError('compute requires its immediately preceding preload descriptor')
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,fb.const(isa.GARBAGE),a['depth'],descriptor_columns[0]))
            if not advanced:
                ctx.fence()
        elif instruction=='gemmini_mvout':
            store_tile(ctx,dst,a['acc'],a['rows'],a['cols'],n,a['dram_row'],a['dram_col'],attrs)
        elif instruction==STORE:
            full=attrs.get('output_dtype',ctx.spec(dst)['dtype'])=='i32'
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(n)),a['dram_col'])
            ctx.emit(*isa.mvout(ctx.ptr(dst,offset),isa.acc_addr(a['acc'],full=full,fb=fb),
                a['rows'],a['cols'],fb=fb))
        else:
            raise ValueError('unknown terminal dense instruction '+instruction)
    schedule.emit_llvm(fb,emit)
    receipt=dict(schedule.realization)
    if control_report is not None:
        receipt['command_control']=control_report
    if policy_receipt is not None:
        receipt['regional_policy']=policy_receipt
    if geometry_receipt is not None:
        receipt['geometry_policy']=geometry_receipt
    if store_epoch_report is not None:
        receipt['store_epochs']=store_epoch_report
    if accumulator_init:
        receipt['accumulator_initialization']={'source':accumulator_init,'layout':'row-major-i32','row_stride_bytes':n*4,'broadcast_bias_added':bool(bias)}
    receipt['requested_pipeline']=selection
    receipt['selected_pipeline']=realized_selection if advanced else 'compact'
    if refusal:
        receipt['pipeline_refusal']=refusal
    if fallback:
        receipt['pipeline_fallback']=fallback
    if thin_receipt is not None:
        receipt['thin_storage_policy']=thin_receipt
    ctx.metadata.setdefault('dense_schedules',[]).append(receipt)
    return receipt
