"""General bounded dense lowering through the shared compact schedule IR."""
from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path
from merlin.compiler.selection.knobs import knob
from merlin.compiler.schedule.blocks import Load,Preload,Compute,Store,InitializeAccumulator
from merlin.compiler.schedule.compact_dense import DenseLoopAdapter,DenseLoopStorage,DenseMemoryView,DenseBiasSource,schedule_compact_dense
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
    uniform_columns=None
    uniform_selection=attrs.get('dense_uniform_column_panels',knob(
        'DENSE_UNIFORM_COLUMN_PANELS',default='off',values=('off','direct'),
        root=Path(__file__).resolve().parent.parent))
    if uniform_selection not in ('off','direct'):
        raise ValueError('DENSE_UNIFORM_COLUMN_PANELS must be off or direct')
    if per_column and uniform_selection=='direct' and attrs.get('output_dtype',ctx.spec(dst)['dtype'])=='i8':
        from .dense_uniform_columns import prove_uniform_column_panels
        uniform_columns=prove_uniform_column_panels(n,attrs)
    # Keep the original source per-column classification for independent
    # numerical-state/epoch admissions. Only the source-bound proof allows the
    # original DIM-wide producer geometry instead of a lane-zero producer.
    column_geometry=per_column and uniform_columns is None
    if per_column and uniform_columns is None and column_route=='raw_reload':
        from .dense_column import lower_column_readout
        receipt=lower_column_readout(ctx,lhs,rhs,dst,m=m,k=k,n=n,attributes=attrs)
        ctx.metadata.setdefault('dense_schedules',[]).append(receipt)
        return receipt
    configured=knob('DENSE_PIPELINE',default='compact',values=('compact','retained','segmented','streaming','capacity'),root=Path(__file__).resolve().parent.parent)
    selection=attrs.get('dense_pipeline',os.environ.get('DENSE_PIPELINE',configured))
    if selection not in ('compact','retained','segmented','streaming','capacity'):
        raise ValueError('DENSE_PIPELINE must be compact, retained, segmented, streaming or capacity')
    from .dense_ready import select_ready_readout
    ready_readout,ready_report=select_ready_readout(attrs,attrs.get('output_dtype',ctx.spec(dst)['dtype']),per_column)
    schedule=None
    refusal=None
    capacity_receipt=None
    geometry_receipt=None
    realized_selection=selection
    fallback=None
    if selection in ('retained','streaming','capacity') and not per_column and m<=isa.DIM and k<=isa.DIM and n<=isa.DIM:
        refusal='single primitive product has no retained operand reuse or reduction overlap'
    elif selection=='streaming':
        from .dense_streaming import streaming_schedule
        schedule,refusal=streaming_schedule(int(m),int(k),int(n),seeded,column_geometry)
    elif selection in ('retained','segmented','capacity'):
        from .dense_pipeline import retained_schedule,RETAINED_CAPACITY_REFUSAL
        if selection=='capacity':
            from .dense_pipeline_policy import execute_capacity_pipeline
            schedule,refusal,capacity_receipt=execute_capacity_pipeline(
                int(m),int(k),int(n),seeded,column_geometry,ready_readout=ready_readout)
            realized_selection=capacity_receipt['selected']
            from .dense_segmented_geometry import execute_segmented_geometry
            schedule,refusal,geometry_receipt=execute_segmented_geometry(
                int(m),int(k),int(n),seeded,column_geometry,schedule,refusal,
                ready_readout=ready_readout)
        else:
            schedule,refusal=retained_schedule(int(m),int(k),int(n),seeded,column_geometry,selection=='segmented',ready_readout=ready_readout)
        if selection in ('retained','capacity') and schedule is None and refusal==RETAINED_CAPACITY_REFUSAL:
            # The retained planner has checked every legal full-K block under
            # its assigned storage. Preserve all legal retained realizations;
            # only this capacity refusal permits the bounded streaming family.
            from .dense_streaming import streaming_schedule
            from merlin.compiler.schedule.compact_verification import DenseVerificationUnavailable
            try:
                schedule,streaming_refusal=streaming_schedule(int(m),int(k),int(n),seeded,column_geometry)
            except DenseVerificationUnavailable as exc:
                # This optional route has no source proof. Preserve its refusal
                # and construct the conservative accelerator schedule below;
                # that schedule must pass its own original source verification.
                schedule=None
                streaming_refusal=type(exc).__name__+': '+str(exc)
            fallback={'requested':'retained','attempted':'streaming','reason':refusal}
            if schedule is not None:
                realized_selection='streaming'
            else:
                fallback['refusal']=streaming_refusal
    advanced=schedule is not None
    if schedule is None:
        schedule=_schedule(int(m),int(k),int(n),seeded,column_geometry)
    from .dense_store_epochs import select_store_epochs,BEGIN,END,STORE
    store_epoch_report=None
    ready_active=bool(schedule.realization.get('ready_readout_selected'))
    if not column_geometry and not ready_active:
        schedule,store_epoch_report=select_store_epochs(schedule,attrs,n,
            attrs.get('output_dtype',ctx.spec(dst)['dtype']))
    uniform_report=None
    if uniform_columns is not None:
        from .dense_uniform_columns import verify_uniform_column_schedule
        uniform_report=verify_uniform_column_schedule(schedule,uniform_columns,m,n,attrs)
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(k,channel=0))
    ctx.emit(*isa.config_ld(n,channel=1))
    if bias and not accumulator_init:
        ctx.emit(*isa.config_ld(0,channel=2,shrink=False))
    if ready_active:
        from .dense_ready import uniform_store_command
        ctx.emit(*uniform_store_command(attrs,n,attrs.get('output_dtype',ctx.spec(dst)['dtype'])))
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
        elif instruction=='gemmini_mvout' and not ready_active:
            store_tile(ctx,dst,a['acc'],a['rows'],a['cols'],n,a['dram_row'],a['dram_col'],attrs,uniform_columns=uniform_columns)
        elif instruction==STORE or (instruction=='gemmini_mvout' and ready_active):
            full=attrs.get('output_dtype',ctx.spec(dst)['dtype'])=='i32'
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(n)),a['dram_col'])
            ctx.emit(*isa.mvout(ctx.ptr(dst,offset),isa.acc_addr(a['acc'],full=full,fb=fb),
                a['rows'],a['cols'],fb=fb))
        else:
            raise ValueError('unknown terminal dense instruction '+instruction)
    schedule.emit_llvm(fb,emit)
    receipt=dict(schedule.realization)
    if capacity_receipt is not None:
        receipt['pre_geometry_capacity_selection']=capacity_receipt
    if geometry_receipt is not None:
        receipt['segmented_geometry_stage']=geometry_receipt
    receipt['ready_readout_selection']=dict(ready_report,applied=ready_active)
    if uniform_report is not None:
        receipt['uniform_column_panels']=uniform_report
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
    ctx.metadata.setdefault('dense_schedules',[]).append(receipt)
    return receipt
