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
    try:
        return schedule_compact_dense(p,r,s,_adapter(),name='dense',dtypes=('i8','i8','i32'),static_budget=128,external_buffers_disjoint=True,verification_timeout_ms=2000,bias_source=DenseBiasSource(n,'i32') if bias else None)
    except ValueError as exc:
        # An unavailable symbolic proof may use the independent exact primitive
        # checker. Concrete invariant failures still refuse.
        from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
        if not isinstance(exc,DenseVerificationUnavailable):
            raise
        from .dense_compact_verification import schedule_primitive_dense
        return schedule_primitive_dense(p,r,s,_adapter(),bias_source=DenseBiasSource(n,'i32') if bias else None)


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
    configured=knob('DENSE_PIPELINE',default='compact',values=('compact','retained','segmented','streaming','adaptive'),root=Path(__file__).resolve().parent.parent)
    selection=attrs.get('dense_pipeline',os.environ.get('DENSE_PIPELINE',configured))
    if selection not in ('compact','retained','segmented','streaming','adaptive'):
        raise ValueError('DENSE_PIPELINE must be compact, retained, segmented, streaming or adaptive')
    schedule=None
    refusal=None
    realized_selection=selection
    fallback=None
    adaptive_receipt=None
    # Advanced schedules require their own optional symbolic proof. Do not
    # enter a planner whose dependency is absent; the exact primitive route
    # below remains an accelerator lowering with conservative completion.
    try:
        import z3
    except ModuleNotFoundError as exc:
        if exc.name!='z3':
            raise
        refusal='optional z3 proof dependency unavailable for advanced dense scheduling'
    else:
        from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
        try:
            if selection in ('retained','streaming','adaptive') and not per_column and m<=isa.DIM and k<=isa.DIM and n<=isa.DIM:
                refusal='single primitive product has no retained operand reuse or reduction overlap'
            elif selection=='adaptive':
                from .dense_pipeline import PANEL_GEOMETRIES,RETAINED_CAPACITY_REFUSAL
                from .dense_pipeline_policy import execute_adaptive_dense_frontier
                geometry_default=knob('DENSE_PANEL_GEOMETRY',default='balanced',values=tuple(PANEL_GEOMETRIES),root=Path(__file__).resolve().parent.parent)
                geometry=os.environ.get('DENSE_PANEL_GEOMETRY',geometry_default)
                split=knob('DENSE_STORAGE_SPLIT',default='half',values=('half','adaptive'),root=Path(__file__).resolve().parent.parent)
                operand_default=knob('DENSE_RETAINED_OPERAND',default='lhs',values=('lhs','weight','capacity'),root=Path(__file__).resolve().parent.parent)
                operand=os.environ.get('DENSE_RETAINED_OPERAND',operand_default)
                if operand not in ('lhs','weight','capacity'):
                    raise ValueError('DENSE_RETAINED_OPERAND must be lhs, weight or capacity')
                limit_default=knob('DENSE_SEGMENT_K_LIMIT',default='16',values=('8','16','32','64'),root=Path(__file__).resolve().parent.parent)
                segment_k_limit=int(os.environ.get('DENSE_SEGMENT_K_LIMIT',limit_default))
                # Adaptive admits the bounded capacity policy after frontier selection;
                # unconditional weight retention remains an explicit retained-family option.
                from .dense_segment_options import resolve_segment_geometry
                segment_geometry=resolve_segment_geometry(geometry)
                schedule,refusal,adaptive_receipt=execute_adaptive_dense_frontier(int(m),int(k),int(n),seeded,per_column,geometry=geometry,storage_split=split,
                    retained_operand='capacity' if operand=='capacity' else 'lhs',segment_k_limit=segment_k_limit,segmented_geometry=segment_geometry)
                realized_selection=adaptive_receipt['policy']['selected_variant']
                if schedule is None and refusal==RETAINED_CAPACITY_REFUSAL:
                    from .dense_streaming import streaming_schedule
                    schedule,streaming_refusal=streaming_schedule(int(m),int(k),int(n),seeded,per_column)
                    fallback={'requested':'adaptive','attempted':'streaming','reason':refusal}
                    if schedule is not None:
                        realized_selection='streaming'
                    else:
                        fallback['refusal']=streaming_refusal
            elif selection=='streaming':
                from .dense_streaming import streaming_schedule
                schedule,refusal=streaming_schedule(int(m),int(k),int(n),seeded,per_column)
            elif selection in ('retained','segmented'):
                from .dense_pipeline import retained_schedule,RETAINED_CAPACITY_REFUSAL
                schedule,refusal=retained_schedule(int(m),int(k),int(n),seeded,per_column,selection=='segmented')
                if selection=='segmented' and schedule is None:
                    # Optional frontier plans keep the incumbent when their exact
                    # source/storage domain is unavailable. Preserve its ordinary
                    # retained-to-streaming capacity fallback as well.
                    schedule,retained_refusal=retained_schedule(int(m),int(k),int(n),seeded,per_column,False,frontier='off')
                    fallback={'requested':'segmented','attempted':'retained','reason':refusal}
                    if schedule is not None:
                        realized_selection='retained'
                    else:
                        fallback['refusal']=retained_refusal
                        if retained_refusal==RETAINED_CAPACITY_REFUSAL:
                            from .dense_streaming import streaming_schedule
                            schedule,streaming_refusal=streaming_schedule(int(m),int(k),int(n),seeded,per_column)
                            fallback['attempted']='retained_then_streaming'
                            if schedule is not None:
                                realized_selection='streaming'
                            else:
                                fallback['streaming_refusal']=streaming_refusal
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
        except DenseVerificationUnavailable as exc:
            schedule=None
            refusal='optional advanced symbolic proof unavailable: '+str(exc)
    advanced=schedule is not None
    if schedule is None:
        schedule=_schedule(int(m),int(k),int(n),seeded,per_column)
    from .dense_store_epochs import select_store_epochs,BEGIN,END,STORE
    store_epoch_report=None
    from .dense_compact_verification import PrimitiveDenseResult
    if isinstance(schedule,PrimitiveDenseResult):
        store_epoch_report={'applied':False,'refusal':'optional SMT store-epoch optimization unavailable; primitive full drains retained'}
    elif not per_column:
        schedule,store_epoch_report=select_store_epochs(schedule,attrs,n,
            attrs.get('output_dtype',ctx.spec(dst)['dtype']))
    load_run_default=knob('DENSE_PRIMITIVE_LOAD_RUNS',default='off',
        values=('off','unroll4'),root=Path(__file__).resolve().parent.parent)
    load_runs=attrs.get('dense_primitive_load_runs',os.environ.get('DENSE_PRIMITIVE_LOAD_RUNS',load_run_default))
    if load_runs not in ('off','unroll4'):
        raise ValueError('DENSE_PRIMITIVE_LOAD_RUNS must be off or unroll4')
    if load_runs=='unroll4':
        from .wide_load_runs import select_primitive_load_runs
        schedule=select_primitive_load_runs(schedule,_adapter(),factor=4)
    lhs_stride=ctx.row_stride(lhs,k)
    rhs_stride=ctx.row_stride(rhs,n)
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(lhs_stride,channel=0))
    ctx.emit(*isa.config_ld(rhs_stride,channel=1))
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
            stride=lhs_stride if left else rhs_stride
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
    if store_epoch_report is not None:
        receipt['store_epochs']=store_epoch_report
    if accumulator_init:
        receipt['accumulator_initialization']={'source':accumulator_init,'layout':'row-major-i32','row_stride_bytes':n*4,'broadcast_bias_added':bool(bias)}
    receipt['requested_pipeline']=selection
    receipt['selected_pipeline']=realized_selection if advanced else 'compact'
    if adaptive_receipt is not None:
        receipt['adaptive_frontier']=adaptive_receipt
    if refusal:
        receipt['pipeline_refusal']=refusal
    if fallback:
        receipt['pipeline_fallback']=fallback
    ctx.metadata.setdefault('dense_schedules',[]).append(receipt)
    return receipt
