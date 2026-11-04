"""Checked library dense scheduling, lowered through the shared Gemmini encoder.

The library owns source axes, blocking, retained panels, reduction order and tails.
This adapter owns ISA effects and separately binds the hardware readout epilogue.
No numerical work runs on the host.
"""
from __future__ import annotations
import hashlib
import json
import os
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Load, Preload, Compute, Store, InitializeAccumulator
from merlin.compile.scheduling.compact_dense import (DenseLoopAdapter, DenseLoopStorage, DenseMemoryView,
    DenseBiasSource, schedule_compact_dense)
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.sched.ir import Call, as_expr
import gemmini_encode as E
from gemmini_readout import source_scales, column_scale_runs, emit_readout, _relu

ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = Path(__file__).resolve().parent/'data'/'rtl_facts.json'
if not FACTS_PATH.is_file(): FACTS_PATH = ROOT/'rtl_facts/facts.json'


def dense_adapter(*, readout_identity='raw-i32'):
    """A reversible effect boundary; projection reads the actual call operands."""
    def call(name, names, values):
        return Call(name, tuple(zip(names, (as_expr(int(v) if type(v) is bool else v) for v in values))))
    def encode(op):
        if isinstance(op, Load):
            return call('gemmini.mvin.'+op.role,
                ('block0','block1','src_row','src_col','rows','cols','local','step'),
                (*op.block,op.dram_row,op.dram_col,op.rows,op.cols,op.row,op.step))
        if isinstance(op, InitializeAccumulator):
            return call('gemmini.mvin.bias',('block0','block1','src_col','rows','cols','local','row_stride','col_stride','src_row','additional_bias'),
                (*op.block,op.dram_col,op.rows,op.cols,op.row,op.source_row_stride,op.source_col_stride,op.dram_row,op.additional_bias))
        if isinstance(op, Preload):
            return call('gemmini.preload',('weight','block0','block1','acc','continue','rows','cols','depth'),
                (op.weight_row,*op.weight_block,op.accumulator_row,op.accumulate,op.rows,op.cols,op.weight_rows))
        if isinstance(op, Compute):
            return call('gemmini.compute',('lhs','block0','block1','rows','depth'),
                (op.input_row,*op.input_block,op.rows,op.cols))
        if isinstance(op, Store):
            return call('gemmini.mvout',('row','col','rows','cols','acc'),
                (op.dram_row,op.dram_col,op.rows,op.cols,op.accumulator_row))
        raise TypeError(type(op))
    def project(c):
        a=dict(c.args)
        if c.instr in ('gemmini.mvin.lhs','gemmini.mvin.weight'):
            return Load(c.instr.rsplit('.',1)[1],(a['block0'],a['block1']),a['src_row'],a['src_col'],
                a['rows'],a['cols'],a['local'],a['step'])
        if c.instr=='gemmini.mvin.bias':
            return InitializeAccumulator((a['block0'],a['block1']),a['src_col'],a['rows'],a['cols'],a['local'],a['row_stride'],a['col_stride'],a['src_row'],a['additional_bias'])
        if c.instr=='gemmini.preload':
            return Preload(a['weight'],(a['block0'],a['block1']),a['acc'],a['continue'],a['rows'],a['cols'],a['depth'])
        if c.instr=='gemmini.compute':
            return Compute(a['lhs'],(a['block0'],a['block1']),a['rows'],a['depth'],True)
        if c.instr=='gemmini.mvout':
            return Store(a['row'],a['col'],a['rows'],a['cols'],a['acc'])
        raise ValueError('unrecognized dense effect '+c.instr)
    facts=FACTS_PATH.read_bytes()
    return DenseLoopAdapter(encode,project,'gemmini-rtl:'+hashlib.sha256(facts).hexdigest(),
        'source-ordered-i8-i8-i32:'+readout_identity,True,True,64,
        qualification_identity=None)


@lru_cache(maxsize=1)
def geometry():
    from merlin.targetgen import address_space as AS
    facts=json.loads(FACTS_PATH.read_text())
    space=AS.derive_address_space('gemmini',facts=facts)
    op,acc=AS.operand_store(space),AS.accumulator_store(space)
    if op.store is None or acc.store is None or space.array_rows!=space.array_cols:
        raise ValueError('dense needs selected square-array operand and accumulator geometry')
    return space.array_rows,op.store,op.capacity_rows(),acc.store,acc.capacity_rows()


def plan_dense(m,k,n,*,bias=False,initial=False,readout_identity='raw-i32',block_limit=None,outer_order=None):
    edge,op,oprows,acc,accrows=geometry()
    p=DenseProblem(m,k,n,edge,edge,edge,1,1,4,bias=bias or initial)
    split=oprows//2
    s=DenseLoopStorage(DenseMemoryView('lhs',op.name,op.row_bytes,0,split),
        DenseMemoryView('rhs',op.name,op.row_bytes,split,oprows-split),
        DenseMemoryView('acc',acc.name,acc.row_bytes,0,accrows))
    # Bounded search policy, independent of tensor size; capacities constrain every candidate.
    # Exposed alternatives are compiler knobs, and require paired timing before selection.
    limits=block_limit or tuple(int(x) for x in os.environ.get('GEMMINI_DENSE_BLOCK_LIMIT','8,1,8').split(','))
    if len(limits)!=3 or min(limits)<=0:
        raise ValueError('dense block limit must contain three positive M,K,N integers')
    r=DenseResources(oprows*op.row_bytes,accrows*acc.row_bytes,
        max_m_blocks=limits[0],max_k_blocks=limits[1],max_n_blocks=limits[2],
        partial_output_tiles=True,partial_output_blocks=True)
    adapter=dense_adapter(readout_identity=readout_identity)
    order=outer_order or tuple(os.environ.get('GEMMINI_DENSE_OUTER_ORDER','m,n').split(','))
    result=schedule_compact_dense(p,r,s,adapter,name='dense',dtypes=('i8','i8','i32'),
        static_budget=128,external_buffers_disjoint=True,outer_order=order,
        verification_timeout_ms=20000,bias_source=seed_source(m,n,bias=bias,initial=initial))
    return result,adapter,p


def seed_source(m,n,*,bias=False,initial=False):
    return (DenseBiasSource(n,'i32',rows=m,row_stride=n,additional_bias=bias) if initial
        else DenseBiasSource(n,'i32') if bias else None)


_CHECKED_EMISSION = ContextVar('gemmini_dense_checked_emission', default=None)


@contextmanager
def consume_checked_dense(result, adapter, problem, *, mode, readout_identity):
    """Bind an exploration artifact to the unchanged production terminal emitter.

    Scoped to this context/thread; an exact single-source contraction must consume
    the artifact once. Ordinary compilation never installs this override.
    """
    result.revalidate()
    state = dict(result=result, adapter=adapter, problem=problem, mode=mode,
                 readout_identity=readout_identity, consumed=0)
    token = _CHECKED_EMISSION.set(state)
    try:
        yield state
        if state['consumed'] != 1:
            raise ValueError('checked exploration artifact must be consumed exactly once')
    finally:
        _CHECKED_EMISSION.reset(token)


def emit_dense(fn,lhs,rhs,dst,m,k,n,*,attrs=None,bias=None,initial=None,lhs_dtype='i8',rhs_dtype='i8',output_dtype='i32'):
    """Emit row-major signed dense contraction and optional hardware scalar readout.

    Bias is a contiguous per-column i32 pointer, broadcast in the accumulator.
    Initial is an MxN i32 matrix read before the contraction, with optional
    initial_stride in i32 elements. Bias, when also supplied, is added after
    this matrix overwrite and before the first product contribution.
    attrs: scale (scalar fp32), relu (bool), activation ('relu'/'none'/0/1).
    The contraction checker proves exact logical i32 accumulation and memory axes;
    the terminal readout config implements the independently stated scalar epilogue.
    """
    attrs=dict(attrs or {})
    if lhs_dtype not in ('i8','int8') or rhs_dtype not in ('i8','int8'):
        raise ValueError('Gemmini dense requires signed i8 operands')
    if output_dtype not in ('i32','int32','i8','int8'):
        raise ValueError('Gemmini dense output must be i32 or i8')
    narrow=output_dtype in ('i8','int8')
    scale=source_scales(attrs)
    if isinstance(scale,(list,tuple)):
        if len(scale)!=n: raise ValueError('dense scale vector must contain one coefficient per output column')
        if len(set(scale))!=1:
            if not narrow: raise ValueError('per-column scale requires i8 output')
            receipts=[]
            for start,width,coefficient in column_scale_runs(attrs,n):
                subattrs={key:value for key,value in attrs.items() if key not in ('acc_scale_by_column','acc_scale','scale','output_scale')}
                subattrs.update(scale=coefficient,lhs_stride=attrs.get('lhs_stride',k),
                    rhs_stride=attrs.get('rhs_stride',n),dst_stride=attrs.get('dst_stride',n),
                    initial_stride=attrs.get('initial_stride',n))
                receipts.append(emit_dense(fn,lhs,E.pointer_offset(fn,rhs,start),E.pointer_offset(fn,dst,start),
                    m,k,width,attrs=subattrs,bias=None if bias is None else E.pointer_offset(fn,bias,start*4),
                    initial=None if initial is None else E.pointer_offset(fn,initial,start*4),
                    lhs_dtype=lhs_dtype,rhs_dtype=rhs_dtype,output_dtype=output_dtype))
            return {'library':'checked-dense-source-column-intervals','source_shape':(m,k,n),'intervals':receipts}
        scale=scale[0]
    scale=float(scale)
    activation=attrs.get('activation',0)
    relu=bool(attrs.get('relu',False)) or activation in (1,'relu','ReLU') or 'relu' in attrs.get('epilogue',[])
    shift=int(attrs.get('requant_shift',attrs.get('shift',0)))
    if shift:
        raise ValueError('dense hardware shift requires a qualified source requantization path')
    if not narrow and (scale!=1.0 or relu):
        raise ValueError('full i32 readout does not apply scaling or activation')
    readout=json.dumps({'dtype':output_dtype,'scale':scale,'relu':relu},sort_keys=True)
    checked=_CHECKED_EMISSION.get()
    if checked is None:
        result,adapter,p=plan_dense(m,k,n,bias=bias is not None,initial=initial is not None,readout_identity=readout)
    else:
        p=checked['problem']
        if (p.m,p.k,p.n)!=(m,k,n) or p.bias!=(bias is not None or initial is not None) or checked['readout_identity']!=readout:
            raise ValueError('exploration artifact source axes/initializer/readout differ from actual source')
        if checked['consumed']:
            raise ValueError('exploration artifact consumed more than once')
        checked['consumed']+=1
        result,adapter=checked['result'],checked['adapter']
        result.revalidate()
    from merlin.compile.tuning_knobs import knob
    dense_schedule=attrs.get('dense_schedule',knob('dense_schedule',default='compact',
        values=('compact','panel_drained','panel_epochs','ready_readout','segmented_ready','auto','auto_ready','array_epochs'),root=Path(__file__).resolve().parent))
    if checked is not None:dense_schedule=checked['mode']
    schedule_receipt=None
    auto_consumed=False
    gemv_columns=knob('gemv_column_blocks',default='off',values=('off',16,32,64),
        root=Path(__file__).resolve().parent)
    if checked is None and p.m==1 and gemv_columns!='off':
        from gemmini_gemv import plan_gemv
        try:
            result,schedule_receipt=plan_gemv(result,adapter,p,geometry(),FACTS_PATH,
                column_blocks=gemv_columns,bias=bias is not None,initial=initial is not None,
                outer_order=tuple(os.environ.get('GEMMINI_DENSE_OUTER_ORDER','m,n').split(',')))
            dense_schedule='panel_epochs'
            auto_consumed=True
        except ValueError as error:
            # The original checked source plan remains available when this
            # explicit alternative lacks a required capacity or proof premise.
            schedule_receipt={'gemv_unavailable':str(error)}
    longk_columns=knob('dense_longk_columns',default=32,values=('off',8,16,32),
        root=Path(__file__).resolve().parent)
    short_panels=knob('dense_short_panels',default='off',values=('off','on'),
        root=Path(__file__).resolve().parent)
    if (short_panels=='on' and dense_schedule in ('auto','auto_ready')
            and checked is None and not auto_consumed and gemv_columns=='off'
            and longk_columns==32):
        from gemmini_dense_short_policy import plan_short_panel
        from gemmini_dense_readout import ReadoutUnavailable
        try:
            result,dense_schedule,schedule_receipt=plan_short_panel(result,adapter,p,
                geometry(),FACTS_PATH,bias=bias is not None,initial=initial is not None,
                attrs=attrs,output_dtype=output_dtype,
                outer_order=tuple(os.environ.get('GEMMINI_DENSE_OUTER_ORDER','m,n').split(',')))
            auto_consumed=True
        except ReadoutUnavailable as error:
            schedule_receipt={'short_panel_unavailable':str(error)}
    if (dense_schedule in ('auto','auto_ready') and checked is None and not auto_consumed
            and gemv_columns=='off' and p.tiles[0]==1 and longk_columns!='off'):
        from gemmini_dense_longk_experiment import plan_longk_output
        if output_dtype in ('i8','int8') and knob('dense_scalar_longk',default='on',
                values=('off','on'),root=Path(__file__).resolve().parent)=='on':
            from gemmini_dense_scalar import plan_fused_dense as plan_longk_output
        from gemmini_dense_readout import ReadoutUnavailable
        longk_wide=knob('dense_longk_wide',default='off',values=('off',48,64,'auto_fullrow'),
            root=Path(__file__).resolve().parent)
        if longk_wide in (48,64) or (longk_wide=='auto_fullrow' and p.m==geometry()[0]):
            from functools import partial
            from gemmini_dense_longk_wide import plan_wide_or_fallback
            plan_longk_output=partial(plan_wide_or_fallback,plan_longk_output,
                wide_column_blocks=48 if longk_wide==48 else 64)
        try:
            longk_result,longk_receipt=plan_longk_output(result,adapter,p,geometry(),FACTS_PATH,
                column_blocks=longk_columns,bias=bias is not None,initial=initial is not None,
                attrs=attrs,output_dtype=output_dtype,
                outer_order=tuple(os.environ.get('GEMMINI_DENSE_OUTER_ORDER','m,n').split(',')))
            edge,operand,operand_rows,accumulator,accumulator_rows=geometry()
            block=longk_receipt['longk_output_experiment']['bounded_source_block']
            lhs_rows=block['m']*p.tiles[1]*edge
            full_rhs_rows=block['n']*p.tiles[1]*edge
            if full_rhs_rows<=(operand_rows-lhs_rows)//2:
                raise ReadoutUnavailable('complete K fits one physical RHS segment')
            result,schedule_receipt=longk_result,longk_receipt
            dense_schedule='panel_epochs'
            auto_consumed=True
        except ReadoutUnavailable as error:
            schedule_receipt={'longk_unavailable':str(error)}
    if dense_schedule in ('auto','auto_ready') and checked is None and not auto_consumed:
        from dense_policy import auto_schedule,ready_auto_schedule
        if dense_schedule=='auto_ready':
            rhs_reuse=knob('dense_rhs_reuse',default='off',values=('off','ready','wide'),
                root=Path(__file__).resolve().parent)
            if rhs_reuse == 'off':
                from gemmini_dense_coalesced_policy import m1_coalesced_auto_schedule
                select_dense=m1_coalesced_auto_schedule
                reuse_options={}
            else:
                from gemmini_dense_rhs_ready_policy import rhs_ready_auto_schedule
                select_dense=rhs_ready_auto_schedule
                reuse_options=dict(allow_scalar_readout=True,aggregate_transfers=rhs_reuse=='wide')
            result,dense_schedule,schedule_receipt=select_dense(
                result,adapter,p,geometry(),FACTS_PATH,bias=bias is not None,
                initial=initial is not None,attrs=attrs,output_dtype=output_dtype,
                outer_order=tuple(os.environ.get('GEMMINI_DENSE_OUTER_ORDER','m,n').split(',')),
                **reuse_options)
        else:
            result,dense_schedule,schedule_receipt=auto_schedule(result,adapter,p,geometry(),FACTS_PATH,
                bias=bias is not None,initial=initial is not None)
        auto_consumed=True
    asynchronous=dense_schedule!='compact'
    if asynchronous and checked is None and not auto_consumed:
        from gemmini_dense_pipeline import plan_pipeline
        arguments=dict(bias=bias is not None,initial=initial is not None,
            outer_order=tuple(os.environ.get('GEMMINI_DENSE_OUTER_ORDER','m,n').split(',')))
        if dense_schedule=='array_epochs':
            from gemmini_dense_array import plan_array
            try:
                result,schedule_receipt,adapter=plan_array(result,adapter,p,geometry(),FACTS_PATH,**arguments)
            except ValueError as error:
                result,schedule_receipt=plan_pipeline(result,adapter,p,geometry(),FACTS_PATH,mode='panel_epochs',**arguments)
                schedule_receipt.update(requested_mode='array_epochs',fallback_reason=str(error))
                dense_schedule='panel_epochs'
        elif dense_schedule in ('ready_readout','segmented_ready'):
            from gemmini_dense_readout import plan_ready_readout,ReadoutUnavailable
            planner=plan_ready_readout
            if dense_schedule=='segmented_ready':
                from gemmini_dense_segmented import plan_segmented_readout
                planner=plan_segmented_readout
                overlap=knob('segmented_readout_overlap',default='off',values=('off','disjoint','coalesced'),
                        root=Path(__file__).resolve().parent)
                if overlap=='disjoint':
                    from gemmini_dense_overlap import plan_overlap
                    planner=plan_overlap
                elif overlap=='coalesced':
                    from gemmini_dense_transfers import plan_transfers
                    planner=plan_transfers
            requested=dense_schedule
            try:
                if (dense_schedule=='segmented_ready' and overlap=='coalesced'
                        and (int(attrs.get('lhs_stride',k))!=k or int(attrs.get('rhs_stride',n))!=n)):
                    raise ReadoutUnavailable('coalesced transfers require original contiguous row-major operands')
                result,schedule_receipt=planner(result,adapter,p,geometry(),FACTS_PATH,**arguments)
            except ReadoutUnavailable as error:
                result,schedule_receipt=plan_pipeline(result,adapter,p,geometry(),FACTS_PATH,mode='panel_epochs',**arguments)
                schedule_receipt.update(requested_mode=requested,fallback_reason=str(error))
        else:
            result,schedule_receipt=plan_pipeline(result,adapter,p,geometry(),FACTS_PATH,mode=dense_schedule,**arguments)
    lower=dense_terminal(fn,lhs,rhs,dst,p,attrs=attrs,bias=bias,initial=initial,
        output_dtype=output_dtype,asynchronous=asynchronous)
    if dense_schedule=='array_epochs':
        from gemmini_dense_array import wrap_terminal
        lower=wrap_terminal(fn,lower,p)
    if asynchronous:
        if knob('command_pack_bounds',default='off',values=('off','on'),
                root=Path(__file__).resolve().parent)=='on':
            from gemmini_command_unroll import emit_bounded_effects
            command_receipt=emit_bounded_effects(result,fn,lower,complete_epoch=lambda:E.fence(fn))
            schedule_receipt={**(schedule_receipt or {}),'command_pack_bounds':command_receipt['pack_bounds']}
        else:
            from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
            result.emit_index_program(FnBuilderControlAdapter(fn),lower,complete_epoch=lambda:E.fence(fn))
    else:
        command_unroll=knob('command_small_loop_unroll',default='off',values=('off','on','deep16','auto'),
                root=Path(__file__).resolve().parent)
        if command_unroll!='off':
            from gemmini_command_unroll import emit_compact
            command_receipt=emit_compact(result,fn,lower,mode=command_unroll)
            schedule_receipt={**(schedule_receipt or {}),'command_loop_unroll':command_receipt}
        else:
            if knob('command_pack_bounds',default='off',values=('off','on'),
                    root=Path(__file__).resolve().parent)=='on':
                from gemmini_command_unroll import emit_bounded_compact
                command_receipt=emit_bounded_compact(result,fn,lower)
                schedule_receipt={**(schedule_receipt or {}),'command_pack_bounds':command_receipt['pack_bounds']}
            else:
                result.emit_llvm(fn,lower)
    E.fence(fn)
    from gemmini_activity import dense_activity
    return {'library':'schedule_compact_dense' if not asynchronous else 'schedule_dense_pipeline',
        'dense_schedule':dense_schedule,'schedule_receipt':schedule_receipt,'realization':result.realization,
        'source_shape':(m,k,n),'readout':json.loads(readout),
        'activity':dense_activity(result,adapter,p,output_dtype=output_dtype)}


def dense_terminal(fn,lhs,rhs,dst,p,*,attrs=None,bias=None,initial=None,output_dtype='i32',asynchronous=False):
    """One terminal effect encoder for private and shared-RHS checked schedules."""
    attrs=dict(attrs or {});m,k,n=p.m,p.k,p.n
    narrow=output_dtype in ('i8','int8');scale=source_scales(attrs);relu=_relu(attrs)
    scale=float(scale[0] if isinstance(scale,(list,tuple)) else scale)
    lhs_stride=int(attrs.get('lhs_stride',k))
    rhs_stride=int(attrs.get('rhs_stride',n))
    dst_stride=int(attrs.get('dst_stride',n))
    initial_stride=int(attrs.get('initial_stride',n))
    if lhs_stride<k or rhs_stride<n or dst_stride<n or initial_stride<n:
        raise ValueError('dense physical row stride is smaller than its logical row')
    E.config_ex(fn,dataflow='ws')
    E.config_load(fn,lhs_stride,stream=0)
    E.config_load(fn,rhs_stride,stream=1)
    if initial is not None or bias is not None:
        E.config_load(fn,initial_stride*4 if initial is not None else 0,stream=2,shrunk=False)
    E.config_store(fn,dst_stride*(1 if narrow else 4),scale=scale,activation=int(relu))
    def add(a,b):
        return fn.add_i(a if not isinstance(a,int) else fn.const(a),b if not isinstance(b,int) else fn.const(b))
    def mul(a,b):
        return fn.mul_i(a if not isinstance(a,int) else fn.const(a),b if not isinstance(b,int) else fn.const(b))
    def lower(name,a):
        if name in ('gemmini.mvin.lhs','gemmini.mvin.weight'):
            left=name.endswith('.lhs'); stride=lhs_stride if left else rhs_stride
            offset=add(mul(a['src_row'],stride),a['src_col'])
            E.mvin(fn,E.pointer_offset(fn,lhs if left else rhs,offset),a['local'],a['rows'],a['cols'],stream=0 if left else 1)
            if not asynchronous: E.fence(fn)
        elif name=='gemmini.mvin.bias':
            if initial is not None:
                # A combined initializer is a checked matrix read followed by
                # a separately declared modular column-vector addition.
                if bias is not None:E.config_load(fn,initial_stride*4,stream=2,shrunk=False)
                offset=mul(add(mul(a['src_row'],initial_stride),a['src_col']),4)
                E.mvin(fn,E.pointer_offset(fn,initial,offset),add(a['local'],E.ACC),a['rows'],a['cols'],stream=2)
                if bias is not None:
                    E.fence(fn)
                    E.config_load(fn,0,stream=2,shrunk=False)
                    E.mvin(fn,E.pointer_offset(fn,bias,mul(a['src_col'],4)),add(a['local'],E.ACC_ADD),a['rows'],a['cols'],stream=2)
            else:
                E.mvin(fn,E.pointer_offset(fn,bias,mul(a['src_col'],4)),add(a['local'],E.ACC),a['rows'],a['cols'],stream=2)
            if not asynchronous: E.fence(fn)
        elif name=='gemmini.preload':
            E.preload(fn,a['weight'],add(add(a['acc'],E.ACC),mul(a['continue'],E.ACCUMULATE)),
                a['depth'],a['cols'],a['rows'],a['cols'])
        elif name=='gemmini.compute':
            E.compute(fn,a['lhs'],E.GARBAGE,a['rows'],a['depth'],p.tile_k,p.tile_n,accumulate=False)
            if not asynchronous: E.fence(fn)
        elif name=='gemmini.mvout':
            offset=mul(add(mul(a['row'],dst_stride),a['col']),1 if narrow else 4)
            emit_readout(fn,E.pointer_offset(fn,dst,offset),a['acc'],a['rows'],a['cols'],
                output_stride=dst_stride*(1 if narrow else 4),attrs={'scale':scale,'relu':relu},output_dtype=output_dtype)
            if not asynchronous: E.fence(fn)
        else: raise ValueError('unrecognized emitted dense effect '+name)
    return lower


def emit_resident_group(fn,lhs,rhs,dst,m,k,n,*,attrs=None,source_identity,
                        external_buffers_disjoint=True,scheduler=None):
    """Emit a source-bound shared-RHS region, or decline before any emission.

    The owner supplies adjacent independent products with one immutable RHS,
    equal geometry and readout attributes, distinct liveouts and disjoint ABI
    buffers. Original source grouping/eviction boundaries remain caller-owned.
    No execution qualification is inferred from these source premises.
    """
    from dataclasses import replace
    from merlin.sched.ir import Ptr
    from merlin.compile.scheduling.dense_shared_rhs import schedule_dense_shared_rhs
    from merlin.compile.scheduling.dense_shared_rhs_check import DenseRHSReuseLaw
    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    lhs,dst=tuple(lhs),tuple(dst);attrs=dict(attrs or {})
    if not 2<=len(lhs)==len(dst)<=128 or external_buffers_disjoint is not True:return None
    if (not isinstance(source_identity,str) or len(source_identity)!=64 or
        any(c not in '0123456789abcdef' for c in source_identity)):
        raise ValueError('resident group needs original source SHA256 identity')
    dtype=attrs.get('output_dtype','i32')
    if dtype!='i32' or source_scales(attrs)!=1 or _relu(attrs):return None
    if any(attrs.get(key) for key in ('shift','requant_shift','bias')):return None
    if any(stage not in () for stage in attrs.get('epilogue',())):return None
    if any(key in attrs and int(attrs[key])!=stride for key,stride in
        (('lhs_stride',k),('rhs_stride',n),('dst_stride',n))):return None
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry();split=operand_rows//2
    p=DenseProblem(m,k,n,edge,edge,edge,1,1,4)
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=8,max_k_blocks=p.tiles[1],max_n_blocks=8,
        partial_output_tiles=True,partial_output_blocks=True)
    adapter=dense_adapter(readout_identity=json.dumps({'dtype':'i32','scale':1.0,'relu':False},sort_keys=True))
    def encode_member(member,op):
        call=adapter.encode(op);ptr=None
        if isinstance(op,Load):
            weight=op.role=='weight'
            ptr=Ptr('rhs' if weight else 'lhs'+str(member),
                (as_expr(op.dram_row)*(n if weight else k)+op.dram_col))
        elif isinstance(op,Store):
            ptr=Ptr('output'+str(member),(as_expr(op.dram_row)*n+op.dram_col)*4)
        return replace(call,instr=('shared' if member is None else str(member))+'::'+call.instr,
            args=call.args+((('source_pointer',ptr),) if ptr is not None else ()))
    def project_member(call):
        member,name=call.instr.split('::',1)
        projected=replace(call,instr=name,args=tuple((key,value) for key,value in call.args if not isinstance(value,Ptr)))
        return (None if member=='shared' else int(member)),adapter.project(projected)
    law_identity=hashlib.sha256(json.dumps(dict(source=source_identity,
        shapes=[m,k,n],members=len(lhs),rhs_view=storage.rhs.__dict__,
        ownership='source immutable RHS until eviction; private local stores and disjoint external buffers'),
        sort_keys=True).encode()).hexdigest()
    law=DenseRHSReuseLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
        adapter.qualification_identity,'source-derived-private-rhs:'+law_identity,
        rhs_unchanged_by_member_operations=True,independent_member_interleaving=True,
        suppressed_loads_no_faults=True,suppressed_loads_no_extra_observations=True)
    if scheduler is None:scheduler=schedule_dense_shared_rhs
    result=scheduler((p,)*len(lhs),resources,(storage,)*len(lhs),adapter,
        encode_member=encode_member,project_member=project_member,source_identity=source_identity,
        name='resident_dense',dtypes=('i8','i8','i32'),static_budget=4096,
        external_buffers_disjoint=True,retention_law=law,verification_timeout_ms=20000)
    if result is None:return None
    result.revalidate()
    # Preparation is complete before touching the builder. Shared prefix and
    # every source member then use the exact incumbent terminal effect encoder.
    lowers=tuple(dense_terminal(fn,a,rhs,d,p,attrs=attrs,asynchronous=True) for a,d in zip(lhs,dst))
    def lower(tag,values):
        member,name=tag.split('::',1)
        lowers[0 if member=='shared' else int(member)](name,values)
    emission=result.emit_index_program(FnBuilderControlAdapter(fn),lower,complete_epoch=lambda:E.fence(fn))
    E.fence(fn)
    from gemmini_activity import dense_activity
    return dict(library='schedule_dense_shared_rhs',source_identity=source_identity,
        source_shape=(m,k,n),members=len(lhs),realization=result.realization,
        schedule_plan=result.plan_text,checked_kernel_sha256=result.kernel.digest(),
        emission=emission,retention_law=law.record(),qualification_identity=None,
        activity=dense_activity(result,adapter,p,output_dtype='i32',project_member=project_member),
        qualification='source/effect integer proof; target execution qualification remains external')
