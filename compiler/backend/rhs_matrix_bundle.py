"""Checked pointer-only packing of independent same-RHS matrix products.

A virtual row i*M+r is exactly member i's original row r. Only LHS DMA and
raw output DMA are split at member boundaries; every MAC, reduction order,
RHS transfer and accumulator seed comes from the checked retained schedule.
No host allocation, tensor copy, value-dependent dispatch or arithmetic runs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json

from merlin.compile.scheduling.batched_dense import (
    BatchMatrixView, BatchedDenseProblem, counted_batched_dense_products,
)
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.sched.ir import Call, Const, Loop, TensorArg, check_structure
from merlin.sched.ir.expr import evaluate, free_vars
from merlin.sched.primitives import Cursor, unroll
from merlin.sched.primitives.loop_handles import uniquify_loop_binders
from . import isa
from .dense_pipeline import _retained_schedule
from .dense_readout import _pack
from .rhs_row_bundle_epochs import _readout_epochs, _strip_completion
from .column_batches import _admission

DRAIN = 'merlin_dense_panel_drain'
COMPLETE = Call(DRAIN, ())


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _source(ctx, group):
    """Exact raw source admission, including inert parsed provenance only."""
    members = tuple(group)
    if not 2 <= len(members) <= 16 or ctx.cb.get('params',{}).get('tensor_aliases'):
        return None
    rhs = members[0].get('rhs')
    lhs = [row.get('lhs') for row in members]
    outputs = [row.get('dst') for row in members]
    if (rhs is None or None in lhs or None in outputs or len(set(lhs)) != len(lhs)
            or rhs in lhs or len(set(outputs)) != len(outputs)
            or set(outputs)&(set(lhs)|{rhs}) or any(row.get('rhs') != rhs for row in members)):
        return None
    shape = None
    known = {'epilogue','output_dtype','acc_scale','relu','bias','accumulator_init',
             'acc_scale_by_column','requant_shift','bound_lsb',
             'prov.region_id','prov.op','prov.family',
             'merlin.source_op_index','merlin.global_task'}
    for row in members:
        attrs = row.get('attributes',{})
        if any(key not in known for key in attrs):
            return None
        scale = attrs.get('acc_scale',1.0)
        if (type(scale) not in (int,float) or scale != 1 or
                attrs.get('epilogue',()) not in ((),[]) or attrs.get('output_dtype','i32') != 'i32' or
                attrs.get('relu',False) is not False or attrs.get('bias') is not None or
                attrs.get('accumulator_init') is not None or attrs.get('acc_scale_by_column') is not None or
                attrs.get('requant_shift') is not None or attrs.get('bound_lsb',0) != 0):
            return None
        a,b,c = (ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
        if (a['dtype'] != 'i8' or b['dtype'] != 'i8' or c['dtype'] != 'i32'
                or len(a['shape']) != 2 or len(b['shape']) != 2 or len(c['shape']) != 2):
            return None
        m,k = a['shape']; kk,n = b['shape']
        if (any(type(v) is not int or v <= 0 for v in (m,k,kk,n)) or m <= 1
                or k != kk or tuple(c['shape']) != (m,n)
                or m*len(members)>256 or m*k*len(members)*4 >= 1<<63
                or m*n*len(members)*4 >= 1<<63):
            return None
        if shape is not None and shape != (m,k,n): return None
        shape = m,k,n
        if any(key in row and row[key] != value for key,value in zip(('m','k','n'),shape)):
            return None
    return members,shape


def _specialize_rows(kernel):
    """Public ordered unroll only for bounded virtual-M transfer coordinates."""
    hygiene=uniquify_loop_binders(kernel,prefix="matrix_bundle_loop")
    variables = set()
    def collect(items):
        for item in items:
            if type(item) is Loop: collect(item.body)
            elif item.instr in ('gemmini_mvin_lhs','gemmini_mvout'):
                a=dict(item.args)
                variables.update(free_vars(a['dram_row'])|free_vars(a['rows']))
    collect(hygiene.kernel.body)
    def find(items,path=()):
        for item in items:
            if type(item) is Loop:
                at=(*path,item.var)
                nested=find(item.body,at)
                if nested:return nested
                if item.var in variables:return at
        return None
    actual=hygiene.kernel;passes=[]
    while (path:=find(actual.body)) is not None:
        actual=unroll(actual,Cursor(path));passes.append(path)
        if len(passes)>256:raise ValueError('matrix projection exceeded bounded M specialization')
    return actual,dict(hygiene_bindings=hygiene.bindings,unroll_paths=tuple(passes))


def _mapped_kernel(schedule,members,m,k,n,physical):
    """Invert every projected transfer to the actual checked virtual transfer."""
    schedule.revalidate()
    count=len(members);total=count*m
    expected=[('lhs',(total,k),'i8','read'),('rhs',(k,n),'i8','read'),
              ('output',(total,n),'i32','write')]
    if [(a.name,a.shape,a.dtype,a.access) for a in schedule.kernel.args] != expected:
        raise ValueError('matrix bundle requires its exact virtual source ABI')
    if schedule.realization['schedule_ir']['pipeline_kind'] != 'retained_full_k':
        raise ValueError('matrix bundle requires checked full-K retention')
    admission=schedule.realization['admission']
    limits=(admission['max_issued_loads'],admission['pending_loads'],physical['finite_load_entries'])
    if any(type(v) is not int or v<1 for v in limits):
        raise ValueError('matrix bundle lacks positive selected load bounds')
    limit=min(limits)
    specialized,passes=_specialize_rows(schedule.kernel)
    transfers=[]
    def walk(items):
        mapped=[];inverse=[]
        for item in items:
            if type(item) is Loop:
                if item.iter_args or item.yields or item.results:
                    raise ValueError('matrix bundle cannot rewrite carried completion events')
                body,restored=walk(item.body)
                mapped.append(replace(item,body=body));inverse.append(replace(item,body=restored));continue
            if type(item) is not Call or any((item.unit,item.produces,item.awaits,item.stages,item.sets,item.assumes)):
                raise ValueError('matrix bundle requires checked synchronous primitive effects')
            if item.instr in ('gemmini_mvin_lhs','gemmini_mvout'):
                a=dict(item.args);row=evaluate(a['dram_row'],{});rows=evaluate(a['rows'],{})
                if (not 0<=row<row+rows<=total or not 1<=rows<=isa.DIM):
                    raise ValueError('matrix bundle transfer outside original complete row range')
                load=item.instr=='gemmini_mvin_lhs';address='addr' if load else 'acc'
                parts=[];cursor=row
                while cursor<row+rows:
                    member=cursor//m;local=cursor-member*m
                    length=min(row+rows-cursor,m-local)
                    args=dict(a,dram_row=Const(local),rows=Const(length))
                    args[address]=a[address]+(cursor-row)
                    part=replace(item,instr=('row_bundle_lhs_' if load else 'row_bundle_out_')+str(member),args=tuple(args.items()))
                    # Check actual emitted operands, including the reverse view.
                    q=dict(part.args)
                    if (part.instr != ('row_bundle_lhs_' if load else 'row_bundle_out_')+str(member)
                            or q['dram_row'].value+member*m != cursor
                            or q['rows'].value != length or q[address] != a[address]+(cursor-row)
                            or not 0<=local<local+length<=m
                            or any(q[key]!=value for key,value in a.items() if key not in ('dram_row','rows',address))):
                        raise ValueError('matrix bundle cannot invert original pointer projection')
                    parts.append(part);cursor+=length
                if cursor != row+rows:raise ValueError('matrix bundle lost original source rows')
                # The checked source's load epoch can have earlier pending
                # transfers. Complete them before introducing a split epoch.
                if load:mapped.append(COMPLETE)
                for index,part in enumerate(parts):
                    mapped.append(part)
                    if not load or (index+1)%limit==0 or index+1==len(parts):mapped.append(COMPLETE)
                inverse.append(item)
                transfers.append(dict(kind='lhs' if load else 'output',virtual_row=row,rows=rows,
                                      parts=[dict(member=int(p.instr.rsplit('_',1)[1]),
                                            row=dict(p.args)['dram_row'].value,rows=dict(p.args)['rows'].value) for p in parts]))
            elif item.instr in (DRAIN,'gemmini_mvin_weight','gemmini_preload','gemmini_compute_preloaded'):
                if item.instr==DRAIN and item.args:raise ValueError('unknown completion payload')
                mapped.append(item);inverse.append(item)
            else:raise ValueError('matrix bundle only accepts zero-seeded raw products')
        return tuple(mapped),tuple(inverse)
    body,inverse=walk(specialized.body)
    if inverse!=specialized.body:raise ValueError('matrix bundle changed checked complete MAC/source projection')
    args=tuple(TensorArg('lhs_'+str(i),(m,k),'i8','read') for i in range(count))
    args+=(TensorArg('rhs',(k,n),'i8','read'),)
    args+=tuple(TensorArg('output_'+str(i),(m,n),'i32','write') for i in range(count))
    mapped=replace(specialized,name='rhs_matrix_bundle',args=args,body=body)
    # Reuse the same store-only ownership proof as the row-bundle epochs.
    actual,regions=_readout_epochs(mapped,count)
    if _strip_completion(actual.body)!=_strip_completion(mapped.body):
        raise ValueError('matrix store epochs changed data calls or pointer bindings')
    if check_structure(actual):raise ValueError('invalid matrix bundle index scope')
    source=BatchedDenseProblem((count,),BatchMatrixView((count,m,k),(0,)),BatchMatrixView((k,n),()))
    original=counted_batched_dense_products(source,name='original_matrix_products',instruction='original_raw_product')
    proof=dict(schema='authored_rhs_matrix_bundle_projection.v1',original_products=source.record(),
        original_product_index_sha256=original.digest(),original_member_bindings_sha256=_digest(members),
        pointer_bindings=[dict(member=i,lhs=row['lhs'],rhs=row['rhs'],output=row['dst'],source_op_index=row.get('attributes',{}).get('merlin.source_op_index')) for i,row in enumerate(members)],
        virtual_dense_kernel_sha256=schedule.kernel.digest(),specialized_kernel_sha256=specialized.digest(),
        mapped_kernel_sha256=actual.digest(),ordered_unroll_paths=passes,
        pass_order=('checked_retained_schedule','public_binder_hygiene','ordered_M_unroll','exact_member_boundary_projection','uniform_store_epochs'),
        complete_inverse_projection=True,transfers=transfers,store_regions=regions,
        max_split_load_issued=limit,physical_admission=dict(physical,store_configuration_snapshot='not assumed; one immutable raw CONFIG_ST per invocation'),
        source_view='virtual LHS[i*M+r,k]=original lhs_i[r,k]; original output_i[r,n]=virtual output[i*M+r,n]',
        numeric_law='each row retains signed i8 products and original full K order, modulo i32',
        effects='original disjoint SPAD/ACC rows partitioned without gaps or overlap; shared immutable RHS; fresh independent outputs',
        completion='original compute epochs retained; bounded split-load epochs drain before consumption; proven pure-store epochs drain before reuse')
    return actual,proof


@dataclass(frozen=True)
class _Bundle:
    members: tuple
    shape: tuple
    schedule: object
    kernel: object
    receipt: dict

    def revalidate(self):
        kernel,proof=_mapped_kernel(self.schedule,self.members,*self.shape,_admission())
        if kernel!=self.kernel or proof!=self.receipt:
            raise ValueError('matrix bundle source, pointer projection or selected contracts changed')


def _prepare(ctx,group):
    admitted=_source(ctx,group)
    if admitted is None:return None
    members,(m,k,n)=admitted
    schedule,refusal=_retained_schedule(len(members)*m,k,n,False,False,False,'wide','half')
    if schedule is None:return None
    kernel,receipt=_mapped_kernel(schedule,members,m,k,n,_admission())
    return _Bundle(members,(m,k,n),schedule,kernel,receipt)


def lower_matrix_bundle(ctx,group):
    """Lower an uninterrupted independent group, or refuse before mutation.

    The caller supplies exact source-identity RHS buckets and owns their whole
    device interval. Admission is bounded to 2..16 M>1 raw products/256 rows.
    """
    try:
        bundle=_prepare(ctx,group)
        if bundle is None:return None
        bundle.revalidate()
    except (ValueError,KeyError,TypeError,OverflowError):return None
    m,k,n=bundle.shape;fb=ctx.fb
    ctx.fence()
    ctx.current_source_index=bundle.members[0].get('attributes',{}).get('merlin.source_op_index')
    ctx.emit(*isa.config_ex(dataflow=1));ctx.emit(*isa.config_ld(k,channel=0))
    ctx.emit(*isa.config_ld(n,channel=1));ctx.emit(*isa.config_st(n*4))
    descriptor=[None]
    def emit(instruction,arguments):
        a={key:bridge.value(value) for key,value in arguments.items()}
        if instruction==DRAIN:ctx.fence()
        elif instruction.startswith('row_bundle_lhs_'):
            row=bundle.members[int(instruction.removeprefix('row_bundle_lhs_'))]
            ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index')
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(k)),a['dram_col'])
            ctx.emit(*isa.mvin(ctx.ptr(row['lhs'],offset),a['addr'],a['rows'],a['cols'],channel=0,fb=fb))
        elif instruction=='gemmini_mvin_weight':
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(n)),a['dram_col'])
            ctx.emit(*isa.mvin(ctx.ptr(bundle.members[0]['rhs'],offset),a['addr'],a['rows'],a['cols'],channel=1,fb=fb))
        elif instruction=='gemmini_preload':
            descriptor[0]=a['cols']
            acc=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
        elif instruction=='gemmini_compute_preloaded':
            if descriptor[0] is None:raise ValueError('matrix bundle compute lacks descriptor')
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,isa.GARBAGE,a['depth'],descriptor[0]))
        elif instruction.startswith('row_bundle_out_'):
            row=bundle.members[int(instruction.removeprefix('row_bundle_out_'))]
            ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index')
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(n)),a['dram_col'])
            ctx.emit(*isa.mvout(ctx.ptr(row['dst'],offset),isa.acc_addr(a['acc'],full=True,fb=fb),a['rows'],a['cols'],fb=fb))
        else:raise ValueError('unknown matrix bundle terminal instruction '+instruction)
    bridge=FnBuilderIndexAdapter(fb)
    emission=emit_index_program(bundle.kernel,bridge,emit)
    ctx.fence()
    receipt=dict(mechanism='shared_rhs_matrix_bundle',products=len(bundle.members),original_shape=[m,k,n],virtual_shape=[len(bundle.members)*m,k,n],
        bundles=[bundle.receipt],emitted_kernel_sha256=[bundle.kernel.digest()],retained_plan=dict(bundle.schedule.realization),emission=emission,
        qualification='source-checked exact pointer partition; independent execution required')
    ctx.metadata.setdefault('rhs_matrix_bundles',[]).append(receipt)
    return receipt
