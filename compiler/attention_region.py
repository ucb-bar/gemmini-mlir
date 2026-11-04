"""Exact common-RHS head rows through the existing checked dense scheduler.

Separate source tensors remain separate ABI allocations. Only target DMA rows
are scattered/gathered; the shared dense schedule owns every integer product,
segmented reduction, accumulator and completion boundary. No host concatenation
or arithmetic is introduced. Selection requires the exact source proof below.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from xdsl.ir import Block
from xdsl.dialects.builtin import StringAttr,ArrayAttr

ROW_FUSION_ENABLED = True
KV_RESIDENCY_ENABLED = True
KV_READY_FRONTIER_ENABLED = True
KV_KNM_ORDER_ENABLED = False


def enabled():
    return ROW_FUSION_ENABLED or KV_RESIDENCY_ENABLED


@dataclass(frozen=True)
class HeadRowGroup:
    members: tuple
    source_indices: tuple[int, ...]
    rhs: object
    k: int
    n: int
    m: int = 1
    source_sha256: str = ''


def find_head_row_groups(module, *, excluded=(), source_sha256=None):
    """Read library-proved integer dots without changing original source IR.

    Read operands must be original function arguments; every result must remain
    independently returned. Unknown effects, nonzero seeds, alternate scalar
    recurrences, source views and early result observers keep their old route.
    """
    from dense import _capacities
    from merlin.transforms.dense_row_batching_check import (
        geometry, supported_scope, safe_span, RowBatchingUnavailable)
    functions=[op for op in module.body.block.ops if op.name=='func.func']
    if len(functions)!=1 or len(functions[0].regions[0].blocks)!=1:return ()
    block=functions[0].regions[0].block
    operations=tuple(op for op in block.ops if op.name!='func.return')
    # Existing compiler logical names only label the unchanged ABI. Validate
    # their complete arities, then omit those labels on the scope-analysis clone.
    function=functions[0];scope_module=module.clone()
    scope_function=next(op for op in scope_module.body.block.ops if op.name=='func.func')
    for key,arity in (('merlin.logical_inputs',len(block.args)),
                      ('merlin.logical_outputs',len(block.last_op.operands))):
        value=function.attributes.get(key)
        if value is not None:
            if (not isinstance(value,ArrayAttr) or len(value)!=arity
                or any(not isinstance(v,StringAttr) or not v.data for v in value)):
                return ()
            scope_function.attributes.pop(key)
    roles=function.attributes.get('merlin.logical_input_roles')
    if roles is not None:
        # The public logical ABI enrichment defines exactly these ownership
        # labels. They add no arithmetic, alias or mutability permission.
        if (not isinstance(roles,ArrayAttr) or len(roles)!=len(block.args)
            or any(not isinstance(v,StringAttr) or v.data not in
                   ('input','weight','bias','scale') for v in roles)):
            return ()
        scope_function.attributes.pop('merlin.logical_input_roles')
    logical=function.attributes.get('merlin.logical_io')
    if logical is not None:
        if not isinstance(logical,StringAttr) or logical.data!='merlin.logical_io.v1':return ()
        scope_function.attributes.pop('merlin.logical_io')
    scope_operations=tuple(op for op in scope_function.regions[0].block.ops if op.name!='func.return')
    excluded=set(excluded);candidates=[];proven_fills=set()
    for index,op in enumerate(operations):
        if op.name!='linalg.generic' or not supported_scope(scope_operations[index]):continue
        try:dimensions,dtypes=geometry(op,index_bits=64)
        except RowBatchingUnavailable:continue
        if dtypes!=('i8','i8','i32'):continue
        # The exact library scalar reader also proves these initializer fills.
        if op.operands[2].owner.name=='linalg.fill':proven_fills.add(op.operands[2].owner)
        m=dimensions[0]
        if (op in excluded
            or (m==1 and not ROW_FUSION_ENABLED)
            or (m!=1 and (not KV_RESIDENCY_ENABLED
                or type(source_sha256) is not str or len(source_sha256)!=64
                or any(c not in '0123456789abcdef' for c in source_sha256)))
            or any(not isinstance(v.owner,Block) or v.owner is not block for v in op.operands[:2])
            or any(use.operation.name!='func.return' for use in op.results[0].uses)):
            continue
        candidates.append((index,op,dimensions))
    partitions={}
    for index,op,(m,k,n) in candidates:
        partitions.setdefault((op.operands[1],m,k,n),[]).append((index,op))
    limit=_capacities()[1];groups=[]
    for (rhs,m,k,n),rows in partitions.items():
        member_limit=limit if m==1 else 128
        # Preserve source C-order; target rows never replace the logical heads.
        for first in range(0,len(rows),member_limit):
            members=rows[first:first+member_limit]
            if len(members)<2:continue
            lo,hi=members[0][0],members[-1][0]
            span=[op for op in operations[lo:hi+1] if op not in proven_fills]
            if not safe_span(span):continue
            groups.append(HeadRowGroup(tuple(op for _,op in members),
                tuple(index for index,_ in members),rhs,k,n,m,
                source_sha256 if m!=1 else ''))
    return tuple(sorted(groups,key=lambda group:group.source_indices[0]))


def _checked_dense_rows(rows,k,n):
    """Consume the actual dense plan; reject any partial logical row domain."""
    from dense import plan_dense
    from merlin.sched.ir.kernel import instances,concretize
    problem,schedule=plan_dense(rows,k,n,alternative='epochs',panel_shape='balanced')
    schedule.revalidate()
    calls=0;loads=stores=0
    for call,env in instances(schedule.kernel):
        calls+=1
        if calls>200000:raise ValueError('head-row schedule exceeds bounded projection budget')
        f=concretize(call,env)
        if call.instr=='lhs_load':
            if f['src_row']!=0 or f['rows']!=rows or f.get('step',1)!=1:
                raise ValueError('head-row adapter requires complete contiguous logical input rows')
            loads+=1
        elif call.instr=='store':
            if f['row']!=0 or f['rows']!=rows:
                raise ValueError('head-row adapter requires complete logical output rows')
            stores+=1
        elif call.instr not in ('weight_load','preload','compute','merlin_dense_panel_drain'):
            raise ValueError('head-row adapter cannot ignore an unfamiliar dense effect')
    if not loads or not stores:raise ValueError('head-row dense schedule lost its source reads or outputs')
    return problem,schedule,dict(dynamic_dense_calls=calls,
        logical_lhs_loads=loads,logical_stores=stores,
        actual_lhs_loads=loads*rows,actual_stores=stores*rows)


class _SeparateRows:
    """Bind checked logical rows to original separate tensors using Encoder."""
    def __init__(self,enc,group,pointers):
        self.enc,self.group,self.pointers=enc,group,pointers
    def __getattr__(self,name):return getattr(self.enc,name)
    def dense_effect(self,instruction,fields,pointers,problem):
        if instruction not in ('lhs_load','store'):
            return self.enc.dense_effect(instruction,fields,pointers,problem)
        # _checked_dense_rows established row origin zero and full member count
        # over every actual schedule instance, before any target instruction.
        previous=getattr(self.enc.fn,'current_source_index',None)
        for row,(op,index) in enumerate(zip(self.group.members,self.group.source_indices,strict=True)):
            self.enc.fn.current_source_index=index
            f=dict(fields)
            if instruction=='lhs_load':
                f.update(src_row=0,rows=1,dst=self.enc.add(f['dst'],row))
                p={**pointers,'lhs':self.pointers[op.operands[0]]}
            else:
                f.update(row=0,rows=1,acc=self.enc.add(f['acc'],row))
                p={**pointers,'out':self.pointers[op.results[0]]}
            self.enc.dense_effect(instruction,f,p,problem)
        self.enc.fn.current_source_index=previous


def emit_head_row_group(enc,group,pointers):
    """Emit all original results, or return None before issuing any instruction.

    The caller supplies the ordinary function ABI's disjoint allocations, valid
    source buffers and no intervening device execution during this synchronous
    emitter. Source grouping is admitted only by find_head_row_groups above.
    """
    from dense import emit_dense,_capacities
    if (type(group) is not HeadRowGroup
        or not 2<=len(group.members)<=(_capacities()[1] if group.m==1 else 128)
        or any(v not in pointers for op in group.members for v in (*op.operands[:2],*op.results))):
        return None
    if group.m!=1:
        from resident_chain import emit_resident_products,plan_resident_products
        products=[dict(lhs=pointers[op.operands[0]],out=pointers[op.results[0]],
            m=group.m,k=group.k,n=group.n,out_dtype='i32',
            source_op_index=index)
            for op,index in zip(group.members,group.source_indices,strict=True)]
        try:
            planned=plan_resident_products(products,
                source_identity=group.source_sha256,external_buffers_disjoint=True,
                ready_frontier=KV_READY_FRONTIER_ENABLED and not KV_KNM_ORDER_ENABLED,
                compute_order=("k","n","m") if KV_KNM_ORDER_ENABLED else None)
        except ValueError:
            # The checked full-RHS plan must admit storage and every index
            # before emission. Library plan failures retain ordinary lowering.
            return None
        if planned is None:return None
        emitted=emit_resident_products(enc,pointers[group.rhs],products,
            source_identity=group.source_sha256,external_buffers_disjoint=True,
            ready_frontier=KV_READY_FRONTIER_ENABLED and not KV_KNM_ORDER_ENABLED,
                compute_order=("k","n","m") if KV_KNM_ORDER_ENABLED else None)
        if emitted is None:return None
        return {**emitted,'selected_schedule':'common_rhs_multirow_heads',
            'source_op_indices':list(group.source_indices),
            'm':group.m,'k':group.k,'n':group.n,
            'source_arithmetic':'library static_zero_integer_matmul_geometry',
            'binding':'independent original inputs and outputs; exact common immutable RHS SSA',
            'complete_source_outputs':len(group.members),
            'implementation_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    try:problem,schedule,projection=_checked_dense_rows(len(group.members),group.k,group.n)
    except ValueError:return None
    wrapped=_SeparateRows(enc,group,pointers)
    first=group.members[0]
    emitted=emit_dense(wrapped,pointers[first.operands[0]],pointers[group.rhs],
        pointers[first.results[0]],len(group.members),group.k,group.n,
        alternative='epochs',panel_shape='balanced')
    return dict(selected_schedule='common_rhs_head_rows',source_op_indices=list(group.source_indices),
        logical_rows=len(group.members),k=group.k,n=group.n,
        binding='independent original i8 inputs and i32 outputs; exact shared source RHS',
        source_arithmetic='library static_zero_integer_matmul_geometry',
        physical_rows='one DMA row per original allocation; no host packing',
        complete_source_outputs=len(group.members),projection=projection,
        schedule_kernel_sha256=schedule.kernel.digest(),dense_emission=emitted,
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
