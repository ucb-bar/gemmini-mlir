"""Capacity-bounded retained-LHS alternatives with explicit completion epochs.

The compact incumbent and these alternatives share their operation adapter and
terminal encoder. Execute completion order comes from the selected public RTL
ordering facts. Full drains protect every operand ownership change; no generic
RAW/WAR scoreboard premise or latency spacing replaces completion.
"""
from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import hashlib,json,os
from pathlib import Path
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,DenseBiasSource,emit_compact_dense
from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission,dense_retained_operand_hoists,dense_retained_compute_order
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
from merlin.compile.tuning_knobs import knob
from . import isa

RETAINED_CAPACITY_REFUSAL='full-K retained panel exceeds assigned operand storage'
# Finite wall-clock budget for the unchanged segmented source/epoch proof.
# Ragged N groups introduce additional quantified residency/continuation cases;
# the connected M1/K2048/N5728 plan takes about 16s to construct and verify.
# This changes neither the program nor its original correctness obligations.
SEGMENTED_VERIFICATION_TIMEOUT_MS=30000


PANEL_GEOMETRIES={'balanced':(4,4),'wide':(8,8),'tall':(8,4),'wide_n':(4,8),
                  'wide_16':(4,16),'tall_16':(16,4)}


def retained_schedule(m,k,n,bias,column_scales=False,segmented=False,*,geometry=None,frontier=None,retained_operand=None,segment_k_limit=None,segment_partition=None):
    """Resolve a panel resource bound before entering the geometry-keyed cache."""
    configured=knob('DENSE_PANEL_GEOMETRY',default='balanced',values=tuple(PANEL_GEOMETRIES),
                    root=Path(__file__).resolve().parents[1])
    selection=geometry if geometry is not None else os.environ.get('DENSE_PANEL_GEOMETRY',configured)
    if selection not in PANEL_GEOMETRIES:
        raise ValueError('DENSE_PANEL_GEOMETRY must be one of '+', '.join(PANEL_GEOMETRIES))
    if segmented:
        from .dense_segment_options import resolve_segment_geometry
        selection=resolve_segment_geometry(selection)
    split=knob('DENSE_STORAGE_SPLIT',default='half',values=('half','adaptive'),
               root=Path(__file__).resolve().parents[1])
    configured_frontier=knob('DENSE_SEGMENTED_FRONTIER',default='off',
        values=('off','k_tiles','output_groups'),root=Path(__file__).resolve().parents[1])
    frontier=frontier if frontier is not None else os.environ.get('DENSE_SEGMENTED_FRONTIER',configured_frontier)
    if frontier not in ('off','k_tiles','output_groups'):
        raise ValueError('invalid dense segmented frontier')
    configured_limit=knob('DENSE_SEGMENT_K_LIMIT',default='16',values=('8','16','32','64'),
        root=Path(__file__).resolve().parents[1])
    segment_k_limit=segment_k_limit if segment_k_limit is not None else int(os.environ.get('DENSE_SEGMENT_K_LIMIT',configured_limit))
    if type(segment_k_limit) is not int or segment_k_limit not in (8,16,32,64):
        raise ValueError('DENSE_SEGMENT_K_LIMIT must be 8, 16, 32 or 64')
    configured_operand=knob('DENSE_RETAINED_OPERAND',default='lhs',values=('lhs','weight','capacity'),
        root=Path(__file__).resolve().parents[1])
    retained_operand=retained_operand if retained_operand is not None else os.environ.get('DENSE_RETAINED_OPERAND',configured_operand)
    if retained_operand not in ('lhs','weight','capacity'):
        raise ValueError('DENSE_RETAINED_OPERAND must be lhs, weight or capacity')
    if retained_operand=='capacity' and not segmented:
        from .dense_retention_policy import execute_capacity_retention
        return execute_capacity_retention(m,k,n,bias,column_scales,geometry=selection,storage_split=split)
    from .dense_segment_options import resolve_segment_partition
    partition=resolve_segment_partition(segment_partition) if segmented else 'legacy'
    return _retained_schedule(m,k,n,bias,column_scales,segmented,selection,split,frontier,retained_operand,segment_k_limit,partition)


def _retained_schedule(m,k,n,bias,column_scales,segmented,geometry,storage_split='half',frontier='off',retained_operand='lhs',segment_k_limit=16,segment_partition='legacy',rhs_residency=None):
    from .dense_segment_residency import resolve_rhs_residency
    option=resolve_rhs_residency(rhs_residency) if segmented else 'off'
    return _retained_schedule_cached(m,k,n,bias,column_scales,segmented,geometry,
        storage_split,frontier,retained_operand,segment_k_limit,segment_partition,option)


@lru_cache(maxsize=256)
def _retained_schedule_cached(m,k,n,bias,column_scales,segmented,geometry,storage_split='half',frontier='off',retained_operand='lhs',segment_k_limit=16,segment_partition='legacy',rhs_residency='off'):
    from .dense_segment_options import SEGMENT_PARTITIONS,segment_k_cap
    if segment_partition not in SEGMENT_PARTITIONS:
        raise ValueError('invalid dense segment partition')
    if type(segment_k_limit) is not int or segment_k_limit not in (8,16,32,64):
        raise ValueError('dense segment K limit must be one of 8,16,32,64')
    from .dense import _adapter
    a=_adapter()
    dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,1 if column_scales else dim,1,1,4,bias=bias)
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,isa.SPAD_ROWS//2),DenseMemoryView('rhs','spad',dim,isa.SPAD_ROWS//2,isa.SPAD_ROWS//2),DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS))
    max_m,max_n=PANEL_GEOMETRIES[geometry]
    r=DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*4*dim,max_m_blocks=max_m,max_n_blocks=max_n,max_k_blocks=p.tiles[1],partial_output_tiles=True,partial_output_blocks=True)
    partition_receipt=None
    if storage_split=='adaptive' and not segmented:
        from .dense_storage_partition import select_full_k_partition
        partition=select_full_k_partition(p,r,max_m_blocks=max_m,max_n_blocks=max_n)
        if partition is None:
            return None,RETAINED_CAPACITY_REFUSAL
        s,adaptive_block,partition_receipt=partition
        candidates=[adaptive_block]
    elif segmented:
        # Retain complete LHS while double buffering bounded RHS segments.
        # This is a separate experimental family, never an incumbent default.
        from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission,dense_segmented_blocks
        from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission
        cap=segment_k_cap(p.tiles[1],segment_k_limit,segment_partition,frontier)
        if cap<1:
            return None,'segmented reduction requires more than one K tile'
        r=replace(r,max_k_blocks=cap)
        candidates=dense_segmented_blocks(p,r,s)
        if frontier=='output_groups':
            candidates=[b for b in candidates if b.k==p.tiles[1] and b.n<p.tiles[2]]
            if not candidates:
                return None,'two complete RHS output groups do not fit assigned storage'
    else:
        candidates=[b for b in dense_blocks(p,r,k_limit=lambda m,n:s.k_limit(p,r,m,n)) if b.k==p.tiles[1] and s.admits(p,b,r)]
    if not candidates:
        return None,RETAINED_CAPACITY_REFUSAL
    block=min(candidates,key=lambda b:(b.loads,-b.m*b.n,b.m,b.n))
    retained_choice=None
    if not segmented and retained_operand=='weight':
        from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_choices
        choices=dense_retained_choices(p,r,s,roles=('weight',))
        # Storage partition selection remains the authority for its exact
        # assigned windows. The alternative changes only source residency.
        if storage_split=='adaptive':
            choices=tuple(c for c in choices if c.block==block)
        if not choices:
            return None,RETAINED_CAPACITY_REFUSAL
        retained_choice=choices[0]
        block=retained_choice.block
    evidence=json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order']!='in_order':
        return None,'selected execute completion order is unknown'
    # Pending capacities are selected structural queue depths (the command
    # queue holds 8, paired execution control holds at least two instructions).
    issued_loads=max(1,(block.m+block.n)*block.k)
    if segmented:
        # The first pure-load epoch fills the COMPLETE resident LHS panel;
        # subsequent RHS epochs fill one of two disjoint bounded K slots.
        # Issued counts bound the finite counted program, not simultaneous
        # pending commands. Preserve the selected 2/8 ready/valid capacities.
        issued_loads=max(issued_loads,block.m*p.tiles[1])
    law=DenseRetainedReductionAdmission(a.configuration_identity,a.numerical_policy_identity,evidence['identity'],max(1,block.m*block.k*block.n),issued_loads,2,8)
    bias_source=DenseBiasSource(n,'i32') if bias else None
    if segmented:
        overlap=DensePanelOverlapAdmission(a.configuration_identity,a.numerical_policy_identity,evidence['identity']+':disjoint-owned-operand-windows',law.max_issued_compute_pairs,law.max_issued_loads,2,8)
        combined=DenseSegmentedAdmission(overlap,law,evidence['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
        source=hashlib.sha256(json.dumps([m,k,n,bias,'i8-i8-i32'],sort_keys=True).encode()).hexdigest()
        plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,bias_source=bias_source,
            pipeline_issue_order={'off':'compute_then_prefetch',
                'k_tiles':'prefetch_rhs_between_k_tiles',
                'output_groups':'prefetch_rhs_between_output_groups'}[frontier])
        result=schedule_dense_pipeline(p,r,s,a,name='dense_segmented',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=combined,bias_source=bias_source,readout_mode='separate',static_budget=128,timeout_ms=SEGMENTED_VERIFICATION_TIMEOUT_MS)
        _record_geometry(result,geometry,block,p,r,s,segmented=True)
        result.realization['verification_timeout_ms']=SEGMENTED_VERIFICATION_TIMEOUT_MS
        result.realization['requested_frontier']=frontier
        result.realization['segment_k_limit']=segment_k_limit
        result.realization['segment_partition']=segment_partition
        result.realization['effective_segment_k_cap']=cap
        result.realization['segmented_epoch_bounds']={
            'retained_lhs_loads':block.m*p.tiles[1],
            'rhs_slot_loads':block.k*block.n,
            'retained_lhs_rows':block.m*p.tiles[1]*dim,
            'rhs_slot_rows':block.k*block.n*dim,'rhs_slots':2,
            'max_issued_loads':law.max_issued_loads,
            'max_issued_compute_pairs':law.max_issued_compute_pairs,
            'pending_loads':law.pending_loads,'pending_compute_pairs':law.pending_compute_pairs}
        if rhs_residency=='boundary':
            from .dense_segment_residency import select_boundary_rhs
            result=select_boundary_rhs(result,p,block,r,s,a,timeout_ms=SEGMENTED_VERIFICATION_TIMEOUT_MS)
        return result,None
    outer_order=('m','n') if retained_operand=='lhs' else ('n','m')
    raw=emit_compact_dense(p,block,s,a,name='dense_retained',dtypes=('i8','i8','i32'),bias_source=bias_source,outer_order=outer_order)
    transforms=dense_retained_operand_hoists(raw,a,role=retained_operand)
    transforms+=dense_retained_compute_order(propose_loop_transforms(raw,transforms),a,order=('k','n','m'))
    source=hashlib.sha256(json.dumps({'m':m,'k':k,'n':n,'bias':bias,'dtype':['i8','i8','i32']},sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),outer_order=outer_order,loop_transforms=transforms,pipeline_kind='retained_full_k',pipeline_prefetch=False,bias_source=bias_source)
    result=schedule_dense_pipeline(p,r,s,a,name='dense_retained',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,bias_source=bias_source,readout_mode='separate',static_budget=128)
    _record_geometry(result,geometry,block,p,r,s)
    result.realization['storage_partition']=partition_receipt or {'selection':'half',
        'lhs_window_rows':s.lhs.capacity,'rhs_window_rows':s.rhs.capacity}
    result.realization['retained_operand']=retained_operand
    if retained_choice is not None:
        actual_loads=_retained_load_counts(result.kernel,a)
        if actual_loads != (retained_choice.lhs_loads,retained_choice.rhs_loads):
            raise ValueError('retained weight proposal disagrees with actual counted operand loads')
        result.realization['retained_choice']={'outer_order':list(outer_order),
            'lhs_loads':retained_choice.lhs_loads,'rhs_loads':retained_choice.rhs_loads,
            'actual_lhs_loads':actual_loads[0],'actual_rhs_loads':actual_loads[1],
            'requested_operand_bytes':retained_choice.requested_operand_bytes}
    return result,None


def _retained_load_counts(kernel,adapter):
    """Count actual projected transfers without expanding workload loops."""
    from merlin.sched.ir import Loop
    from merlin.compile.scheduling.block_schedule import Load
    loads={'lhs':0,'weight':0}
    def walk(body,multiplicity):
        for statement in body:
            if type(statement) is Loop:
                walk(statement.body,multiplicity*statement.extent)
            elif statement.instr!='merlin_dense_panel_drain':
                operation=adapter.project(statement)
                if type(operation) is Load:
                    loads[operation.role]+=multiplicity
    walk(kernel.body,1)
    return loads['lhs'],loads['weight']


def _record_geometry(result,geometry,block,problem,resources,storage,*,segmented=False):
    from .dense_panel_predicates import describe_dense_panel
    # Consume the exact selected model, rather than reconstructing a candidate
    # from diagnostic receipts. Admission failures stop before LLVM emission.
    facts=describe_dense_panel(problem,resources,storage,block,segmented=segmented)
    result.realization['panel_geometry']={
        'selection':geometry,'max_m_blocks':PANEL_GEOMETRIES[geometry][0],
        'max_n_blocks':PANEL_GEOMETRIES[geometry][1],
        'selected_block_tiles':[block.m,block.k,block.n],
        'accumulator_rows':facts.accumulator.live,
        'accumulator_capacity_rows':facts.accumulator.assigned,
        'lhs_window_rows':facts.lhs.assigned,'rhs_window_rows':facts.rhs.assigned}
    # Keep JSON receipts detached from the immutable typed admission facts.
    result.realization['panel_facts']=facts.as_dict()


_retained_schedule.cache_clear=_retained_schedule_cached.cache_clear
_retained_schedule.cache_info=_retained_schedule_cached.cache_info


# Retain the cache inspection API for callers comparing explicit alternatives.
retained_schedule.cache_clear=_retained_schedule.cache_clear
retained_schedule.cache_info=_retained_schedule.cache_info
