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


def _checked_pipeline(*args, **kwargs):
    """Unavailable optional proofs retain the caller's checked compact path.

    Counterexamples and malformed admissions still propagate. Solver UNKNOWN
    or a finite wall-clock deadline does not authorize the proposed schedule.
    """
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    # This optional API's retained/segmented epoch verifier imports z3 directly.
    # Normalize only absence of that selected proof backend, before proposing
    # any emission. Other import failures and concrete proof failures propagate.
    try:
        import z3  # selected proof backend; no schedule is admitted here
    except ModuleNotFoundError as error:
        if error.name != 'z3':
            raise
        return None, 'optional dense pipeline verification unavailable: selected SMT backend z3 is unavailable'
    try:
        return schedule_dense_pipeline(*args, **kwargs), None
    except DenseVerificationUnavailable as error:
        return None, 'optional dense pipeline verification unavailable: '+str(error)


PANEL_GEOMETRIES={'balanced':(4,4),'wide':(8,8),'tall':(8,4),'wide_n':(4,8),
                  'wide_16':(4,16),'tall_16':(16,4)}


def retained_schedule(m,k,n,bias,column_scales=False,segmented=False,*,geometry=None,segment_k_tiles=16):
    """Resolve a panel resource bound before entering the geometry-keyed cache."""
    configured=knob('DENSE_PANEL_GEOMETRY',default='balanced',values=tuple(PANEL_GEOMETRIES),
                    root=Path(__file__).resolve().parents[1])
    selection=geometry if geometry is not None else os.environ.get('DENSE_PANEL_GEOMETRY',configured)
    if selection not in PANEL_GEOMETRIES:
        raise ValueError('DENSE_PANEL_GEOMETRY must be one of '+', '.join(PANEL_GEOMETRIES))
    split=knob('DENSE_STORAGE_SPLIT',default='half',values=('half','adaptive'),
               root=Path(__file__).resolve().parents[1])
    if segment_k_tiles != 'capacity' and (type(segment_k_tiles) is not int or segment_k_tiles < 1):
        raise ValueError('segment_k_tiles must be positive or capacity')
    return _retained_schedule(m,k,n,bias,column_scales,segmented,selection,split,segment_k_tiles)


@lru_cache(maxsize=256)
def _retained_schedule(m,k,n,bias,column_scales,segmented,geometry,storage_split='half',segment_k_tiles=16):
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
        # The incumbent keeps its 16-tile segment. The capacity alternative
        # lets the shared planner derive the largest physically fitting RHS
        # pair for each M/N block; it still keeps more than one K segment.
        cap=p.tiles[1]-1 if segment_k_tiles=='capacity' else min(segment_k_tiles,p.tiles[1]-1)
        if cap<1:
            return None,'segmented reduction requires more than one K tile'
        r=replace(r,max_k_blocks=cap)
        candidates=dense_segmented_blocks(p,r,s)
    else:
        candidates=[b for b in dense_blocks(p,r,k_limit=lambda m,n:s.k_limit(p,r,m,n)) if b.k==p.tiles[1] and s.admits(p,b,r)]
    if not candidates:
        return None,RETAINED_CAPACITY_REFUSAL
    block=min(candidates,key=lambda b:(b.loads,-b.m*b.n,b.m,b.n))
    if segmented and segment_k_tiles=='capacity':
        # Seal the physical bound into the resource view consumed by the plan
        # and typed panel receipt. Both must describe the actual selected K
        # domain, rather than the larger exploratory candidate bound.
        r=replace(r,max_k_blocks=block.k)
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
        plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,bias_source=bias_source)
        result,refusal=_checked_pipeline(p,r,s,a,name='dense_segmented',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=combined,bias_source=bias_source,readout_mode='separate',static_budget=128,timeout_ms=SEGMENTED_VERIFICATION_TIMEOUT_MS)
        if result is None:
            return None,refusal
        _record_geometry(result,geometry,block,p,r,s)
        result.realization['verification_timeout_ms']=SEGMENTED_VERIFICATION_TIMEOUT_MS
        result.realization['segmented_epoch_bounds']={
            'segment_k_selection':segment_k_tiles,
            'retained_lhs_loads':block.m*p.tiles[1],
            'rhs_slot_loads':block.k*block.n,
            'retained_lhs_rows':block.m*p.tiles[1]*dim,
            'rhs_slot_rows':block.k*block.n*dim,'rhs_slots':2,
            'max_issued_loads':law.max_issued_loads,
            'max_issued_compute_pairs':law.max_issued_compute_pairs,
            'pending_loads':law.pending_loads,'pending_compute_pairs':law.pending_compute_pairs}
        return result,None
    raw=emit_compact_dense(p,block,s,a,name='dense_retained',dtypes=('i8','i8','i32'),bias_source=bias_source)
    transforms=dense_retained_operand_hoists(raw,a,role='lhs')
    transforms+=dense_retained_compute_order(propose_loop_transforms(raw,transforms),a,order=('k','n','m'))
    source=hashlib.sha256(json.dumps({'m':m,'k':k,'n':n,'bias':bias,'dtype':['i8','i8','i32']},sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),loop_transforms=transforms,pipeline_kind='retained_full_k',pipeline_prefetch=False,bias_source=bias_source)
    result,refusal=_checked_pipeline(p,r,s,a,name='dense_retained',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,bias_source=bias_source,readout_mode='separate',static_budget=128)
    if result is None:
        return None,refusal
    _record_geometry(result,geometry,block,p,r,s)
    result.realization['storage_partition']=partition_receipt or {'selection':'half',
        'lhs_window_rows':s.lhs.capacity,'rhs_window_rows':s.rhs.capacity}
    return result,None


def _record_geometry(result,geometry,block,problem,resources,storage):
    from .dense_panel_predicates import describe_dense_panel
    # Consume the exact selected model, rather than reconstructing a candidate
    # from diagnostic receipts. Admission failures stop before LLVM emission.
    facts=describe_dense_panel(problem,resources,storage,block)
    result.realization['panel_geometry']={
        'selection':geometry,'max_m_blocks':PANEL_GEOMETRIES[geometry][0],
        'max_n_blocks':PANEL_GEOMETRIES[geometry][1],
        'selected_block_tiles':[block.m,block.k,block.n],
        'accumulator_rows':facts.accumulator.live,
        'accumulator_capacity_rows':facts.accumulator.assigned,
        'lhs_window_rows':facts.lhs.assigned,'rhs_window_rows':facts.rhs.assigned}
    # Keep JSON receipts detached from the immutable typed admission facts.
    result.realization['panel_facts']=facts.as_dict()


# Retain the cache inspection API for callers comparing explicit alternatives.
retained_schedule.cache_clear=_retained_schedule.cache_clear
retained_schedule.cache_info=_retained_schedule.cache_info
