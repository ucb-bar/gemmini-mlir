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
                  'wide_16':(4,16),'tall_16':(16,4),
                  'wide_32_frontier':(2,32),'wide_64_frontier':(1,64),
                  'resident_axis':(isa.ACC_ROWS//isa.DIM,isa.ACC_ROWS//isa.DIM)}
PANEL_SELECTIONS=(*PANEL_GEOMETRIES,'regional')


def _wide_loads(result, problem, block, resources, storage, adapter):
    """Consume the checked actual transfer program before terminal lowering."""
    selection=knob('DENSE_WIDE_LOADS',default='off',values=('off','on','regional'),
                   root=Path(__file__).resolve().parents[1])
    if selection=='off':
        return result
    if selection=='regional' and not (problem.m>=128 or problem.k>=512):
        # Complete modeled gains cover large advancing-row and deep-reduction
        # families. Matched original RTL checks show mixed small-call latency,
        # including regressions. Keep their tested terminal path; this is a
        # conservative evidence boundary, not a storage legality condition.
        return replace(result,realization=dict(result.realization,wide_loads={
            'selection':selection,'applied':False,
            'reason':'small-call incumbent retained outside supported wide-load classes'}))
    from .dense_wide_loads import select_dense_wide_loads
    facts=json.loads(Path(__file__).with_name('dense_load_facts.json').read_text())
    try:
        selected,receipt=select_dense_wide_loads(result,problem,block,resources,
            storage,adapter,facts,selection='on',timeout_ms=30000)
    except (ValueError,TimeoutError) as error:
        selected=result
        receipt={'selection':selection,'applied':False,
                 'refusal':type(error).__name__+': '+str(error)}
    # The frozen original owner remains untouched if a rewrite is refused.
    return replace(selected,realization=dict(selected.realization,
        wide_loads=dict(receipt,selection=selection)))


def retained_schedule(m,k,n,bias,column_scales=False,segmented=False,*,geometry=None):
    """Resolve a panel resource bound before entering the geometry-keyed cache."""
    configured=knob('DENSE_PANEL_GEOMETRY',default='balanced',values=PANEL_SELECTIONS,
                    root=Path(__file__).resolve().parents[1])
    selection=geometry if geometry is not None else os.environ.get('DENSE_PANEL_GEOMETRY',configured)
    if selection not in PANEL_SELECTIONS:
        raise ValueError('DENSE_PANEL_GEOMETRY must be one of '+', '.join(PANEL_SELECTIONS))
    split=knob('DENSE_STORAGE_SPLIT',default='half',values=('half','adaptive'),
               root=Path(__file__).resolve().parents[1])
    if selection=='regional':
        from .dense_regional_policy import regional_schedule
        return regional_schedule(m,k,n,bias,column_scales,segmented,split,
                                 retained_planner=_retained_schedule)
    return _retained_schedule(m,k,n,bias,column_scales,segmented,selection,split)


@lru_cache(maxsize=256)
def _retained_schedule(m,k,n,bias,column_scales,segmented,geometry,storage_split='half'):
    from .dense import _adapter
    a=_adapter()
    dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,1 if column_scales else dim,1,1,4,bias=bias)
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,isa.SPAD_ROWS//2),DenseMemoryView('rhs','spad',dim,isa.SPAD_ROWS//2,isa.SPAD_ROWS//2),DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS))
    max_m,max_n=PANEL_GEOMETRIES[geometry]
    r=DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*4*dim,max_m_blocks=max_m,max_n_blocks=max_n,max_k_blocks=p.tiles[1],partial_output_tiles=True,partial_output_blocks=True)
    partition_receipt=None
    if segmented:
        from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission,dense_segmented_blocks
        from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission
    frontier=geometry.endswith('_frontier') and segmented
    if frontier:
        # Registered experimental geometry: the full LHS owns only its actual
        # padded rows; both advancing RHS slots share the remaining capacity.
        from .dense_storage_partition import select_segmented_partition
        r=replace(r,max_k_blocks=max(1,p.tiles[1]-1))
        partition=select_segmented_partition(p,r)
        if partition is None:
            return None,RETAINED_CAPACITY_REFUSAL
        s,adaptive_block,partition_receipt=partition
        r=replace(r,max_k_blocks=adaptive_block.k)
        candidates=[adaptive_block]
    elif storage_split=='adaptive' and not segmented:
        from .dense_storage_partition import select_full_k_partition
        partition=select_full_k_partition(p,r,max_m_blocks=max_m,max_n_blocks=max_n)
        if partition is None:
            return None,RETAINED_CAPACITY_REFUSAL
        s,adaptive_block,partition_receipt=partition
        candidates=[adaptive_block]
    elif segmented:
        # Retain complete LHS while double buffering bounded RHS segments.
        # This is a separate experimental family, never an incumbent default.
        cap=min(16,p.tiles[1]-1)
        if cap<1:
            return None,'segmented reduction requires more than one K tile'
        r=replace(r,max_k_blocks=cap)
        candidates=dense_segmented_blocks(p,r,s)
    else:
        candidates=[b for b in dense_blocks(p,r,k_limit=lambda m,n:s.k_limit(p,r,m,n)) if b.k==p.tiles[1] and s.admits(p,b,r)]
    if not candidates:
        return None,RETAINED_CAPACITY_REFUSAL
    block=min(candidates,key=lambda b:(b.loads,-b.m*b.n,b.m,b.n))
    residency=None
    if geometry=='resident_axis' and not segmented:
        # Enumerate the library's actual hoisted full/tail families, including
        # their separate tail reloads. This is an opt-in traffic hypothesis;
        # no unobserved timing or default schedule promotion is implied.
        from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_choices
        choices=dense_retained_choices(p,r,s)
        if not choices:
            return None,RETAINED_CAPACITY_REFUSAL
        residency=choices[0]
        block=residency.block
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
    if residency is not None and residency.role=='weight':
        from merlin.compile.scheduling.dense_panel_pipeline import schedule_dense_m_panels
        source=hashlib.sha256(json.dumps({'m':m,'k':k,'n':n,'bias':bias,'dtype':['i8','i8','i32']},sort_keys=True).encode()).hexdigest()
        attempt=schedule_dense_m_panels(p,r,s,a,law,n_tiles=block.n,m_tiles=block.m,
            name='dense_rhs_resident',dtypes=('i8','i8','i32'),source_identity=source,
            external_buffers_disjoint=True,bias_source=bias_source,
            static_budget=128,timeout_ms=20000)
        if attempt.result is None:
            return None,'RHS-resident M panels refused: '+str(attempt.feedback)
        result=attempt.result
        result=_wide_loads(result,p,block,r,s,a)
        _record_geometry(result,geometry,block,p,r,s)
        result.realization['resident_axis']={
            'role':residency.role,'outer_order':list(residency.outer_order),
            'requested_operand_bytes':residency.requested_operand_bytes,
            'selection_metric':'actual hoisted full/tail requested bytes; cycles UNKNOWN',
            'm_panel_admission':attempt.feedback}
        return result,None
    if segmented:
        overlap=DensePanelOverlapAdmission(a.configuration_identity,a.numerical_policy_identity,evidence['identity']+':disjoint-owned-operand-windows',law.max_issued_compute_pairs,law.max_issued_loads,2,8)
        combined=DenseSegmentedAdmission(overlap,law,evidence['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
        source=hashlib.sha256(json.dumps([m,k,n,bias,'i8-i8-i32'],sort_keys=True).encode()).hexdigest()
        issue_order='prefetch_rhs_between_k_tiles' if frontier else 'compute_then_prefetch'
        plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,pipeline_issue_order=issue_order,bias_source=bias_source)
        result=schedule_dense_pipeline(p,r,s,a,name='dense_segmented',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=combined,bias_source=bias_source,readout_mode='separate',static_budget=128,timeout_ms=SEGMENTED_VERIFICATION_TIMEOUT_MS)
        result=_wide_loads(result,p,block,r,s,a)
        _record_geometry(result,geometry,block,p,r,s)
        result.realization['verification_timeout_ms']=SEGMENTED_VERIFICATION_TIMEOUT_MS
        result.realization['storage_partition']=partition_receipt or {'selection':'half'}
        result.realization['selected_issue_order']=issue_order
        result.realization['segmented_epoch_bounds']={
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
    result=schedule_dense_pipeline(p,r,s,a,name='dense_retained',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,bias_source=bias_source,readout_mode='separate',static_budget=128)
    result=_wide_loads(result,p,block,r,s,a)
    _record_geometry(result,geometry,block,p,r,s)
    if residency is not None:
        result.realization['resident_axis']={
            'role':residency.role,'outer_order':list(residency.outer_order),
            'requested_operand_bytes':residency.requested_operand_bytes,
            'selection_metric':'actual hoisted full/tail requested bytes; cycles UNKNOWN'}
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
