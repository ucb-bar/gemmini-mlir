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
from merlin.compiler.schedule.compact_dense import DenseLoopStorage,DenseMemoryView,DenseBiasSource,emit_compact_dense
from merlin.compiler.schedule.dense_panel_pipeline import DenseRetainedReductionAdmission,dense_retained_operand_hoists,dense_retained_compute_order
from merlin.compiler.schedule.dense_ready_readout import schedule_dense_pipeline
from merlin.compiler.schedule.schedule_ir import make_dense_schedule_plan
from merlin.compiler.schedule.loop_schedule import propose_loop_transforms
from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
from merlin.compiler.selection.knobs import knob
from . import isa

RETAINED_CAPACITY_REFUSAL='full-K retained panel exceeds assigned operand storage'
# Finite wall-clock budget for the unchanged segmented source/epoch proof.
# Ragged N groups introduce additional quantified residency/continuation cases;
# the connected M1/K2048/N5728 plan takes about 16s to construct and verify.
# This changes neither the program nor its original correctness obligations.
SEGMENTED_VERIFICATION_TIMEOUT_MS=30000


PANEL_GEOMETRIES={'balanced':(4,4),'wide':(8,8),'tall':(8,4),'wide_n':(4,8),
                  'wide_16':(4,16),'tall_16':(16,4)}


def retained_schedule(m,k,n,bias,column_scales=False,segmented=False,*,geometry=None,ready_readout=False):
    """Resolve a panel resource bound before entering the geometry-keyed cache."""
    configured=knob('DENSE_PANEL_GEOMETRY',default='balanced',values=tuple(PANEL_GEOMETRIES),
                    root=Path(__file__).resolve().parents[1])
    selection=geometry if geometry is not None else os.environ.get('DENSE_PANEL_GEOMETRY',configured)
    if selection not in PANEL_GEOMETRIES:
        raise ValueError('DENSE_PANEL_GEOMETRY must be one of '+', '.join(PANEL_GEOMETRIES))
    split=knob('DENSE_STORAGE_SPLIT',default='half',values=('half','adaptive'),
               root=Path(__file__).resolve().parents[1])
    rhs_transfers=knob('DENSE_RHS_TRANSFERS',default='primitive',values=('primitive','block64'),
                       root=Path(__file__).resolve().parents[1])
    operand=knob('DENSE_RETAINED_OPERAND',default='lhs',values=('lhs','rhs_reuse','rhs_amortized'),
                 root=Path(__file__).resolve().parents[1])
    issue=knob('DENSE_SEGMENTED_ISSUE',default='compute_then_prefetch',
        values=('compute_then_prefetch','prefetch_rhs_between_columns','prefetch_rhs_between_k_tiles','prefetch_rhs_grouped_columns'),
        root=Path(__file__).resolve().parents[1])
    placement=knob('DENSE_SEGMENTED_PLACEMENT',default='compact',
        values=('compact','window_halves'),root=Path(__file__).resolve().parents[1])
    segmented_transfers=knob('DENSE_SEGMENTED_TRANSFERS',default='primitive',values=('primitive','block64'),root=Path(__file__).resolve().parents[1])
    segmented_k_cap=knob('DENSE_SEGMENTED_K_CAP',default=16,values=(8,16,32),root=Path(__file__).resolve().parents[1])
    from .dense_segmented_readout import segmented_readout_selection
    segmented_readout_mode=segmented_readout_selection()
    input_pipeline=knob('DENSE_INPUT_PIPELINE',default='off',values=('off','two_slot'),root=Path(__file__).resolve().parents[1])
    storage_policy=knob('DENSE_STORAGE_POLICY',default='incumbent',values=('incumbent','adaptive_rhs'),root=Path(__file__).resolve().parents[1])
    geometry_policy=knob('DENSE_GEOMETRY_POLICY',default='incumbent',values=('incumbent','tall_m'),root=Path(__file__).resolve().parents[1])
    if geometry_policy=='tall_m' and not segmented and selection=='wide' and storage_policy=='incumbent':
        from .dense_geometry_policy import execute_tall_geometry
        return execute_tall_geometry(m,k,n,bias,column_scales,storage_split=split,
            retained_planner=lambda *args:_retained_schedule(*args,rhs_transfers=rhs_transfers,
                retained_operand=operand,segmented_issue=issue,segmented_placement=placement,
                ready_readout=ready_readout,segmented_transfers=segmented_transfers,
                segmented_readout_mode=segmented_readout_mode,input_pipeline=input_pipeline))
    if storage_policy=='adaptive_rhs' and not segmented and split=='half':
        from .dense_storage_policy import DenseStorageSource,execute_storage_policy
        # Preserve all current selected transformations on both actual arms.
        return execute_storage_policy(DenseStorageSource(m,k,n,bias,column_scales,selection),
            lambda *args,**kwargs:_retained_schedule(*args,rhs_transfers=rhs_transfers,
                retained_operand=operand,segmented_issue=issue,segmented_placement=placement,
                ready_readout=ready_readout,segmented_transfers=segmented_transfers,
                segmented_readout_mode=segmented_readout_mode,input_pipeline=input_pipeline,**kwargs))
    return _retained_schedule(m,k,n,bias,column_scales,segmented,selection,split,rhs_transfers,operand,issue,placement,ready_readout,segmented_transfers,segmented_k_cap=segmented_k_cap,segmented_readout_mode=segmented_readout_mode,input_pipeline=input_pipeline)


@lru_cache(maxsize=256)
def _retained_schedule(m,k,n,bias,column_scales,segmented,geometry,storage_split='half',rhs_transfers='primitive',retained_operand='lhs',segmented_issue='compute_then_prefetch',segmented_placement='compact',ready_readout=False,segmented_transfers='primitive',storage_n_cap=None,segmented_k_cap=None,input_pipeline=None,segmented_readout_mode=None):
    if segmented_readout_mode is None:
        from .dense_segmented_readout import segmented_readout_selection
        segmented_readout_mode=segmented_readout_selection()
    if segmented_readout_mode not in ('separate','ready'):
        raise ValueError('invalid segmented readout mode')
    if input_pipeline is None:
        input_pipeline=knob('DENSE_INPUT_PIPELINE',default='off',values=('off','two_slot'),root=Path(__file__).resolve().parents[1])
    if retained_operand=='rhs_amortized' and not segmented:
        lhs,reason=_retained_schedule(m,k,n,bias,column_scales,False,geometry,storage_split,rhs_transfers,'lhs',segmented_issue,segmented_placement,ready_readout,segmented_transfers,storage_n_cap,input_pipeline='off')
        if lhs is None:return lhs,reason
        rhs,reason=_retained_schedule(m,k,n,bias,column_scales,False,geometry,storage_split,rhs_transfers,'rhs_reuse',segmented_issue,segmented_placement,ready_readout,segmented_transfers,storage_n_cap,input_pipeline='off')
        if rhs is None:return lhs,None
        from .dense_operand_residency import select_checked_residency
        selected=select_checked_residency(lhs,rhs)
        if input_pipeline=='two_slot' and ready_readout:
            selected_operand='rhs_reuse' if selected.realization['checked_operand_residency']['selected']=='rhs_reuse' else 'lhs'
            return _retained_schedule(m,k,n,bias,column_scales,False,geometry,storage_split,rhs_transfers,selected_operand,segmented_issue,segmented_placement,ready_readout,segmented_transfers,storage_n_cap,input_pipeline='two_slot')
        return selected,None
    from .dense import _adapter
    a=_adapter()
    dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,1 if column_scales else dim,1,1,4,bias=bias)
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,isa.SPAD_ROWS//2),DenseMemoryView('rhs','spad',dim,isa.SPAD_ROWS//2,isa.SPAD_ROWS//2),DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS))
    max_m,max_n=PANEL_GEOMETRIES[geometry]
    if storage_n_cap is not None:
        if type(storage_n_cap)is not int or storage_n_cap<1 or segmented:
            raise ValueError('full-K storage bound requires a positive N cap and retained source')
        max_n=min(max_n,storage_n_cap)
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
        from merlin.compiler.schedule.dense_segmented_pipeline import DenseSegmentedAdmission,dense_segmented_blocks
        from merlin.compiler.schedule.dense_panel_pipeline import DensePanelOverlapAdmission
        if segmented_k_cap is None:
            # Capacity policy also invokes this non-dispatching planner directly.
            # Bind its actual root-selected resource bound, not only the public dispatcher.
            segmented_k_cap=knob('DENSE_SEGMENTED_K_CAP',default=16,values=(8,16,32),
                root=Path(__file__).resolve().parents[1])
        if type(segmented_k_cap)is not int or segmented_k_cap not in (8,16,32):
            raise ValueError('segmented K bound must be one of the declared integer resource bounds')
        cap=min(segmented_k_cap,p.tiles[1]-1)
        if cap<1:
            return None,'segmented reduction requires more than one K tile'
        r=replace(r,max_k_blocks=cap)
        candidates=dense_segmented_blocks(p,r,s)
    else:
        candidates=[b for b in dense_blocks(p,r,k_limit=lambda m,n:s.k_limit(p,r,m,n)) if b.k==p.tiles[1] and s.admits(p,b,r)]
    if not candidates:
        return None,RETAINED_CAPACITY_REFUSAL
    block=min(candidates,key=lambda b:(b.loads,-b.m*b.n,b.m,b.n))
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
    if ready_readout and (not segmented or segmented_readout_mode=='ready'):
        law=replace(law,max_issued_readouts=block.m*block.n,
            readout_configuration='ordered_snapshot_before_store_issue')
    bias_source=DenseBiasSource(n,'i32') if bias else None
    if segmented:
        overlap=DensePanelOverlapAdmission(a.configuration_identity,a.numerical_policy_identity,evidence['identity']+':disjoint-owned-operand-windows',law.max_issued_compute_pairs,law.max_issued_loads,2,8)
        if ready_readout and segmented_readout_mode=='ready':
            overlap=replace(overlap,max_issued_readouts=block.m*block.n,
                readout_configuration='ordered_snapshot_before_store_issue')
        combined=DenseSegmentedAdmission(overlap,law,evidence['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
        source=hashlib.sha256(json.dumps([m,k,n,bias,'i8-i8-i32'],sort_keys=True).encode()).hexdigest()
        from .dense_segmented_placement import choose_segmented_placement
        offsets,placement_receipt=choose_segmented_placement(p,block,s,segmented_placement)
        plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,
            pipeline_window_offsets=offsets,
            pipeline_issue_order=('prefetch_rhs_between_columns' if segmented_issue=='prefetch_rhs_grouped_columns' else segmented_issue),bias_source=bias_source)
        result=schedule_dense_pipeline(p,r,s,a,name='dense_segmented',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=combined,bias_source=bias_source,readout_mode='separate',static_budget=128,timeout_ms=SEGMENTED_VERIFICATION_TIMEOUT_MS)
        if segmented_issue=='prefetch_rhs_grouped_columns':
            from .dense_segmented_frontier import group_future_frontier
            result,_=group_future_frontier(result,p,block,r,s,a)
        _record_geometry(result,geometry,block,p,r,s)
        result.realization['verification_timeout_ms']=SEGMENTED_VERIFICATION_TIMEOUT_MS
        result.realization['segmented_k_resource_bound']=dict(requested=segmented_k_cap,
            max_k_blocks=cap,selected_k_blocks=block.k,
            rhs_slot_rows=block.k*block.n*dim,rhs_slots=2,
            assigned_rhs_rows=s.rhs.capacity)
        result.realization['segmented_issue_selection']=segmented_issue
        result.realization['segmented_placement_selection']=placement_receipt
        result.realization['segmented_epoch_bounds']={
            'retained_lhs_loads':block.m*p.tiles[1],
            'rhs_slot_loads':block.k*block.n,
            'retained_lhs_rows':block.m*p.tiles[1]*dim,
            'rhs_slot_rows':block.k*block.n*dim,'rhs_slots':2,
            'max_issued_loads':law.max_issued_loads,
            'max_issued_compute_pairs':law.max_issued_compute_pairs,
            'pending_loads':law.pending_loads,'pending_compute_pairs':law.pending_compute_pairs}
        if segmented_transfers=='block64':
            from .dense_segmented_transfers import select_segmented_block_transfers
            result,receipt=select_segmented_block_transfers(result,p,block,r,s,a)
            result.realization['segmented_block_transfers']=receipt
        from .dense_segmented_readout import maybe_segmented_ready_readout
        result=maybe_segmented_ready_readout(result,p,block,r,s,a,eligible=ready_readout,selection=segmented_readout_mode)
        return result,None
    # Compare residency on the SAME capacity-admitted block. Retaining RHS
    # may save repeated immutable loads across complete M groups. Partial M
    # families retain the original reloads; bytes are a construction filter,
    # never a cycle prediction or a source-identity dispatch.
    from .dense_operand_residency import choose_residency
    residency=choose_residency(p,block,selection=retained_operand)
    raw=emit_compact_dense(p,block,s,a,name='dense_retained',dtypes=('i8','i8','i32'),
        outer_order=residency['outer_order'],bias_source=bias_source)
    transforms=dense_retained_operand_hoists(raw,a,role=residency['role'])
    transforms+=dense_retained_compute_order(propose_loop_transforms(raw,transforms),a,order=('k','n','m'))
    source=hashlib.sha256(json.dumps({'m':m,'k':k,'n':n,'bias':bias,'dtype':['i8','i8','i32']},sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),outer_order=residency['outer_order'],loop_transforms=transforms,pipeline_kind='retained_full_k',pipeline_prefetch=False,bias_source=bias_source)
    result=schedule_dense_pipeline(p,r,s,a,name='dense_retained',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,bias_source=bias_source,readout_mode='separate',static_budget=128)
    if ready_readout:
        from .dense_ready import admit_causal_readout
        result=admit_causal_readout(result,p,block,r,s,a)
        from .dense_input_pipeline import maybe_pipeline_input
        result=maybe_pipeline_input(result,p,block,r,s,a,plan=plan,
            pipeline_admission=law,source_identity=source,residency=residency,selection=input_pipeline)
    _record_geometry(result,geometry,block,p,r,s)
    result.realization['operand_residency']=result.realization.get('input_pipeline_residency',residency)
    result.realization['storage_partition']=partition_receipt or {'selection':'half',
        'lhs_window_rows':s.lhs.capacity,'rhs_window_rows':s.rhs.capacity}
    if rhs_transfers=='block64':
        from .dense_rhs_transfers import select_rhs_block_transfers
        result,receipt=select_rhs_block_transfers(result,p,block,r,s,a)
        result.realization['rhs_block_transfers']=receipt
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
