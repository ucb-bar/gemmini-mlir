"""Capacity-sized retained-LHS/two-RHS schedules using the shared checked owner.

This explicit alternative changes placement and counted segment geometry. It
keeps the incumbent integer adapter, independently selected ordering premise,
full completion at every slot reuse, and original complete-K source proof.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView, DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission, DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission, dense_segmented_blocks
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.compile.tuning_knobs import knob
from . import isa


def capacity_candidates(problem):
    """Original legal blocks and exact simultaneously-live physical windows.

    One LHS panel spans complete K. The two RHS slots share the remainder;
    private ACC rows bound N. We enumerate only the library's largest legal K
    for each M/N pair, and never expand dynamic commands or run timing here.
    """
    dim=isa.DIM
    mt,kt,nt=problem.tiles
    # This family targets one row tile. Larger M retains the existing planner.
    if mt != 1 or kt < 2 or problem.tile_n != dim:
        return ()
    lhs_rows=kt*dim
    if lhs_rows+2*dim > isa.SPAD_ROWS:
        return ()
    # If the complete output and a full RHS panel fit together, segmentation
    # creates no operand-capacity opportunity. Keep the retained incumbent.
    full_rhs_tiles=(isa.SPAD_ROWS-lhs_rows)//(kt*dim)
    if nt <= min(full_rhs_tiles,isa.ACC_ROWS//dim):
        return ()
    storage=DenseLoopStorage(
        DenseMemoryView('lhs','spad',dim,0,lhs_rows,alignment=dim),
        DenseMemoryView('rhs','spad',dim,lhs_rows,isa.SPAD_ROWS-lhs_rows,alignment=dim),
        DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS,alignment=dim))
    resources=DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*4*dim,
        max_m_blocks=1,max_n_blocks=min(nt,isa.ACC_ROWS//dim),max_k_blocks=kt,
        partial_output_tiles=True,partial_output_blocks=True)
    return tuple((b,resources,storage) for b in dense_segmented_blocks(problem,resources,storage))


def completion_cost(problem,block):
    """Count source-group and segment boundaries; no claim of hardware cycles.

    Full-K panels prefetch across output groups (one startup plus one drain
    per group); partial-K panels stage once and drain every segment. Store drains
    are independent of block geometry (the later store-epoch pass may batch
    those). Tie-break toward fewer output groups, then less final-K waste.
    """
    mt,kt,nt=problem.tiles
    mg=(mt+block.m-1)//block.m
    ng=(nt+block.n-1)//block.n
    segments=(kt+block.k-1)//block.k
    boundaries=mg*(ng+1) if block.k==kt else mg*ng*(segments+1)
    return (boundaries,mg*ng,segments*block.k-kt,-block.n)


CAPACITY_PANELS=('completion','wide','deep','area')


def capacity_schedule(m,k,n,bias,column_scales=False,*,panel=None):
    """Resolve an explicit geometry alternative before the schedule cache."""
    configured=knob('DENSE_CAPACITY_PANEL',default='completion',values=CAPACITY_PANELS,
        root=Path(isa.__file__).resolve().parents[1])
    selected=panel if panel is not None else os.environ.get('DENSE_CAPACITY_PANEL',configured)
    if selected not in CAPACITY_PANELS:
        raise ValueError('DENSE_CAPACITY_PANEL must be one of '+', '.join(CAPACITY_PANELS))
    return _capacity_schedule(m,k,n,bias,column_scales,selected)


def candidate_cost(problem,block,panel):
    completion=completion_cost(problem,block)
    if panel=='wide':
        return (-block.n,*completion)
    if panel=='deep':
        return (-block.k,*completion)
    if panel=='area':
        return (-block.k*block.n,*completion)
    return completion


@lru_cache(maxsize=256)
def _capacity_schedule(m,k,n,bias,column_scales,panel):
    """Return an actual checked schedule, or a normal family refusal.

    The caller falls back to its retained/streaming incumbent on refusal.
    Capacity only chooses geometry; the library separately checks the source,
    effects, accumulator continuation, exact windows and combined epoch law.
    """
    from .dense import _adapter
    adapter=_adapter()
    dim=isa.DIM
    problem=DenseProblem(m,k,n,dim,dim,1 if column_scales else dim,1,1,4,bias=bias)
    candidates=capacity_candidates(problem)
    if not candidates:
        return None,'capacity segmented family requires one M tile, common scales and resident complete LHS'
    block,resources,storage=min(candidates,key=lambda item:candidate_cost(problem,item[0],panel))
    evidence=json.loads(Path(isa.__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order']!='in_order':
        return None,'selected execute completion order is unknown'
    issued_compute=max(1,block.m*block.k*block.n)
    issued_loads=max(1,block.m*problem.tiles[1],(block.m+block.n)*block.k)
    reduction=DenseRetainedReductionAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity,evidence['identity'],issued_compute,issued_loads,2,8)
    overlap=DensePanelOverlapAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity,evidence['identity']+':disjoint-owned-operand-windows',
        issued_compute,issued_loads,2,8)
    law=DenseSegmentedAdmission(overlap,reduction,
        evidence['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
    source=hashlib.sha256(json.dumps([m,k,n,bias,'i8-i8-i32'],sort_keys=True).encode()).hexdigest()
    bias_source=DenseBiasSource(n,'i32') if bias else None
    plan=make_dense_schedule_plan(problem,resources,storage,adapter,source_identity=source,
        dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),
        pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,
        pipeline_issue_order=('prefetch_rhs_between_output_groups' if block.k==problem.tiles[1]
                              else 'compute_then_prefetch'),bias_source=bias_source)
    result=schedule_dense_pipeline(problem,resources,storage,adapter,name='dense_capacity_segmented',
        dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,
        schedule_plan=plan,pipeline_admission=law,bias_source=bias_source,
        readout_mode='separate',static_budget=128,timeout_ms=30000)
    result.realization['capacity_segmented']={
        'selected_block_tiles':[block.m,block.k,block.n],
        'lhs_rows':storage.lhs.capacity,'rhs_rows':storage.rhs.capacity,
        'rhs_slot_rows':block.k*block.n*dim,'rhs_slots':2,
        'accumulator_rows':block.m*block.n*dim,
        'candidate_count':len(candidates),'panel_selection':panel,
        'lookahead_axis':'output_group' if block.k==problem.tiles[1] else 'reduction_segment',
        'completion_objective':list(completion_cost(problem,block)),
        'selection_basis':'counted segment and output-group completion boundaries; capacity admitted',
        'max_issued_compute_pairs':issued_compute,'max_issued_loads':issued_loads,
        'pending_compute_pairs':2,'pending_loads':8}
    return result,None


capacity_schedule.cache_clear=_capacity_schedule.cache_clear
capacity_schedule.cache_info=_capacity_schedule.cache_info


def regional_capacity_candidate(retained,m,k,n,column_scales=False):
    """Capacity starvation in an actual admitted incumbent, never a case label.

    Only a complete-K retained RHS window holding one N tile selects the
    alternative. Missing retained admission, mixed configurations, multiple M
    tiles and all other capacity classes keep the incumbent route. A positive
    result only proposes capacity_schedule; its original checks must still pass.
    """
    geometry={} if retained is None else retained.realization.get('panel_geometry',{})
    kt=(k+isa.DIM-1)//isa.DIM
    block=geometry.get('selected_block_tiles',())
    rhs_rows=geometry.get('rhs_window_rows',0)
    width=rhs_rows//(kt*isa.DIM)
    eligible=(retained is not None and not column_scales and 0<m<=isa.DIM and n>isa.DIM
        and len(block)==3 and block[0]==1 and block[1]==kt and block[2]==1 and width==1)
    if retained is not None:retained.revalidate()
    return eligible,{'class':'full_k_rhs_capacity_one_tile',
        'eligible':eligible,'retained_admitted':retained is not None,
        'm_tiles':(m+isa.DIM-1)//isa.DIM,'k_tiles':kt,
        'n_tiles':(n+isa.DIM-1)//isa.DIM,'rhs_window_rows':rhs_rows,
        'full_k_rhs_width_tiles':width,'incumbent_block_tiles':list(block),
        'selected':'capacity' if eligible else 'retained',
        'predicate':'one M tile; multiple N tiles; uniform readout; admitted complete-K RHS width exactly one tile'}
