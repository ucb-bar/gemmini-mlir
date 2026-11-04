"""Bounded original-K panel streams through checked shared transformations.

Adaptive segmented storage keeps a complete LHS only while capacity admits it.
Beyond that limit the shared two-window stream reuses bounded K-slice windows,
with real completion at every ownership change. No cross-queue RAW/WAR ordering
is assumed and initialization remains inside the original output-group schedule.
"""
from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import hashlib,json
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission,DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
from . import isa
from .dense_verification import verify_dense_with_budget


def _storage(lhs_rows):
    dim=isa.DIM
    return DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,lhs_rows),
        DenseMemoryView('rhs','spad',dim,lhs_rows,isa.SPAD_ROWS-lhs_rows),
        DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS))


@lru_cache(maxsize=256)
def streaming_schedule(m,k,n,bias,column_scales=False,*,geometry='balanced'):
    """Return (checked result, refusal), compatible with retained_schedule.

    Both plans consume the complete original K problem. Segment/window bounds
    come from selected storage capacities, rather than a shape cutoff. The
    returned emitter uses the incumbent adapter's commands and drain callback.
    """
    from .dense import _adapter
    from .dense_pipeline import PANEL_GEOMETRIES
    if geometry not in PANEL_GEOMETRIES:
        raise ValueError('streaming geometry must name an existing dense panel bound')
    max_m,max_n=PANEL_GEOMETRIES[geometry]
    a=_adapter()
    dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,1 if column_scales else dim,1,1,4,bias=bias)
    r=DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*4*dim,max_m_blocks=max_m,max_n_blocks=max_n,
        max_k_blocks=max(1,p.tiles[1]-1),partial_output_tiles=True,partial_output_blocks=True)
    evidence=json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order']!='in_order':
        return None,'selected execute completion order is unknown'
    candidates=[]
    # A complete LHS panel plus two RHS segments. Every split owns disjoint
    # physical row intervals and keeps the original full source K domain.
    for bm in range(1,min(p.tiles[0],r.max_m_blocks)+1):
        lhs_rows=bm*p.tiles[1]*dim
        if lhs_rows+2*dim>isa.SPAD_ROWS or p.tiles[1]<2:
            continue
        s=_storage(lhs_rows)
        for b in dense_blocks(p,r,k_limit=lambda _m,bn:s.rhs.capacity//(2*bn*dim)):
            if b.m==bm and s.admits(p,b,r):
                candidates.append((b,s))
    if candidates:
        block,s=min(candidates,key=lambda pair:(pair[0].loads,-pair[0].k*pair[0].n,-pair[0].m*pair[0].n))
        r=replace(r,max_k_blocks=block.k)
        kind='retained_lhs_segmented_k'
        reduction=DenseRetainedReductionAdmission(a.configuration_identity,a.numerical_policy_identity,
            evidence['identity'],max(1,block.m*block.k*block.n),max(1,block.m*p.tiles[1],(block.m+block.n)*block.k),2,8)
        overlap=DensePanelOverlapAdmission(a.configuration_identity,a.numerical_policy_identity,
            evidence['identity']+':disjoint-owned-operand-windows',reduction.max_issued_compute_pairs,
            reduction.max_issued_loads,2,8)
        law=DenseSegmentedAdmission(overlap,reduction,
            evidence['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
    else:
        # Arbitrarily deep K uses two bounded primitive windows. Loads and
        # compute touch opposite slots in an epoch; every slot reuse drains.
        s=_storage(isa.SPAD_ROWS//2)
        r=replace(r,max_k_blocks=1)
        choices=[b for b in dense_blocks(p,r) if s.admits(p,b,r)
                 and 2*b.m*dim<=s.lhs.capacity and 2*b.n*dim<=s.rhs.capacity]
        if not choices:
            return None,'two disjoint primitive panel windows exceed assigned operand storage'
        block=min(choices,key=lambda b:(b.loads,-b.m*b.n))
        kind='two_window'
        law=DensePanelOverlapAdmission(a.configuration_identity,a.numerical_policy_identity,
            evidence['identity']+':disjoint-owned-operand-windows',max(1,block.m*block.n),
            max(1,block.m+block.n),2,8)
    source=hashlib.sha256(json.dumps([m,k,n,bias,'i8-i8-i32','capacity-streaming'],sort_keys=True).encode()).hexdigest()
    bias_source=DenseBiasSource(n,'i32') if bias else None
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),
        static_budget=128,block_tiles=(block.m,block.k,block.n),pipeline_kind=kind,
        pipeline_prefetch=True,bias_source=bias_source)
    result=verify_dense_with_budget(lambda timeout: schedule_dense_pipeline(p,r,s,a,name='dense_streaming',dtypes=('i8','i8','i32'),
        source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,
        pipeline_admission=law,bias_source=bias_source,readout_mode='separate',static_budget=128,timeout_ms=timeout),initial_timeout_ms=20000)
    result.realization['streaming_storage']={'panel_geometry':geometry,'pipeline_kind':kind,'lhs_rows':s.lhs.capacity,
        'rhs_rows':s.rhs.capacity,'block_tiles':[block.m,block.k,block.n],
        'original_k_tiles':p.tiles[1],'retains_complete_lhs':kind=='retained_lhs_segmented_k'}
    return result,None
