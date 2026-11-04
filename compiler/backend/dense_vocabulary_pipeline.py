"""Isolated full-K RHS output-panel lookahead through shared dense scheduling.

Two disjoint full-K RHS slots alternate across N panels while the complete
LHS stays resident. The shared source/ownership/epoch verifier consumes the
actual peeled/count-paired schedule. No compiler-selected completion token.
"""
from functools import lru_cache
from pathlib import Path
import hashlib,json
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission,DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission,OUTPUT_GROUP_PREFETCH,dense_segmented_blocks
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.transforms.dense_planning import DenseProblem,DenseResources
from . import isa

@lru_cache(maxsize=128,typed=True)
def vocabulary_schedule(m:int,k:int,n:int,bias:bool,column_scales:bool=False,*,n_tiles:int=3):
    """Return (checked shared schedule, refusal); opt-in construction only."""
    if type(bias)is not bool or type(column_scales)is not bool:
        raise ValueError('vocabulary schedule requires explicit boolean initialization/readout flags')
    if any(type(x)is not int or x<=0 for x in (m,k,n,n_tiles)):
        raise ValueError('vocabulary schedule requires positive static dimensions/bound')
    if m!=1 or bias or column_scales:
        return None,'requires unseeded one-row projection without mixed column readout'
    from .dense import _adapter
    a=_adapter();d=isa.DIM
    p=DenseProblem(m,k,n,d,d,d,1,1,4,bias=False)
    lhs_rows=p.tiles[1]*d
    if lhs_rows+2*p.tiles[1]*d>isa.SPAD_ROWS:
        return None,'complete LHS and two full-K primitive RHS slots exceed owned SRAM'
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',d,0,lhs_rows,alignment=d),
        DenseMemoryView('rhs','spad',d,lhs_rows,isa.SPAD_ROWS-lhs_rows,alignment=d),
        DenseMemoryView('acc','acc',4*d,0,isa.ACC_ROWS,alignment=d))
    r=DenseResources(isa.SPAD_ROWS*d,isa.ACC_ROWS*4*d,max_m_blocks=1,max_n_blocks=n_tiles,
        max_k_blocks=p.tiles[1],partial_output_tiles=True,partial_output_blocks=True)
    choices=[b for b in dense_segmented_blocks(p,r,s) if b.k==p.tiles[1] and p.tiles[2]>b.n]
    if not choices:return None,'two independently checked full-K RHS windows do not fit'
    b=min(choices,key=lambda b:(b.loads,-b.n))
    evidence=json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order']!='in_order':
        return None,'selected execute completion order is unknown'
    reduction=DenseRetainedReductionAdmission(a.configuration_identity,a.numerical_policy_identity,
        evidence['identity'],b.m*b.k*b.n,max(b.m*p.tiles[1],b.k*b.n),2,8)
    overlap=DensePanelOverlapAdmission(a.configuration_identity,a.numerical_policy_identity,
        evidence['identity']+':disjoint-owned-operand-windows',reduction.max_issued_compute_pairs,
        reduction.max_issued_loads,2,8)
    law=DenseSegmentedAdmission(overlap,reduction,evidence['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
    source=hashlib.sha256(json.dumps([m,k,n,bias,'i8-i8-i32'],sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),
        static_budget=128,block_tiles=(b.m,b.k,b.n),pipeline_kind='retained_lhs_segmented_k',
        pipeline_prefetch=True,pipeline_issue_order=OUTPUT_GROUP_PREFETCH)
    result=schedule_dense_pipeline(p,r,s,a,name='dense_vocabulary',dtypes=('i8','i8','i32'),
        source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,
        readout_mode='separate',static_budget=128,timeout_ms=30000)
    result.realization['vocabulary_pipeline']=dict(selection='full_k_output_group_lookahead',
        selected_block_tiles=[b.m,b.k,b.n],n_tile_bound=n_tiles,lhs_rows=lhs_rows,
        rhs_slot_rows=b.k*b.n*d,rhs_slots=2,acc_rows=b.m*b.n*d,
        operand_capacity_rows=isa.SPAD_ROWS,accumulator_capacity_rows=isa.ACC_ROWS,
        source_predicate='M=1, no initializer/mixed column readout; two full-K RHS slots and multiple N groups physically fit',
        completion='shared full epoch completion before all RHS/ACC reuse; existing store epochs retain closing completion')
    return result,None
