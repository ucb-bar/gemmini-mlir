"""Capacity-bounded lookahead across complete output-column panels.

The original full-K LHS stays resident while two disjoint full-K RHS windows
alternate. The shared segmented consumer checks every source effect and keeps
its full completion epochs. Only its explicit output-group issue order differs;
the ordinary dense adapter and terminal encoder own all target instructions.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseBiasSource, DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission, DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission, OUTPUT_GROUP_PREFETCH, dense_segmented_blocks
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.transforms.dense_planning import DenseProblem, DenseResources

from . import isa
from .dense_pipeline import PANEL_GEOMETRIES, _record_geometry


@lru_cache(maxsize=256)
def output_group_schedule(m, k, n, bias, column_scales=False, *, geometry='wide'):
    """Return a checked output-lookahead program, or a capacity refusal.

    Full-K double buffering is selected only inside the actual operand/ACC
    domain. A refusal is deliberately left to the caller's incumbent route;
    this function never changes the original K or fabricates a shorter problem.
    """
    from .dense import _adapter

    if geometry not in PANEL_GEOMETRIES:
        raise ValueError('unknown output-prefetch panel geometry')
    a = _adapter()
    dim = isa.DIM
    p = DenseProblem(m, k, n, dim, dim, 1 if column_scales else dim, 1, 1, 4, bias=bias)
    max_m, max_n = PANEL_GEOMETRIES[geometry]
    r = DenseResources(isa.SPAD_ROWS*dim, isa.ACC_ROWS*4*dim,
        max_m_blocks=max_m, max_n_blocks=max_n, max_k_blocks=p.tiles[1],
        partial_output_tiles=True, partial_output_blocks=True)
    candidates = []
    for bm in range(1, min(p.tiles[0], max_m)+1):
        lhs_rows = bm*p.tiles[1]*dim
        if lhs_rows + 2*p.tiles[1]*dim > isa.SPAD_ROWS:
            continue
        s = DenseLoopStorage(
            DenseMemoryView('lhs', 'spad', dim, 0, lhs_rows),
            DenseMemoryView('rhs', 'spad', dim, lhs_rows, isa.SPAD_ROWS-lhs_rows),
            DenseMemoryView('acc', 'acc', 4*dim, 0, isa.ACC_ROWS))
        candidates.extend((b, s) for b in dense_segmented_blocks(p, r, s)
            if b.m == bm and b.k == p.tiles[1] and p.tiles[2] > b.n)
    if not candidates:
        return None, 'full-K retained LHS and two RHS output panels exceed assigned storage or have no following N group'
    block, s = min(candidates, key=lambda pair:
        (pair[0].loads, -pair[0].m*pair[0].n, pair[0].m, pair[0].n))
    evidence = json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order'] != 'in_order':
        return None, 'selected execute completion order is unknown'
    # These are finite issued-work bounds, not enlarged hardware queue depths.
    issued_pairs = block.m*block.k*block.n
    issued_loads = max(block.m*block.k, block.k*block.n)
    reduction = DenseRetainedReductionAdmission(a.configuration_identity,
        a.numerical_policy_identity, evidence['identity'], issued_pairs, issued_loads, 2, 8)
    overlap = DensePanelOverlapAdmission(a.configuration_identity,
        a.numerical_policy_identity, evidence['identity']+':disjoint-owned-operand-windows',
        issued_pairs, issued_loads, 2, 8)
    law = DenseSegmentedAdmission(overlap, reduction,
        evidence['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
    source = hashlib.sha256(json.dumps(dict(m=m, k=k, n=n, bias=bias,
        dtype=['i8', 'i8', 'i32']), sort_keys=True).encode()).hexdigest()
    bias_source = DenseBiasSource(n, 'i32') if bias else None
    plan = make_dense_schedule_plan(p, r, s, a, source_identity=source,
        dtypes=('i8', 'i8', 'i32'), static_budget=128,
        block_tiles=(block.m, block.k, block.n),
        pipeline_kind='retained_lhs_segmented_k', pipeline_prefetch=True,
        pipeline_issue_order=OUTPUT_GROUP_PREFETCH, bias_source=bias_source)
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    try:
        result = schedule_dense_pipeline(p, r, s, a, name='dense_output_prefetch',
        dtypes=('i8', 'i8', 'i32'), source_identity=source,
        external_buffers_disjoint=True, schedule_plan=plan, pipeline_admission=law,
            bias_source=bias_source, readout_mode='separate', static_budget=128, timeout_ms=30000)
    except DenseVerificationUnavailable as error:
        return None, 'optional output-group verification unavailable: '+str(error)
    _record_geometry(result, geometry, block, p, r, s)
    result.realization['output_group_prefetch'] = dict(
        selected=True, original_k_tiles=p.tiles[1], rhs_slots=2,
        block_tiles=[block.m, block.k, block.n],
        lhs_rows=block.m*block.k*dim, rhs_slot_rows=block.k*block.n*dim,
        complete_source_identity=source, actual_kernel_sha256=result.kernel.digest(),
        completion='original full completion before each operand or accumulator reuse',
        target_encoder='shared dense adapter and terminal encoder')
    return result, None


@lru_cache(maxsize=256)
def capacity_stream_schedule(m, k, n, bias, column_scales=False, *, k_tiles=1):
    """Checked two-window stream sized from original ACC and operand capacity.

    This alternative trades full-LHS retention for a larger output panel.
    Dense block load counts rank legal plans; they do not claim lower latency.
    Every primitive K ownership change retains the existing completion epoch.
    """
    from merlin.transforms.dense_planning import dense_blocks
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    from .dense import _adapter
    a=_adapter();dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,1 if column_scales else dim,1,1,4,bias=bias)
    if column_scales:
        return None,'capacity stream leaves per-column readout to its original exact route'
    if type(k_tiles) is not int or k_tiles not in (1,4,8,16,32):
        raise ValueError('capacity stream K tiles must be one of the bounded explored choices')
    if k_tiles>p.tiles[1]:
        return None,'K-panel choice exceeds the original reduction domain'
    if k_tiles>1 and k%(k_tiles*dim):
        return None,'multi-tile capacity stream requires complete K panels; original partial-K route remains available'
    output_tile_budget=isa.ACC_ROWS//dim
    r=DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*4*dim,
        max_m_blocks=min(p.tiles[0],output_tile_budget),
        max_n_blocks=min(p.tiles[2],output_tile_budget),max_k_blocks=k_tiles,
        partial_output_tiles=True,partial_output_blocks=True)
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,isa.SPAD_ROWS//2),
        DenseMemoryView('rhs','spad',dim,isa.SPAD_ROWS//2,isa.SPAD_ROWS-isa.SPAD_ROWS//2),
        DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS))
    choices=[b for b in dense_blocks(p,r) if b.k==k_tiles and s.admits(p,b,r)
        and 2*b.m*b.k*dim<=s.lhs.capacity and 2*b.n*b.k*dim<=s.rhs.capacity]
    if not choices:return None,'capacity stream has no legal two-window original output panel'
    block=min(choices,key=lambda b:(b.loads,-b.m*b.n,b.m,b.n))
    evidence=json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order']!='in_order':
        return None,'capacity stream requires original in-order execute completion'
    law=DensePanelOverlapAdmission(a.configuration_identity,a.numerical_policy_identity,
        evidence['identity']+':disjoint-owned-operand-windows',block.m*block.k*block.n,
        (block.m+block.n)*block.k,2,8)
    source=hashlib.sha256(json.dumps(dict(m=m,k=k,n=n,bias=bias,
        dtype=['i8','i8','i32'],family='capacity-two-window',k_panel=k_tiles),sort_keys=True).encode()).hexdigest()
    bias_source=DenseBiasSource(n,'i32') if bias else None
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),
        static_budget=128,block_tiles=(block.m,block.k,block.n),
        pipeline_kind='two_window' if k_tiles==1 else 'k_panel_two_window',pipeline_prefetch=True,bias_source=bias_source)
    consumer=schedule_dense_pipeline
    if k_tiles>1:
        from merlin.compile.scheduling.dense_k_panel_pipeline import schedule_dense_k_panel_pipeline
        reduction=DenseRetainedReductionAdmission(a.configuration_identity,a.numerical_policy_identity,
            evidence['identity'],block.m*block.k*block.n,(block.m+block.n)*block.k,2,8)
        law=DenseSegmentedAdmission(law,reduction,
            evidence['identity']+':pure-inorder-execute-with-disjoint-future-operand-panels')
        def consumer(*args,pipeline_admission,readout_mode,**kwargs):
            if readout_mode!='separate':raise ValueError('K-panel capacity stream preserves exact separate readout')
            return schedule_dense_k_panel_pipeline(*args,pipeline_admission,**kwargs)
    try:
        result=consumer(p,r,s,a,name='dense_capacity_stream',
            dtypes=('i8','i8','i32'),source_identity=source,
            external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,
            bias_source=bias_source,readout_mode='separate',static_budget=128,timeout_ms=30000)
    except DenseVerificationUnavailable as error:
        return None,'optional capacity-stream verification unavailable: '+str(error)
    result.realization['capacity_stream']=dict(pipeline_kind='two_window' if k_tiles==1 else 'k_panel_two_window',
        block_tiles=[block.m,block.k,block.n],original_k_tiles=p.tiles[1],
        accumulator_tile_budget=output_tile_budget,ranked_by='dense_blocks original load count',
        primitive_loads=block.loads,original_source_identity=source,
        actual_kernel_sha256=result.kernel.digest(),
        completion='existing full completion at every K-panel ownership change',
        performance_qualification='untimed traffic/drain tradeoff; no native latency claim')
    return result,None
