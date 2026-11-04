"""Explicit retained-LHS, double-buffered RHS candidate for long reductions.

The shared segmented owner checks every source product and actual local window.
Selected ordering facts supply the combined disjoint-load/ordered-commit premises;
this candidate has no target execution certificate and is never selected by auto.
"""
from dataclasses import asdict, replace
import hashlib
import json

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import (
    DensePanelOverlapAdmission, DenseRetainedReductionAdmission, DenseReadyReadoutAdmission)
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission, dense_segmented_blocks
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan, dense_schedule_text, parse_dense_schedule_plan
from merlin.transforms.dense_planning import DenseResources
from gemmini_dense_readout import ReadoutUnavailable


def plan_segmented_readout(incumbent,adapter,problem,geometry,facts_path,*,bias=False,initial=False,
                           outer_order=('m','n'),readout_mode='ready'):
    """Keep incumbent M/N blocking; derive the largest legal RHS K segment.

    This is a capacity-derived candidate, not a timing-based block preference.
    Unsupported geometry or source/fact contracts decline before target emission.
    """
    from gemmini_dense import seed_source
    from gemmini_dense_pipeline import ordering_premises
    if outer_order!=('m','n') or readout_mode not in ('ready','separate'):
        raise ReadoutUnavailable('segmented candidate requires M/N order and an explicit readout mode')
    if problem.tiles[1]<2:
        raise ReadoutUnavailable('segmented candidate requires at least two K tiles')
    if initial:
        raise ReadoutUnavailable('segmented owner has no admitted matrix-initializer boundary')
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry
    old=incumbent.choice.block
    lhs_rows=old.m*problem.tiles[1]*edge
    if lhs_rows>=operand_rows:
        raise ReadoutUnavailable('full source LHS panel leaves no RHS segment storage')
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,lhs_rows),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,lhs_rows,operand_rows-lhs_rows),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=old.m,max_k_blocks=problem.tiles[1]-1,max_n_blocks=old.n,
        partial_output_tiles=True,partial_output_blocks=True)
    block=next((b for b in dense_segmented_blocks(problem,resources,storage)
                if (b.m,b.n)==(old.m,old.n)),None)
    if block is None:
        raise ReadoutUnavailable('incumbent M/N panel cannot fit retained LHS and two RHS segments')
    if max(block.m*block.k*block.n,(block.m+block.n)*block.k)>4096:
        raise ReadoutUnavailable('segmented candidate exceeds its declared compiler issue budget')
    resources=replace(resources,max_k_blocks=block.k)
    flags,pending,receipt=ordering_premises(facts_path)
    if not all(flags.values()):
        raise ReadoutUnavailable('combined segmented ordering is not established by selected facts')
    identity='selected-command-ordering:'+receipt['ordering_sha256']
    common=dict(max_issued_readouts=block.m*block.n,pending_readouts=pending['store'],
                readout_configuration='ordered_snapshot_before_store_issue')
    overlap=DensePanelOverlapAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':disjoint-future-rhs-loads',4096,4096,pending['execute']//2,pending['load'],**common)
    reduction=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':ordered-accumulator-commits',4096,4096,pending['execute']//2,pending['load'],**common)
    admission=DenseSegmentedAdmission(overlap,reduction,
        identity+':ordered-accumulator-commits-with-disjoint-future-rhs-loads')
    readout=None if readout_mode=='separate' else DenseReadyReadoutAdmission(
        adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':descriptor-acc-write-to-store-completion',identity+':ordered-store-configuration',
        block.m*block.n,pending['store'])
    initializer=seed_source(problem.m,problem.n,bias=bias,initial=initial)
    source_identity=hashlib.sha256(json.dumps(dict(source='row-major-source-ordered-dense',
        problem=asdict(problem),initializer=None if initializer is None else asdict(initializer),
        numerical_policy=adapter.numerical_policy_identity),sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(problem,resources,storage,adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=256,block_tiles=(block.m,block.k,block.n),
        outer_order=outer_order,pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,
        bias_source=initializer)
    text=dense_schedule_text(plan)
    try:
        result=schedule_dense_pipeline(problem,resources,storage,adapter,name='dense_segmented_'+readout_mode,
            source_identity=source_identity,dtypes=('i8','i8','i32'),static_budget=256,
            external_buffers_disjoint=True,schedule_plan=parse_dense_schedule_plan(text),
            pipeline_admission=admission,batch_readouts=True,bias_source=initializer,
            readout_mode=readout_mode,readout_admission=readout,timeout_ms=60000)
    except ValueError as error:
        raise ReadoutUnavailable('checked segmented composition declined: '+str(error)) from error
    result.revalidate()
    receipt.update(mode='segmented_'+readout_mode,library='schedule_dense_pipeline(retained_lhs_segmented_k)',
        source_identity=source_identity,block=(block.m,block.k,block.n),
        lhs_rows=lhs_rows,rhs_rows=operand_rows-lhs_rows,k_segments=(problem.tiles[1]+block.k-1)//block.k,
        schedule_ir=text,transformed_schedule_sha256=result.kernel.digest(),
        combined_admission=asdict(admission),readout_admission=None if readout is None else asdict(readout),
        qualification_identity=None)
    return result,receipt
