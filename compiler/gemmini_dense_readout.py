"""Fact-bound final-K readout through the library's retained dense owner.

This candidate stages the original full K panel, then moves each original
store immediately after its last reduction. Geometry does not grant ordering:
the selected descriptor-to-readout completion edge supplies the causal law.
Unsupported storage/ordering/transform regions retain the checked panel mode.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json

from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission,DenseReadyReadoutAdmission
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan,dense_schedule_text,parse_dense_schedule_plan
from merlin.transforms.dense_planning import DenseResources,dense_blocks


class ReadoutUnavailable(ValueError):
    """A source/fact/storage region outside this candidate's admitted domain."""


def plan_ready_readout(incumbent,adapter,problem,geometry,facts_path,*,bias=False,initial=False,outer_order=('m','n')):
    from gemmini_dense import seed_source
    from gemmini_dense_pipeline import ordering_premises
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry
    split=operand_rows//2
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    old=incumbent.choice.block
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=old.m,max_k_blocks=problem.tiles[1],max_n_blocks=old.n,
        partial_output_tiles=True,partial_output_blocks=True)
    # Preserve the incumbent M/N block. Only K residency changes, when the
    # original separately assigned operand stores admit the complete panel.
    blocks=dense_blocks(problem,resources,k_limit=lambda m,n:storage.k_limit(problem,resources,m,n))
    block=next((b for b in blocks if (b.m,b.n,b.k)==(old.m,old.n,problem.tiles[1])),None)
    if block is None:raise ReadoutUnavailable('incumbent M/N block cannot retain complete K in its assigned stores')
    flags,pending,receipt=ordering_premises(facts_path)
    required=('ordered_accumulator_writes','accumulator_result_visibility','ordered_configuration_capture',
        'finite_queue_progress','complete_device_drain','exclusive_private_state')
    missing=[key for key in required if not flags[key]]
    if missing:raise ReadoutUnavailable('unestablished causal readout premises: '+','.join(missing))
    if max(block.m*block.k*block.n,(block.m+block.n)*block.k)>4096:
        raise ReadoutUnavailable('retained readout exceeds the declared finite compiler issue budget')
    identity='selected-command-ordering:'+receipt['ordering_sha256']
    law=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':retained-read-only-operands',4096,4096,pending['execute']//2,pending['load'],
        max_issued_readouts=block.m*block.n,pending_readouts=pending['store'],
        readout_configuration='ordered_snapshot_before_store_issue')
    readout_law=DenseReadyReadoutAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':descriptor-acc-write-to-store-completion',identity+':ordered-store-configuration',
        block.m*block.n,pending['store'])
    initializer=seed_source(problem.m,problem.n,bias=bias,initial=initial)
    source_identity=hashlib.sha256(json.dumps({'source':'row-major-source-ordered-dense',
        'problem':asdict(problem),'initializer':None if initializer is None else asdict(initializer),
        'numerical_policy':adapter.numerical_policy_identity},sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(problem,resources,storage,adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=256,block_tiles=(block.m,block.k,block.n),
        outer_order=outer_order,pipeline_kind='retained_full_k',pipeline_prefetch=False,bias_source=initializer)
    text=dense_schedule_text(plan)
    try:
        result=schedule_dense_pipeline(problem,resources,storage,adapter,name='dense_ready_readout',
            source_identity=source_identity,dtypes=('i8','i8','i32'),static_budget=256,
            external_buffers_disjoint=True,schedule_plan=parse_dense_schedule_plan(text),
            pipeline_admission=law,batch_readouts=True,bias_source=initializer,
            readout_mode='ready',readout_admission=readout_law,timeout_ms=60000)
    except ValueError as error:
        raise ReadoutUnavailable('checked ready-readout composition declined: '+str(error)) from error
    result.revalidate()
    if result.kernel.digest()==result.original.kernel.digest():
        raise ReadoutUnavailable('ready-readout composition did not transform actual commands')
    receipt.update(mode='ready_readout',source_identity=source_identity,block=(block.m,block.k,block.n),
        library='schedule_dense_pipeline(retained_full_k,readout_mode=ready,batch_readouts=True)',
        source_schedule_sha256=result.original.kernel.digest(),transformed_schedule_sha256=result.kernel.digest(),
        schedule_ir=text,readout_admission=asdict(readout_law),
        causal_edge=receipt['basis']['accumulator_result_visibility'])
    return result,receipt
