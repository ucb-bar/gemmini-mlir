"""Isolated variable-partition adapter over existing shared RHS retention owner."""
from dataclasses import asdict
import hashlib
import json

from merlin.compile.scheduling.compact_dense import (
    DenseLoopStorage, DenseMemoryView, emit_compact_dense)
from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
from merlin.compile.scheduling.dense_panel_pipeline import (
    DenseRetainedReductionAdmission, dense_retained_operand_hoists)
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import (
    make_dense_schedule_plan, dense_schedule_text, parse_dense_schedule_plan)
from merlin.transforms.dense_planning import DenseResources, dense_blocks
from gemmini_dense_readout import ReadoutUnavailable


def plan_capacity_panels(incumbent, adapter, problem, geometry, facts_path, *,
                    bias=False, initial=False, outer_order=('m', 'n'), lhs_partition_rows=None):
    """Return (checked result, receipt), preserving the original adapter/terminal.

    Retain the incumbent M/N block, expand K to its complete source extent,
    execute output panels in N/M order, and hoist RHS loads out of repeated M.
    The complete retained panel must fit its original disjoint operand half.
    Unsupported domains refuse before any emission; callers retain incumbents.
    """
    from gemmini_dense import seed_source
    from gemmini_dense_pipeline import ordering_premises
    incumbent.revalidate()
    if tuple(outer_order) != ('m', 'n'):
        raise ReadoutUnavailable('RHS panel experiment requires original M/N source owner')
    if bool(problem.bias) != bool(bias or initial):
        raise ReadoutUnavailable('RHS panel initializer differs from original source')
    if (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4):
        raise ReadoutUnavailable('RHS panel source requires signed i8 operands and i32 sums')
    edge, operand, operand_rows, accumulator, accumulator_rows = geometry
    split = operand_rows // 2 if lhs_partition_rows is None else lhs_partition_rows
    if type(split) is not int or split < edge or split >= operand_rows or split % edge:
        raise ReadoutUnavailable('capacity partition must be a disjoint tile-aligned operand range')
    old = incumbent.choice.block
    # A remainder family has its own loads; only full groups share hoists.
    # This is reuse applicability, not a measured performance cutoff.
    if problem.tiles[0] // old.m < 2:
        raise ReadoutUnavailable('RHS panel requires at least two full original outer M groups')
    storage = DenseLoopStorage(
        DenseMemoryView('lhs', operand.name, operand.row_bytes, 0, split),
        DenseMemoryView('rhs', operand.name, operand.row_bytes, split, operand_rows-split),
        DenseMemoryView('acc', accumulator.name, accumulator.row_bytes, 0, accumulator_rows))
    # Capacity-only library witness is applied to a physical N panel, never
    # mislabeled as complete original matrix residency.
    panel = plan_dense_rhs_residency(problem.m, problem.k,
        min(problem.n, old.n * edge), block=edge, operand_rows=operand_rows,
        accumulator_rows=accumulator_rows, rhs_base=split, max_static_rhs_tiles=4096)
    if panel is None:
        raise ReadoutUnavailable('complete source K times incumbent N panel exceeds assigned RHS store')
    resources = DenseResources(operand_rows * operand.row_bytes,
        accumulator_rows * accumulator.row_bytes, max_m_blocks=old.m,
        max_k_blocks=problem.tiles[1], max_n_blocks=old.n,
        partial_output_tiles=True, partial_output_blocks=True)
    block = next((b for b in dense_blocks(problem, resources,
        k_limit=lambda m,n: storage.k_limit(problem, resources, m, n))
        if (b.m,b.k,b.n) == (old.m,problem.tiles[1],old.n)), None)
    if block is None:
        raise ReadoutUnavailable('full-K LHS/RHS panels do not fit original disjoint storage')
    if max(block.m*block.k*block.n, (block.m+block.n)*block.k) > 4096:
        raise ReadoutUnavailable('retained full-K panel exceeds finite compiler issue budget')
    flags, pending, receipt = ordering_premises(facts_path)
    required = ('ordered_accumulator_writes', 'accumulator_result_visibility',
        'ordered_configuration_capture', 'finite_queue_progress',
        'complete_device_drain', 'exclusive_private_state')
    if not all(flags[key] for key in required):
        raise ReadoutUnavailable('selected RHS panel commit/configuration/completion premises incomplete')
    identity = 'selected-command-ordering:' + receipt['ordering_sha256']
    law = DenseRetainedReductionAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity, identity+':retained-read-only-operands',
        4096, 4096, pending['execute']//2, pending['load'],
        max_issued_readouts=block.m*block.n, pending_readouts=pending['store'],
        readout_configuration='ordered_snapshot_before_store_issue')
    initializer = seed_source(problem.m, problem.n, bias=bias, initial=initial)
    source_identity = hashlib.sha256(json.dumps(dict(
        source='row-major-source-ordered-dense', problem=asdict(problem),
        initializer=None if initializer is None else asdict(initializer),
        numerical_policy=adapter.numerical_policy_identity), sort_keys=True).encode()).hexdigest()
    order = ('n', 'm')
    raw = emit_compact_dense(problem, block, storage, adapter, name='dense_rhs_panels',
        dtypes=('i8','i8','i32'), outer_order=order, bias_source=initializer)
    transforms = dense_retained_operand_hoists(raw, adapter, role='weight')
    if not transforms:
        raise ReadoutUnavailable('shared owner found no invariant RHS output-group loads')
    plan = make_dense_schedule_plan(problem, resources, storage, adapter,
        source_identity=source_identity, dtypes=('i8','i8','i32'), static_budget=256,
        block_tiles=(block.m,block.k,block.n), outer_order=order,
        loop_transforms=transforms, pipeline_kind='retained_full_k',
        pipeline_prefetch=False, bias_source=initializer)
    text = dense_schedule_text(plan)
    result = schedule_dense_pipeline(problem, resources, storage, adapter,
        name='dense_rhs_panels', source_identity=source_identity, dtypes=('i8','i8','i32'),
        static_budget=256, external_buffers_disjoint=True,
        schedule_plan=parse_dense_schedule_plan(text), pipeline_admission=law,
        batch_readouts=True, bias_source=initializer, timeout_ms=60000)
    result.revalidate()
    receipt.update(lhs_partition_rows=split, mode='rhs_panels', library='retained_full_k/dense_retained_operand_hoists(weight)',
        source_identity=source_identity, block=(block.m,block.k,block.n),
        outer_order=order, panel_residency=asdict(panel),
        panel_rhs_rows=panel.rhs_rows, retained_lhs_rows=block.m*block.k*edge,
        original_incumbent_sha256=incumbent.kernel.digest(),
        unhoisted_schedule_sha256=raw.digest(), transformed_schedule_sha256=result.kernel.digest(),
        loop_transforms=[asdict(t) for t in transforms], schedule_ir=text,
        source_and_epochs=result.realization, qualification_identity=None,
        timing='UNKNOWN; no cost ranking inferred from load counts')
    return result, receipt
