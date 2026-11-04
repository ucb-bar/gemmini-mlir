"""Optional source-checked wide-column GEMV using the existing dense emitter.

Column limits are bounded alternatives, not a cycle ranking. All seed, reduction,
readout and terminal effects stay in the shared dense/panel-epoch implementation.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json

from merlin.compile.scheduling.compact_dense import (
    DenseLoopStorage, DenseMemoryView, schedule_compact_dense,
)
from merlin.transforms.dense_planning import DenseResources

COLUMN_LIMITS = (16, 32, 64)


def plan_gemv(incumbent, adapter, problem, geometry, facts_path, *,
              column_blocks=64, bias=False, initial=False, outer_order=('m', 'n')):
    """Return (checked panel-epoch result, receipt) for an original M=1 problem.

    Same inputs and output contract as gemmini_dense_pipeline.plan_pipeline.
    A caller must emit this returned artifact through consume_checked_dense with
    mode='panel_epochs', or the ordinary shared dense asynchronous terminal.
    Missing applicability raises; the caller retains its original schedule.
    """
    from gemmini_dense import seed_source
    from gemmini_dense_pipeline import plan_pipeline
    if type(column_blocks) is not int or column_blocks not in COLUMN_LIMITS:
        raise ValueError('GEMV column block limit must be 16, 32 or 64')
    edge, operand, operand_rows, accumulator, accumulator_rows = geometry
    if (problem.m != 1 or (problem.tile_m, problem.tile_k, problem.tile_n) != (edge, edge, edge)
            or (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4)
            or problem.bias != bool(bias or initial)):
        raise ValueError('wide GEMV requires the original signed-i8 M=1 dense problem and seed contract')
    if tuple(outer_order) not in (('m', 'n'), ('n', 'm')):
        raise ValueError('GEMV preserves an explicit shared dense output-axis order')
    incumbent.revalidate()
    split = operand_rows // 2
    storage = DenseLoopStorage(
        DenseMemoryView('lhs', operand.name, operand.row_bytes, 0, split),
        DenseMemoryView('rhs', operand.name, operand.row_bytes, split, operand_rows-split),
        DenseMemoryView('acc', accumulator.name, accumulator.row_bytes, 0, accumulator_rows))
    # The shared planner may select a smaller block for tails or capacities.
    # Each complete N block needs two RHS K panels and one ACC tile per column.
    resources = DenseResources(operand_rows*operand.row_bytes,
        accumulator_rows*accumulator.row_bytes, max_m_blocks=1, max_k_blocks=1,
        max_n_blocks=column_blocks, partial_output_tiles=True, partial_output_blocks=True)
    compact = schedule_compact_dense(problem, resources, storage, adapter,
        name='gemv_wide', dtypes=('i8', 'i8', 'i32'), static_budget=128,
        external_buffers_disjoint=True, outer_order=tuple(outer_order),
        verification_timeout_ms=20000,
        bias_source=seed_source(problem.m, problem.n, bias=bias, initial=initial))
    compact.revalidate()
    block = compact.choice.block
    if (block.m != 1 or block.k != 1 or block.n > column_blocks
            or 2*block.m*edge > split or 2*block.n*edge > operand_rows-split
            or block.m*block.n*edge > accumulator_rows):
        raise ValueError('wide GEMV does not fit the selected two-panel/ACC capacities')
    checked, receipt = plan_pipeline(compact, adapter, problem, geometry, facts_path,
        mode='panel_epochs', bias=bias, initial=initial, outer_order=tuple(outer_order))
    checked.revalidate()
    receipt.update(gemv=dict(requested_column_blocks=column_blocks,
        actual_block=asdict(block), original_problem=asdict(problem),
        selected_resources=dict(operand_rows=operand_rows, accumulator_rows=accumulator_rows,
            lhs_double_window_rows=2*block.m*edge, rhs_double_window_rows=2*block.n*edge,
            accumulator_live_rows=block.m*block.n*edge),
        incumbent_block=asdict(incumbent.choice.block),
        initializer=dict(bias=bool(bias), initial=bool(initial)),
        execution_qualification='NOT_ESTABLISHED',
        terminal='unchanged gemmini_dense.dense_terminal via consume_checked_dense'))
    receipt['gemv']['source_contract_sha256'] = hashlib.sha256(json.dumps(dict(
        problem=asdict(problem), numerical_policy=adapter.numerical_policy_identity,
        initializer=receipt['gemv']['initializer']), sort_keys=True).encode()).hexdigest()
    return checked, receipt
