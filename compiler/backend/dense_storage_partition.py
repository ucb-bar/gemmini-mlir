"""Physical full-K panel partitions for the existing retained dense pipeline.

Capacity is evaluated with simultaneously live padded LHS and RHS rows. This
module selects storage and a shared DenseBlock only; it introduces no command,
completion, array-persistence or source-semantic law.
"""
from __future__ import annotations

from dataclasses import replace

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks

from . import isa


def select_full_k_partition(problem, resources, *, max_m_blocks=None, max_n_blocks=None):
    """Return ``(storage, block, receipt)`` or None when no full-K panel fits.

    Resources describe available bytes after external reservations. This
    standalone dense route owns a contiguous scratchpad prefix and a separate
    accumulator prefix; callers with other placements must keep their original
    planner. Geometry bounds narrow the supplied resources and never enlarge
    their declared candidate domain. The original resources remain unchanged.
    """
    if type(problem) is not DenseProblem or type(resources) is not DenseResources:
        raise ValueError('full-K partition requires typed dense problem and resources')
    if (problem.tile_m != isa.DIM or problem.tile_k != isa.DIM
            or problem.tile_n not in (1, isa.DIM)
            or (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4)):
        return None
    bounds = []
    for supplied, existing in ((max_m_blocks, resources.max_m_blocks),
                               (max_n_blocks, resources.max_n_blocks)):
        if supplied is not None and (type(supplied) is not int or supplied <= 0):
            raise ValueError('full-K partition geometry bounds must be positive integers')
        bounds.append(existing if supplied is None else min(supplied, existing))
    selected_resources = replace(resources, max_m_blocks=bounds[0], max_n_blocks=bounds[1])
    dim = isa.DIM
    # Actual scratchpad rows are DIM bytes even for a one-column readout tile;
    # the accumulator has DIM i32 lanes per physical row.
    operand_rows = min(isa.SPAD_ROWS, resources.operand_bytes // dim)
    accumulator_rows = min(isa.ACC_ROWS, resources.accumulator_bytes // (4 * dim))
    if operand_rows <= 0 or accumulator_rows <= 0:
        return None
    candidates = dense_blocks(problem, selected_resources,
        k_limit=lambda bm, bn: operand_rows // (bm * problem.tile_m + bn * problem.tile_k))
    admitted = []
    for block in candidates:
        if block.k != problem.tiles[1]:
            continue
        lhs_rows = block.m * block.k * problem.tile_m
        rhs_rows = block.k * block.n * problem.tile_k
        acc_rows = block.m * block.n * problem.tile_m
        if lhs_rows + rhs_rows > operand_rows or acc_rows > accumulator_rows:
            continue
        storage = DenseLoopStorage(
            DenseMemoryView('lhs', 'spad', dim, 0, lhs_rows, alignment=dim),
            DenseMemoryView('rhs', 'spad', dim, lhs_rows, operand_rows-lhs_rows, alignment=dim),
            DenseMemoryView('acc', 'acc', 4*dim, 0, accumulator_rows, alignment=dim))
        if storage.admits(problem, block, selected_resources):
            admitted.append((block, storage, lhs_rows, rhs_rows, acc_rows))
    if not admitted:
        return None
    block, storage, lhs_rows, rhs_rows, acc_rows = min(admitted,
        key=lambda candidate: (candidate[0].loads, -candidate[0].m*candidate[0].n,
                               candidate[0].m, candidate[0].n))
    receipt = dict(selection='adaptive', primitive_tiles=list(problem.tiles),
        geometry_bounds=list(bounds), selected_block_tiles=[block.m, block.k, block.n],
        physical_rows=dict(lhs=lhs_rows, rhs=rhs_rows, accumulator=acc_rows),
        assigned_windows=dict(lhs=[0, lhs_rows], rhs=[lhs_rows, operand_rows],
                              accumulator=[0, accumulator_rows]),
        capacity_rows=dict(operand=operand_rows, accumulator=accumulator_rows),
        unused_operand_rows=operand_rows-lhs_rows-rhs_rows,
        requested_operand_bytes=block.requested_operand_bytes,
        source_bias=problem.bias,
        legality='simultaneous padded LHS/RHS and private ACC footprints; disjoint backing intervals',
        execution='original retained source, ownership drains and terminal encoder required')
    return storage, block, receipt
