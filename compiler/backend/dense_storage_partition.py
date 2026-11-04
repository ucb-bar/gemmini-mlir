"""Physical full-K and segmented partitions for checked retained pipelines.

Capacity includes every simultaneously live padded operand window. This
module selects storage and a shared DenseBlock only; it introduces no command,
completion, array-persistence or source-semantic law.
"""
from __future__ import annotations

from dataclasses import replace

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks

from . import isa


def select_segmented_partition(problem, resources):
    """Assign full LHS and two RHS slots before selecting segmented geometry.

    A fixed half-store split wastes rows for skinny M panels. Enumerate only
    the existing bounded M/N candidate domain, placing each complete LHS at
    row zero and the two alternating RHS slots in the remaining suffix. The
    shared segmented planner supplies its own physical K limit and validates
    both slots, including padding, alignment and accumulator ownership.
    """
    from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
    if type(problem) is not DenseProblem or type(resources) is not DenseResources:
        raise ValueError('segmented partition requires typed dense problem and resources')
    if (problem.tile_m != isa.DIM or problem.tile_k != isa.DIM
            or problem.tile_n not in (1, isa.DIM)
            or (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4)
            or problem.tiles[1] < 2):
        return None
    dim = isa.DIM
    operand_rows = min(isa.SPAD_ROWS, resources.operand_bytes // dim)
    accumulator_rows = min(isa.ACC_ROWS, resources.accumulator_bytes // (4 * dim))
    admitted = []
    for bm in range(1, min(problem.tiles[0], resources.max_m_blocks) + 1):
        lhs_rows = bm * problem.tiles[1] * dim
        if lhs_rows + 2 * dim > operand_rows or accumulator_rows < dim:
            continue
        storage = DenseLoopStorage(
            DenseMemoryView('lhs', 'spad', dim, 0, lhs_rows, alignment=dim),
            DenseMemoryView('rhs', 'spad', dim, lhs_rows, operand_rows-lhs_rows, alignment=dim),
            DenseMemoryView('acc', 'acc', 4*dim, 0, accumulator_rows, alignment=dim))
        for block in dense_segmented_blocks(problem, resources, storage):
            if block.m == bm:
                admitted.append((block, storage))
    if not admitted:
        return None
    # Preserve the existing geometry preference. For a fixed output geometry,
    # larger K segments eliminate ownership drains without increasing traffic.
    block, storage = min(admitted,
        key=lambda item: (item[0].loads, -item[0].m*item[0].n,
                          -item[0].k, item[0].m, item[0].n))
    rhs_slot_rows = block.k * block.n * dim
    receipt = dict(selection='adaptive_segmented', primitive_tiles=list(problem.tiles),
        selected_block_tiles=[block.m, block.k, block.n],
        physical_rows=dict(lhs=storage.lhs.capacity, rhs_slot=rhs_slot_rows,
                           rhs_slots=2, accumulator=block.m*block.n*dim),
        assigned_windows=dict(lhs=[0, storage.lhs.capacity],
            rhs=[storage.rhs.base, storage.rhs.base+storage.rhs.capacity],
            accumulator=[0, accumulator_rows]),
        capacity_rows=dict(operand=operand_rows, accumulator=accumulator_rows),
        unused_operand_rows=operand_rows-storage.lhs.capacity-2*rhs_slot_rows,
        legality='shared segmented full-LHS/two-RHS physical assignment',
        execution='original segmented source, combined epoch law and terminal encoder required')
    return storage, block, receipt


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
