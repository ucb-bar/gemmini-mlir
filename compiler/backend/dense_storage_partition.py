"""Physical operand partitions for the existing checked dense pipelines.

Capacity is evaluated with simultaneously live padded LHS and RHS rows. This
module selects storage and a shared DenseBlock only; it introduces no command,
completion, array-persistence or source-semantic law.
"""
from __future__ import annotations

from dataclasses import replace

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks

from . import isa


def _rank(block):
    return block.loads, -block.m * block.n, block.m, block.n


def half_storage(resources):
    """The incumbent selected physical windows, limited by caller resources."""
    dim = isa.DIM
    rows = min(isa.SPAD_ROWS, resources.operand_bytes // dim)
    acc = min(isa.ACC_ROWS, resources.accumulator_bytes // (4 * dim))
    middle = rows // 2
    return DenseLoopStorage(
        DenseMemoryView('lhs', 'spad', dim, 0, middle),
        DenseMemoryView('rhs', 'spad', dim, middle, rows-middle),
        DenseMemoryView('acc', 'acc', 4*dim, 0, acc))


def _preserve_incumbent(problem, resources, selected, *, segmented):
    """Keep the original physical assignment when extra space buys no block.

    An address-only relocation adds no source reuse and can change bank traffic.
    The alternative must improve the same existing geometric candidate ordering.
    """
    if selected is None:
        return None
    storage, block, receipt = selected
    incumbent = half_storage(resources)
    if segmented:
        from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
        candidates = dense_segmented_blocks(problem, resources, incumbent)
    else:
        candidates = tuple(b for b in dense_blocks(problem, resources,
            k_limit=lambda m, n: incumbent.k_limit(problem, resources, m, n))
            if b.k == problem.tiles[1] and incumbent.admits(problem, b, resources))
    if candidates:
        old = min(candidates, key=_rank)
        if _rank(old) <= _rank(block):
            return incumbent, old, dict(selection='adaptive', applied=False,
                reason='incumbent half windows already provide the selected reuse geometry',
                selected_block_tiles=[old.m, old.k, old.n],
                lhs_window_rows=incumbent.lhs.capacity,
                rhs_window_rows=incumbent.rhs.capacity)
    return storage, block, dict(receipt, applied=True)


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
    return _preserve_incumbent(problem, selected_resources,
        (storage, block, receipt), segmented=False)


def select_segmented_partition(problem, resources):
    """Reserve full live LHS rows and give two RHS slots the legal remainder.

    This only chooses original DenseMemoryViews and library DenseBlocks. The
    library's dense_segmented_blocks calls its exact _assignment validator for
    every proposal: complete retained LHS, both disjoint RHS slots, alignment,
    physical bytes, private accumulator rows, and source/geometry constraints.
    The ordinary combined execution admission and source proof remain required.
    """
    from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
    if type(problem) is not DenseProblem or type(resources) is not DenseResources:
        raise ValueError('segmented partition requires typed dense problem and resources')
    if (problem.tile_m != isa.DIM or problem.tile_k != isa.DIM
            or problem.tile_n not in (1, isa.DIM)
            or (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4)):
        return None
    dim = isa.DIM
    rows = min(isa.SPAD_ROWS, resources.operand_bytes // dim)
    acc = min(isa.ACC_ROWS, resources.accumulator_bytes // (4*dim))
    admitted = []
    for bm in range(1, min(problem.tiles[0], resources.max_m_blocks)+1):
        lhs_rows = bm * problem.tiles[1] * problem.tile_m
        if lhs_rows >= rows or acc <= 0:
            continue
        storage = DenseLoopStorage(
            DenseMemoryView('lhs', 'spad', dim, 0, lhs_rows, alignment=dim),
            DenseMemoryView('rhs', 'spad', dim, lhs_rows, rows-lhs_rows, alignment=dim),
            DenseMemoryView('acc', 'acc', 4*dim, 0, acc, alignment=dim))
        for block in dense_segmented_blocks(problem, replace(resources, max_m_blocks=bm), storage):
            if block.m == bm:
                admitted.append((storage, block))
    if not admitted:
        return None
    storage, block = min(admitted, key=lambda pair:_rank(pair[1]))
    lhs_rows = block.m * problem.tiles[1] * problem.tile_m
    slot_rows = block.k * block.n * problem.tile_k
    receipt = dict(selection='adaptive', family='retained_lhs_segmented_k',
        primitive_tiles=list(problem.tiles), selected_block_tiles=[block.m, block.k, block.n],
        physical_rows=dict(lhs=lhs_rows, rhs_slot=slot_rows, rhs_slots=2,
                           accumulator=block.m*block.n*problem.tile_m),
        assigned_windows=dict(lhs=[0,lhs_rows], rhs=[lhs_rows,rows], accumulator=[0,acc]),
        live_windows=dict(lhs=[0,lhs_rows], rhs_slots=[
            [lhs_rows,lhs_rows+slot_rows], [lhs_rows+slot_rows,lhs_rows+2*slot_rows]]),
        capacity_rows=dict(operand=rows, accumulator=acc),
        unused_operand_rows=rows-lhs_rows-2*slot_rows,
        legality='library dense_segmented_blocks exact full-LHS/two-RHS assignment; disjoint aligned windows',
        execution='unchanged combined ordered reduction/future-load law and full ownership drains')
    return _preserve_incumbent(problem, resources, (storage,block,receipt), segmented=True)
