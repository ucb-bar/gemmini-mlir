"""Immutable geometry facts for existing, capacity-admitted dense candidates.

These are occupancy and source-work descriptions, not cycle estimates or a
schedule selector. Partial panels remain explicit. Neither hypothetical padded
work count below claims that the terminal encoder executes masked-off products.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from merlin.compile.scheduling.compact_dense import DenseLoopStorage
from merlin.transforms.dense_planning import DenseBlock, DenseProblem, DenseResources, dense_blocks

from . import isa


@dataclass(frozen=True)
class AxisPanelFacts:
    extent: int
    primitive: int
    primitive_tiles: int
    block_tiles: int
    panel_extent: int
    panels: int
    complete_panels: int
    tail_panels: int
    tile_complete_panels: int
    tile_tail_panels: int
    tile_tail_extent: int
    last_panel_extent: int
    last_panel_tiles: int
    primitive_tail_extent: int

    @property
    def primitive_aligned(self):
        return self.primitive_tail_extent == 0

    @property
    def panel_aligned(self):
        return self.tail_panels == 0


@dataclass(frozen=True)
class RowOccupancy:
    base: int
    live: int
    assigned: int
    unit_bytes: int

    @property
    def free(self):
        return self.assigned - self.live


@dataclass(frozen=True)
class DensePanelFacts:
    m: AxisPanelFacts
    k: AxisPanelFacts
    n: AxisPanelFacts
    lhs: RowOccupancy
    rhs: RowOccupancy
    accumulator: RowOccupancy
    full_k: bool
    bias: bool
    output_panels: int
    complete_output_panels: int
    m_tail_only_output_panels: int
    n_tail_only_output_panels: int
    both_tail_output_panels: int
    source_macs: int
    primitive_padded_macs: int
    panel_padded_macs: int
    source_output_elements: int
    primitive_padded_output_elements: int
    panel_padded_output_elements: int
    logical_lhs_loads: int
    logical_rhs_loads: int
    logical_bias_loads: int
    logical_output_tiles: int

    @property
    def primitive_aligned(self):
        return self.m.primitive_aligned and self.k.primitive_aligned and self.n.primitive_aligned

    @property
    def output_panels_aligned(self):
        return self.m.panel_aligned and self.n.panel_aligned

    @property
    def simultaneous_operand_rows(self):
        return self.lhs.live + self.rhs.live

    def as_dict(self):
        """Detached receipt data; mutations cannot change the typed facts."""
        return dict(asdict(self), primitive_aligned=self.primitive_aligned,
                    output_panels_aligned=self.output_panels_aligned,
                    simultaneous_operand_rows=self.simultaneous_operand_rows,
                    work_units='MACs; padded counts describe rectangular envelopes only',
                    load_counts='DenseBlock logical counts before schedule hoisting; not emitted DMA counts')


def _axis(extent, primitive, block_tiles):
    tiles = (extent + primitive - 1) // primitive
    span = primitive * block_tiles
    complete, remainder = divmod(extent, span)
    panels = complete + int(remainder != 0)
    tile_complete, tile_remainder = divmod(tiles, block_tiles)
    last_extent = extent - (panels - 1) * span
    return AxisPanelFacts(extent, primitive, tiles, block_tiles, span, panels,
        complete, int(remainder != 0), tile_complete, int(tile_remainder != 0),
        tile_remainder, last_extent, (last_extent + primitive - 1) // primitive,
        extent % primitive)


def describe_dense_panel(problem: DenseProblem, resources: DenseResources,
                         storage: DenseLoopStorage, block: DenseBlock, *, segmented=False) -> DensePanelFacts:
    """Describe one actual candidate, or raise ValueError before any mutation.

    Shared candidate enumeration and exact storage admission remain authoritative.
    ``complete_panels`` counts panels with a full scalar extent. In contrast,
    ``tile_complete_panels`` counts full runs of primitive tile descriptors; the
    last descriptor can still have a scalar tail. This distinction matters for
    M=1 and for grouped tails such as N tiles not divisible by seven.

    Live rows describe this one block, matching ``storage.row_counts``. Extra
    windows used by a prefetch/segmented pipeline require its own lifetime law;
    these geometry facts do not replace that law or certify such overlap.
    """
    if (type(problem) is not DenseProblem or type(resources) is not DenseResources
            or type(storage) is not DenseLoopStorage or type(block) is not DenseBlock):
        raise ValueError('panel facts require typed original dense planning objects')
    problem.__post_init__()
    resources.__post_init__()
    storage.__post_init__()
    if any(type(value) is not int or value < 0 for value in vars(block).values()):
        raise ValueError('panel block metadata must contain nonnegative integers')
    if any(type(value) is not int or value <= 0 for value in (block.m, block.k, block.n)):
        raise ValueError('panel facts require positive integer block extents')
    if storage.accumulator is None:
        raise ValueError('panel facts require a private accumulator assignment')
    views = (storage.lhs, storage.rhs, storage.accumulator)
    for view in views:
        view.__post_init__()
    if (problem.tile_m != isa.DIM or problem.tile_k != isa.DIM
            or problem.tile_n not in (1, isa.DIM)
            or (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4)
            or storage.lhs.backing != storage.rhs.backing
            or storage.accumulator.backing == storage.lhs.backing
            or tuple(view.unit_bytes for view in views) != (isa.DIM, isa.DIM, 4*isa.DIM)
            or any(view.base + view.capacity > isa.SPAD_ROWS for view in views[:2])
            or views[2].base + views[2].capacity > isa.ACC_ROWS):
        raise ValueError('panel assignment differs from selected dense physical geometry')
    if not storage.admits(problem, block, resources):
        raise ValueError('panel candidate exceeds or aliases its assigned physical storage')
    if segmented:
        # Segmented planning reserves two physical RHS slots. Its candidate
        # domain can choose a smaller K panel than the single-slot maximum.
        from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
        candidates = dense_segmented_blocks(problem, resources, storage)
    else:
        candidates = dense_blocks(problem, resources,
            k_limit=lambda bm, bn: storage.k_limit(problem, resources, bm, bn))
    if block not in candidates:
        raise ValueError('panel candidate is outside the exact shared planning domain')
    m, k, n = (_axis(extent, primitive, count) for extent, primitive, count in
        ((problem.m, problem.tile_m, block.m), (problem.k, problem.tile_k, block.k),
         (problem.n, problem.tile_n, block.n)))
    live_rows = storage.row_counts(problem, block)
    occupancies = tuple(RowOccupancy(view.base, live, view.capacity, view.unit_bytes)
                        for view, live in zip(views, live_rows))
    primitive_m = m.primitive_tiles * m.primitive
    primitive_k = k.primitive_tiles * k.primitive
    primitive_n = n.primitive_tiles * n.primitive
    panel_m, panel_k, panel_n = (axis.panels * axis.panel_extent for axis in (m, k, n))
    return DensePanelFacts(m, k, n, *occupancies,
        block.k == problem.tiles[1], problem.bias, m.panels*n.panels,
        m.complete_panels*n.complete_panels,
        m.tail_panels*n.complete_panels, m.complete_panels*n.tail_panels,
        m.tail_panels*n.tail_panels, problem.m*problem.k*problem.n,
        primitive_m*primitive_k*primitive_n, panel_m*panel_k*panel_n,
        problem.m*problem.n, primitive_m*primitive_n, panel_m*panel_n,
        block.lhs_loads, block.rhs_loads, block.bias_loads, block.output_tiles)
