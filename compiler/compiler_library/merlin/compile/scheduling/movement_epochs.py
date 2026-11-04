"""Exact flat byte movement through bounded disjoint private tile epochs.

This is a two-phase movement consumer, not a loaded-sum contraction. The target
binds the original span pointers and exact signed-byte/zero transport policy.
No inter-queue overlap or same-cell pending ordering is granted: all loads drain
before readouts, and all readouts drain before any private tile is reused.
"""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class MovementEpochLaw:
    hardware_identity: str
    numerical_identity: str
    qualification_identity: str
    disjoint_loads_pending: bool = False
    disjoint_stores_pending: bool = False
    immutable_completed_configuration: bool = False
    finite_queue_progress: bool = False
    complete_device_drain: bool = False
    exclusive_private_state: bool = False

    def revalidate(self):
        if any(type(v) is not str or not v for v in (
            self.hardware_identity, self.numerical_identity, self.qualification_identity)):
            raise ValueError('movement epochs require original hardware/numeric/owner identities')
        if any(getattr(self, name) is not True for name in (
            'disjoint_loads_pending', 'disjoint_stores_pending',
            'immutable_completed_configuration', 'finite_queue_progress',
            'complete_device_drain', 'exclusive_private_state')):
            raise ValueError('movement epochs require complete original ordering and ownership premises')


@dataclass(frozen=True)
class MovementEpochPlan:
    elements: int
    source_kind: str
    tile: int
    accumulator_rows: int
    load_issue_bound: int
    store_issue_bound: int
    law: MovementEpochLaw

    def __post_init__(self):
        if (type(self.law) is not MovementEpochLaw or self.source_kind not in ('copy', 'zero')
                or any(type(v) is not int or not 0 < v < (1 << 62) for v in (
                    self.elements, self.tile, self.accumulator_rows,
                    self.load_issue_bound, self.store_issue_bound))
                or self.accumulator_rows < self.tile
                or self.elements + self.tile * self.tile >= 1 << 63):
            raise ValueError('movement epochs need an exact bounded source span and private tile capacity')
        self.law.revalidate()

    @property
    def live_tiles(self):
        return min(self.accumulator_rows // self.tile,
                   self.load_issue_bound, self.store_issue_bound)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def report(self):
        self.__post_init__()
        return dict(schema='merlin.byte_movement_epochs.v1', plan_sha256=self.identity,
            elements=self.elements, source_kind=self.source_kind, live_tiles=self.live_tiles,
            private_rows=[0, self.live_tiles * self.tile],
            source_intervals='complete contiguous span, each byte loaded or zeroed and stored once',
            phase_order=['load', 'complete', 'store', 'complete'], law=asdict(self.law))


def plan_movement_epochs(elements, source_kind, *, tile, accumulator_rows,
                         load_issue_bound, store_issue_bound, law):
    """Check all premises before emitting or configuring the target."""
    return MovementEpochPlan(elements, source_kind, tile, accumulator_rows,
                             load_issue_bound, store_issue_bound, law)


def emit_movement_epochs(plan, builder, lower):
    """Lower every source rectangle and real completion through one target seam.

    ``builder`` supplies counted loop/op; lower receives load/store/complete.
    Configuration must already be completed and immutable. Rectangle offsets
    are source-span byte offsets, while private rows are phase-local slots.
    The target must implement complete as a real whole-device drain.
    """
    if (type(plan) is not MovementEpochPlan or not callable(lower)
            or any(not callable(getattr(builder, name, None)) for name in ('loop', 'op'))):
        raise ValueError('movement epochs require a checked plan, counted builder and target seam')
    plan.__post_init__()
    identity = plan.identity
    tile, live = plan.tile, plan.live_tiles
    size = tile * tile
    full, remainder = divmod(plan.elements, size)
    groups, tail = divmod(full, live)

    def group(first, count):
        for kind in ('load', 'store'):
            def slot(index):
                lower(kind, dict(offset=builder.op('add', first, builder.op('mul', index, size)),
                    local_row=builder.op('mul', index, tile), rows=tile, columns=tile))
            builder.loop(0, count, 1, slot)
            lower('complete', dict(phase=kind))

    if groups:
        builder.loop(0, groups, 1,
            lambda index: group(builder.op('mul', index, live * size), live))
    # At most live-1 full tiles and two exact ragged rectangles remain. Keep
    # their complete source intervals explicit; no masked/uninitialized lane
    # is stored. These bounded tails can occupy successive private tile slots.
    rectangles = [(groups * live * size + i * size, tile, tile) for i in range(tail)]
    rows, columns = divmod(remainder, tile)
    if rows:
        rectangles.append((full * size, rows, tile))
    if columns:
        rectangles.append((full * size + rows * tile, 1, columns))
    for first in range(0, len(rectangles), live):
        batch = rectangles[first:first + live]
        for kind in ('load', 'store'):
            for index, (offset, rows, columns) in enumerate(batch):
                lower(kind, dict(offset=offset, local_row=index * tile,
                    rows=rows, columns=columns))
            lower('complete', dict(phase=kind))
    plan.__post_init__()
    if plan.identity != identity:
        raise ValueError('movement source plan changed during emission')
    return plan.report()
