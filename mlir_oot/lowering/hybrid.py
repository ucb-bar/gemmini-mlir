"""Splitting a program into the alternating MESH and OFF-MESH stages one kernel issues.

A capsule may interleave a region this target's readout cannot execute with a contraction that
belongs on its systolic array. `lanes.NON_READOUT_FAMILIES` records WHY an off-mesh region is
off-mesh -- `AccumulatorScale` gates its normalisation paths on `has_normalizations`, which this
elaborated design leaves at its default, so no reciprocal square root and no row-wise exponential
exist on the store path.

Neither whole-program route answers such a capsule. Running it all on the scalar lane computes the
right numbers while the array issues nothing, which is not what a capsule asking for the array's own
instruction classes is asking for; lowering it all to the mesh has no encoding for the off-mesh
stage. This module states the third route: ONE kernel whose command stream has HOLES in it, each
hole an off-mesh stage the code generator expands into scalar blocks in place.

The split is read off the program's own operation order -- consecutive operations that belong on the
same lane are one stage -- so it is the capsule's dataflow that decides the shape of the kernel, not
a table of capsule names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.workload import Op, Workload
from .lanes import NON_READOUT_FAMILIES

#: Operations that carry no command of their own: a residency pack/evict is bookkeeping the
#: contraction it serves already accounts for. They never START a stage, they join the one running.
NEUTRAL_KINDS = frozenset({"resident_pack", "evict", "matmul"})

#: Stages a REWRITE introduced rather than the capsule: splitting a wide operand into digits the
#: mesh's operand port encodes, and recombining the partial products. Integer bookkeeping at the
#: datapath's edge, which is the scalar lane's job exactly as a normalisation is.
STAGING_KINDS = frozenset({"int_digits", "digit_combine", "row_weight", "row_normalize",
                            "softmax_weighted_sum"})

#: Every region only the scalar lane can carry.
LANE_KINDS = NON_READOUT_FAMILIES | STAGING_KINDS

MESH, SCALAR = "mesh", "scalar"


@dataclass
class Stage:
    """One run of consecutive operations that belong on the same lane."""

    lane: str
    ops: list[Op] = field(default_factory=list)


def lane_of(op: Op) -> str:
    """Which lane ``op``'s region belongs on, read off the interface mnemonic alone."""
    return SCALAR if op.kind in LANE_KINDS else MESH


def stages(wl: Workload) -> list[Stage]:
    """``wl.ops`` grouped into consecutive same-lane runs, in program order."""
    out: list[Stage] = []
    for op in wl.ops:
        if op.kind in NEUTRAL_KINDS and out:
            out[-1].ops.append(op)
            continue
        lane = lane_of(op)
        if out and out[-1].lane == lane:
            out[-1].ops.append(op)
        else:
            out.append(Stage(lane, [op]))
    return out


def scalar_stages(wl: Workload) -> list[Stage]:
    """Only the off-mesh stages, in the order their markers appear in the command stream."""
    return [s for s in stages(wl) if s.lane == SCALAR]


def is_hybrid(wl: Workload) -> bool:
    """True when this program needs BOTH lanes inside one kernel.

    Both halves have to carry real work: a program whose only mesh stage is a residency pack has
    nothing on the array, and one with no off-mesh stage is an ordinary mesh program.
    """
    seen = stages(wl)
    has_scalar = any(s.lane == SCALAR for s in seen)
    has_mesh = any(
        s.lane == MESH and any(o.kind not in NEUTRAL_KINDS for o in s.ops) for s in seen
    )
    return has_scalar and has_mesh
