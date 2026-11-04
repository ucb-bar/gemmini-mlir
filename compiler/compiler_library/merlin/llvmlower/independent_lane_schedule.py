"""Order-only schedules for independent scalar SSA lanes.

The caller binds typed operation payloads to these nodes. This contract grants
no arithmetic rewrite, memory alias permission or target instruction capability.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LaneOperation:
    name: str
    operands: tuple[str, ...]
    result_type: str
    rounding: str


@dataclass(frozen=True)
class LaneEffects:
    nontrapping: bool
    stable_rounding: bool
    no_intermediate_flags_observation: bool
    no_memory_effects: bool


def independent_lane_schedule(operations, inputs, *, lanes, effects, stage_major=False):
    """Return (lane, operation-index) pairs preserving every lane's SSA order.

    Inputs are immutable values independently supplied for each lane; names
    cannot reference another lane. Sticky final flags may be observed: OR of
    the same nontrapping operations is independent of the interleaving.
    """
    if type(lanes) is not int or lanes <= 0 or type(stage_major) is not bool:
        raise ValueError("positive lane count and explicit schedule required")
    if not isinstance(effects, LaneEffects) or any(type(v) is not bool or not v for v in vars(effects).values()):
        raise ValueError("complete independent scalar effect contract required")
    operations, inputs = tuple(operations), tuple(inputs)
    if not operations or not inputs or any(not isinstance(x, str) or not x.isidentifier() for x in inputs):
        raise ValueError("nonempty scalar SSA body and named inputs required")
    known = set(inputs)
    if len(known) != len(inputs):
        raise ValueError("duplicate input")
    for op in operations:
        if not isinstance(op, LaneOperation) or not op.name.isidentifier() or op.name in known:
            raise ValueError("unique scalar SSA result required")
        if not op.result_type or op.rounding not in {"exact", "rne", "rup", "rdn", "rtz", "dynamic"}:
            raise ValueError("typed scalar result and explicit rounding required")
        if not op.operands or any(x not in known for x in op.operands):
            raise ValueError("forward, unknown or cross-lane dependency")
        known.add(op.name)
    if stage_major:
        return tuple((lane, index) for index in range(len(operations)) for lane in range(lanes))
    return tuple((lane, index) for lane in range(lanes) for index in range(len(operations)))
