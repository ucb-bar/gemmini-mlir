"""The routing plan this package publishes as ``params.lane_placement``.

WHERE each region of a program ran is a statement only the compiler can make, and it is the
statement a capsule asserting lane composition is graded against: ``lanes.require`` names lanes
that must have carried work, ``lanes.forbid`` names lanes that must have carried none. So the plan
is published for EVERY region the package places -- the ones it ACCEPTS onto the mesh as well as
the ones it refuses to the host lane -- rather than only for the refusals. A program that
accelerates its whole body and says nothing has an empty ledger, which reads as "no lane carried
anything" and is not what happened.

The lane keys are the routing plan's own vocabulary and the family names are the ABI's
(`contraction`, `attention`, `movement`, `elementwise_map`); both are read off the interface
operation that was placed, never off the capsule.
"""

from __future__ import annotations

from typing import Any

from ..ir.workload import Workload
from ..target import facts

#: The lane this target's systolic mesh is named by in the routing plan.
MESH_LANE = "on_mesh"


class MixedLaneProgram(Exception):
    """A program whose regions belong on different lanes, which this package cannot yet sequence."""

#: interface mnemonic -> the ABI's semantic family for the region it heads. ``commit`` and
#: ``evict`` are not region heads: a commit is the READOUT of the contraction that produced its
#: accumulator and is reported inside that region, and an evict releases a residency budget.
REGION_FAMILY = {
    "matmul": "contraction",
    "conv2d": "contraction",
    "matmul_batched": "contraction",
    "attention_qk": "attention",
    "attention_pv": "attention",
    "movement": "movement",
    "bias_add": "elementwise_map",
    "residual_add": "elementwise_map",
    "rmsnorm": "normalization",
    "softmax": "softmax",
    "rope": "elementwise_map",
}

#: Regions whose arithmetic this target's READOUT cannot execute. Derived from the RTL, not
#: assumed: `AccumulatorScale` gates its LAYERNORM / SOFTMAX / IGELU paths on `has_normalizations`,
#: and the elaborated design this backend compiles for leaves that parameter at its default -- the
#: normalizer is not instantiated, so the store path offers no reciprocal-square-root and no
#: row-wise exponential. Such a region is compiled onto the scalar lane and DECLARED there.
NON_READOUT_FAMILIES = {"rmsnorm", "softmax", "rope"}


def _mesh_reason() -> str:
    from .iface_to_gemmini import MESH_OPERAND_DTYPES, MESH_RESULT_DTYPES

    return (
        f"issued on this target's {facts.DIM}x{facts.DIM} systolic mesh: the RTL-derived datapath "
        f"reads {'/'.join(MESH_OPERAND_DTYPES)} operands (scratchpad UInt<8>) into an i32 "
        f"accumulator (AccumulatorMem SInt<32>) and reads out "
        f"{'/'.join(MESH_RESULT_DTYPES)}, which is the encoding this region's operands and "
        "declared readout are written in"
    )


def placement(wl: Workload, lane: str, reason: str) -> list[dict[str, Any]]:
    """One ledger entry per REGION of ``wl``, all of them placed on ``lane`` for ``reason``.

    A region is headed by the interface operation that carries its work; a ``commit`` is the
    READOUT of the contraction whose accumulator it consumes and is reported inside that region
    rather than as one of its own, and an ``evict`` releases a residency budget.
    """
    if not wl.ops:
        return []
    commit_of = {op.operands["src"]: op for op in wl.ops if op.kind == "commit"}
    entries: list[dict[str, Any]] = []
    for op in wl.ops:
        family = REGION_FAMILY.get(op.kind)
        if family is None:
            continue
        touched = [wl.residents.get(ref, ref) for ref in op.operands.values()]
        operations = 1
        if op.operands.get("rhs") in wl.residents or op.operands.get("weight") in wl.residents:
            operations += 1  # the resident pack that staged the stationary operand
        epilogue = op.epilogue
        if op.kind == "matmul":
            commit = commit_of.get(op.out)
            if commit is None:
                continue
            region = commit.out
            epilogue = commit.epilogue
            operations += 1  # the readout
        else:
            region = op.out
        touched.append(region)
        if epilogue is not None and epilogue.bias:
            touched.append(epilogue.bias)
        entries.append(
            {
                "region": region,
                "lane": lane,
                "family": family,
                "operations": operations,
                "dtypes": sorted({wl.tensors[t].dtype for t in touched if t in wl.tensors}),
                "reason": reason,
            }
        )
    return entries


def mesh_placement(wl: Workload) -> list[dict[str, Any]]:
    """One ledger entry per region this lowering ACCEPTED onto the mesh (admission route A)."""
    return placement(wl, MESH_LANE, _mesh_reason())



#: interface mnemonic -> the operand roles whose dtype the MESH must be able to encode. A bias
#: rides the accumulator preload rather than the operand store, so it is NOT in this list; the
#: entries mirror exactly the four sites the lowering itself checks.
MESH_OPERAND_ROLES = {
    "commit": ("lhs", "rhs"),
    "conv2d": ("ifm", "weight"),
    "matmul_batched": ("a", "w"),
    "attention_qk": ("q", "k"),
    "attention_pv": ("p", "v"),
}


def mesh_refusal(wl: Workload) -> str | None:
    """Why this mesh cannot encode some region of ``wl``, or ``None`` when it can encode them all.

    Asked BEFORE the interface->target rewrite runs, so a region the datapath has no encoding for
    is routed to the scalar lane whole rather than half-rewritten. The check itself is the
    lowering's own :func:`require_mesh_dtypes` -- there is one datapath predicate in this package,
    not two that can drift apart.
    """
    from .contraction import LoweringRefusal
    from .iface_to_gemmini import require_mesh_dtypes, require_movement_dtypes

    #: A program that INTERLEAVES an off-mesh stage with a mesh contraction is answered by
    #: neither whole-program route on its own, but it IS answered by one kernel whose command
    #: stream has holes in it -- see :mod:`mlir_oot.lowering.hybrid`. When that route applies, the
    #: off-mesh stages are not a refusal at all: they are stages this kernel runs off the array,
    #: and only the MESH stages have to satisfy the datapath check below.
    from . import hybrid

    mixed = hybrid.is_hybrid(wl)

    for op in wl.ops:
        if op.kind in NON_READOUT_FAMILIES:
            if mixed:
                continue    # runs off-mesh INSIDE the kernel, at its own lane_stage marker
            return (
                f"{op.kind} needs arithmetic this readout does not implement: AccumulatorScale "
                "gates its normalization paths on `has_normalizations`, which this elaborated "
                "design leaves at its default, so no normalizer is instantiated and the store "
                "path offers neither a reciprocal square root nor a row-wise exponential"
            )
        if op.kind == "movement":
            # Movement never enters the mesh, so it is asked its OWN datapath question: which
            # on-chip container can carry the trip. Same predicate the lowering uses.
            try:
                require_movement_dtypes(
                    wl.tensors[wl.residents.get(op.operands["src"], op.operands["src"])].dtype,
                    wl.tensors[op.out].dtype,
                )
            except LoweringRefusal as exc:
                return str(exc)
            continue
        roles = MESH_OPERAND_ROLES.get(op.kind)
        if roles is None:
            continue
        if op.kind == "commit":
            mm = next(
                (o for o in wl.ops if o.kind == "matmul" and o.out == op.operands["src"]), None
            )
            if mm is None:
                continue
            refs = [mm.operands.get("lhs"), mm.operands.get("rhs")]
        else:
            refs = [op.operands.get(r) for r in roles]
        operands = {}
        for role, ref in zip(roles, refs):
            name = wl.residents.get(ref, ref)
            if name in wl.tensors:
                operands[role] = wl.tensors[name].dtype
        result = wl.tensors[op.out].dtype if op.out in wl.tensors else None
        if result is None:
            continue
        try:
            require_mesh_dtypes(operands, result, op.kind)
        except LoweringRefusal as exc:
            return str(exc)
    return None


def host_placement(wl: Workload, reason: str) -> list[dict[str, Any]]:
    """Route H: one ledger entry per region this lowering refused to the scalar host lane."""
    return placement(wl, "host", reason)


def hybrid_placement(wl: Workload) -> list[dict[str, Any]]:
    """Route X: the ledger for a kernel that issues on the array AND runs stages off it.

    One entry per region, each naming the lane that region actually ran on -- the mesh stages with
    the datapath reason, the off-mesh stages with the reason the readout cannot execute them. A
    program that carries both and reports only one of them is not describing what it did.
    """
    from . import hybrid

    off_mesh_reason = (
        "this region needs arithmetic the readout does not implement: AccumulatorScale gates its "
        "normalization paths on `has_normalizations`, which this elaborated design leaves at its "
        "default, so no normalizer is instantiated and the store path offers neither a reciprocal "
        "square root nor a row-wise exponential. It is compiled into the SAME kernel as a scalar "
        "stage between the array's own commands, so the mesh still carries every region that "
        "belongs on it"
    )
    by_region: dict[str, dict[str, Any]] = {}
    for entry in placement(wl, MESH_LANE, _mesh_reason()):
        by_region[entry["region"]] = entry
    scalar_outs = {op.out for st in hybrid.scalar_stages(wl) for op in st.ops}
    for entry in by_region.values():
        if entry["region"] in scalar_outs:
            entry["lane"] = "host"
            entry["reason"] = off_mesh_reason
    return list(by_region.values())
