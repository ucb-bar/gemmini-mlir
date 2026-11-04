"""Semantic admission before target emission.

Integer shift requantization is distinct from floating-point readout scaling.
The selected device route has no admitted implementation for that stage.
"""
from __future__ import annotations

import copy


def explicit_decline(command_buffer):
    for command in command_buffer.get("commands", []):
        attrs = command.get("attributes") or {}
        stages = attrs.get("epilogue", []) or []
        if isinstance(stages, str):
            stages = [stages]
        if "requant" not in [str(stage).lower() for stage in stages]:
            continue
        declined = copy.deepcopy(command_buffer)
        declined["commands"] = []
        declined["params"] = {}
        declined["declined"] = {
            "op": "requant",
            "reason": (
                "Integer shift requantization is not implemented by the selected "
                "accelerator route. Floating-point accumulator readout uses "
                "round-to-nearest-even and cannot substitute for the declared "
                "integer rounding and shift semantics. No target work was emitted."
            ),
        }
        return declined
    return None


def declare_interface_placement(command_buffer):
    """Declare executable interface regions that have no linalg source plan.

    Resident handles and deferred accumulator expressions are storage/control
    nodes. Their compute is emitted by COMMIT, so count that region once.
    """
    params = command_buffer.setdefault("params", {})
    if "lane_placement" in params:
        return
    regions = []
    for index, command in enumerate(command_buffer.get("commands", [])):
        opcode = command.get("opcode")
        attrs = command.get("attributes") or {}
        if opcode in ("RES_PACK", "MATMUL", "MATMUL_RESIDENT", "EVICT",
                      "BARRIER", "FENCE", "SYNC"):
            continue
        host = opcode == "HOST_REGION" or (
            opcode == "VECTOR_MAP" and attrs.get("op") == "host_region")
        regions.append({
            "lane": "host" if host else "unit",
            "family": attrs.get("op", opcode),
            "reason": (
                "Explicit interface region lowered through shared host tensor code"
                if host else
                "Explicit interface region lowered through accelerator DMA/compute/readout"
            ) + f"; instruction_index={index}",
        })
    params["lane_placement"] = regions
