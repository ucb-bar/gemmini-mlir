"""A self-contained structural validator for the command buffer.

The package ships no copy of the bench's JSON schema and imports nothing from the harness, so this
mirrors the schema's CLOSED vocabularies (opcode enum, epilogue-stage enum, role enum) and its
route rule: a buffer either declines, or declares a host lane with its outputs, or emits commands.
"""

from __future__ import annotations

from typing import Any

OPCODES = frozenset(
    (
        "RES_PACK", "MATMUL_RESIDENT", "MATMUL", "COMMIT", "EVICT", "VECTOR_MAP", "VREDUCE",
        "RMSNORM", "ATTENTION_QK", "BATCHED_MATMUL", "LAYERNORM", "SOFTMAX", "GELU", "SOFTCAP",
        "GEGLU", "ATTENTION_FULL", "ROPE", "CONV", "MATMUL_BATCHED", "BIAS_ADD", "RESIDUAL_ADD",
        "CONV2D", "MOVEMENT", "ATTENTION_PV", "K_CHAIN", "DEPTHWISE_CONV2D",
    )
)
EPILOGUE_STAGES = frozenset(("bias_add", "bias", "requant", "acc_scale", "relu", "maxpool"))
ROLES = frozenset(("input", "weight", "bias", "output", "scale", "intermediate"))


def problems(cb: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k in ("abi_version", "target", "commands"):
        if k not in cb:
            out.append(f"missing required top-level field {k!r}")
    tensors = cb.get("tensors") or {}
    for name, spec in tensors.items():
        if "shape" not in spec or "dtype" not in spec:
            out.append(f"tensor {name!r}: shape and dtype are required")
        if spec.get("role") and spec["role"] not in ROLES:
            out.append(f"tensor {name!r}: role {spec['role']!r} is not in the ABI's role set")
        for d in spec.get("shape", []):
            if not isinstance(d, int) or d < 1:
                out.append(f"tensor {name!r}: extent {d!r} is not a positive integer")
    commands = cb.get("commands", [])
    for i, c in enumerate(commands):
        if c.get("opcode") not in OPCODES:
            out.append(f"commands[{i}]: opcode {c.get('opcode')!r} is not in the ABI")
        for key in c:
            if key not in ("opcode", "operands", "attributes"):
                out.append(f"commands[{i}]: unexpected key {key!r}")
        for stage in (c.get("attributes") or {}).get("epilogue", []):
            if stage not in EPILOGUE_STAGES:
                out.append(f"commands[{i}]: epilogue stage {stage!r} is not in the ABI vocabulary")
        if "declined" not in cb:
            for role, operand in (c.get("operands") or {}).items():
                if operand not in tensors and not operand.startswith(("res", "acc")):
                    out.append(f"commands[{i}]: operand {role}={operand!r} names no declared tensor")
    if "declined" in cb:
        if not cb["declined"].get("reason"):
            out.append("declined: a reason is required")
        if commands:
            out.append("a buffer may not both decline and emit commands")
    elif not commands:
        outputs = [n for n, s in tensors.items() if s.get("role") == "output"]
        if not outputs:
            out.append(
                "an empty command list with no `declined` and no declared output is a program "
                "dropped on the floor, not a route"
            )
    return out
