"""Exact source binding and emitted-range plans for mixed whole programs.

Source ownership comes from the lowering's explicit mappings. Instruction ranges
come from actual emission. This module never guesses either from operation names,
provenance strings, or the presence of a source operation in the input.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

_LIBRARY = Path(__file__).resolve().parent / "compiler_library"
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))

from merlin.targetgen.oot_starterkit.plan import (
    source_operation_inventory, _source_plan_problems,
)


class AttributionError(ValueError):
    pass


def _source_key(origin: Mapping[str, Any] | None) -> tuple:
    if not isinstance(origin, Mapping):
        raise AttributionError("source value has no exact defining argument or operation result")
    if set(origin) == {"arg_index"} and type(origin["arg_index"]) is int:
        return ("arg", origin["arg_index"])
    if set(origin) == {"op_index", "result_index"} and all(type(origin[k]) is int for k in origin):
        return ("op", origin["op_index"], origin["result_index"])
    raise AttributionError(f"invalid source value identity: {origin!r}")


def _bindings(inventory: dict, cb: dict) -> tuple[list[str], list[dict], list[str]]:
    mapping = cb.get("params", {}).get("tensor_source_map")
    if not isinstance(mapping, Mapping):
        raise AttributionError("params.tensor_source_map is required; exact value identities cannot be inferred")
    by_source: dict[tuple, str] = {}
    source_values = []
    for name, origin in mapping.items():
        if name not in cb["tensors"]:
            raise AttributionError(f"source binding names an absent materialized tensor {name!r}")
        key = _source_key(origin)
        if key in by_source:
            raise AttributionError(f"source value {key!r} is bound to multiple materialized tensors")
        by_source[key] = name
        if key[0] == "op":
            source_values.append({"op_index": key[1], "result_index": key[2], "tensor": name})
    try:
        entries = [by_source[("arg", i)] for i in range(len(inventory["arguments"]))]
        outputs = [by_source[_source_key(value["source"])] for value in inventory["returns"]]
    except KeyError as exc:
        raise AttributionError(f"required source argument or returned value lacks materialized binding: {exc.args[0]}") from exc
    source_values.sort(key=lambda row: (row["op_index"], row["result_index"]))
    return entries, source_values, outputs


def prepare_source_bindings(source: str | bytes, command_buffer: dict) -> dict:
    """Declare the actual pointer ABI before emission; return the exact inventory.

    Every materialized buffer, including intermediates, receives a pointer. The
    entry bindings are read-only; produced values and scratch buffers are write
    allocations. The function mutates only ``kernel_abi`` and source digest/count
    metadata. It does not claim that any task has been emitted.
    """
    inventory = source_operation_inventory(source)
    entries, _, outputs = _bindings(inventory, command_buffer)
    if set(entries) & set(outputs):
        raise AttributionError("source return aliases an entry argument; explicit owned copy lowering is required")
    command_buffer["kernel_abi"] = {
        "kind": "whole_program",
        "args": [{"tensor": name, "access": "read" if name in entries else "write"}
                 for name in command_buffer["tensors"]],
        "outputs": outputs,
    }
    params = command_buffer.setdefault("params", {})
    params["source_sha256"] = inventory["source_sha256"]
    params["source_op_count"] = inventory["source_op_count"]
    return inventory


def task_definitions(command_buffer: dict) -> list[dict]:
    """Group existing command attribution into ordered, exact source-owning tasks.

    The lowering must annotate every command with ``source_task_index`` and
    ``source_op_indices``. All commands for a task must be contiguous and agree
    on their source ownership. A task may span residency, compute and readout.
    Instruction intervals remain absent until the emitter records them.
    """
    explicit = command_buffer.get("params", {}).get("source_task_definitions")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise AttributionError("source_task_definitions must be a nonempty ordered task list")
        tasks = deepcopy(explicit)
        for ident, row in enumerate(tasks):
            if row.get("task_index") != ident or not row.get("source_op_indices") or not row.get("kind"):
                raise AttributionError("source_task_definitions lacks exact ordered source task records")
        return tasks
    tasks = []
    previous = None
    for index, command in enumerate(command_buffer.get("commands", [])):
        attrs = dict(command.get("attributes") or {})
        attrs.update(command_buffer.get("params", {}).get("command_source_attribution", {}).get(str(index), {}))
        ident, owned = attrs.get("source_task_index"), attrs.get("source_op_indices")
        if type(ident) is not int or ident < 0 or not isinstance(owned, list) or not owned:
            raise AttributionError(f"command {index} lacks explicit source task ownership")
        if any(type(i) is not int or i < 0 for i in owned) or len(set(owned)) != len(owned):
            raise AttributionError(f"command {index} has invalid source_op_indices")
        if ident != previous:
            if ident != len(tasks):
                raise AttributionError("source_task_index must enumerate contiguous command groups in emission order")
            tasks.append({"task_index": ident, "kind": "source_tensor_computation",
                          "source_op_indices": sorted(owned), "command_indices": []})
            previous = ident
        task = tasks[-1]
        if task["source_op_indices"] != sorted(owned):
            raise AttributionError(f"task {ident} command members disagree on exact source ownership")
        task["command_indices"].append(index)
    if not tasks:
        raise AttributionError("source graph has no emitted command tasks")
    return tasks


def materialized_task_accesses(inventory: dict, command_buffer: dict,
                               source_op_indices: Sequence[int]) -> tuple[list[str], list[str]]:
    """Read/write boundary implied by explicit materialized source SSA bindings.

    Nonmaterialized scalar/init values contribute no buffer. Values produced and
    consumed within one task need no cross-task read boundary. The emitter may
    provide more precise physical accesses in the final task records instead.
    """
    _bindings(inventory, command_buffer)  # Validate the exact map before using it.
    bindings = {_source_key(origin): name for name, origin in
                command_buffer["params"]["tensor_source_map"].items()}
    owned = set(source_op_indices)
    reads, writes = set(), set()
    for index in source_op_indices:
        if type(index) is not int or not 0 <= index < inventory["source_op_count"]:
            raise AttributionError(f"task refers to absent source operation {index!r}")
        operation = inventory["operations"][index]
        for operand in operation["operands"]:
            key = _source_key(operand["source"])
            if key in bindings and (key[0] == "arg" or key[1] not in owned):
                reads.add(bindings[key])
        for j, _ in enumerate(operation["results"]):
            name = bindings.get(("op", index, j))
            if name is not None:
                writes.add(name)
    # Genuine compiler-only storage is identified by its explicit declaration
    # AND a destination in this task's actual logical command stream.
    temporaries = {row["tensor"] for row in command_buffer.get("params", {}).get("compiler_temporaries", ())}
    for command in command_buffer.get("commands", ()):
        owners = command.get("attributes", {}).get("source_op_indices", ())
        destination = command.get("operands", {}).get("dst")
        if owners and set(owners).issubset(owned) and destination in temporaries:
            writes.add(destination)
    order = {name: i for i, name in enumerate(command_buffer["tensors"])}
    return sorted(reads, key=order.get), sorted(writes, key=order.get)


def _schema_findings(plan: dict) -> list[str]:
    from jsonschema import Draft202012Validator
    path = Path(__file__).resolve().parent / "contracts" / "mixed_program_plan.schema.json"
    if not path.is_file():
        raise AttributionError("compiler package is missing contracts/mixed_program_plan.schema.json")
    schema = json.loads(path.read_text())
    schema.pop("$id", None)
    return [f"plan schema {error.json_path}: {error.message}" for error in
            Draft202012Validator(schema).iter_errors(plan)]


def build_plan(source: str | bytes, command_buffer: dict, tasks: Sequence[Mapping], *,
               schedule_instruction_count: int,
               prologue_instruction_range: Sequence[int],
               epilogue_instruction_range: Sequence[int],
               compiler_temporaries: Sequence[Mapping] = ()) -> dict:
    """Construct and check a plan using actual emitted task instruction intervals.

    Each task supplies source indices, nonempty instruction_start/end, kind, and
    explicit reads/writes. Missing/unknown intervals are errors, never synthetic
    placeholder instructions. Return the plan without mutating the input buffer.
    """
    cb = deepcopy(command_buffer)
    inventory = prepare_source_bindings(source, cb)
    entries, source_values, outputs = _bindings(inventory, cb)
    rows = []
    for ident, supplied in enumerate(tasks):
        row = deepcopy(dict(supplied))
        if row.get("task_index", ident) != ident:
            raise AttributionError("task indices differ from emission order")
        row["task_index"] = ident
        indices = row.get("source_op_indices")
        if not isinstance(indices, list) or not indices:
            raise AttributionError(f"task {ident} lacks exact source ownership")
        if any(type(i) is not int or not 0 <= i < inventory["source_op_count"] for i in indices):
            raise AttributionError(f"task {ident} refers to absent source operation")
        regions = sorted({inventory["operations"][i]["region_id"] for i in indices
                          if inventory["operations"][i]["region_id"] is not None})
        if "source_region_ids" in row and sorted(row["source_region_ids"]) != regions:
            raise AttributionError(f"task {ident} region IDs differ from its actual source operations")
        row["source_region_ids"] = regions
        rows.append(row)
    plan = {"schema": "mixed_program_plan_v1", "source_sha256": inventory["source_sha256"],
            "source_op_count": inventory["source_op_count"], "tasks": rows,
            "schedule_instruction_count": schedule_instruction_count,
            "prologue_instruction_range": list(prologue_instruction_range),
            "epilogue_instruction_range": list(epilogue_instruction_range),
            "entry_bindings": entries, "source_values": source_values,
            "output_bindings": outputs}
    if compiler_temporaries:
        plan["compiler_temporaries"] = [dict(row) for row in compiler_temporaries]
    findings = _schema_findings(plan)
    if not findings:
        findings = _source_plan_problems(inventory, cb, plan)
    if findings:
        raise AttributionError("; ".join(findings))
    return plan


def attach_plan(source: str | bytes, command_buffer: dict, tasks: Sequence[Mapping], **ranges) -> dict:
    """Attach a fully checked plan and whole-program pointer ABI in place."""
    plan = build_plan(source, command_buffer, tasks, **ranges)
    prepare_source_bindings(source, command_buffer)
    command_buffer.setdefault("params", {})["global_program_plan"] = plan
    return plan


def validate_plan(source: str | bytes, command_buffer: dict, lowered_mlir: str | None = None,
                  *, lowered_abi: Mapping | None = None) -> dict:
    """Package-local schema plus reviewed generic source/LLVM structural checks."""
    from merlin.targetgen.oot_starterkit.plan import _lowered_task_problems
    from xdsl.utils.exceptions import ParseError, VerifyException
    try:
        inventory = source_operation_inventory(source)
        plan = command_buffer.get("params", {}).get("global_program_plan")
        if not isinstance(plan, Mapping):
            raise AttributionError("params.global_program_plan is required")
        findings = _schema_findings(dict(plan))
        if not findings:
            findings = _source_plan_problems(inventory, command_buffer, plan)
            if lowered_mlir is not None:
                findings += _lowered_task_problems(lowered_mlir, plan, command_buffer, lowered_abi)
        return {"ok": not findings, "findings": findings,
                "source_sha256": inventory["source_sha256"],
                "source_op_count": inventory["source_op_count"],
                "scope": "public structural preflight only"}
    except (ValueError, UnicodeError, ParseError, VerifyException) as exc:
        return {"ok": False, "findings": [str(exc)]}
