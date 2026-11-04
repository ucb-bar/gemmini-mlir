"""Bind exact source tasks to the actual LLVM operations emitted for a kernel.

Ranges count static LLVM body operations, including constants and executable
control flow, excluding function returns, block labels, and globals. Blocks are
scheduled by the CFG's strongly connected components, so loop exits follow their actual bodies even
when an inliner prints continuation blocks first. Calls to submitted definitions
are expanded in call order for outlined mixed programs.
This is an authoring inventory, never a runtime instruction or cycle count.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

_LIBRARY = Path(__file__).resolve().parent / "compiler_library"
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))

from source_attribution import (
    AttributionError, attach_plan, materialized_task_accesses,
    prepare_source_bindings, task_definitions, validate_plan,
)
from merlin.common.ir_lock import IR_LOCK
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.frontends.linalg_mlir import parse_mlir_text


def _integer_attr(operation, name):
    attr = operation.attributes.get(name)
    if attr is None:
        return None
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int:
        raise AttributionError(f"{operation.name} has a noninteger {name}")
    return value


def _symbol(operation):
    value = operation.properties.get("sym_name") or operation.attributes.get("sym_name")
    return getattr(value, "data", None)


def _callee(operation):
    value = operation.properties.get("callee") or operation.attributes.get("callee")
    root = getattr(value, "root_reference", None)
    return getattr(root, "data", None)


def _scheduled_operations(module, entry_symbol):
    functions = {_symbol(op): op for op in module.body.block.ops
                 if op.name == "llvm.func" and op.body.blocks}
    entry = functions.get(entry_symbol)
    if entry is None:
        raise AttributionError(f"lowered artifact has no defined LLVM entry {entry_symbol!r}")

    def walk(fn, stack):
        name = _symbol(fn)
        if name in stack:
            raise AttributionError("recursive lowered calls have no finite static task schedule")
        # Schedule the real CFG's loop components before their continuations.
        # A plain DFS/RPO can place a loop exit before the loop body depending
        # on conditional successor order. Condensing strongly connected blocks
        # produces a DAG, whose topological order respects every forward edge.
        blocks = list(fn.body.blocks)
        allowed = set(blocks)
        successors = {block: list(block.last_op.successors) if block.last_op is not None else []
                      for block in blocks}
        if any(successor not in allowed for targets in successors.values() for successor in targets):
            raise AttributionError("lowered CFG edge leaves the selected function")
        visited, postorder = set(), []
        pending = [(blocks[0], False)]
        while pending:
            block, finished = pending.pop()
            if finished:
                postorder.append(block)
                continue
            if block in visited:
                continue
            visited.add(block)
            pending.append((block, True))
            pending.extend((successor, False) for successor in reversed(successors[block]))
        if visited != allowed:
            raise AttributionError("lowered function contains unreachable static blocks")
        predecessors = {block: [] for block in blocks}
        for block, targets in successors.items():
            for target in targets:
                predecessors[target].append(block)
        components, component_of = [], {}
        for root in reversed(postorder):
            if root in component_of:
                continue
            ident = len(components)
            component, work = [], [root]
            while work:
                block = work.pop()
                if block in component_of:
                    continue
                component_of[block] = ident
                component.append(block)
                work.extend(predecessors[block])
            components.append(component)
        edges = {i: set() for i in range(len(components))}
        indegree = {i: 0 for i in edges}
        for block, targets in successors.items():
            left = component_of[block]
            for target in targets:
                right = component_of[target]
                if left != right and right not in edges[left]:
                    edges[left].add(right)
                    indegree[right] += 1
        placement = {block: i for i, block in enumerate(blocks)}
        priority = {i: min(placement[block] for block in component) for i, component in enumerate(components)}
        ready = [i for i in edges if indegree[i] == 0]
        scheduled = []
        while ready:
            ident = min(ready, key=priority.get)
            ready.remove(ident)
            scheduled.extend(sorted(components[ident], key=placement.get))
            for target in edges[ident]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(scheduled) != len(blocks):
            raise AttributionError("lowered CFG condensation failed to schedule every actual block")
        for block in scheduled:
            for op in block.ops:
                if op.name == "llvm.return":
                    continue
                if op.regions:
                    raise AttributionError("nested LLVM regions need explicit schedule ownership")
                yield op
                if op.name == "llvm.call":
                    callee = _callee(op)
                    if callee in functions:
                        yield from walk(functions[callee], (*stack, name))
                    elif callee is None:
                        raise AttributionError("indirect LLVM call has no exact static task inventory")
    return list(walk(entry, ()))


def _shared_entry_definitions(function):
    """Recognize only the reviewed CFG checker's def-use-proved shared plumbing.

    All recognized instructions remain in the static operation inventory. This
    classification prevents a hoisted immutable definition's source tag from
    opening a computation interval before its actual task executes.
    """
    parent = function.parent_op()
    immutable_globals = set()
    if parent is not None and parent.name == "builtin.module":
        for operation in parent.body.block.ops:
            if operation.name == "llvm.mlir.global" and "constant" in operation.properties:
                name = _symbol(operation)
                if isinstance(name, str):
                    immutable_globals.add(name)
    entry = function.body.blocks[0]
    constants, addresses, shared = set(), set(entry.args), set()
    for operation in entry.ops:
        operands = tuple(operation.operands)
        if operation.name == "llvm.mlir.constant":
            constants.update(operation.results)
            shared.add(operation)
        elif operation.name == "llvm.mlir.addressof":
            reference = operation.properties.get("global_name")
            name = getattr(getattr(reference, "root_reference", None), "data", None)
            if name in immutable_globals:
                addresses.update(operation.results)
                shared.add(operation)
        elif operation.name == "llvm.ptrtoint" and operands and operands[0] in addresses:
            addresses.update(operation.results)
            shared.add(operation)
        elif (operation.name == "llvm.add" and len(operands) == 2
              and ((operands[0] in addresses and operands[1] in constants)
                   or (operands[1] in addresses and operands[0] in constants))):
            addresses.update(operation.results)
            shared.add(operation)
        elif operation.name == "llvm.alloca" and operands and all(value in constants for value in operands):
            addresses.update(operation.results)
            shared.add(operation)
    return shared


def attach_emission_plan(source: str | bytes, command_buffer: dict, lowered_mlir: str, *,
                         entry_symbol: str | None = None, completion_symbol: str | None = "kernel_complete",
                         completion_asm: str = "fence", validate_lowered: bool = True) -> dict:
    """Attach exact ranges measured from emitted LLVM task tags and validate.

    The emitter must already carry correct ``merlin.global_task`` on executable
    operations and direct source indices on device operations. This routine
    never adds tags or invents source/task identities. Untagged shared definitions
    count inside the current static range; the generic CFG checker independently
    rejects untagged arithmetic or other unowned executable work.

    Returns the attached plan. ``validate_lowered=False`` only permits inspecting
    a source/range-consistent draft; it does not assert LLVM attribution validity.
    """
    inventory = prepare_source_bindings(source, command_buffer)
    definitions = task_definitions(command_buffer)
    expected = set(range(len(definitions)))
    with IR_LOCK:
        module = parse_mlir_text(lowered_mlir, ctx=make_llvm_context())
        module.verify()
    symbols = [_symbol(op) for op in module.body.block.ops if op.name == "llvm.func" and op.body.blocks]
    if entry_symbol is None:
        entry_symbol = command_buffer.get("params", {}).get("_host_symbol")
    if entry_symbol is None:
        choices = [name for name in symbols if name != completion_symbol]
        if len(choices) != 1:
            raise AttributionError("multiple LLVM definitions require an explicit entry_symbol")
        entry_symbol = choices[0]
    shared = set()
    for function in module.body.block.ops:
        if function.name == "llvm.func" and function.body.blocks:
            shared.update(_shared_entry_definitions(function))
    operations = _scheduled_operations(module, entry_symbol)
    if not operations:
        raise AttributionError("lowered entry emitted no scheduled operations")
    starts, ends = {}, {}
    phase = -1
    previous_order = -1
    owners = {index: task["task_index"] for task in definitions for index in task["source_op_indices"]}
    wrapper_prologue_end = 0
    wrapper_epilogue_start = len(operations)
    for index, operation in enumerate(operations):
        tagged = _integer_attr(operation, "merlin.global_task")
        # Def-use-proved immutable entry plumbing can be hoisted from later
        # tasks. Count it, but do not mistake its source tag for computation.
        # In particular llvm.mlir.zero is NOT silently exempted: the actual
        # lowering must explicitly own any such instruction as prologue/task.
        range_tag = None if operation in shared else tagged
        if range_tag is not None:
            tagged = range_tag
            if tagged not in expected | {-1, -2}:
                raise AttributionError(f"lowered operation names absent global task {tagged}")
            order = len(definitions) if tagged == -2 else tagged
            if order < previous_order:
                raise AttributionError("LLVM static task order reverses; task ranges would be noncontiguous")
            if tagged != phase:
                if phase in expected:
                    ends[phase] = index
                if tagged in expected:
                    starts[tagged] = index
                if phase == -1:
                    wrapper_prologue_end = index
                if tagged == -2:
                    wrapper_epilogue_start = index
            phase = tagged
            previous_order = order
        source_index = _integer_attr(operation, "merlin.source_op_index")
        if source_index is not None and (tagged is None or owners.get(source_index) != tagged):
            raise AttributionError("lowered source_op_index differs from exact owning global task")
    if phase in expected:
        ends[phase] = len(operations)
    if set(starts) != expected or set(ends) != expected:
        missing = sorted(expected - starts.keys())
        raise AttributionError(f"planned tasks have no actual contiguous emitted range: {missing}")
    tasks = []
    temporaries = command_buffer.get("params", {}).get("compiler_temporaries", ())
    for row in definitions:
        task = dict(row)
        ident = task["task_index"]
        task["instruction_start"], task["instruction_end"] = starts[ident], ends[ident]
        if "reads" not in task or "writes" not in task:
            task["reads"], task["writes"] = materialized_task_accesses(
                inventory, command_buffer, task["source_op_indices"])
        # A declared compiler temporary has no source SSA binding, but its
        # provenance identifies the actual computation task that materializes it.
        # Within-task consumption does not cross the task's read boundary.
        declared_temporaries = {temp["tensor"] for temp in temporaries}
        internal = []
        for command_index in task.get("command_indices", ()):
            command = command_buffer["commands"][command_index]
            destination = command.get("operands", {}).get("dst")
            if destination in declared_temporaries:
                internal.append(destination)
        task["writes"] = list(dict.fromkeys([*task["writes"], *internal]))
        tasks.append(task)
    plan = attach_plan(source, command_buffer, tasks,
                       schedule_instruction_count=len(operations),
                       prologue_instruction_range=[0, wrapper_prologue_end],
                       epilogue_instruction_range=[wrapper_epilogue_start, len(operations)],
                       compiler_temporaries=temporaries)
    if validate_lowered:
        abi = {"entry_symbol": entry_symbol,
               "completion_symbol": completion_symbol if completion_symbol in symbols else None,
               "completion_asm": completion_asm if completion_symbol in symbols else None}
        result = validate_plan(source, command_buffer, lowered_mlir, lowered_abi=abi)
        if not result["ok"]:
            raise AttributionError("emitted plan validation failed: " + "; ".join(result["findings"]))
    return plan


def emit_attributed(source: str | bytes, command_buffer: dict, emit_callback, **kwargs) -> str:
    """Prepare the true pointer ABI, invoke the emitter, and attach checked ranges.

    ``emit_callback(cb)`` must return LLVM text whose task scopes were set by the
    backend using ``task_definitions(cb)``. It is responsible for actual lowering;
    this adapter performs no instruction generation or attribution repair.
    """
    prepare_source_bindings(source, command_buffer)
    lowered = emit_callback(command_buffer)
    if not isinstance(lowered, str):
        raise TypeError("attributed emit callback must return LLVM dialect text")
    attach_emission_plan(source, command_buffer, lowered, **kwargs)
    return lowered


def normalize_compiler_prologue(lowered_mlir: str) -> str:
    """Own proven compiler descriptor seeds and null-size constants in prologue.

    The stock lowering hoists pointer-based allocation-size constants and
    poison seeds for compiler memref descriptors. They remain real LLVM
    instructions with wrapper ownership. Descriptor seeds must have no source
    identity and feed only insertvalue aggregate operands. Require a pointer
    zero without source-operation identity, whose
    entire pointer-use chain is static GEP followed by ptrtoint. Loads/stores,
    dynamic GEP offsets, and pointers escaping this size chain are refused.
    This never reclassifies tensor arithmetic or source-defined computation.
    """
    from io import StringIO
    from xdsl.dialects.builtin import IntegerAttr, i64
    from merlin.common.dense_float_ir import DenseFloatPrinter
    with IR_LOCK:
        module = parse_mlir_text(lowered_mlir, ctx=make_llvm_context())
        module.verify()
    changed = False
    for function in module.body.block.ops:
        if function.name != "llvm.func" or not function.body.blocks:
            continue
        for operation in function.body.blocks[0].ops:
            if operation.operands or len(operation.results) != 1:
                continue
            if "merlin.source_op_index" in operation.attributes or any(
                    name.startswith("prov.") for name in operation.attributes):
                continue
            uses = list(operation.results[0].uses)
            if not uses:
                continue
            name = operation.name
            if name == "builtin.unregistered":
                name = getattr(operation.attributes.get("op_name__"), "data", name)
            result_type = str(operation.results[0].type)
            # Stock memref lowering builds descriptor aggregates from poison.
            # The seed itself computes no source value: every use must only
            # replace a descriptor field, never observe/escape the seed.
            if (name == "llvm.mlir.poison" and result_type.startswith("!llvm.struct<")
                    and "!llvm.ptr" in result_type
                    and all(use.operation.name == "llvm.insertvalue" and use.index == 0
                            for use in uses)):
                operation.attributes["merlin.global_task"] = IntegerAttr(-1, i64)
                changed = True
                continue
            if name != "llvm.mlir.zero" or result_type != "!llvm.ptr":
                continue
            proven = True
            for use in uses:
                gep = use.operation
                if (gep.name != "llvm.getelementptr" or len(gep.operands) != 1
                        or gep.operands[0] is not operation.results[0] or len(gep.results) != 1):
                    proven = False
                    break
                conversions = list(gep.results[0].uses)
                if not conversions or any(conversion.operation.name != "llvm.ptrtoint" for conversion in conversions):
                    proven = False
                    break
            if proven:
                operation.attributes["merlin.global_task"] = IntegerAttr(-1, i64)
                changed = True
    if not changed:
        return lowered_mlir
    module.verify()
    stream = StringIO()
    DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue() + "\n"
