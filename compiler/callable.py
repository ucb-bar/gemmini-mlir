"""Materialize real target objects and adapt their logical ABI to shared C calls.

The ordinary region emitter remains available without native tools. This route
requires explicitly selected MLIR/LLVM tools, a RISC-V ISA and a psABI. It admits
only complete static device regions with no extra host storage or external calls.
It does not build an image or establish numerical or hardware qualification.
No selected-platform completion gate is wired yet, so native products remain
non-callable inputs even after successful object generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path


@dataclass(frozen=True)
class NativeTools:
    translate: Path
    clang: Path
    march: str
    mabi: str


def completion_admission():
    """Report the missing physical contract separately from emitted structure.

    A fence's spelling, side effects and CFG position do not establish that the
    selected SoC drains accelerator transactions or makes device writes visible
    to the CPU. Existing ordering facts describe component queues, and the
    historical harness ABI explicitly grants only structural fence admission.
    No editable backend assertion can authorize this missing platform evidence.
    """
    return {
        "status": "unavailable",
        "reason": "no independently admitted selected-platform accelerator completion and CPU-visibility contract",
        "structural_fence_preserved": True,
        "synchronous_call_admitted": False,
    }


def selected_tools(parameters: dict) -> NativeTools:
    """Validate explicit tool/ISA selection before publishing any products."""
    if parameters.get("gemmini.emit") == "c-call":
        raise ValueError(
            "callable emission is unavailable: no selected-platform accelerator completion and CPU visibility contract"
        )
    fields = {"gemmini.emit", "gemmini.mlir_translate", "gemmini.clang", "gemmini.march", "gemmini.mabi"}
    if set(parameters) != fields or parameters.get("gemmini.emit") != "native-object":
        raise ValueError("native emission requires gemmini.emit=native-object, mlir_translate, clang, march and mabi")
    paths = []
    for key in ("gemmini.mlir_translate", "gemmini.clang"):
        value = parameters[key]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{key} requires an explicit absolute executable path")
        path = Path(value)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"selected native tool is missing or not executable: {path}")
        paths.append(path)
    march, mabi = parameters["gemmini.march"], parameters["gemmini.mabi"]
    if (not isinstance(march, str) or not march.startswith("rv64") or len(march) <= 4
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in march)
            or not isinstance(mabi, str) or mabi not in {"lp64", "lp64f", "lp64d"}):
        raise ValueError("callable emission requires an explicit rv64 ISA and lp64/lp64f/lp64d psABI")
    return NativeTools(*paths, march, mabi)


def _digest(path):
    from merlin.compiler.artifact import digest
    return digest(path)


def _boundary(source, commands, module, symbol):
    """Reconcile original tensor ports, command pointer order and actual SSA uses."""
    from xdsl.dialects import func, llvm

    from merlin.compiler.ir.logical_abi import kernel_arg_order, logical_abi
    from merlin.compiler.backend.cabi import CRegionABI, TensorPort
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    source_module = parse_mlir_text(source)
    source_module.verify()
    functions = tuple(source_module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not func.FuncOp:
        raise ValueError("callable device emission requires one complete tensor function")
    signature = functions[0].function_type
    inputs = tuple(TensorPort.from_type(t) for t in signature.inputs)
    outputs = tuple(TensorPort.from_type(t) for t in signature.outputs)
    params = commands.get("params", {})
    input_names, output_names = params.get("frontend_entry_bindings"), params.get("frontend_output_bindings")
    plan = params.get("global_program_plan", {})
    if (not isinstance(input_names, list) or not isinstance(output_names, list)
            or len(input_names) != len(inputs) or len(output_names) != len(outputs)
            or not outputs or len(set(input_names + output_names)) != len(input_names + output_names)
            or params.get("storage_encodings") or not plan.get("tasks")
            or any(task.get("kind") != "device" for task in plan["tasks"])
            or plan.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest()):
        raise ValueError("callable emission requires exact source-bound device-only tensor boundaries")
    for name, port in zip(input_names + output_names, inputs + outputs, strict=True):
        spec = commands.get("tensors", {}).get(name, {})
        if spec.get("shape") != list(port.shape) or spec.get("dtype") != port.dtype:
            raise ValueError("source tensor port differs from emitted command ABI")
    selected_abi = logical_abi(Path(__file__).parent / "backend/data/abi.yaml")
    order = kernel_arg_order(commands, selected_abi)
    entries = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.sym_name.data == "kernel_entry"]
    if len(entries) != 1:
        raise ValueError("emitted module has no unique logical kernel entry")
    entry = entries[0]
    if (not entry.body.blocks or entry.function_type.is_variadic
            or type(entry.function_type.output) is not llvm.LLVMVoidType
            or len(entry.function_type.inputs) != len(order)
            or any(type(t) is not llvm.LLVMPointerType for t in entry.function_type.inputs)):
        raise ValueError("emitted entry does not implement the declared logical pointer ABI")
    mapping = {name: f"(void *)inputs[{index}]" for index, name in enumerate(input_names)}
    mapping.update({name: f"outputs[{index}]" for index, name in enumerate(output_names)})
    arguments = []
    for name, arg in zip(order, entry.body.blocks.first.args, strict=True):
        if name in mapping:
            arguments.append(mapping[name])
        elif arg.uses:
            raise ValueError("callable region needs extra host workspace not admitted by this ABI")
        else:
            # Virtual accumulators may retain an unused historical pointer slot.
            arguments.append("(void *)0")
    if set(input_names + output_names) - set(order):
        raise ValueError("emitted logical entry omits a source boundary port")
    abi = CRegionABI(symbol, inputs, outputs, selected_abi.alignment_bytes,
                     ("call.c", "device.o"), resource_kind="accelerator")
    return entry, tuple(arguments), abi


def prepare_call(source, commands, artifact):
    """Return the exact renamed device module and a C wrapper consuming its ABI."""
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    from merlin.development.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    symbol = "gemmini_region_" + hashlib.sha256((source + "\0" + artifact).encode()).hexdigest()[:32]
    entry, arguments, abi = _boundary(source, commands, module, symbol)
    # Mutable globals, private scratch and arbitrary callees require another
    # resource/ABI contract. Never hide them behind the zero-workspace claim.
    if any(not isinstance(op, llvm.FuncOp) for op in module.body.block.ops):
        raise ValueError("callable region has module storage or unsupported top-level operations")
    if any(op.name in {"llvm.call", "llvm.alloca", "llvm.mlir.addressof"} for op in entry.walk()):
        raise ValueError("callable region requires external calls or additional host storage")
    emitted = [op for op in entry.walk() if type(op) is llvm.InlineAsmOp]
    if (not emitted or not any(op.asm_string.data.startswith(".insn r ") for op in emitted)
            or emitted[-1].asm_string.data != "fence"
            or any(op.has_side_effects is None or "~{memory}" not in op.constraints.data for op in emitted)):
        raise ValueError("callable region lacks actual ordered target instructions and completion")
    for op in tuple(module.body.block.ops):
        if op is entry:
            continue
        if op.sym_name.data != "kernel_complete":
            raise ValueError("callable emission has another unbound function")
        module.body.block.erase_op(op)
    device_symbol = symbol + "_device"
    entry.properties["sym_name"] = builtin.StringAttr(device_symbol)
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    signature = ", ".join("void *" for _ in arguments) or "void"
    pointers = [f"inputs[{i}]" for i in range(len(abi.inputs))] + [f"outputs[{i}]" for i in range(len(abi.outputs))]
    checks = " || ".join(f"!{p} || ((uintptr_t){p} & {abi.buffer_alignment - 1}u)" for p in pointers)
    wrapper = (
        "#include <stdint.h>\n"
        f"extern void {device_symbol}({signature});\n"
        f"int {symbol}(void *context, const void *const *inputs, void *const *outputs) {{\n"
        "  (void)context;\n"
        "  if (!inputs || !outputs) return -1;\n"
        f"  if ({checks}) return -1;\n"
        f"  {device_symbol}({', '.join(arguments)});\n"
        "  return 0;\n}\n"
    )
    return stream.getvalue() + "\n", wrapper, abi


def _run(command, directory, label):
    completed = subprocess.run(command, cwd=directory, text=True, capture_output=True, timeout=300)
    (directory / f"{label}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"selected native {label} failed; see {directory / (label + '.log')}")


def materialize(product, source, tools: NativeTools):
    """Lower target instructions; retain non-callable products pending completion admission."""
    from merlin.analysis.artifact_reader import PinnedELF, open_ordinary
    from merlin.analysis.host_isa import elf_flag_corroboration
    from merlin.compiler.artifact import Artifact

    directory = product.directory
    commands = json.loads(product.commands.read_text(encoding="utf-8"))
    module, wrapper, abi = prepare_call(source, commands, product.artifact.read_text(encoding="utf-8"))
    lowered = directory / "callable.mlir"
    lowered.write_text(module, encoding="utf-8")
    call = directory / "call.c"
    call.write_text(wrapper, encoding="utf-8")
    before = {str(path): _digest(path) for path in (tools.translate, tools.clang)}
    source_files = (product.artifact, product.commands, lowered, call)
    source_hashes = {path.name: _digest(path) for path in source_files}
    versions = {}
    for path in (tools.translate, tools.clang):
        versions[str(path)] = subprocess.run([str(path), "--version"], text=True, capture_output=True,
                                             check=True, timeout=30).stdout.strip()
    ll, obj = directory / "device.ll", directory / "device.o"
    translate = [str(tools.translate), "--mlir-to-llvmir", str(lowered), "-o", str(ll)]
    compile_command = [str(tools.clang), "--target=riscv64-unknown-elf", f"-march={tools.march}",
                       f"-mabi={tools.mabi}", "-ffreestanding", "-fno-pic", "-O2", "-c", str(ll), "-o", str(obj)]
    _run(translate, directory, "translate")
    _run(compile_command, directory, "object")
    image = PinnedELF(open_ordinary(obj), str(obj), declared_isa={"value": tools.march, "source": "gemmini.march"})
    try:
        # ET_REL is the standard ELF relocatable-object kind, not a hardware fact.
        object_abi = elf_flag_corroboration(image.e_flags, None)["float_abi"]
        if (int.from_bytes(image.read_at(16, 2), "little") != 1
                or object_abi != {"lp64": "soft", "lp64f": "single", "lp64d": "double"}[tools.mabi]):
            raise ValueError("native result is not a relocatable object with the selected psABI")
        definition = image.function_extent(abi.symbol + "_device")
        definition["scope"] = "relocatable_object_definition"
        object_arch = image.arch_attribute
    finally:
        image.close()
    if (before != {str(path): _digest(path) for path in (tools.translate, tools.clang)}
            or source_hashes != {path.name: _digest(path) for path in source_files}):
        raise ValueError("selected native tool or emitted source bytes changed during callable emission")
    record = directory / "callable.json"
    admission = completion_admission()
    record.write_text(json.dumps({
        "schema": "gemmini.callable.v1", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "emitted_module_sha256": _digest(product.artifact), "callable_module_sha256": _digest(lowered),
        "inputs": source_hashes,
        "tools": {key: {"sha256": digest, "version": versions[key]} for key, digest in before.items()},
        "commands": [translate, compile_command], "isa": tools.march, "abi": tools.mabi,
        "object_arch_attribute": object_arch, "object_float_abi": object_abi,
        "device_symbol": definition, "c_call_requirements": abi.record(), "c_call_admission": admission,
        "execution": "not_run",
        "qualification": "not_established",
    }, indent=2) + "\n", encoding="utf-8")
    artifacts = tuple(Artifact.from_file(path, kind=kind, role=role) for path, kind, role in (
        (lowered, "llvm-dialect", "callable-ir"), (ll, "llvm-ir", "callable-ir"),
        (obj, "object", "region"), (call, "c-source", "wrapper-input"), (record, "json", "callable-record"),
        (directory / "translate.log", "text", "diagnostic"), (directory / "object.log", "text", "diagnostic"),
    ))
    return artifacts, admission
