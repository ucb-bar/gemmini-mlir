"""Actual target emission plus C-call transport; no accelerator execution claims."""

import ctypes
import json
import shutil
import subprocess
from copy import deepcopy
from io import StringIO
from pathlib import Path

import pytest
from gemmini import compile_source, provider
from gemmini.callable import completion_admission, prepare_call, selected_tools
from xdsl.dialects import llvm
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin import api
from merlin.compiler.ir.logical_abi import kernel_arg_order, logical_abi
from merlin.development.targetgen.oot_starterkit.llvm_context import make_llvm_context


def test_actual_device_module_and_wrapper_preserve_order_and_completion(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "examples/matmul.mlir").read_text()
    product = compile_source(source, output=tmp_path / "region")
    commands = json.loads(product.commands.read_text())
    original = product.artifact.read_text()
    renamed, wrapper, abi = prepare_call(source, commands, original)
    assert abi.resource_kind == "accelerator"
    assert [port.dtype for port in abi.inputs] == ["i8", "i8"]
    assert [port.dtype for port in abi.outputs] == ["i32"]
    assert abi.link_inputs == ("call.c", "device.o") and abi.scratch_bytes == 0
    module = Parser(make_llvm_context(), renamed).parse_module()
    module.verify()
    entry = module.body.block.first_op
    assert entry.sym_name.data == abi.symbol + "_device"
    instructions = [op.asm_string.data for op in entry.walk() if isinstance(op, llvm.InlineAsmOp)]
    original_module = Parser(make_llvm_context(), original).parse_module()
    original_entry = next(op for op in original_module.body.block.ops if op.sym_name.data == "kernel_entry")
    assert instructions == [op.asm_string.data for op in original_entry.walk() if isinstance(op, llvm.InlineAsmOp)]
    assert instructions[-1] == "fence" and any(asm.startswith(".insn r ") for asm in instructions)
    changed = deepcopy(commands)
    changed["tensors"][changed["params"]["frontend_entry_bindings"][0]]["dtype"] = "i32"
    with pytest.raises(ValueError, match="differs"):
        prepare_call(source, changed, original)
    for op in tuple(original_entry.walk()):
        if isinstance(op, llvm.InlineAsmOp) and op.asm_string.data == "fence":
            op.parent_block().erase_op(op)
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(original_module)
    with pytest.raises(ValueError, match="completion"):
        prepare_call(source, commands, stream.getvalue())

    # A CPU stub checks only pointer-array transport. It does not emulate the
    # device function, its arithmetic, target instructions or completion.
    if shutil.which("gcc") is None:
        pytest.skip("GCC required for the C wrapper transport check")
    args = tuple(entry.body.blocks.first.args)
    prototype = ", ".join(f"void *p{i}" for i in range(len(args)))
    assigns = "\n".join(f"  observed[{i}] = (uintptr_t)p{i};" for i in range(len(args)))
    stub = ("#include <stdint.h>\n" + f"uintptr_t observed[{len(args)}]; int calls;\n"
            + f"void {abi.symbol}_device({prototype}) {{\n{assigns}\n  ++calls;\n}}\n")
    (tmp_path / "wrapper.c").write_text(wrapper)
    (tmp_path / "stub.c").write_text(stub)
    library = tmp_path / "transport.so"
    subprocess.run(["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(tmp_path / "wrapper.c"), str(tmp_path / "stub.c"), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    call = getattr(loaded, abi.symbol)
    call.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    call.restype = ctypes.c_int
    storage = [ctypes.create_string_buffer(2048 + abi.buffer_alignment) for _ in range(3)]
    addresses = [(ctypes.addressof(value) + abi.buffer_alignment - 1) & -abi.buffer_alignment for value in storage]
    inputs, outputs = (ctypes.c_void_p * 2)(*addresses[:2]), (ctypes.c_void_p * 1)(addresses[2])
    assert call(None, None, outputs) == -1
    inputs[0] = addresses[0] + 1
    assert call(None, inputs, outputs) == -1
    assert ctypes.c_int.in_dll(loaded, "calls").value == 0
    inputs[0] = addresses[0]
    assert call(None, inputs, outputs) == 0
    params = commands["params"]
    expected = dict(zip(params["frontend_entry_bindings"] + params["frontend_output_bindings"], addresses, strict=True))
    declaration = logical_abi(Path(provider.__file__).parent / "backend/data/abi.yaml")
    order = kernel_arg_order(commands, declaration)
    observed = (ctypes.c_size_t * len(args)).in_dll(loaded, "observed")
    assert list(observed) == [expected.get(name, 0) for name in order]
    assert ctypes.c_int.in_dll(loaded, "calls").value == 1


def test_missing_selected_tools_refuse_before_region_publication(tmp_path):
    parameters = {"gemmini.emit": "native-object", "gemmini.mlir_translate": str(tmp_path / "missing-translate"),
                  "gemmini.clang": str(tmp_path / "missing-clang"), "gemmini.march": "rv64gc", "gemmini.mabi": "lp64d"}
    with pytest.raises(ValueError, match="missing or not executable"):
        selected_tools(parameters)
    source = (Path(__file__).resolve().parents[1] / "examples/matmul.mlir").read_text()
    with pytest.raises(ValueError, match="missing or not executable"):
        api.compile(source, backend=provider, output=tmp_path / "refused",
                    options=api.CompileOptions(parameters=parameters))
    assert not (tmp_path / "refused").exists()
    with pytest.raises(ValueError, match="no selected-platform accelerator completion"):
        api.compile(source, backend=provider, output=tmp_path / "call-refused",
                    options=api.CompileOptions(parameters={"gemmini.emit": "c-call"}))
    assert not (tmp_path / "call-refused").exists()


def test_native_artifact_success_cannot_authorize_synchronous_dispatch(tmp_path, monkeypatch):
    """Provider gate only; the substitute producer is not native-tool evidence."""
    from gemmini import callable as native

    from merlin.compiler.backend.cabi import CRegionABI

    source = (Path(__file__).resolve().parents[1] / "examples/matmul.mlir").read_text()
    # The real region emitter still runs. Substitute only the unavailable native
    # phase to establish that a successful phase cannot publish synchronous ABI.
    selected = native.NativeTools(Path("/not-executed/translate"), Path("/not-executed/clang"), "rv64gc", "lp64d")
    monkeypatch.setattr(native, "selected_tools", lambda parameters: selected)
    monkeypatch.setattr(native, "materialize", lambda *args: ((), completion_admission()))
    result = api.compile(source, backend=provider, output=tmp_path / "native-inputs",
                         options=api.CompileOptions(parameters={"gemmini.emit": "native-object"}))
    assert result.metadata["scope"] == "native_region_inputs"
    assert result.metadata["c_call_admission"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="no supported explicit C call contract"):
        CRegionABI.from_compilation(result)
