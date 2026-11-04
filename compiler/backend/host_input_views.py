"""Bind the original logical shim's checked readonly dense-offset ABI.

This adapter changes descriptor layouts only. It establishes no borrowed
lifetime, accelerator completion, or numerical effects.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess


def prepare(source, shim_path, groups, *, work):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.device_input_views import _dense_tensor, rewrite_readonly_dense_offsets
    from merlin.llvmlower.device_shim import emit_logical_translation_unit
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects import func

    def identity(path):
        path = Path(path).resolve()
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    work = Path(work)
    work.mkdir(exist_ok=True)
    shim_path = Path(shim_path).resolve()
    original = emit_logical_translation_unit('gemmini', groups, kernel_symbol_for=lambda s: s + '_kernel')
    if original.skipped or shim_path.read_text() != original.text:
        raise ValueError('dense input views lost the original logical shim')
    # Compile the exact unchanged emitter output before granting its offset ABI.
    include = work / 'include'
    include.mkdir(exist_ok=True)
    (include / 'string.h').write_text('void *memset(void *, int, __SIZE_TYPE__);\n')
    obj = work / 'device_shim.o'
    command = [str(clang()), '--target=riscv64-unknown-elf', '-march=rv64gc', '-mabi=lp64d',
        '-O1', '-ffreestanding', '-I', str(include), '-c', str(shim_path), '-o', str(obj)]
    before = identity(shim_path)
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise ValueError('dense input view original shim compilation failed: ' + run.stderr[-4000:])
    if identity(shim_path) != before:
        raise ValueError('dense input view original shim changed during compilation')
    module = parse_mlir_text(source)
    calls = {}
    for operation in module.walk():
        if isinstance(operation, func.CallOp):
            calls.setdefault(operation.callee.root_reference.data, []).append(operation)
    declarations = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    selected = {}
    for symbol, (buffers, call_buffers) in groups.items():
        declaration = declarations.get(symbol)
        types = tuple(declaration.function_type.inputs) if declaration is not None else ()
        if len(types) != len(call_buffers) or len(buffers) != len(call_buffers):
            raise ValueError('dense input view lost the original declaration roster')
        indices = []
        for index, (buffer, call, typ) in enumerate(zip(buffers, call_buffers, types, strict=True)):
            if tuple(typ.get_shape()) != buffer.shape or str(typ.element_type) != buffer.dtype or call != dict(
                    role=buffer.role, shape=list(buffer.shape), dtype=buffer.dtype):
                raise ValueError('dense input view declaration differs from the original shim')
            if (buffer.kind == 'input' and len(buffer.shape) in (2, 3) and calls.get(symbol)
                    and all(_dense_tensor(c.arguments[index]) for c in calls[symbol])):
                indices.append(index)
        if indices:
            selected[symbol] = tuple(indices)
    applied = rewrite_readonly_dense_offsets(module, selected)
    rewritten = text(module, generic=True)
    report = dict(schema='submission.logical_dense_input_views.v1', arguments=applied,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        selected_source_sha256=hashlib.sha256(rewritten.encode()).hexdigest(),
        shim_source=before, shim_object=identity(obj), command=command,
        layout='fixed dense row strides; dynamic element offset',
        effects='unchanged original output, alias and completion contracts',
        numerical_execution='not_established', performance_cycles=None)
    (work / 'selection.json').write_text(json.dumps(report, indent=2))
    return rewritten, report
