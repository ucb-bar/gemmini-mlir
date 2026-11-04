"""Exact scalar host lowering through Merlin's upstream MLIR pipeline.

This owns no tensor evaluator or numerical approximation. The existing ranked
source entry transports logical pointers; stock upstream lowering retains the
source's scalar order and emits libm calls for nonlinear floating operations.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess
import tempfile

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnitAttr, IntegerAttr, i64
from xdsl.parser import Parser
from xdsl.printer import Printer
from merlin.llvmlower.source_region_host import tensor_abi
from merlin.llvmlower.host_builder import FnBuilder
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context


def _text(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _invoke(arguments, *, stage):
    process = subprocess.run(list(map(str, arguments)), capture_output=True, text=True, timeout=240)
    if process.returncode:
        raise ValueError('stock host ' + stage + ' failed: ' + process.stderr[-12000:])


def lower_stock_host(module, symbol, *, task_index=0, optimization_report=None):
    """Lower one original host tensor function to an inspectable LLVM module.

    The integer accelerator reader selects device work before reaching this API.
    Its caller supplies the complete original host island, without reconstructing
    a model from operation names or source provenance.
    """
    from host_lowering import _stock_tool, _owner
    from merlin.llvmlower.pipeline import _UPSTREAM_PASSES

    module = module.clone()
    owner = _owner(module)
    graph_symbol = symbol + '_graph'
    logical_inputs, logical_outputs = tensor_abi(module)
    logical_buffers = (*logical_inputs, *logical_outputs)
    owner.properties['sym_name'] = StringAttr(graph_symbol)
    owner.attributes['llvm.emit_c_interface'] = UnitAttr()
    # Retain the library's serial scalar phase order. Its private runner markers
    # are replaced by their ordinary upstream ownership/deallocation passes.
    passes = []
    for name in _UPSTREAM_PASSES:
        if name == '__DEALLOC__':
            passes.extend(('ownership-based-buffer-deallocation', 'buffer-deallocation-simplification',
                           'bufferization-lower-deallocations'))
        else:
            passes.append(name)
    with tempfile.TemporaryDirectory(prefix='gemmini-host-stock-') as directory:
        work = Path(directory)
        source, lowered = work / 'source.mlir', work / 'source.llvm.mlir'
        source.write_text(_text(module))
        from host_completion_policy import run_stock_pipeline
        run_stock_pipeline(source, lowered, passes, tool=_stock_tool('mlir-opt'),
                           invoke=_invoke, print_module=_text, report=optimization_report)
        tensor_module = Parser(make_llvm_context(), lowered.read_text()).parse_module()
    tensor_module.verify()
    # Definitions own duplicate declarations. LLVM module flags describe the C
    # compilation invocation; they are not executable source work or transport.
    objects = []
    defined = {op.sym_name.data for parsed in (tensor_module,) for op in parsed.body.block.ops
               if isinstance(op, llvm.FuncOp) and op.body.blocks}
    seen_declarations = set()
    for parsed in (tensor_module,):
        for op in tuple(parsed.body.block.ops):
            if op.name == 'llvm.module_flags':
                continue
            if isinstance(op, llvm.FuncOp) and not op.body.blocks:
                if op.sym_name.data in defined or op.sym_name.data in seen_declarations:
                    continue
                seen_declarations.add(op.sym_name.data)
            parsed.body.block.detach_op(op)
            objects.append(op)
            if isinstance(op, llvm.FuncOp) and op.body.blocks:
                for nested in op.walk():
                    if nested is not op and nested.name != 'llvm.return':
                        nested.attributes['merlin.global_task'] = IntegerAttr(task_index, i64)
    graph = next(op for op in objects if isinstance(op, llvm.FuncOp) and op.sym_name.data == graph_symbol)
    pointer = llvm.LLVMPointerType()
    builder = FnBuilder([pointer] * len(logical_buffers))
    arguments = []
    for ptr, declaration in zip(builder.entry.args, logical_buffers, strict=True):
        shape = declaration['shape']
        strides = []
        stride = 1
        for extent in reversed(shape):
            strides.insert(0, stride)
            stride *= extent
        arguments.extend((ptr, ptr, builder.const(0)))
        arguments.extend(builder.const(extent) for extent in shape)
        arguments.extend(builder.const(stride) for stride in strides)
    if tuple(value.type for value in arguments) != tuple(graph.function_type.inputs):
        raise ValueError('stock flattened ranked descriptor ABI differs from original static tensors')
    builder.add(llvm.CallOp(graph_symbol, *arguments))
    wrapper = llvm.FuncOp(symbol, llvm.LLVMFunctionType([pointer] * len(logical_buffers), llvm.LLVMVoidType()),
                          linkage=llvm.LinkageAttr('external'), body=builder.finish())
    for operation in wrapper.walk():
        if operation is not wrapper and operation.name != 'llvm.return':
            operation.attributes['merlin.global_task'] = IntegerAttr(task_index, i64)
    objects.append(wrapper)
    product = ModuleOp(objects)
    product.verify()
    return _text(product)
