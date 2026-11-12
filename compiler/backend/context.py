"""Logical-buffer LLVM emission and a single terminal RoCC encoder.

Tensor arithmetic belongs to the selected device/host lowerers.  This context
only provides addresses, storage, ordinary control flow and command submission.
"""
from __future__ import annotations

from math import prod
from pathlib import Path
import re

from xdsl.dialects import builtin, llvm
from xdsl.dialects.builtin import IntAttr, IntegerType, i8, i64
from xdsl.ir import Block, Region, SSAValue
from xdsl.printer import Printer
from io import StringIO

from merlin.compiler.ir.logical_abi import kernel_arg_order, logical_abi
from merlin.compiler.lower.model.host_builder import FnBuilder, PTR


class EmissionContext:
    def __init__(self, command_buffer, symbol="kernel_entry"):
        self.cb = command_buffer
        self.symbol = symbol
        abi = logical_abi(Path(__file__).parent / "data/abi.yaml")
        self.arg_names = kernel_arg_order(command_buffer, abi)
        self.fb = FnBuilder([PTR] * len(self.arg_names))
        self.buffers = dict(zip(self.arg_names, self.fb.entry.args, strict=True))
        self.tensors = dict(command_buffer.get("tensors", {}))
        self.globals = []
        self.emitted = []
        self._has_opening_fence = False
        self.metadata = {}
        self.current_source_index = None
        for name, spec in command_buffer.get("constants", {}).items():
            self.constant(name, spec["shape"], spec["dtype"], spec["values"])
        for name, alias in command_buffer.get("params", {}).get("tensor_aliases", {}).items():
            self.register(name, self.buffer(alias["source"]), alias["shape"], alias["dtype"])

    def spec(self, name):
        if name not in self.tensors:
            raise ValueError(f"no typed tensor declaration for {name!r}")
        return self.tensors[name]

    def buffer(self, name):
        if name not in self.buffers:
            spec = self.spec(name)
            self.alloc(name, spec["shape"], spec["dtype"])
        return self.buffers[name]

    def ptr(self, name, elem_offset=0):
        base = self.buffer(name)
        if isinstance(elem_offset, int) and elem_offset == 0:
            return base
        dtype = self.spec(name)["dtype"]
        if not dtype.startswith("i"):
            raise ValueError(f"unbound element address type {dtype}")
        ty = IntegerType(int(dtype[1:]))
        index = self.fb.const(elem_offset) if isinstance(elem_offset, int) else elem_offset
        return self.fb.gep(base, index, ty)

    def register(self, name, pointer, shape, dtype):
        self.tensors[name] = {"shape": list(shape), "dtype": dtype, "role": "intermediate"}
        self.buffers[name] = pointer
        return pointer

    def alloc(self, name, shape, dtype):
        if name in self.buffers:
            return self.buffers[name]
        nbytes = prod(shape) * ((int(dtype[1:]) + 7) // 8)
        # Private globals avoid stack-frame size depending on model dimensions.
        # All live contents are initialized by their producer on every call.
        ty = llvm.LLVMArrayType(IntAttr(max(1, nbytes)), i8)
        block = Block()
        zero = llvm.ZeroOp(result_types=[ty])
        block.add_op(zero)
        block.add_op(llvm.ReturnOp(zero.results[0]))
        unique = re.sub(r"[^a-zA-Z0-9_]", "_", f"{self.symbol}_scratch_{len(self.globals)}_{name}")
        glob = llvm.GlobalOp(ty, unique, "internal", alignment=64, body=Region(block))
        self.globals.append(glob)
        addr = llvm.AddressOfOp(unique, PTR)
        self.fb.entry.insert_op_before(addr, self.fb.entry.first_op) if self.fb.entry.first_op else self.fb.entry.add_op(addr)
        return self.register(name, addr.result, shape, dtype)

    def constant(self, name, shape, dtype, values):
        """Materialize immutable source/configuration bytes at link time."""
        if name in self.buffers:
            return self.buffers[name]
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType
        width = int(dtype[1:])
        element = IntegerType(width)
        count = prod(shape)
        if len(values) != count:
            raise ValueError("constant initializer extent differs from its typed storage")
        signed = [((int(v) + (1 << (width - 1))) % (1 << width)) - (1 << (width - 1)) for v in values]
        value = DenseIntOrFPElementsAttr.from_list(TensorType(element, [count]), signed)
        unique = re.sub(r"[^a-zA-Z0-9_]", "_", f"{self.symbol}_literal_{len(self.globals)}_{name}")
        glob = llvm.GlobalOp(llvm.LLVMArrayType(IntAttr(count), element), unique,
                             "internal", constant=True, value=value, alignment=64)
        self.globals.append(glob)
        address = llvm.AddressOfOp(unique, PTR)
        self.fb.entry.insert_op_before(address, self.fb.entry.first_op) if self.fb.entry.first_op else self.fb.entry.add_op(address)
        return self.register(name, address.result, shape, dtype)

    def word(self, value):
        if isinstance(value, int):
            # IntegerAttr accepts signed representations; preserve all 64 bits.
            value = value & ((1 << 64) - 1)
            return self.fb.const(value - (1 << 64) if value >= (1 << 63) else value)
        if isinstance(value, llvm.LLVMPointerType):
            raise TypeError("expected SSA pointer, not its type")
        value = SSAValue.get(value)
        if value.type == PTR:
            return self.fb.add(llvm.PtrToIntOp(value, i64)).results[0]
        if value.type != i64:
            return self.fb.add(llvm.ZExtOp(value, i64)).results[0]
        return value

    def emit(self, funct, rs1=0, rs2=0):
        from .isa import CUSTOM_OPCODE
        if not self.emitted and not self._has_opening_fence:
            self.fence()
        op = llvm.InlineAsmOp(
            f".insn r {CUSTOM_OPCODE}, 3, {int(funct)}, x0, $0, $1",
            "r,r,~{memory}", [self.word(rs1), self.word(rs2)],
            has_side_effects=True,
        )
        if self.cb.get("params", {}).get("frontend_entry_bindings") is not None:
            op.attributes["merlin.global_task"] = builtin.IntegerAttr(0, i64)
            if self.current_source_index is not None:
                op.attributes["merlin.source_op_index"] = builtin.IntegerAttr(self.current_source_index, i64)
        self.fb.add(op)
        self.emitted.append(int(funct))
        return op

    def fence(self):
        from .completion import emit_fence
        emit_fence(self.fb)
        self._has_opening_fence = True

    def finish(self):
        self.fence()
        func = llvm.FuncOp(self.symbol, llvm.LLVMFunctionType([PTR] * len(self.arg_names)),
                           linkage=llvm.LinkageAttr("external"), body=self.fb.finish())
        if self.cb.get("params", {}).get("frontend_entry_bindings") is not None:
            for operation in func.walk():
                if operation is not func:
                    operation.attributes["merlin.global_task"] = builtin.IntegerAttr(0, i64)
        functions = [func]
        if self.symbol == "kernel_entry":
            done = FnBuilder([])
            from .completion import emit_fence
            emit_fence(done)
            functions.append(llvm.FuncOp("kernel_complete", llvm.LLVMFunctionType([]), linkage=llvm.LinkageAttr("external"), body=done.finish()))
        module = builtin.ModuleOp([*self.globals, *functions])
        module.verify()
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        return stream.getvalue() + "\n"
