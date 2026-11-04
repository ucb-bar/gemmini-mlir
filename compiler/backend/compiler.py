"""Source command graph to a compact LLVM CFG containing basic device commands."""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, IntegerAttr, StringAttr, i8, i16, i32, i64, f16, f32, f64
from xdsl.ir import Block, Region, SSAValue
from xdsl.printer import Printer

from merlin.compile.logical_abi import logical_abi, logical_interface
from merlin.llvmlower.host_builder import FnBuilder, PTR
from merlin.compile.tuning_knobs import knob
from . import isa

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"i8": i8, "i16": i16, "i32": i32, "i64": i64, "f16": f16, "f32": f32, "f64": f64}


class TaggedBuilder(FnBuilder):
    def __init__(self, types):
        super().__init__(types)
        self.task_index = None
        self.source_op_index = None
        self.region_id = None

    def tag(self, op):
        if self.task_index is not None:
            op.attributes["merlin.global_task"] = IntegerAttr(self.task_index, i64)
        if self.source_op_index is not None:
            op.attributes["merlin.source_op_index"] = IntegerAttr(self.source_op_index, i64)
        if self.region_id:
            op.attributes["prov.region_id"] = StringAttr(self.region_id)
        return op

    def add(self, op):
        return super().add(self.tag(op))


class Context:
    def __init__(self, cb):
        self.cb = cb
        self.abi = logical_abi(ROOT / "contracts" / "mlir_oot_backend_contract.yaml")
        self.buffers = logical_interface(cb, self.abi)
        self.f = TaggedBuilder([PTR] * len(self.buffers))
        self.pointers = {b.name: a for b, a in zip(self.buffers, self.f.entry.args)}
        self.globals = []
        for b in self.buffers:
            cb.setdefault("tensors", {}).setdefault(b.name, {
                "shape": list(b.shape), "dtype": b.dtype,
                "role": "output" if b.kind == "output" else "input"})
        if not cb.get("kernel_abi"):
            cb["kernel_abi"] = {
                "kind": "whole_program",
                "args": [{"tensor": b.name, "access": "write" if b.kind == "output" else "read"}
                         for b in self.buffers],
                "outputs": [b.name for b in self.buffers if b.kind == "output"]}
        self.storage_plan = None
        self.storage_slots = {}
        if knob("storage_reuse", default=False, values=(False, True), root=ROOT):
            from .allocation import plan_storage
            self.storage_plan = plan_storage(cb, self.pointers)
            cb.setdefault("params", {})["storage_plan"] = self.storage_plan

    def ptr(self, name):
        if isinstance(name, SSAValue):
            return name
        if name in self.pointers:
            return self.pointers[name]
        spec = self.cb["tensors"].get(name)
        if spec is None:
            raise ValueError(f"no storage declaration for {name!r}")
        return self.alloc(name, spec["shape"], spec["dtype"])

    def alloc(self, name, shape, dtype, *, constant=False):
        if name in self.pointers:
            return self.pointers[name]
        count = math.prod(shape)
        if count <= 0 or dtype not in TYPES:
            raise ValueError(f"invalid internal allocation {name}: {shape} {dtype}")
        if self.storage_plan and not constant:
            planned = self.storage_plan["tensors"].get(name)
            if planned is not None:
                slot = planned["slot"]
                description = self.storage_plan["slots"][slot]
                widths = {"i8": 1, "i16": 2, "i32": 4, "i64": 8,
                          "f16": 2, "f32": 4, "f64": 8}
                if dtype != description["dtype"] or count * widths[dtype] > description["bytes"]:
                    raise ValueError(f"allocation disagrees with storage proof for {name}")
                if slot in self.storage_slots:
                    self.pointers[name] = self.storage_slots[slot]
                    return self.pointers[name]
                slot_name = "__storage_slot_" + str(slot)
                while slot_name in self.cb["tensors"] or slot_name in self.pointers:
                    slot_name += "_"
                pointer = self.alloc(slot_name,
                                     [description["bytes"] // widths[dtype]], dtype)
                self.storage_slots[slot] = self.pointers[name] = pointer
                return pointer
        ty = llvm.LLVMArrayType(count, TYPES[dtype])
        init = Block()
        z = llvm.ZeroOp(result_types=[ty])
        init.add_ops([z, llvm.ReturnOp(z.results[0])])
        symbol = "scratch_" + str(len(self.globals))
        self.globals.append(llvm.GlobalOp(ty, symbol, "internal", constant=constant,
                                         alignment=64, body=Region(init)))
        address = llvm.AddressOfOp(symbol, PTR)
        address.attributes["merlin.global_task"] = IntegerAttr(-1, i64)
        # An address dominates its uses even when first requested inside a loop.
        if self.f.entry.last_op and self.f.entry.last_op.has_trait(llvm.IsTerminator):
            self.f.entry.insert_op_before(address, self.f.entry.last_op)
        else:
            self.f.entry.add_op(address)
        self.pointers[name] = address.results[0]
        return address.results[0]


def _attrs(command):
    return command.get("attributes") or {}


def compile_buffer(command_buffer):
    from .admission import explicit_decline, declare_interface_placement

    declined = explicit_decline(command_buffer)
    if declined is not None:
        return declined, "module {}\n"

    from .epilogue import lower_matmul_epilogue
    from .convolution import lower_conv
    from .elementwise import lower_elementwise
    from .schedule import install
    from .graph import fuse_sibling_matmuls
    from .resident import plan_resident_weights, ATTRIBUTE as RESIDENT_WEIGHT

    cb = fuse_sibling_matmuls(copy.deepcopy(command_buffer))
    declare_interface_placement(cb)
    resident_plan = plan_resident_weights(cb, enabled=knob(
        "resident_weights", default=False, values=(False, True), root=ROOT))
    ctx = Context(cb)
    schedule = install(ctx.f)
    ctx.f.task_index = -1
    isa.fence(ctx.f)
    isa.emit(ctx.f, 7, 0, 0)
    body = Block()
    ctx.f.region.add_block(body)
    ctx.f.add(llvm.BrOp(body))
    ctx.f.blk = body
    residents, accumulators = {}, {}
    for command_index, command in enumerate(cb["commands"]):
        op = command["opcode"]
        operands, attrs = command.get("operands", {}), _attrs(command)
        ctx.f.task_index = attrs.get("task_index")
        ctx.f.source_op_index = attrs.get("source_op_index")
        ctx.f.region_id = attrs.get("source_region_id") or attrs.get("prov.region_id")
        if op == "RES_PACK":
            residents[operands["dst"]] = operands["src"]
        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            accumulators[operands["dst"]] = command
        elif op == "COMMIT":
            matmul = accumulators[operands["src"]]
            args = matmul["operands"]
            lhs, rhs = args["lhs"], residents.get(args["rhs"], args["rhs"])
            shape_a, shape_b = cb["tensors"][lhs]["shape"], cb["tensors"][rhs]["shape"]
            m, k, n = shape_a[-2], shape_a[-1], shape_b[-1]
            combined = {**_attrs(matmul), **attrs}
            if "bias" in operands:
                combined["bias"] = operands["bias"]
            if command_index in resident_plan:
                combined[RESIDENT_WEIGHT] = resident_plan[command_index]
            lower_matmul_epilogue(ctx, lhs, rhs, operands["dst"], m, k, n, combined)
        elif op == "CONV2D":
            lower_conv(ctx, command, residents)
        elif op in ("MOVEMENT", "RESIDUAL_ADD", "BIAS_ADD"):
            lower_elementwise(ctx, command)
        elif op in ("BATCHED_MATMUL", "ATTENTION_QK", "ATTENTION_PV"):
            from .batched import lower_batched
            lower_batched(ctx, command, residents)
        elif op == "VECTOR_MAP" and attrs.get("op") == "shared_rhs_matmul":
            from .shared_rhs import lower_shared_rhs
            lower_shared_rhs(ctx, command)
        elif op == "VECTOR_MAP" and attrs.get("op") == "sibling_matmul":
            from .siblings import lower_siblings
            lower_siblings(ctx, command)
        elif op == "HOST_REGION" or (op == "VECTOR_MAP" and attrs.get("op") == "host_region"):
            from .host import lower_host_task
            for name in attrs.get("input_tensors", []) + attrs.get("output_tensors", []):
                ctx.ptr(name)
            lower_host_task(ctx.f, command, ctx.pointers, allocate=ctx.alloc)
        elif op == "EVICT":
            residents.pop(operands.get("handle"), None)
        elif op in ("BARRIER", "FENCE", "SYNC"):
            isa.fence(ctx.f)
        else:
            raise ValueError(f"target lowering is not implemented for {op}")
        # Region composition also creates join branches directly on blocks.
        # Bind those executable edges to the source task that created them.
        for lowered in ctx.f.region.walk():
            if "merlin.global_task" not in lowered.attributes:
                ctx.f.tag(lowered)
    ctx.f.task_index = -2
    ctx.f.source_op_index = ctx.f.region_id = None
    isa.fence(ctx.f)
    cb.setdefault("params", {})["terminal_schedule"] = schedule.snapshot()
    function = llvm.FuncOp("kernel_entry", llvm.LLVMFunctionType([PTR]*len(ctx.buffers), llvm.LLVMVoidType()),
                           linkage=llvm.LinkageAttr("external"), body=ctx.f.finish())
    completion = FnBuilder([])
    isa.fence(completion)
    complete_func = llvm.FuncOp("kernel_complete", llvm.LLVMFunctionType([], llvm.LLVMVoidType()),
                               linkage=llvm.LinkageAttr("external"), body=completion.finish())
    module = ModuleOp([*ctx.globals, function, complete_func])
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return cb, stream.getvalue() + "\n"
