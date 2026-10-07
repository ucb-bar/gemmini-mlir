"""``gemmini`` dialect -> an LLVM-dialect module of raw ``.insn`` inline assembly.

The emitted module defines ``gemmini_kernel`` and takes one ``!llvm.ptr`` per DRAM buffer, in the
ABI order the lowering picked. Every instruction is the canonical form stock clang/LLVM assembles:

    %c = llvm.mlir.constant(<imm>) : i64
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, <funct>, x0, $0, $1", "r,r" %a, %c

Both operands are SSA values -- an immediate through ``llvm.mlir.constant`` and a DRAM address
through ``llvm.ptrtoint`` of the matching pointer argument (plus a constant tile offset). No DRAM
address is ever a literal: the harness allocates the buffers at run time.

A tiled contraction issues the same commands once per tile, so the stream is RE-ROLLED into loops
first (:mod:`mlir_oot.codegen.reroll`): the issued command sequence is unchanged, but the emitted
code stops growing with the payload. Inside a loop the two operand words are recomputed from the
induction variable, which is addressing, not compute.
"""

from __future__ import annotations

from io import StringIO

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i8, i64
from xdsl.ir import Block, Region, SSAValue
from xdsl.printer import Printer

from ..target import dialect as gd
from ..target import isa
from .isa_stream import flatten
from .reroll import Rolled, reroll

KERNEL_SYMBOL = "gemmini_kernel"
#: llvm.icmp predicate 6 = "ult" (unsigned less-than), the loop-back condition.
ICMP_ULT = 6


class CodegenError(Exception):
    pass


class LLVMEmitter:
    def __init__(self) -> None:
        self.ptr = llvm.LLVMPointerType()

    def emit(self, module: ModuleOp, quant=None, lane_program=None,
             scratch: dict[str, int] | None = None) -> ModuleOp:
        kernel = next((o for o in module.body.block.ops if isinstance(o, gd.KernelOp)), None)
        if kernel is None:
            raise CodegenError("the target module defines no gemmini.kernel")
        arg_tensors = [a.type.tensor.data for a in kernel.body.block.args]
        stream = flatten(kernel, arg_tensors)
        # A HYBRID kernel is issued straight-line. Re-rolling keeps the emitted code from growing
        # with the payload, which is why every other program takes it -- but it also means the
        # kernel BODY no longer states the commands it issues one for one, and a consumer reading
        # the artifact statically then measures a five-tile store loop as a single store. A hybrid
        # program's command stream is bounded by what its off-mesh stages can carry, so it is small
        # enough to state in full, and stating it in full is worth more than the code it saves.
        has_stage = any(isinstance(n, str) and n.startswith("lane:") for n in stream)
        nodes = list(stream) if has_stage else reroll(stream)

        # Route Q stages an operand on the stack: it is a kernel-internal buffer, so it carries a
        # DRAM address like any other tensor but is NOT one of the pointers the harness passes in.
        staged = {s.dst for s in quant.staged} if quant is not None else set()
        # Route W's digit staging is the same kind of buffer: addressed like a DRAM tensor, but
        # allocated by the kernel rather than handed in, so it is NOT one of the ABI pointers.
        scratch = dict(scratch or {})
        staged |= set(scratch)
        params = [n for n in arg_tensors if n not in staged]
        entry = Block(arg_types=[self.ptr] * len(params))
        blocks = [entry]
        prelude: list = []
        consts: dict[int, SSAValue] = {}
        bases: dict[str, SSAValue] = {}
        addrs: dict[tuple[str, int], SSAValue] = {}
        # Route X: the off-mesh stages this kernel's markers expand into.
        lane_wl, lane_stages = lane_program if lane_program is not None else (None, [])
        helpers: dict = {}
        tables: dict = {}
        stage_fns: list = []
        own: dict[str, SSAValue] = {}
        for name, nbytes in scratch.items():
            size = llvm.ConstantOp(IntegerAttr(int(nbytes), i64), i64)
            prelude.append(size)
            alloca = llvm.AllocaOp(size.results[0], i8, alignment=64)
            prelude.append(alloca)
            own[name] = alloca.res
        lane = None
        if quant is not None:
            from .quant_lane import QuantLane

            lane = QuantLane(quant, lambda v: const(v), blocks, prelude, helpers)
            lane.allocate(entry)

        def const(v: int) -> SSAValue:
            key = int(v) & isa.MASK64
            if key not in consts:
                op = llvm.ConstantOp(IntegerAttr(key, i64), i64)
                prelude.append(op)
                consts[key] = op.results[0]
            return consts[key]

        def ptr_of(tensor: str) -> SSAValue:
            """The POINTER for ``tensor``: the harness's argument, or route Q's own staging."""
            if tensor in own:
                return own[tensor]
            if lane is not None and tensor in lane.staging:
                return lane.staging[tensor]
            try:
                return entry.args[params.index(tensor)]
            except ValueError as exc:
                raise CodegenError(f"tensor {tensor!r} is not a kernel pointer") from exc

        def base_of(tensor: str) -> SSAValue:
            if tensor not in bases:
                op = llvm.PtrToIntOp(ptr_of(tensor), i64)
                prelude.append(op)
                bases[tensor] = op.output
            return bases[tensor]

        def operand(o) -> SSAValue:
            if isinstance(o, isa.Imm):
                return const(o.value)
            key = (o.tensor, o.byte_offset)
            if key not in addrs:
                b = base_of(o.tensor)
                if o.byte_offset == 0:
                    addrs[key] = b
                else:
                    op = llvm.AddOp(b, const(o.byte_offset))
                    prelude.append(op)
                    addrs[key] = op.results[0]
            return addrs[key]

        # -- pre-materialise every loop-invariant operand ---------------------
        for node in nodes:
            items = node.body if isinstance(node, Rolled) else [node]
            for k, instr in enumerate(items):
                if isinstance(instr, str):
                    continue
                steps = node.steps[k] if isinstance(node, Rolled) else (0, 0)
                for o, s in ((instr.rs1, steps[0]), (instr.rs2, steps[1])):
                    if s == 0:
                        operand(o)
                    else:
                        const(s)
                        if isinstance(o, isa.ArgAddr):
                            operand(o)
                        else:
                            const(o.value)
            if isinstance(node, Rolled):
                const(node.trip)
                const(1)
                const(0)

        cur = entry
        if lane is not None:
            cur = lane.prologue(cur, ptr_of)

        def issue(blk: Block, instr: isa.Instr, rs1: SSAValue, rs2: SSAValue) -> None:
            blk.add_op(
                llvm.InlineAsmOp(instr.asm, "r,r", [rs1, rs2], [], has_side_effects=True)
            )

        for node in nodes:
            if isinstance(node, str):
                if node.startswith("lane:"):
                    from .scalar_lane import compile_stage

                    idx = int(node.split(":", 1)[1])
                    if lane_wl is None or idx >= len(lane_stages):
                        raise CodegenError(
                            f"the stream marks lane stage {idx} with no stage to fill it")
                    sym = f"__merlin_lane_stage_{idx}"
                    fn, names = compile_stage(
                        lane_wl, lane_stages[idx], sym, helpers, tables)
                    stage_fns.append(fn)
                    cur.add_op(llvm.CallOp(sym, *[ptr_of(n) for n in names]))
                    continue
                cur.add_op(llvm.InlineAsmOp("fence", "", [], [], has_side_effects=True))
                continue
            if isinstance(node, isa.Instr):
                issue(cur, node, operand(node.rs1), operand(node.rs2))
                continue
            body = Block(arg_types=[i64])
            tail = Block(arg_types=[])
            blocks += [body, tail]
            cur.add_op(llvm.BrOp(body, const(0)))
            iv = body.args[0]
            #: one `iv * step` per DISTINCT step in the body, reused by every operand that advances
            #: at that rate -- an unrolled body repeats the same handful of strides.
            scaled: dict[int, SSAValue] = {}

            def stride(s: int) -> SSAValue:
                if s not in scaled:
                    mul = llvm.MulOp(iv, const(s))
                    body.add_op(mul)
                    scaled[s] = mul.results[0]
                return scaled[s]

            for k, instr in enumerate(node.body):
                s1, s2 = node.steps[k]
                ops_: list[SSAValue] = []
                for o, s in ((instr.rs1, s1), (instr.rs2, s2)):
                    if s == 0:
                        ops_.append(operand(o))
                        continue
                    start = operand(o) if isinstance(o, isa.ArgAddr) else const(o.value)
                    add = llvm.AddOp(start, stride(s))
                    body.add_op(add)
                    ops_.append(add.results[0])
                issue(body, instr, ops_[0], ops_[1])
            nxt = llvm.AddOp(iv, const(1))
            body.add_op(nxt)
            cmp = llvm.ICmpOp(nxt.results[0], const(node.trip), IntegerAttr(ICMP_ULT, i64))
            body.add_op(cmp)
            body.add_op(llvm.CondBrOp(cmp.res, body, [nxt.results[0]], tail, []))
            cur = tail

        if lane is not None:
            cur = lane.epilogue(cur, ptr_of)
        cur.add_op(llvm.ReturnOp())
        for op in reversed(prelude):
            entry.insert_op_before(op, entry.first_op) if entry.first_op else entry.add_op(op)
        fn = llvm.FuncOp(
            KERNEL_SYMBOL,
            llvm.LLVMFunctionType([self.ptr] * len(params)),
            linkage=llvm.LinkageAttr("external"),
            body=Region(blocks),
        )
        out = ModuleOp([*tables.values(), *helpers.values(), *stage_fns, fn])
        out.verify()
        return out


def to_text(module: ModuleOp) -> str:
    s = StringIO()
    Printer(stream=s).print_op(module)
    return s.getvalue() + "\n"
