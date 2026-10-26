"""Emit a rerolled primitive stream as a RISC-V device kernel and compile it.

Every distinct command form is first verified as a Gemmini-dialect op against the
RTL-derived limits (``verify_forms``).  The stream is encoded to RoCC words,
rerolled into ordinary CPU loops (``reroll``), checked to re-expand to the exact
words, and emitted as LLVM inline asm whose operands are affine in the loop
induction variables.  Kernel ABI: ``void <symbol>(ptr arg0, ptr arg1, ...)`` over
the stream's runtime arguments in declaration order (dense row-major buffers).
Compile-time constant operands become private read-only globals.  A DRAM operand
is always a pointer argument (or constant global) plus an offset; the only literal
DRAM address is the zero-fill primitive's address 0.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from xdsl.dialects import llvm
from xdsl.dialects.builtin import (BytesAttr, DenseIntOrFPElementsAttr, Float32Type, FloatAttr, IntegerAttr, ModuleOp,
                                   StringAttr, TensorType, i8, i64)

from ..codegen.builder import PTR, FnBuilder
from ..ir import gemmini_dialect as G
from ..tables import isa
from .reroll import Rec, reroll
from .stream import Ins, Stream

_OPS = {
    "flush": G.FlushOp, "config_ex": G.ConfigExOp, "config_ld": G.ConfigLdOp, "config_st": G.ConfigStOp,
    "mvin": G.MvinOp, "mvout": G.MvoutOp, "preload": G.PreloadOp, "compute": G.ComputeOp, "fence": G.FenceOp,
}
_DROP = {"zero", "multiplier_bits"}
_MASK = (1 << 64) - 1


def _attr(value):
    if isinstance(value, bool):
        return IntegerAttr(int(value), i64)
    if isinstance(value, int):
        return IntegerAttr(value, i64)
    if isinstance(value, float):
        return FloatAttr(value, Float32Type())
    return StringAttr(str(value))


def encode(ins: Ins) -> Rec:
    """The RoCC words of one command; a DRAM rs1 excludes its argument's base pointer."""
    a = ins.a
    k = ins.kind
    if k == "fence":
        return Rec(("fence",), ())
    if k == "flush":
        f, r1, r2 = isa.flush(0)
    elif k == "config_ex":
        f, r1, r2 = isa.config_ex(dataflow=a("dataflow"), sys_act=a("act"), acc_scale=a("acc_scale"),
                                  a_stride=a("a_stride"), c_stride=a("c_stride"),
                                  a_transpose=a("a_transpose"), b_transpose=a("b_transpose"))
    elif k == "config_ld":
        f, r1, r2 = isa.config_ld(stride=a("stride"), scale=a("scale"), shrunk=a("shrunk"),
                                  block_stride=a("block_stride"), pixel_repeats=a("pixel_repeats"),
                                  load_id=a("load_id"))
    elif k == "config_st":
        if a('multiplier_bits') is None:
            f, r1, r2 = isa.config_st(stride=a('stride'), acc_act=a('acc_act'), acc_scale=a('acc_scale'))
        else:
            from .readout_encoding import config_st
            if a('multiplier_bits') != isa.f32_bits(a('acc_scale')):
                raise ValueError('readout scale differs from its supplied multiplier bits')
            from .pool_plan import POOL_FIELDS
            pool = {k: a(k) for k in POOL_FIELDS if a(k) is not None}
            f, r1, r2 = config_st(a('stride'), a('acc_act'), a('multiplier_bits'), pool)
    elif k == "mvin":
        f, _, r2 = isa.mvin(local_addr=a("local"), cols=a("cols"), rows=a("rows"), load_id=a("load_id"))
        r1 = 0 if a("zero") else isa._u64(ins.off)
    elif k == "mvout":
        f, _, r2 = isa.mvout(local_addr=a("local"), cols=a("cols"), rows=a("rows"))
        r1 = isa._u64(ins.off)
    elif k == "preload":
        f, r1, r2 = isa.preload(bd_addr=a("bd"), c_addr=a("c"), bd_cols=a("bd_cols"), bd_rows=a("bd_rows"),
                                c_cols=a("c_cols"), c_rows=a("c_rows"))
    elif k == "compute":
        f, r1, r2 = isa.compute(a_addr=a("a"), bd_addr=a("bd"), a_cols=a("a_cols"), a_rows=a("a_rows"),
                                bd_cols=a("bd_cols"), bd_rows=a("bd_rows"), accumulate=a("accumulate"))
    else:
        raise ValueError(f"no encoding for {k}")
    isa.assert_legal(f)
    arg = None if (k == "mvin" and a("zero")) else ins.arg
    return Rec((f, arg), (int(r1), int(r2)))


def _signed(v: int) -> int:
    v &= _MASK
    return v - (1 << 64) if v >> 63 else v


def verify_forms(stream: Stream) -> int:
    """Verify one Gemmini-dialect op per distinct static command form against the RTL limits."""
    seen, ops = set(), []
    fb = FnBuilder([PTR])
    from .stream import CompactInstructions
    for ins in (stream.ins.templates() if isinstance(stream.ins, CompactInstructions) else stream.ins):
        form = (ins.kind, tuple((k, v) for k, v in ins.attrs if k not in _DROP))
        if form in seen:
            continue
        seen.add(form)
        operands = [fb.entry.args[0]] if ins.kind in ("mvin", "mvout") else []
        op = _OPS[ins.kind](operands=[operands], result_types=[[]])
        op.attributes.update({k: _attr(v) for k, v in form[1]})
        fb.add(op)
        ops.append(op)
    fn = llvm.FuncOp("forms", llvm.LLVMFunctionType([PTR]), linkage=llvm.LinkageAttr("external"), body=fb.finish())
    ModuleOp([fn]).verify()
    return len(ops)


class _Emitter:
    def __init__(self, stream: Stream, symbol: str, meta=None):
        self.meta = meta if meta is not None else {}
        self.meta["host_readout_contract"] = "exact_rational_RNE_v1"
        self.stream = stream
        self.symbol = symbol
        self.ptr_type = PTR
        runtime = [a for a in stream.args if a.role != "const"]
        self.fb = FnBuilder([PTR] * len(runtime))
        self.base = {}
        self.globals = []
        k = 0
        for idx, arg in enumerate(stream.args):
            if arg.role == "const":
                data = stream.consts[arg.name].reshape(-1).view(np.uint8)
                name = f"{symbol}__{arg.name}"
                value = DenseIntOrFPElementsAttr(TensorType(i8,[data.size]),BytesAttr(data.tobytes()))
                self.globals.append(llvm.GlobalOp(llvm.LLVMArrayType(data.size, i8), name,
                                                  "internal", constant=True, value=value, alignment=64))
                ptr = self.fb.add(llvm.AddressOfOp(name, PTR)).results[0]
            else:
                ptr = self.fb.entry.args[k]
                k += 1
            self.base[idx] = self.fb.add(llvm.PtrToIntOp(ptr, i64)).results[0]

    def _value(self, const: int, terms, base=None):
        fb = self.fb
        out = base
        c = _signed(const)
        if c or out is None:
            cv = fb.const(c)
            out = cv if out is None else fb.add_i(out, cv)
        for iv, step in terms:
            step = _signed(step)
            if step:
                out = fb.add_i(out, fb.mul_i(iv, fb.const(step)))
        return out

    def leaf(self, rec: Rec, terms):
        if rec.key == ("fence",):
            self.fb.add(llvm.InlineAsmOp("fence", "~{memory}", [], [], has_side_effects=True))
            return
        funct, arg = rec.key
        rs1 = self._value(rec.values[0], [(iv, s[0]) for iv, s in terms],
                          self.base[arg] if arg is not None else None)
        rs2 = self._value(rec.values[1], [(iv, s[1]) for iv, s in terms])
        self.fb.add(llvm.InlineAsmOp(isa.asm_string(funct), "r,r,~{memory}", [rs1, rs2], [], has_side_effects=True))

    def nodes(self, nodes, terms):
        """``terms``: [(iv, [step per flat value of ``nodes``])]."""
        pos = 0
        for node in nodes:
            width = len(flat_values_of(node))
            sub = [(iv, steps[pos:pos + width]) for iv, steps in terms]
            if isinstance(node, Rec):
                self.leaf(node, sub)
            else:
                def body(iv, node=node, sub=sub):
                    self.nodes(node.body, sub + [(iv, list(node.step))])
                self.fb.for_loop(0, node.count, 1, body)
            pos += width

    def module(self, tree) -> ModuleOp:
        if self.stream.packs:
            from .host_readout import emit_packs
            emit_packs(self)
        if self.stream.readouts:
            from .host_readout import emit
            groups={}
            for r in self.stream.readouts:
                groups.setdefault(len(self.stream.ins) if r.after is None else r.after,[]).append(r)
            start=0
            for boundary,readouts in sorted(groups.items()):
                self.nodes(instruction_tree(self.stream.ins[start:boundary]),[])
                emit(self,readouts)
                start=boundary
            self.nodes(instruction_tree(self.stream.ins[start:]),[])
        else:
            self.nodes(tree,[])
        fn = llvm.FuncOp(self.symbol, llvm.LLVMFunctionType([PTR] * len(self.fb.entry.args)),
                         linkage=llvm.LinkageAttr("external"), body=self.fb.finish())
        module = ModuleOp([*self.globals, fn])
        module.verify()
        return module


def flat_values_of(node):
    from .reroll import flat_values
    return flat_values(node)


def instruction_tree(instructions):
    from .stream import CompactInstructions, InstructionLoop
    from .reroll import Loop
    if not isinstance(instructions, CompactInstructions):
        return reroll([encode(i) for i in instructions])
    def nodes(items):
        result = []
        for item in items:
            if isinstance(item, InstructionLoop):
                body = nodes(item.body)
                strides = dict(item.strides)
                templates = CompactInstructions(item.body).templates()
                step = tuple(v for ins in templates for v in (strides.get(ins.arg, 0), 0))
                result.append(Loop(item.count, tuple(body), step))
            else:
                result.append(encode(item))
        return result
    return nodes(instructions.nodes)


def build_module(stream: Stream, symbol: str, *, check_expand: bool = True, meta=None) -> tuple[ModuleOp, list]:
    if meta is None:
        meta = {}
    from .reroll import iter_expand
    from itertools import zip_longest
    tree = instruction_tree(stream.ins)
    if check_expand and any(a != b for a, b in zip_longest(iter_expand(tree), (encode(i) for i in stream.ins))):
        raise AssertionError("loop tree does not reproduce the recipe stream")
    verify_forms(stream)
    module = _Emitter(stream, symbol, meta).module(tree)
    module.attributes["gemmini.core_aten_kernel"] = StringAttr(symbol)
    if stream.readouts:
        module.attributes["gemmini.host_readout_contract"] = StringAttr("exact_rational_RNE_v1")
    return module, tree


def llvm_bin_dir(explicit: str | Path | None = None) -> Path:
    for cand in (explicit, os.environ.get("GEMMINI_LLVM_BIN"),
                 (Path(os.environ["MERLIN_MLIR_INSTALL"]) / "bin") if os.environ.get("MERLIN_MLIR_INSTALL") else None):
        if cand:
            return Path(cand)
    found = shutil.which("mlir-translate")
    if found:
        return Path(found).parent
    raise RuntimeError("no LLVM/MLIR bin directory: set GEMMINI_LLVM_BIN")


def compile_to_object(module: ModuleOp, workdir: Path, *, llvm_bin: Path | None = None, stem: str = "kernel") -> dict:
    """Lower the target module to RoCC inline asm and assemble a RISC-V object."""
    from ..no_fsm_audit import audit_elf

    workdir.mkdir(parents=True, exist_ok=True)
    bindir = llvm_bin_dir(llvm_bin)
    llvm_mlir = workdir / f"{stem}.llvm.mlir"
    llvm_ir = workdir / f"{stem}.ll"
    obj = workdir / f"{stem}.o"
    from .storage import module_text
    llvm_mlir.write_text(module_text(module) + "\n")
    commands = [[str(bindir / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir), "-o", str(llvm_ir)]]
    if (bindir / "clang").exists():
        commands.append([str(bindir / "clang"), "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d",
                         "-mcmodel=medany", "-O2", "-c", str(llvm_ir), "-o", str(obj)])
    else:
        commands.append([str(bindir / "llc"), "-mtriple=riscv64-unknown-elf", "-mattr=+m,+a,+f,+d,+c",
                         "-target-abi=lp64d", "-code-model=medium", "-O2", "-filetype=obj", str(llvm_ir),
                         "-o", str(obj)])
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    audit = audit_elf(obj.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("device object contains a forbidden Gemmini command")
    return {"object": str(obj), "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
            "argv": commands, "nofsm": audit["status"]}
