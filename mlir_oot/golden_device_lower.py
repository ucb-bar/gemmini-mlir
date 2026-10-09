"""Lower verified primitive Gemmini xDSL ops to the RISC-V device artifact.

The same target module is used for inspection and compilation.  CPU loop blocks
remain ordinary LLVM CFG; only Gemmini primitive ops become side-effecting RoCC
inline asm.  Any target op without an exact lowering is refused.
"""

from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import i64
from xdsl.rewriter import Rewriter

from .codegen.builder import iconst
from .ir import gemmini_dialect as G
from .tables import isa, rtl_facts as F


def _encoded(op: G._GemminiOp) -> tuple[int, int | None, int] | None:
    a = op.a
    if isinstance(op, G.FenceOp):
        return None
    if isinstance(op, G.FlushOp):
        return isa.flush(op.a("skip", 0))
    if isinstance(op, G.ConfigExOp):
        return isa.config_ex(
            dataflow=a("dataflow"), sys_act=a("act", isa.NO_ACTIVATION),
            sys_shift=a("sys_shift", 0), acc_scale=a("acc_scale", 1.0),
            a_stride=a("a_stride", 1), c_stride=a("c_stride", 1),
            a_transpose=bool(a("a_transpose", 0)),
            b_transpose=bool(a("b_transpose", 0)),
            set_only_strides=bool(a("set_only_strides", 0)),
        )
    if isinstance(op, G.ConfigLdOp):
        return isa.config_ld(stride=a("stride"), scale=a("scale", 1.0),
                             load_id=a("load_id"), block_stride=a("block_stride",isa.DIM),
                             pixel_repeats=a("pixel_repeats",1), shrunk=bool(a("shrunk",0)))
    if isinstance(op, G.ConfigStOp):
        return isa.config_st(stride=a("stride"), acc_act=a("acc_act"),
                             acc_scale=a("acc_scale"),
                             **{key:a(key,0) for key in ("pool_stride","pool_size","pool_out_dim",
                                 "porows","pocols","orows","ocols","upad","lpad")})
    if isinstance(op, G.MvinOp):
        return isa.mvin(local_addr=a("local"), cols=a("cols"),
                        rows=a("rows"), load_id=a("load_id"))
    if isinstance(op, G.MvoutOp):
        return isa.mvout(local_addr=a("local"), cols=a("cols"), rows=a("rows"))
    if isinstance(op, G.PreloadOp):
        return isa.preload(bd_addr=a("bd"), c_addr=a("c"),
                           bd_cols=a("bd_cols"), bd_rows=a("bd_rows"),
                           c_cols=a("c_cols"), c_rows=a("c_rows"))
    if isinstance(op, G.ComputeOp):
        return isa.compute(a_addr=a("a"), bd_addr=a("bd", isa.GARBAGE_ADDR),
                           a_cols=a("a_cols"), a_rows=a("a_rows"),
                           bd_cols=a("bd_cols", F.DIM), bd_rows=a("bd_rows", F.DIM),
                           accumulate=bool(a("accumulate", False)))
    raise ValueError(f"no exact device lowering for {op.name}")


def lower(module):
    """Rewrite a golden target module in place, then verify no Gemmini op remains."""
    module.verify()
    _verify_dynamic_preload_rows(module)
    for op in list(module.walk()):
        if not isinstance(op, G._GemminiOp):
            continue
        if isinstance(op, G.PreloadOp) and "bd" in op.dynamic_rows():
            rows = dict(zip(op.dynamic_rows(), op.operands_, strict=True))
            funct, rs1_base, rs2_base = isa.preload(
                bd_addr=0,
                c_addr=isa.acc_addr(0, accumulate=bool(op.a("c_accumulate", 0)))
                if "c" in rows else op.a("c"),
                bd_cols=op.a("bd_cols"), bd_rows=op.a("bd_rows"),
                c_cols=op.a("c_cols"), c_rows=op.a("c_rows"),
            )
            isa.assert_legal(funct)
            b_high, c_high = iconst(rs1_base), iconst(rs2_base)
            b_packed = llvm.OrOp(rows["bd"], b_high.results[0])
            ops = [b_high, b_packed, c_high]
            rhs = c_high.results[0]
            if "c" in rows:
                c_packed = llvm.OrOp(rows["c"], c_high.results[0])
                ops.append(c_packed)
                rhs = c_packed.results[0]
            ops.append(llvm.InlineAsmOp(isa.asm_string(funct), "r,r",
                [b_packed.results[0], rhs], [], has_side_effects=True))
            Rewriter.replace_op(op, ops)
            continue
        if isinstance(op, G.PreloadOp) and len(op.operands_) == 1:
            funct, rs1, rs2_base = isa.preload(
                bd_addr=op.a("bd"),
                c_addr=isa.acc_addr(0, accumulate=bool(op.a("c_accumulate", 0))),
                bd_cols=op.a("bd_cols"), bd_rows=op.a("bd_rows"),
                c_cols=op.a("c_cols"), c_rows=op.a("c_rows"),
            )
            isa.assert_legal(funct)
            other, high = iconst(rs1), iconst(rs2_base)
            packed = llvm.OrOp(op.operands_[0], high.results[0])
            Rewriter.replace_op(op, [other, high, packed, llvm.InlineAsmOp(
                isa.asm_string(funct), "r,r", [other.results[0], packed.results[0]], [],
                has_side_effects=True,
            )])
            continue
        if isinstance(op, G.ComputeOp) and len(op.operands_) == 1:
            funct, rs1_base, rs2 = isa.compute(
                a_addr=0, bd_addr=op.a("bd", isa.GARBAGE_ADDR),
                a_cols=op.a("a_cols"), a_rows=op.a("a_rows"),
                bd_cols=op.a("bd_cols", F.DIM), bd_rows=op.a("bd_rows", F.DIM),
                accumulate=bool(op.a("accumulate", False)),
            )
            isa.assert_legal(funct)
            high = iconst(rs1_base)
            packed = llvm.OrOp(op.operands_[0], high.results[0])
            other = iconst(rs2)
            Rewriter.replace_op(op, [high, packed, other, llvm.InlineAsmOp(
                isa.asm_string(funct), "r,r", [packed.results[0], other.results[0]], [],
                has_side_effects=True,
            )])
            continue
        encoded = _encoded(op)
        if encoded is None:
            Rewriter.replace_op(op, llvm.InlineAsmOp(
                "fence", "", [], [], has_side_effects=True))
            continue
        funct, rs1, rs2 = encoded
        isa.assert_legal(funct)
        ops = []
        if rs1 is None:
            ptr = llvm.PtrToIntOp(op.operands_[0], i64)
            ops.append(ptr)
            lhs = ptr.results[0]
        else:
            c1 = iconst(rs1)
            ops.append(c1)
            lhs = c1.results[0]
        c2 = iconst(rs2)
        ops.append(c2)
        ops.append(llvm.InlineAsmOp(isa.asm_string(funct), "r,r",
                                    [lhs, c2.results[0]], [],
                                    has_side_effects=True))
        Rewriter.replace_op(op, ops)
    if any(isinstance(op, G._GemminiOp) for op in module.walk()):
        raise AssertionError("device lowering left a Gemmini op behind")
    module.verify()
    return module


def _verify_dynamic_preload_rows(module):
    """Close every declared B/C range against the actual ordinary CPU CFG.

    Resource attributes do not establish runtime row bounds. This option accepts
    only statically resolvable command loops; unknown values/control flow refuse
    before mutation. Every dynamic declaration must be observed in a completed
    function trace. Unreachable or uncontained declarations have no row proof.
    """
    from merlin.llvmlower.static_llvm_cfg import StaticInt, StaticPointer, trace_static_function

    dynamic = {op for op in module.walk()
               if isinstance(op, G.PreloadOp) and op.operands_}
    observed = set()
    for function in module.walk():
        if not isinstance(function, llvm.FuncOp) or not any(
                isinstance(op, G.PreloadOp) and op.operands_ for op in function.walk()):
            continue
        arguments = [StaticPointer(i, StaticInt(0, 64))
                     for i, _ in enumerate(function.body.blocks.first.args)]
        for step in trace_static_function(function, arguments,
                observe=lambda op: isinstance(op, G._GemminiOp), pointer_index_bits=64):
            op = step.operation
            if isinstance(op, G.PreloadOp) and op.operands_:
                observed.add(op)
                for key, row in zip(op.dynamic_rows(), step.inputs, strict=True):
                    if (not isinstance(row, StaticInt)
                            or not op.a(key + "_min", 0) <= row.value <= op.a(key + "_max")
                            or row.value % op.a(key + "_alignment", 1)):
                        message = ("executed dynamic C violates declared accumulator row range"
                                   if key == "c" else "executed dynamic B violates declared scratchpad row range/alignment")
                        raise ValueError(message)
            elif isinstance(op, G.ComputeOp) and op.operands_:
                row = step.inputs[0]
                if not isinstance(row, StaticInt) or not 0 <= row.value <= op.a("a_max"):
                    raise ValueError("executed dynamic A violates declared scratchpad row range")
    if observed != dynamic:
        label = "B/C" if any("bd" in op.dynamic_rows() for op in dynamic) else "C"
        raise ValueError("dynamic " + label + " declaration has no complete reachable function trace")
