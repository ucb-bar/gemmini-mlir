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
        return isa.flush()
    if isinstance(op, G.ConfigExOp):
        return isa.config_ex(dataflow=a("dataflow"))
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
    for op in list(module.walk()):
        if not isinstance(op, G._GemminiOp):
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
