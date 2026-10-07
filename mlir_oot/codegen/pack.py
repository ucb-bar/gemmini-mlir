"""Dense im2col staging for narrow-channel convolution bands."""

from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i8, i64
from xdsl.ir import Block, SSAValue

from ..lowering.conv import PackSpec


def emit_pack(cur: Block, spec: PackSpec, blocks: list[Block], const,
              ptr_of) -> Block:
    """Copy one band into dense reduction rows using a compact host loop nest."""
    g = spec.geometry
    src, dst = ptr_of(spec.source), ptr_of(spec.target)
    ptr = llvm.LLVMPointerType()

    def put(blk: Block, op) -> SSAValue:
        blk.add_op(op)
        return op.results[0]

    def add(blk: Block, a: SSAValue, b: SSAValue) -> SSAValue:
        return put(blk, llvm.AddOp(a, b))

    def mul(blk: Block, a: SSAValue, b: SSAValue) -> SSAValue:
        return put(blk, llvm.MulOp(a, b))

    def gep(blk: Block, base: SSAValue, off: SSAValue) -> SSAValue:
        return put(blk, llvm.GEPOp(base, [llvm.GEP_USE_SSA_VAL], i8,
                                    ssa_indices=[off], result_type=ptr))

    row_head = Block(arg_types=[i64])
    tap_head = Block(arg_types=[i64])
    yes, no, join, row_latch, tail = (Block() for _ in range(5))
    blocks.extend([row_head, tap_head, yes, no, join, row_latch, tail])
    cur.add_op(llvm.BrOp(row_head, const(0)))

    row = row_head.args[0]
    p = add(row_head, row, const(spec.row0))
    oh = put(row_head, llvm.SDivOp(p, const(g.wo)))
    ow = put(row_head, llvm.SRemOp(p, const(g.wo)))
    ih0 = add(row_head, mul(row_head, oh, const(g.sh)), const(-g.pad_t))
    iw0 = add(row_head, mul(row_head, ow, const(g.sw)), const(-g.pad_l))
    dst_row = mul(row_head, row, const(spec.target_pitch))
    zero = put(row_head, llvm.TruncOp(const(0), i8))
    row_head.add_op(llvm.BrOp(tap_head, const(0)))

    tap = tap_head.args[0]
    kh = put(tap_head, llvm.SDivOp(tap, const(g.kw)))
    kw = put(tap_head, llvm.SRemOp(tap, const(g.kw)))
    ih = add(tap_head, ih0, mul(tap_head, kh, const(g.dh)))
    iw = add(tap_head, iw0, mul(tap_head, kw, const(g.dw)))
    ch = put(tap_head, llvm.ICmpOp(ih, const(g.h), IntegerAttr(6, i64)))
    cw = put(tap_head, llvm.ICmpOp(iw, const(g.w), IntegerAttr(6, i64)))
    valid = put(tap_head, llvm.AndOp(ch, cw))
    dst_tap = add(tap_head, dst_row, mul(tap_head, tap, const(g.ci)))
    tap_head.add_op(llvm.CondBrOp(valid, yes, [], no, []))

    src_pixel = mul(yes, add(yes, mul(yes, ih, const(g.w)), iw),
                    const(spec.source_pitch))
    for c in range(g.ci):
        v = put(yes, llvm.LoadOp(gep(yes, src, add(yes, src_pixel, const(c))), i8))
        yes.add_op(llvm.StoreOp(v, gep(yes, dst, add(yes, dst_tap, const(c)))))
        no.add_op(llvm.StoreOp(zero, gep(no, dst, add(no, dst_tap, const(c)))))
    yes.add_op(llvm.BrOp(join))
    no.add_op(llvm.BrOp(join))

    next_tap = add(join, tap, const(1))
    more_taps = put(join, llvm.ICmpOp(next_tap, const(g.kh * g.kw), IntegerAttr(6, i64)))
    join.add_op(llvm.CondBrOp(more_taps, tap_head, [next_tap], row_latch, []))
    next_row = add(row_latch, row, const(1))
    more_rows = put(row_latch, llvm.ICmpOp(next_row, const(spec.rows), IntegerAttr(6, i64)))
    row_latch.add_op(llvm.CondBrOp(more_rows, row_head, [next_row], tail, []))
    return tail
