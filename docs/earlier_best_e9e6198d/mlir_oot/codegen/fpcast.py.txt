"""Float -> signed-integer conversion built from ops the LLVM dialect actually registers.

The narrowing cast ``llvm.fptosi`` is NOT part of the LLVM dialect this package's IR travels
through, and an unregistered operation is not a cosmetic problem: a consumer that parses the module
STRUCTURALLY fails on it and falls back to scanning the text, which finds the ``.insn`` strings but
resolves none of their operands -- so a kernel that is entirely correct reads back as a trace of
instructions with unknown fields. Measured on this package's own emitted artifact: every operand of
every command came back ``kind: unknown`` for exactly this reason.

So the conversion is emitted as an internal helper function built from registered arithmetic, and
called where the cast used to be. It is the textbook decomposition of an IEEE-754 binary32:

    value = (-1)**sign * significand * 2**(exponent - 23)

with the significand shifted into place. It truncates TOWARD ZERO (a right shift discards the
fractional bits, which is what the narrowing cast does), and it SATURATES rather than producing an
undefined value for an input outside the integer's range -- undefined is not something a bring-up
kernel should be able to observe.
"""

from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, f32, i32, i64
from xdsl.ir import Block, Region, SSAValue

#: llvm.icmp predicates: 1 = ne, 2 = slt, 4 = sgt, 8 = ugt
ICMP_SLT, ICMP_SGT, ICMP_UGT = 2, 4, 8

HELPER = "__merlin_f32_to_i32_trunc"

INT_MAX = 0x7FFFFFFF
MANT_BITS = 23
EXP_BIAS = 127


def _emit(blk: Block, op):
    blk.add_op(op)
    return op.results[0]


def _c(blk: Block, v: int) -> SSAValue:
    return _emit(blk, llvm.ConstantOp(IntegerAttr(int(v), i32), i32))


def _build(blk: Block, v: SSAValue) -> SSAValue:
    bits = _emit(blk, llvm.BitcastOp(v, i32))
    neg = llvm.ICmpOp(bits, _c(blk, 0), IntegerAttr(ICMP_SLT, i64))
    blk.add_op(neg)
    mag = _emit(blk, llvm.AndOp(bits, _c(blk, INT_MAX)))
    exp = _emit(blk, llvm.SubOp(_emit(blk, llvm.LShrOp(mag, _c(blk, MANT_BITS))),
                                _c(blk, EXP_BIAS)))
    mant = _emit(blk, llvm.OrOp(_emit(blk, llvm.AndOp(mag, _c(blk, (1 << MANT_BITS) - 1))),
                                _c(blk, 1 << MANT_BITS)))
    sh = _emit(blk, llvm.SubOp(exp, _c(blk, MANT_BITS)))
    rs = _emit(blk, llvm.SubOp(_c(blk, 0), sh))
    # value = mant * 2**(exp-23): shift LEFT when the exponent outruns the significand,
    # RIGHT when it does not -- and a right shift discards the fraction, which IS the truncation.

    def held(amount: SSAValue) -> SSAValue:
        """``amount`` kept below the register width, so neither shift is ever undefined.

        The results the clamped shifts would have produced past the width are discarded wholesale
        by the range guards below, so holding the amount changes no value this returns.
        """
        wide = llvm.ICmpOp(amount, _c(blk, 31), IntegerAttr(ICMP_UGT, i64))
        blk.add_op(wide)
        return _emit(blk, llvm.SelectOp(wide.res, _c(blk, 31), amount))

    left = _emit(blk, llvm.ShlOp(mant, held(sh)))
    right = _emit(blk, llvm.LShrOp(mant, held(rs)))
    up = llvm.ICmpOp(sh, _c(blk, 0), IntegerAttr(ICMP_SGT, i64))
    blk.add_op(up)
    r = _emit(blk, llvm.SelectOp(up.res, left, right))
    # |v| < 1 truncates to zero; |v| at or past the integer's range saturates at its end.
    small = llvm.ICmpOp(exp, _c(blk, 0), IntegerAttr(ICMP_SLT, i64))
    blk.add_op(small)
    r = _emit(blk, llvm.SelectOp(small.res, _c(blk, 0), r))
    huge = llvm.ICmpOp(exp, _c(blk, 30), IntegerAttr(ICMP_SGT, i64))
    blk.add_op(huge)
    r = _emit(blk, llvm.SelectOp(huge.res, _c(blk, INT_MAX), r))
    return _emit(blk, llvm.SelectOp(neg.res, _emit(blk, llvm.SubOp(_c(blk, 0), r)), r))


def helper(helpers: dict) -> None:
    """Materialise the conversion once into ``helpers``, keyed by its symbol name."""
    if HELPER in helpers:
        return
    blk = Block(arg_types=[f32])
    blk.add_op(llvm.ReturnOp(_build(blk, blk.args[0])))
    helpers[HELPER] = llvm.FuncOp(
        HELPER, llvm.LLVMFunctionType([f32], i32),
        linkage=llvm.LinkageAttr("internal"), body=Region([blk]),
    )


def to_i32(blk: Block, helpers: dict, v: SSAValue) -> SSAValue:
    """``(i32) v`` -- truncating toward zero, saturating at the range ends."""
    helper(helpers)
    call = llvm.CallOp(HELPER, v, return_type=i32)
    blk.add_op(call)
    return call.results[0]


def to_i64(blk: Block, helpers: dict, v: SSAValue) -> SSAValue:
    """The same conversion widened to the index width the address algebra uses."""
    return _emit(blk, llvm.SExtOp(to_i32(blk, helpers, v), i64))
