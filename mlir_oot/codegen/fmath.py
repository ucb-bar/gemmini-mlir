"""Transcendental functions built from BASE single-precision arithmetic.

A host-lane region needs ``exp``, ``erf`` and ``rsqrt``. Emitting an intrinsic would make the
artifact depend on a math library the bare-metal harness does not link, so each one is expanded
here into add / multiply / divide / compare / select plus integer bit manipulation -- operations the
target's own scalar core provides. The approximations and their error bounds are the standard ones
and are named where they come from.
"""

from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32, i32
from xdsl.ir import Block, SSAValue

#: 1.5 * 2**23 -- adding it to a float in range forces round-to-nearest-even into the mantissa's
#: low bits, so the integer part can be read straight out of the bit pattern.
SHIFTER = 12582912.0
SHIFTER_BITS = 0x4B400000
LN2_HI = 0.693359375
LN2_LO = -2.12194440e-4
INV_LN2 = 1.4426950408889634


class FloatBuilder:
    """Emits scalar f32 arithmetic into one block, pooling constants."""

    def __init__(self, block: Block) -> None:
        self.b = block
        self._fp: dict[float, SSAValue] = {}
        self._ip: dict[int, SSAValue] = {}

    def _add(self, op):
        self.b.add_op(op)
        return op.results[0]

    def fc(self, v: float) -> SSAValue:
        if v not in self._fp:
            self._fp[v] = self._add(llvm.ConstantOp(FloatAttr(float(v), f32), f32))
        return self._fp[v]

    def ic(self, v: int) -> SSAValue:
        if v not in self._ip:
            self._ip[v] = self._add(llvm.ConstantOp(IntegerAttr(int(v), i32), i32))
        return self._ip[v]

    def add(self, a, b):
        return self._add(llvm.FAddOp(a, b))

    def sub(self, a, b):
        return self._add(llvm.FSubOp(a, b))

    def mul(self, a, b):
        return self._add(llvm.FMulOp(a, b))

    def div(self, a, b):
        return self._add(llvm.FDivOp(a, b))

    def neg(self, a):
        return self.sub(self.fc(0.0), a)

    def cmp(self, pred: str, a, b):
        return self._add(llvm.FCmpOp(a, b, pred))

    def select(self, c, a, b):
        return self._add(llvm.SelectOp(c, a, b))

    def maximum(self, a, b):
        return self.select(self.cmp("ogt", a, b), a, b)

    def bits(self, x):
        return self._add(llvm.BitcastOp(x, i32))

    def unbits(self, x):
        return self._add(llvm.BitcastOp(x, f32))

    def iadd(self, a, b):
        return self._add(llvm.AddOp(a, b))

    def isub(self, a, b):
        return self._add(llvm.SubOp(a, b))

    def ishl(self, a, b):
        return self._add(llvm.ShlOp(a, b))

    def iashr(self, a, b):
        return self._add(llvm.AShrOp(a, b))

    def horner(self, x, coeffs: list[float]):
        """``coeffs`` highest degree first."""
        acc = self.fc(coeffs[0])
        for c in coeffs[1:]:
            acc = self.add(self.mul(acc, x), self.fc(c))
        return acc

    # -- transcendentals ---------------------------------------------------
    def exp(self, x):
        """``e**x`` by Cody-Waite range reduction plus a degree-5 minimax-ish polynomial.

        ``k = round(x / ln2)`` is obtained with the shifter trick, the reduced argument
        ``r = x - k*ln2`` is evaluated in two pieces so the subtraction stays exact, and the power
        of two is re-applied by writing ``k`` into the exponent field. Accurate to a few ulp over
        the range a normalised activation reaches.
        """
        lo, hi = self.fc(-87.0), self.fc(88.0)
        x = self.select(self.cmp("olt", x, lo), lo, x)
        x = self.select(self.cmp("ogt", x, hi), hi, x)
        y = self.add(self.mul(x, self.fc(INV_LN2)), self.fc(SHIFTER))
        kf = self.sub(y, self.fc(SHIFTER))
        ki = self.isub(self.bits(y), self.ic(SHIFTER_BITS))
        r = self.sub(x, self.mul(kf, self.fc(LN2_HI)))
        r = self.sub(r, self.mul(kf, self.fc(LN2_LO)))
        poly = self.horner(
            r, [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0]
        )
        scale = self.unbits(self.ishl(self.iadd(ki, self.ic(127)), self.ic(23)))
        return self.mul(poly, scale)

    def rsqrt(self, x):
        """``1/sqrt(x)`` from the reciprocal-square-root seed plus three Newton steps."""
        seed = self.unbits(self.isub(self.ic(0x5F3759DF), self.iashr(self.bits(x), self.ic(1))))
        half = self.mul(x, self.fc(0.5))
        y = seed
        for _ in range(3):
            y = self.mul(y, self.sub(self.fc(1.5), self.mul(half, self.mul(y, y))))
        return y

    def erf(self, x):
        """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), with the odd symmetry applied by select."""
        neg = self.cmp("olt", x, self.fc(0.0))
        a = self.select(neg, self.neg(x), x)
        t = self.div(self.fc(1.0), self.add(self.fc(1.0), self.mul(self.fc(0.3275911), a)))
        poly = self.horner(
            t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592]
        )
        poly = self.mul(poly, t)
        y = self.sub(self.fc(1.0), self.mul(poly, self.exp(self.neg(self.mul(a, a)))))
        return self.select(neg, self.neg(y), y)
