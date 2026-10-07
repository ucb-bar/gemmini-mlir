"""The scalar prologue and epilogue route Q wraps its mesh program in.

Three emitted loops per staged operand and one per committed output. None of them is arithmetic on
the payload's behalf -- the multiply-accumulate is entirely on the mesh -- they are the format
conversion an integer datapath needs at its edges:

1. ``scan``     one pass over the operand: the largest magnitude, and the smallest NON-ZERO one.
2. ``deviate``  one pass measuring how far the operand sits off the grid the scan proposes.
3. ``stage``    one pass writing the i8 image the DMA will read.

Step 2 is what makes the route exact for the operands it was built for. A dequantised int8 tensor
holds values ``k * s`` for integers ``|k| <= 127``; its smallest non-zero magnitude is ``m * s`` and
its largest is ``K * s``, so ``round(amax / vmin)`` recovers ``K/m`` and ``amax / that`` recovers the
step ``s`` itself whenever ``m`` is 1 -- which it is for any tensor that uses its own range. The
grid is then REPRODUCED rather than re-quantised, and the contraction is bit-exact against the float
program. When the operand does not lie on such a grid the deviation pass sees it and the scale falls
back to ``amax / 127``, the ordinary symmetric quantisation, whose reconstruction error is at most
half of a step that is itself no coarser than the operand's own.

The readout runs the other way: the committed accumulator word is an exact integer dot product, so
one multiply by the product of the operand scales puts it back on the float grid.
"""

from __future__ import annotations

from typing import Callable

from xdsl.dialects import llvm
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, i8, i16, i32, i64, f32
from xdsl.ir import Block, SSAValue

from ..lowering.quantize import QMAX, QuantPlan, Staged, Rescale
from . import fpcast

#: llvm.icmp predicates: 6 = ult, 2 = sgt, 4 = slt
ICMP_ULT, ICMP_SGT, ICMP_SLT = 6, 2, 4
#: How far off the proposed grid an operand may sit and still be called ON it. A float32 value that
#: IS an exact multiple of the step divides to an integer within a few ulp; a value that is not is
#: off by a substantial fraction of a step. Anything between is refused to the fallback scale.
GRID_TOLERANCE = 1.0 / 1024.0


class QuantLane:
    """Emits route Q's edges into the same function body the instruction stream lives in."""

    def __init__(self, plan: QuantPlan, const_i: Callable[[int], SSAValue],
                 blocks: list[Block], prelude: list, helpers: dict | None = None) -> None:
        self.plan = plan
        #: module-level helper functions this lane calls (the narrowing float cast)
        self.helpers = helpers if helpers is not None else {}
        self.ic = const_i
        self.blocks = blocks
        self.prelude = prelude
        self.ptr = llvm.LLVMPointerType()
        self._fc: dict[float, SSAValue] = {}
        #: staged tensor name -> the alloca holding its i8 image
        self.staging: dict[str, SSAValue] = {}
        #: staged tensor name -> the f32 scale the prologue derived for it
        self.scale: dict[str, SSAValue] = {}

    # -- small builders ----------------------------------------------------
    def fc(self, v: float) -> SSAValue:
        if v not in self._fc:
            op = llvm.ConstantOp(FloatAttr(float(v), f32), f32)
            self.prelude.append(op)
            self._fc[v] = op.results[0]
        return self._fc[v]

    @staticmethod
    def _emit(blk: Block, op):
        blk.add_op(op)
        return op.results[0]

    def gep(self, blk: Block, base: SSAValue, elem_t, off: SSAValue) -> SSAValue:
        return self._emit(blk, llvm.GEPOp(base, [llvm.GEP_USE_SSA_VAL], elem_t,
                                          ssa_indices=[off], result_type=self.ptr))

    def load_float(self, blk: Block, base: SSAValue, dtype: str, off: SSAValue) -> SSAValue:
        """One element of a float operand, widened to f32 whatever container it is stored in."""
        if dtype == "f32":
            ld = llvm.LoadOp(self.gep(blk, base, f32, off), f32)
            blk.add_op(ld)
            return ld.dereferenced_value
        ld = llvm.LoadOp(self.gep(blk, base, i16, off), i16)
        blk.add_op(ld)
        wide = self._emit(blk, llvm.ZExtOp(ld.dereferenced_value, i32))
        shifted = self._emit(blk, llvm.ShlOp(wide, self._emit(
            blk, llvm.ConstantOp(IntegerAttr(16, i32), i32))))
        return self._emit(blk, llvm.BitcastOp(shifted, f32))

    # -- loop forms --------------------------------------------------------
    def loop(self, cur: Block, trip: int, carries: list[SSAValue],
             body: Callable[[Block, SSAValue, list[SSAValue]], tuple[Block, list[SSAValue]]],
             ) -> tuple[Block, list[SSAValue]]:
        """``for iv in range(trip)`` carrying ``carries`` -> (tail block, carried values)."""
        if trip <= 0:
            return cur, list(carries)
        types = [c.type for c in carries]
        head = Block(arg_types=[i64] + types)
        tail = Block(arg_types=types)
        self.blocks += [head, tail]
        cur.add_op(llvm.BrOp(head, self.ic(0), *carries))
        iv = head.args[0]
        end, nxt_carries = body(head, iv, list(head.args[1:]))
        step = self._emit(end, llvm.AddOp(iv, self.ic(1)))
        cmp = llvm.ICmpOp(step, self.ic(trip), IntegerAttr(ICMP_ULT, i64))
        end.add_op(cmp)
        end.add_op(llvm.CondBrOp(cmp.res, head, [step] + nxt_carries, tail, nxt_carries))
        return tail, list(tail.args)

    # -- the three operand passes -----------------------------------------
    def _scan(self, cur: Block, s: Staged, base: SSAValue) -> tuple[Block, SSAValue, SSAValue]:
        """(largest |v|, smallest non-zero |v|) over the operand's whole allocated extent."""
        huge = self.fc(3.4028234663852886e38)
        zero = self.fc(0.0)

        def body(blk: Block, iv: SSAValue, carry: list[SSAValue]):
            amax, vmin = carry
            v = self.load_float(blk, base, s.src_dtype, iv)
            a = self._emit(blk, llvm.FAbsOp(v))
            gt = llvm.FCmpOp(a, amax, "ogt")
            blk.add_op(gt)
            new_max = self._emit(blk, llvm.SelectOp(gt.res, a, amax))
            pos = llvm.FCmpOp(a, zero, "ogt")
            blk.add_op(pos)
            lt = llvm.FCmpOp(a, vmin, "olt")
            blk.add_op(lt)
            take = self._emit(blk, llvm.AndOp(pos.res, lt.res))
            new_min = self._emit(blk, llvm.SelectOp(take, a, vmin))
            return blk, [new_max, new_min]

        tail, carried = self.loop(cur, s.elems, [zero, huge], body)
        return tail, carried[0], carried[1]

    def _deviation(self, cur: Block, s: Staged, base: SSAValue,
                   inv: SSAValue) -> tuple[Block, SSAValue]:
        """The largest distance from ``v * inv`` to the nearest integer, over the operand."""
        zero = self.fc(0.0)

        def body(blk: Block, iv: SSAValue, carry: list[SSAValue]):
            worst, = carry
            v = self.load_float(blk, base, s.src_dtype, iv)
            scaled = self._emit(blk, llvm.FMulOp(v, inv))
            nearest = self._round(blk, scaled)
            diff = self._emit(blk, llvm.FSubOp(scaled, nearest))
            dev = self._emit(blk, llvm.FAbsOp(diff))
            gt = llvm.FCmpOp(dev, worst, "ogt")
            blk.add_op(gt)
            return blk, [self._emit(blk, llvm.SelectOp(gt.res, dev, worst))]

        tail, carried = self.loop(cur, s.elems, [zero], body)
        return tail, carried[0]

    def _stage(self, cur: Block, s: Staged, base: SSAValue, dst: SSAValue,
               inv: SSAValue) -> Block:
        """Write the i8 image the DMA reads, saturating at the mesh's symmetric operand range."""

        def body(blk: Block, iv: SSAValue, carry: list[SSAValue]):
            v = self.load_float(blk, base, s.src_dtype, iv)
            scaled = self._emit(blk, llvm.FMulOp(v, inv))
            q = self._to_int(blk, scaled)
            q = self._clamp(blk, q, -QMAX, QMAX)
            byte = self._emit(blk, llvm.TruncOp(q, i8))
            blk.add_op(llvm.StoreOp(byte, self.gep(blk, dst, i8, iv)))
            return blk, []

        tail, _ = self.loop(cur, s.elems, [], body)
        return tail

    # -- rounding helpers --------------------------------------------------
    def _to_int(self, blk: Block, v: SSAValue) -> SSAValue:
        """``round(v)`` half-away-from-zero, as an i64. ``fptosi`` truncates toward zero, so the
        half is added with the value's own sign before the truncation."""
        half = self._emit(blk, llvm.FCopySignOp(self.fc(0.5), v))
        biased = self._emit(blk, llvm.FAddOp(v, half))
        return fpcast.to_i64(blk, self.helpers, biased)

    def _round(self, blk: Block, v: SSAValue) -> SSAValue:
        """The same rounding, returned on the float grid."""
        return self._emit(blk, llvm.SIToFPOp(self._to_int(blk, v), f32))

    def _clamp(self, blk: Block, v: SSAValue, lo: int, hi: int) -> SSAValue:
        hi_c, lo_c = self.ic(hi), self.ic(lo)
        gt = llvm.ICmpOp(v, hi_c, IntegerAttr(ICMP_SGT, i64))
        blk.add_op(gt)
        v = self._emit(blk, llvm.SelectOp(gt.res, hi_c, v))
        lt = llvm.ICmpOp(v, lo_c, IntegerAttr(ICMP_SLT, i64))
        blk.add_op(lt)
        return self._emit(blk, llvm.SelectOp(lt.res, lo_c, v))

    # -- the scale a staged operand is given -------------------------------
    def _scale_for(self, cur: Block, s: Staged, base: SSAValue) -> tuple[Block, SSAValue]:
        cur, amax, vmin = self._scan(cur, s, base)
        one = self.fc(1.0)
        qmax = self.fc(float(QMAX))
        zero = self.fc(0.0)
        # the ordinary symmetric scale, and the one the operand's own grid proposes
        fallback = self._emit(cur, llvm.FDivOp(amax, qmax))
        levels_f = self._emit(cur, llvm.FDivOp(amax, vmin))
        levels = self._to_int(cur, levels_f)
        levels = self._clamp(cur, levels, 1, QMAX)
        grid = self._emit(cur, llvm.FDivOp(amax, self._emit(
            cur, llvm.SIToFPOp(levels, f32))))
        inv_grid = self._emit(cur, llvm.FDivOp(one, grid))
        cur, dev = self._deviation(cur, s, base, inv_grid)
        on_grid = llvm.FCmpOp(dev, self.fc(GRID_TOLERANCE), "ole")
        cur.add_op(on_grid)
        chosen = self._emit(cur, llvm.SelectOp(on_grid.res, grid, fallback))
        # an all-zero operand has no scale to derive; 1.0 keeps the readout's product finite
        empty = llvm.FCmpOp(amax, zero, "ole")
        cur.add_op(empty)
        scale = self._emit(cur, llvm.SelectOp(empty.res, one, chosen))
        return cur, scale

    # -- entry points ------------------------------------------------------
    def allocate(self, entry: Block) -> None:
        """One stack buffer per staged operand, sized to the DRAM image the DMA will read."""
        for s in self.plan.staged:
            size = llvm.ConstantOp(IntegerAttr(s.elems, i64), i64)
            self.prelude.append(size)
            alloca = llvm.AllocaOp(size.results[0], i8, alignment=64)
            self.prelude.append(alloca)
            self.staging[s.dst] = alloca.res

    def prologue(self, cur: Block, base_of: Callable[[str], SSAValue]) -> Block:
        for s in self.plan.staged:
            src = base_of(s.src)
            cur, scale = self._scale_for(cur, s, src)
            self.scale[s.dst] = scale
            inv = self._emit(cur, llvm.FDivOp(self.fc(1.0), scale))
            cur = self._stage(cur, s, src, self.staging[s.dst], inv)
        return cur

    def epilogue(self, cur: Block, base_of: Callable[[str], SSAValue]) -> Block:
        for r in self.plan.rescales:
            cur = self._rescale(cur, r, base_of(r.out))
        return cur

    def _rescale(self, cur: Block, r: Rescale, base: SSAValue) -> Block:
        # the accumulator counts in units of (step of lhs) x (step of rhs)
        scale: SSAValue | None = None
        for name in r.operands:
            s = self.scale[name]
            scale = s if scale is None else self._emit(cur, llvm.FMulOp(scale, s))
        if scale is None:
            scale = self.fc(1.0)

        def body(blk: Block, iv: SSAValue, carry: list[SSAValue]):
            ld = llvm.LoadOp(self.gep(blk, base, i32, iv), i32)
            blk.add_op(ld)
            wide = self._emit(blk, llvm.SExtOp(ld.dereferenced_value, i64))
            as_f = self._emit(blk, llvm.SIToFPOp(wide, f32))
            out = self._emit(blk, llvm.FMulOp(as_f, scale))
            blk.add_op(llvm.StoreOp(out, self.gep(blk, base, f32, iv)))
            return blk, []

        tail, _ = self.loop(cur, r.rows * r.pitch, [], body)
        return tail
