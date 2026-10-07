"""Host-lane codegen for a ``merlin_iface`` region the mesh has no operand encoding for.

:mod:`mlir_oot.codegen.host_lane` compiles a ``linalg-on-tensors`` region by unrolling it to one
SSA value per element, which is affordable for the elementwise and reduction shapes that grammar
carries. A CONTRACTION is not: a 32x32x32 matmul is 32768 multiply-adds, and a program that grows
with the payload is the defect the re-roller exists to prevent on the accelerator path.

So this module compiles the interface's own operation model into a LOOP NEST over the declared
extents: the emitted function is a handful of blocks whatever the extents are, and the extents
themselves are the loop bounds. It is used only where :func:`require_mesh_dtypes` refuses -- the
mesh reads i8 operands into an i32 accumulator, so a float contraction has no encoding there -- and
the placement is DECLARED in ``params.lane_placement`` with that reason. Refusing to the host lane
and computing the region is a different statement from declining it: the program still commits the
tensor it was asked for.

Arithmetic is single-precision throughout, loaded from and stored back to each tensor's declared
container using the same DRAM row pitch every other address in this package goes through.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Callable

from xdsl.dialects import llvm
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    TensorType,
    f32,
    i16,
    i32,
    i64,
)
from xdsl.ir import Block, Region, SSAValue

from ..ir.workload import Epilogue, Workload
from . import fpcast
from ..lowering.conv import geometry_from
from ..target import layout

KERNEL_SYMBOL = "gemmini_kernel"
#: llvm.icmp predicate 6 = "ult", the loop-back condition; 2 = "slt" for a signed bound test.
ICMP_ULT = 6
ICMP_SLT = 2

#: Containers this lane can load from and store to, and the byte width each one occupies.
LANE_DTYPES = {"f32": 4, "bf16": 2, "i8": 1, "i16": 2, "i32": 4}

#: The INTEGER containers, each with its element width and the range a store saturates into. The
#: lane's arithmetic is single precision throughout, so an integer tensor is widened on load and
#: rounded back on store -- which is the same round-to-nearest-even the accelerator's own readout
#: applies, so a region that moves between the two lanes does not change rounding behaviour.
INT_CONTAINERS = {
    "i8": (8, -128, 127),
    "i16": (16, -(1 << 15), (1 << 15) - 1),
    "i32": (32, -(1 << 31), (1 << 31) - 1),
}

#: 2**23 -- adding and subtracting this forces an f32 onto the integer grid under the default
#: round-to-nearest-even mode, which is exactly `roundeven` without needing an intrinsic.
_ROUND_MAGIC = 8388608.0


class ScalarLaneError(Exception):
    """A region this lane does not implement; the caller states it as a decline."""


def _elems(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


@dataclass
class Embed:
    """Where an EMBEDDED lane writes: the blocks, constants and helpers of a host function.

    Route X compiles an off-mesh stage into the SAME ``gemmini_kernel`` the accelerator commands
    live in, so the lane cannot own the function -- it borrows the code generator's block list, its
    entry-block constant pool (so every constant still dominates every loop body) and its helper
    table, and it resolves a tensor to the pointer the host already holds for it.
    """

    blocks: list[Block]
    prelude: list
    helpers: dict
    ptr_of: Callable[[str], SSAValue]
    #: module-level constant tables the stage needs (name -> llvm.mlir.global)
    tables: dict = field(default_factory=dict)


class ScalarLane:
    """Emits an interface region as a loop nest.

    Standalone (``embed is None``) it owns a whole ``gemmini_kernel``; embedded it appends its
    blocks to a function the code generator is already building.
    """

    def __init__(self, wl: Workload, args: list[str], embed: Embed | None = None,
                 helpers: dict | None = None, tables: dict | None = None) -> None:
        self.wl = wl
        self.args = args
        self.ptr = llvm.LLVMPointerType()
        self.embed = embed
        if embed is None:
            self.entry = Block(arg_types=[self.ptr] * len(args))
            self.blocks: list[Block] = [self.entry]
            self.prelude: list = []
            # a stage compiled INTO a host module shares that module's pools, so a helper or a
            # constant table is emitted once however many stages reach for it
            self.helpers: dict[str, llvm.FuncOp] = helpers if helpers is not None else {}
            self.tables: dict[str, llvm.GlobalOp] = tables if tables is not None else {}
        else:
            self.entry = None
            self.blocks = embed.blocks
            self.prelude = embed.prelude
            self.helpers = embed.helpers
            self.tables = embed.tables
        self._ic: dict[int, SSAValue] = {}
        self._fc: dict[float, SSAValue] = {}

    # -- constants (emitted into the entry block, so they dominate every loop body) --
    def ic(self, v: int) -> SSAValue:
        if v not in self._ic:
            op = llvm.ConstantOp(IntegerAttr(int(v), i64), i64)
            self.prelude.append(op)
            self._ic[v] = op.results[0]
        return self._ic[v]

    def fc(self, v: float) -> SSAValue:
        if v not in self._fc:
            op = llvm.ConstantOp(FloatAttr(float(v), f32), f32)
            self.prelude.append(op)
            self._fc[v] = op.results[0]
        return self._fc[v]

    # -- integer index algebra ------------------------------------------------
    def mul(self, blk: Block, a: SSAValue, b: SSAValue) -> SSAValue:
        op = llvm.MulOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def add(self, blk: Block, a: SSAValue, b: SSAValue) -> SSAValue:
        op = llvm.AddOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def sub(self, blk: Block, a: SSAValue, b: SSAValue) -> SSAValue:
        op = llvm.SubOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def scale(self, blk: Block, v: SSAValue, k: int) -> SSAValue:
        """``v * k`` with the two degenerate factors folded away."""
        if k == 0:
            return self.ic(0)
        if k == 1:
            return v
        return self.mul(blk, v, self.ic(k))

    def affine(self, blk: Block, terms: list[tuple[SSAValue, int]], bias: int = 0) -> SSAValue:
        """``sum(v * k) + bias`` as one chain of i64 arithmetic."""
        acc: SSAValue | None = None
        for v, k in terms:
            if k == 0:
                continue
            t = self.scale(blk, v, k)
            acc = t if acc is None else self.add(blk, acc, t)
        if acc is None:
            return self.ic(bias)
        if bias:
            acc = self.add(blk, acc, self.ic(bias))
        return acc

    # -- memory ---------------------------------------------------------------
    def _arg(self, name: str) -> SSAValue:
        if self.embed is not None:
            try:
                return self.embed.ptr_of(name)
            except Exception as exc:
                raise ScalarLaneError(f"host lane: {name!r} is not a kernel pointer") from exc
        try:
            return self.entry.args[self.args.index(name)]
        except ValueError as exc:
            raise ScalarLaneError(f"host lane: {name!r} is not a kernel argument") from exc

    def _container(self, name: str) -> str:
        dt = self.wl.tensors[name].dtype
        if dt not in LANE_DTYPES:
            raise ScalarLaneError(
                f"host lane: no scalar container for {name!r}'s declared dtype {dt!r}"
            )
        return dt

    def load(self, blk: Block, name: str, off: SSAValue) -> SSAValue:
        dt = self._container(name)
        if dt in INT_CONTAINERS:
            bits, _, _ = INT_CONTAINERS[dt]
            ity = IntegerType(bits)
            gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], ity,
                             ssa_indices=[off], result_type=self.ptr)
            blk.add_op(gep)
            ld = llvm.LoadOp(gep.result, ity)
            blk.add_op(ld)
            v = ld.dereferenced_value
            if bits < 32:
                ext = llvm.SExtOp(v, i32)
                blk.add_op(ext)
                v = ext.results[0]
            cvt = llvm.SIToFPOp(v, f32)
            blk.add_op(cvt)
            return cvt.results[0]
        if dt == "f32":
            gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], f32,
                             ssa_indices=[off], result_type=self.ptr)
            blk.add_op(gep)
            ld = llvm.LoadOp(gep.result, f32)
            blk.add_op(ld)
            return ld.dereferenced_value
        gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], i16, ssa_indices=[off],
                         result_type=self.ptr)
        blk.add_op(gep)
        ld = llvm.LoadOp(gep.result, i16)
        blk.add_op(ld)
        return self._widen_bf16(blk, ld.dereferenced_value)

    def round_even(self, blk: Block, v: SSAValue) -> SSAValue:
        """``roundeven(v)`` on the float grid, via the add/subtract-2**23 identity."""
        magic = llvm.FCopySignOp(self.fc(_ROUND_MAGIC), v)
        blk.add_op(magic)
        up = llvm.FAddOp(v, magic.results[0])
        blk.add_op(up)
        down = llvm.FSubOp(up.results[0], magic.results[0])
        blk.add_op(down)
        return down.results[0]

    def fclamp(self, blk: Block, v: SSAValue, lo: float, hi: float) -> SSAValue:
        for pred, bound in (("olt", lo), ("ogt", hi)):
            c = llvm.FCmpOp(v, self.fc(bound), pred)
            blk.add_op(c)
            sel = llvm.SelectOp(c.res, self.fc(bound), v)
            blk.add_op(sel)
            v = sel.results[0]
        return v

    def store(self, blk: Block, name: str, off: SSAValue, v: SSAValue,
              rounding: str = "trunc") -> None:
        """Store ``v`` into ``name``'s declared container.

        ``rounding`` selects how a float value lands on an INTEGER grid. ``trunc`` is the default
        because that is what a cast to an integer type does -- the narrowing conversion truncates
        toward zero -- and it is the rounding the quantized op definitions in this corpus are
        written against. ``even`` requests round-to-nearest-even instead, for a stage whose own
        definition names that rounding (the accumulator readout's `acc_scale`, for instance).
        """
        dt = self._container(name)
        if dt in INT_CONTAINERS:
            bits, lo, hi = INT_CONTAINERS[dt]
            ity = IntegerType(bits)
            r = v if rounding == "trunc" else self.round_even(blk, v)
            r = self.fclamp(blk, r, float(lo), float(hi))
            val = fpcast.to_i32(blk, self.helpers, r)
            if bits < 32:
                tr = llvm.TruncOp(val, ity)
                blk.add_op(tr)
                val = tr.results[0]
            gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], ity,
                             ssa_indices=[off], result_type=self.ptr)
            blk.add_op(gep)
            blk.add_op(llvm.StoreOp(val, gep.result))
            return
        if dt == "f32":
            gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[off],
                             result_type=self.ptr)
            blk.add_op(gep)
            blk.add_op(llvm.StoreOp(v, gep.result))
            return
        narrow = self._narrow_bf16(blk, v)
        gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], i16, ssa_indices=[off],
                         result_type=self.ptr)
        blk.add_op(gep)
        blk.add_op(llvm.StoreOp(narrow, gep.result))

    def offset(self, blk: Block, name: str, rows: list[tuple[SSAValue, int]],
               col: SSAValue | None, row_bias: int = 0, col_bias: int = 0) -> SSAValue:
        """``row * pitch + col`` for the row-major matrix view of ``name``."""
        pitch = layout.row_pitch(self.wl.tensors[name].shape)
        terms = [(v, k * pitch) for v, k in rows]
        if col is not None:
            terms.append((col, 1))
        return self.affine(blk, terms, row_bias * pitch + col_bias)

    # -- bf16 <-> f32 (emitted once, called per element) ----------------------
    def _typed_helper(self, name: str, arg_t, res_t, build) -> None:
        if name in self.helpers:
            return
        blk = Block(arg_types=[arg_t])
        res = build(blk, blk.args[0])
        blk.add_op(llvm.ReturnOp(res))
        self.helpers[name] = llvm.FuncOp(
            name, llvm.LLVMFunctionType([arg_t], res_t),
            linkage=llvm.LinkageAttr("internal"), body=Region([blk]),
        )

    def _widen_bf16(self, blk: Block, half: SSAValue) -> SSAValue:
        def build(b, x):
            ext = llvm.ZExtOp(x, i32)
            b.add_op(ext)
            c16 = llvm.ConstantOp(IntegerAttr(16, i32), i32)
            b.add_op(c16)
            sh = llvm.ShlOp(ext.results[0], c16.results[0])
            b.add_op(sh)
            bc = llvm.BitcastOp(sh.results[0], f32)
            b.add_op(bc)
            return bc.results[0]

        self._typed_helper("gemmini_lane_bf16_to_f32", i16, f32, build)
        op = llvm.CallOp("gemmini_lane_bf16_to_f32", half, return_type=f32)
        blk.add_op(op)
        return op.results[0]

    def _narrow_bf16(self, blk: Block, v: SSAValue) -> SSAValue:
        def build(b, x):
            def c(n):
                op = llvm.ConstantOp(IntegerAttr(n, i32), i32)
                b.add_op(op)
                return op.results[0]

            bits = llvm.BitcastOp(x, i32)
            b.add_op(bits)
            lsb = llvm.LShrOp(bits.results[0], c(16))
            b.add_op(lsb)
            lsb1 = llvm.AndOp(lsb.results[0], c(1))
            b.add_op(lsb1)
            r0 = llvm.AddOp(bits.results[0], c(0x7FFF))
            b.add_op(r0)
            r = llvm.AddOp(r0.results[0], lsb1.results[0])
            b.add_op(r)
            hi = llvm.LShrOp(r.results[0], c(16))
            b.add_op(hi)
            tr = llvm.TruncOp(hi.results[0], i16)
            b.add_op(tr)
            return tr.results[0]

        self._typed_helper("gemmini_lane_f32_to_bf16", f32, i16, build)
        op = llvm.CallOp("gemmini_lane_f32_to_bf16", v, return_type=i16)
        blk.add_op(op)
        return op.results[0]

    # -- control flow ---------------------------------------------------------
    def nest(self, cur: Block, extents: list[int],
             body: Callable[[Block, list[SSAValue]], Block]) -> Block:
        """A perfect loop nest over ``extents``; ``body`` fills the innermost block."""
        return self._nest(cur, list(extents), [], body)

    def _nest(self, cur, extents, ivs, body):
        if not extents:
            return body(cur, list(ivs))
        n = int(extents[0])
        if n <= 0:
            return cur
        blk = Block(arg_types=[i64])
        tail = Block(arg_types=[])
        self.blocks += [blk, tail]
        cur.add_op(llvm.BrOp(blk, self.ic(0)))
        iv = blk.args[0]
        end = self._nest(blk, extents[1:], ivs + [iv], body)
        nxt = llvm.AddOp(iv, self.ic(1))
        end.add_op(nxt)
        cmp = llvm.ICmpOp(nxt.results[0], self.ic(n), IntegerAttr(ICMP_ULT, i64))
        end.add_op(cmp)
        end.add_op(llvm.CondBrOp(cmp.res, blk, [nxt.results[0]], tail, []))
        return tail

    def reduce(self, cur: Block, trip: int, init: SSAValue,
               body: Callable[[Block, SSAValue, SSAValue], tuple[Block, SSAValue]]):
        """``acc = init; for t in range(trip): acc = body(t, acc)`` -> (tail block, acc)."""
        if trip <= 0:
            return cur, init
        blk = Block(arg_types=[i64, f32])
        tail = Block(arg_types=[f32])
        self.blocks += [blk, tail]
        cur.add_op(llvm.BrOp(blk, self.ic(0), init))
        iv, acc = blk.args[0], blk.args[1]
        end, new_acc = body(blk, iv, acc)
        nxt = llvm.AddOp(iv, self.ic(1))
        end.add_op(nxt)
        cmp = llvm.ICmpOp(nxt.results[0], self.ic(trip), IntegerAttr(ICMP_ULT, i64))
        end.add_op(cmp)
        end.add_op(llvm.CondBrOp(cmp.res, blk, [nxt.results[0], new_acc], tail, [new_acc]))
        return tail, tail.args[0]

    def guard(self, cur: Block, cond: SSAValue, init: SSAValue,
              body: Callable[[Block], tuple[Block, SSAValue]]):
        """``cond ? body() : init`` -> (join block, value)."""
        then = Block(arg_types=[])
        join = Block(arg_types=[f32])
        self.blocks += [then, join]
        cur.add_op(llvm.CondBrOp(cond, then, [], join, [init]))
        end, val = body(then)
        end.add_op(llvm.BrOp(join, val))
        return join, join.args[0]

    # -- tensor addressing ----------------------------------------------------
    def row_terms(self, shape, idxs: list[SSAValue]) -> list[tuple[SSAValue, int]]:
        """Affine terms of the ROW index for ``shape``'s row-major matrix view.

        The leading extents of a tensor collapse to one row index; ``idxs`` supplies one induction
        variable per leading extent, outermost first.
        """
        lead = [int(d) for d in shape[:-1]]
        if len(idxs) != len(lead):
            raise ScalarLaneError(
                f"host lane: {len(idxs)} index/indices for a tensor with {len(lead)} leading extents"
            )
        terms: list[tuple[SSAValue, int]] = []
        for k, iv in enumerate(idxs):
            coeff = 1
            for d in lead[k + 1:]:
                coeff *= d
            terms.append((iv, coeff))
        return terms

    def elem_off(self, blk: Block, name: str, idxs: list[SSAValue],
                 col: SSAValue | None) -> SSAValue:
        shape = self.wl.tensors[name].shape
        pitch = layout.row_pitch(shape)
        terms = [(v, k * pitch) for v, k in self.row_terms(shape, idxs)]
        if col is not None:
            terms.append((col, 1))
        return self.affine(blk, terms)

    # -- arithmetic -----------------------------------------------------------
    def fmul(self, blk, a, b):
        op = llvm.FMulOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def fadd(self, blk, a, b):
        op = llvm.FAddOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def icmp(self, blk, a, b, pred: int):
        op = llvm.ICmpOp(a, b, IntegerAttr(pred, i64))
        blk.add_op(op)
        return op.res

    def iand(self, blk, a, b):
        op = llvm.AndOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def epilogue(self, blk: Block, acc: SSAValue, ep: Epilogue | None,
                 col: SSAValue) -> tuple[Block, SSAValue]:
        """Apply the declared readout stages, in the order the interface listed them."""
        if ep is None:
            return blk, acc
        for stage in ep.stages:
            if stage in ("bias_add", "bias"):
                if not ep.bias:
                    raise ScalarLaneError("host lane: a bias stage that names no tensor")
                off = self.elem_off(blk, ep.bias, [], col)
                acc = self.fadd(blk, acc, self.load(blk, ep.bias, off))
            elif stage == "acc_scale":
                if ep.acc_scale is None:
                    raise ScalarLaneError("host lane: an acc_scale stage that declares no scale")
                acc = self.fmul(blk, acc, self.fc(float(ep.acc_scale)))
            elif stage == "relu":
                cmp = llvm.FCmpOp(acc, self.fc(0.0), "ogt")
                blk.add_op(cmp)
                sel = llvm.SelectOp(cmp.res, acc, self.fc(0.0))
                blk.add_op(sel)
                acc = sel.results[0]
            else:
                raise ScalarLaneError(
                    f"host lane: the {stage!r} readout stage is integer-exact and has no "
                    "single-precision expansion here"
                )
        return blk, acc

    # -- regions --------------------------------------------------------------
    def contraction(self, cur: Block, out: str, batch: list[int], m: int, n: int, k: int,
                    lhs: str, rhs: str, *, rhs_transposed: bool = False,
                    lhs_batched: bool = True, rhs_batched: bool = True,
                    ep: Epilogue | None = None) -> Block:
        """``out[b, i, j] = epilogue(sum_t lhs[b, i, t] * rhs[b, t, j])`` as a loop nest."""
        nb = len(batch)

        def body(blk: Block, ivs: list[SSAValue]) -> Block:
            bat, i, j = ivs[:nb], ivs[nb], ivs[nb + 1]

            def red(b2: Block, t: SSAValue, acc: SSAValue):
                a_idx = (list(bat) if lhs_batched else []) + [i]
                a_off = self.elem_off(b2, lhs, a_idx, t)
                if rhs_transposed:
                    b_idx = (list(bat) if rhs_batched else []) + [j]
                    b_off = self.elem_off(b2, rhs, b_idx, t)
                else:
                    b_idx = (list(bat) if rhs_batched else []) + [t]
                    b_off = self.elem_off(b2, rhs, b_idx, j)
                prod = self.fmul(b2, self.load(b2, lhs, a_off), self.load(b2, rhs, b_off))
                return b2, self.fadd(b2, acc, prod)

            tail, acc = self.reduce(blk, k, self.fc(0.0), red)
            tail, val = self.epilogue(tail, acc, ep, j)
            self.store(tail, out, self.elem_off(tail, out, list(bat) + [i], j), val)
            return tail

        return self.nest(cur, list(batch) + [m, n], body)

    def elementwise(self, cur: Block, out: str, src: str,
                    bias: str | None = None) -> Block:
        """A row-major copy, optionally with one length-N vector added to every row."""
        shape = self.wl.tensors[out].shape
        rows, cols = layout.row_count(shape), int(shape[-1]) if shape else 1

        def body(blk: Block, ivs: list[SSAValue]) -> Block:
            r, c = ivs
            off_in = self.affine(blk, [(r, layout.row_pitch(self.wl.tensors[src].shape)), (c, 1)])
            v = self.load(blk, src, off_in)
            if bias is not None:
                v = self.fadd(blk, v, self.load(blk, bias, self.elem_off(blk, bias, [], c)))
            self.store(blk, out, self.affine(blk, [(r, layout.row_pitch(shape)), (c, 1)]), v)
            return blk

        return self.nest(cur, [rows, cols], body)

    def fdiv(self, blk, a, b):
        op = llvm.FDivOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def fsqrt(self, blk, a):
        op = llvm.FSqrtOp(a)
        blk.add_op(op)
        return op.results[0]

    def rmsnorm(self, cur: Block, op) -> Block:
        """``y[i, j] = x[i, j] / sqrt(mean_k(x[i, k]**2) + eps) * gamma[j]``.

        A row reduction followed by a row-wise elementwise map, both in single precision, with the
        store rounding once into the declared output container. The row RMS is computed ONCE per
        row and reused across the row's columns, so the emitted nest is O(rows * cols) work rather
        than recomputing the reduction per element.
        """
        src = self.wl.tensors[op.operands["src"]]
        gamma = self.wl.tensors[op.operands["gamma"]]
        out = self.wl.tensors[op.out]
        if len(src.shape) < 2 or tuple(out.shape) != tuple(src.shape):
            raise ScalarLaneError(
                f"rmsnorm normalises a tensor over its trailing axis at the output's own extent; "
                f"got {tuple(src.shape)} -> {tuple(out.shape)}"
            )
        k = int(src.shape[-1])
        if int(gamma.shape[-1]) != k:
            raise ScalarLaneError(
                f"rmsnorm's gamma must span the normalised axis ({k}); got {tuple(gamma.shape)}"
            )
        eps = float(op.attrs.get("eps", 0.0))
        rows = layout.row_count(src.shape)
        s_pitch = layout.row_pitch(src.shape)
        o_pitch = layout.row_pitch(out.shape)
        g_pitch = layout.row_pitch(gamma.shape)

        # The reduction may already have been contracted on the mesh (:mod:`.lowering.sumsq`). When
        # it has, this lane READS it rather than recomputing it: the staged tile holds band `i//DIM`
        # of X @ X^T, whose DIAGONAL entry for row `i` is column `i mod DIM` of the band's row
        # `i mod DIM`. The band edge is a power of two, so the residue is a mask rather than a
        # division. Nothing else about the row's arithmetic changes.
        staged = op.operands.get("sumsq")
        if staged is not None and staged not in self.wl.tensors:
            raise ScalarLaneError(
                f"rmsnorm names a staged reduction '{staged}' the workload does not declare"
            )
        ss_pitch = layout.row_pitch(self.wl.tensors[staged].shape) if staged else 0

        def row_body(blk: Block, ivs: list[SSAValue]) -> Block:
            (r,) = ivs

            if staged is not None:
                lane = self.iand(blk, r, self.ic(ss_pitch - 1))
                off = self.affine(blk, [(r, ss_pitch), (lane, 1)])
                tail, ss = blk, self.load(blk, staged, off)
            else:
                def sq(b: Block, t: SSAValue, acc: SSAValue):
                    v = self.load(b, src.name, self.affine(b, [(r, s_pitch), (t, 1)]))
                    return b, self.fadd(b, acc, self.fmul(b, v, v))

                tail, ss = self.reduce(blk, k, self.fc(0.0), sq)
            ms = self.fdiv(tail, ss, self.fc(float(k)))
            denom = self.fsqrt(tail, self.fadd(tail, ms, self.fc(eps)))

            def col_body(b: Block, jvs: list[SSAValue]) -> Block:
                (c,) = jvs
                v = self.load(b, src.name, self.affine(b, [(r, s_pitch), (c, 1)]))
                g = self.load(b, gamma.name, self.affine(b, [(c, 1)], 0))
                y = self.fmul(b, self.fdiv(b, v, denom), g)
                self.store(b, out.name, self.affine(b, [(r, o_pitch), (c, 1)]), y)
                return b

            return self.nest(tail, [k], col_body)

        _ = g_pitch
        return self.nest(cur, [rows], row_body)

    # -- compile-time constant tables -----------------------------------------
    def table(self, name: str, values: list[float]) -> SSAValue:
        """A read-only f32 table of ``values``, materialised once and addressed as a pointer.

        A stage whose coefficients depend only on the DECLARED extents -- rotary angles are the
        case here -- has them all at compile time. Emitting them as data keeps the loop nest a loop
        (the emitted code does not grow with the payload) while the values themselves stay exactly
        what this compiler computed, which matters when the result is compared as an exact integer:
        an in-kernel polynomial for the same function would have to agree with the definition to the
        last ulp before the truncation, and a table has nothing to agree with.
        """
        if name not in self.tables:
            arr = llvm.LLVMArrayType(len(values), f32)
            self.tables[name] = llvm.GlobalOp(
                arr, name, "internal", constant=True,
                value=DenseIntOrFPElementsAttr.from_list(
                    TensorType(f32, [len(values)]), [float(v) for v in values]
                ),
            )
        op = llvm.AddressOfOp(name, self.ptr)
        self.prelude.append(op)
        return op.results[0]

    def table_load(self, blk: Block, base: SSAValue, off: SSAValue) -> SSAValue:
        gep = llvm.GEPOp(base, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[off],
                         result_type=self.ptr)
        blk.add_op(gep)
        ld = llvm.LoadOp(gep.result, f32)
        blk.add_op(ld)
        return ld.dereferenced_value

    def rope(self, cur: Block, op) -> Block:
        """Rotary position embedding over the trailing axis, in the corpus's own HALF-SPLIT form.

        With ``half = d // 2`` and ``freq[j] = theta ** (-j / half)``, position ``p`` rotates the
        pair ``(x[p, j], x[p, j + half])``::

            y[p, j]        = x[p, j] * cos(p*freq[j])        - x[p, j+half] * sin(p*freq[j])
            y[p, j+half]   = x[p, j+half] * cos(p*freq[j])   + x[p, j]      * sin(p*freq[j])

        which is ``x * cos + rotate_half(x) * sin`` with ``cos``/``sin`` duplicated across the two
        halves, so the angle for column ``j`` is indexed by ``j mod half``. Every angle is fixed by
        the DECLARED extents and the declared ``theta``, so both trig tables are built here rather
        than approximated in the kernel. The position is the index along the second-to-last axis,
        so a batched operand restarts its positions per matrix rather than running on through them.
        """
        src = self.wl.tensors[op.operands["src"]]
        out = self.wl.tensors[op.out]
        if len(src.shape) < 2 or tuple(out.shape) != tuple(src.shape):
            raise ScalarLaneError(
                f"rope maps a tensor onto its own extent over the trailing axis; got "
                f"{tuple(src.shape)} -> {tuple(out.shape)}"
            )
        d = int(src.shape[-1])
        if d % 2:
            raise ScalarLaneError(
                f"rope rotates PAIRS across the halves of its trailing axis, which needs an even "
                f"extent; the declared one is {d}"
            )
        half = d // 2
        theta = float(op.attrs.get("theta", 10000.0))
        rows = layout.row_count(src.shape)
        positions = int(src.shape[-2])
        s_pitch = layout.row_pitch(src.shape)
        o_pitch = layout.row_pitch(out.shape)

        cos_t, sin_t = [], []
        for r in range(rows):
            pos = r % positions
            for j in range(half):
                ang = pos * (theta ** (-(j / half)))
                cos_t.append(math.cos(ang))
                sin_t.append(math.sin(ang))
        tag = f"__rope_{op.out}_{rows}x{half}"
        cos_p = self.table(tag + "_cos", cos_t)
        sin_p = self.table(tag + "_sin", sin_t)

        def body(blk: Block, ivs: list[SSAValue]) -> Block:
            r, j = ivs
            ang_off = self.affine(blk, [(r, half), (j, 1)])
            c = self.table_load(blk, cos_p, ang_off)
            sn = self.table_load(blk, sin_p, ang_off)
            lo = self.affine(blk, [(r, s_pitch), (j, 1)])
            hi = self.affine(blk, [(r, s_pitch), (j, 1)], half)
            x1 = self.load(blk, src.name, lo)
            x2 = self.load(blk, src.name, hi)
            y1 = self.fadd(blk, self.fmul(blk, x1, c),
                           self.fmul(blk, self.fneg(blk, x2), sn))
            y2 = self.fadd(blk, self.fmul(blk, x2, c), self.fmul(blk, x1, sn))
            self.store(blk, out.name, self.affine(blk, [(r, o_pitch), (j, 1)]), y1)
            self.store(blk, out.name, self.affine(blk, [(r, o_pitch), (j, 1)], half), y2)
            return blk

        return self.nest(cur, [rows, half], body)

    def fneg(self, blk, a):
        op = llvm.FSubOp(self.fc(0.0), a)
        blk.add_op(op)
        return op.results[0]

    # -- exact integer staging (route W) ---------------------------------------
    def load_int(self, blk: Block, name: str, off: SSAValue) -> SSAValue:
        """One element of an INTEGER tensor, sign-extended to i32 and kept integral.

        The lane's ordinary :meth:`load` converts to single precision, which is right for arithmetic
        written in the reals but wrong here: splitting a 32-bit word into radix-256 digits and
        putting the partial products back together has to be bit-exact, and f32 cannot hold every
        i32 to begin with.
        """
        dt = self._container(name)
        if dt not in INT_CONTAINERS:
            raise ScalarLaneError(
                f"exact integer staging needs an integer container; {name!r} is declared {dt!r}"
            )
        bits, _, _ = INT_CONTAINERS[dt]
        ity = IntegerType(bits)
        gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], ity, ssa_indices=[off],
                         result_type=self.ptr)
        blk.add_op(gep)
        ld = llvm.LoadOp(gep.result, ity)
        blk.add_op(ld)
        v = ld.dereferenced_value
        if bits < 32:
            ext = llvm.SExtOp(v, i32)
            blk.add_op(ext)
            v = ext.results[0]
        return v

    def store_int(self, blk: Block, name: str, off: SSAValue, v: SSAValue) -> None:
        """Store an i32 into ``name``'s integer container, WRAPPING as that container does."""
        dt = self._container(name)
        if dt not in INT_CONTAINERS:
            raise ScalarLaneError(
                f"exact integer staging needs an integer container; {name!r} is declared {dt!r}"
            )
        bits, _, _ = INT_CONTAINERS[dt]
        ity = IntegerType(bits)
        val = v
        if bits < 32:
            tr = llvm.TruncOp(val, ity)
            blk.add_op(tr)
            val = tr.results[0]
        gep = llvm.GEPOp(self._arg(name), [llvm.GEP_USE_SSA_VAL], ity, ssa_indices=[off],
                         result_type=self.ptr)
        blk.add_op(gep)
        blk.add_op(llvm.StoreOp(val, gep.result))

    def _i32(self, blk: Block, v: int) -> SSAValue:
        op = llvm.ConstantOp(IntegerAttr(int(v), i32), i32)
        blk.add_op(op)
        return op.results[0]

    def int_digits(self, cur: Block, op) -> Block:
        """Split a wide integer tensor into the BALANCED radix-256 digits the operand port encodes.

        ``r_0 = x``; ``r_{s+1} = floor((r_s + 128) / 256)``; ``d_s = r_s - 256 * r_{s+1}``, which puts
        every digit in ``[-128, 127]``. Floor division by a power of two is the arithmetic right
        shift, so the whole split is two shifts and a subtract per digit. One pass over the operand
        writes all of the digits, so the payload is read once however many of them there are.
        """
        src = self.wl.tensors[op.operands["src"]]
        outs = list(op.attrs["outs"])
        bits = int(op.attrs.get("radix_bits", 8))
        radix, half = 1 << bits, 1 << (bits - 1)
        rows = layout.row_count(src.shape)
        pitch = layout.row_pitch(src.shape)
        for name in outs:
            d = self.wl.tensors[name]
            if layout.row_count(d.shape) != rows or layout.row_pitch(d.shape) != pitch:
                raise ScalarLaneError(
                    f"a digit image spans its operand exactly; {name!r} is {tuple(d.shape)} where "
                    f"{src.name!r} is {tuple(src.shape)}"
                )

        def body(blk: Block, ivs: list[SSAValue]) -> Block:
            (r, c) = ivs
            off = self.affine(blk, [(r, pitch), (c, 1)])
            rem = self.load_int(blk, src.name, off)
            for name in outs:
                biased = llvm.AddOp(rem, self._i32(blk, half))
                blk.add_op(biased)
                nxt = llvm.AShrOp(biased.results[0], self._i32(blk, bits))
                blk.add_op(nxt)
                back = llvm.ShlOp(nxt.results[0], self._i32(blk, bits))
                blk.add_op(back)
                digit = llvm.SubOp(rem, back.results[0])
                blk.add_op(digit)
                self.store_int(blk, name, off, digit.results[0])
                rem = nxt.results[0]
            return blk

        _ = radix
        return self.nest(cur, [rows, pitch], body)

    def digit_combine(self, cur: Block, op) -> Block:
        """Put the per-digit partial products back together: ``y = sum_s p_s << (bits * s)``.

        Summed in the output's own container, so the carry out of it is dropped exactly where the
        declared arithmetic drops it -- which is what makes the split of a wide operand into a fixed
        number of digits exact rather than approximate.
        """
        bits = int(op.attrs.get("radix_bits", 8))
        srcs = [op.operands[k] for k in sorted(op.operands, key=lambda k: int(k[3:]))]
        out = self.wl.tensors[op.out]
        rows = layout.row_count(out.shape)
        pitch = layout.row_pitch(out.shape)

        def body(blk: Block, ivs: list[SSAValue]) -> Block:
            (r, c) = ivs
            off = self.affine(blk, [(r, pitch), (c, 1)])
            acc: SSAValue | None = None
            for s, name in enumerate(srcs):
                v = self.load_int(blk, name, off)
                if s:
                    sh = llvm.ShlOp(v, self._i32(blk, bits * s))
                    blk.add_op(sh)
                    v = sh.results[0]
                if acc is None:
                    acc = v
                else:
                    add = llvm.AddOp(acc, v)
                    blk.add_op(add)
                    acc = add.results[0]
            self.store_int(blk, out.name, off, acc)
            return blk

        return self.nest(cur, [rows, pitch], body)

    def row_weight(self, cur: Block, op) -> Block:
        """``a[i, k] = x[i, k] * gamma[k]`` -- the contraction operand with the row scale hoisted out.

        Exact integer work: both factors are i8, so the product always fits the i16 container this
        stage writes, whatever values the operands carry.
        """
        src = self.wl.tensors[op.operands["src"]]
        gamma = self.wl.tensors[op.operands["gamma"]]
        dst = self.wl.tensors[op.out]
        k = int(src.shape[-1])
        if int(gamma.shape[-1]) != k:
            raise ScalarLaneError(
                f"the hoisted scale spans the contracted axis ({k}); got {tuple(gamma.shape)}"
            )
        rows = layout.row_count(src.shape)
        s_pitch = layout.row_pitch(src.shape)
        d_pitch = layout.row_pitch(dst.shape)

        def body(blk: Block, ivs: list[SSAValue]) -> Block:
            (r, c) = ivs
            x = self.load_int(blk, src.name, self.affine(blk, [(r, s_pitch), (c, 1)]))
            g = self.load_int(blk, gamma.name, self.affine(blk, [(c, 1)]))
            prod = llvm.MulOp(x, g)
            blk.add_op(prod)
            self.store_int(blk, dst.name, self.affine(blk, [(r, d_pitch), (c, 1)]),
                           prod.results[0])
            return blk

        return self.nest(cur, [rows, k], body)

    def row_normalize(self, cur: Block, op) -> Block:
        """``y[i, n] = trunc( z[i, n] / sqrt(mean_k(ref[i, k]**2) + eps) )``.

        The other half of the hoist: the row scale the contraction was relieved of, applied once per
        output element. The divisor is derived from the SAME declared operand the normalisation
        reads, so nothing about the definition has moved -- only where it is evaluated.
        """
        z = self.wl.tensors[op.operands["src"]]
        ref = self.wl.tensors[op.operands["ref"]]
        out = self.wl.tensors[op.out]
        eps = float(op.attrs.get("eps", 0.0))
        k = int(ref.shape[-1])
        rows = layout.row_count(out.shape)
        if layout.row_count(ref.shape) != rows:
            raise ScalarLaneError(
                f"the hoisted scale has one divisor per row of the result; {ref.name!r} carries "
                f"{layout.row_count(ref.shape)} rows where the result has {rows}"
            )
        cols = layout.row_pitch(out.shape)
        z_pitch = layout.row_pitch(z.shape)
        r_pitch = layout.row_pitch(ref.shape)

        def row_body(blk: Block, ivs: list[SSAValue]) -> Block:
            (r,) = ivs

            def sq(b: Block, t: SSAValue, acc: SSAValue):
                v = self.load(b, ref.name, self.affine(b, [(r, r_pitch), (t, 1)]))
                return b, self.fadd(b, acc, self.fmul(b, v, v))

            tail, ss = self.reduce(blk, k, self.fc(0.0), sq)
            denom = self.fsqrt(tail, self.fadd(tail, self.fdiv(tail, ss, self.fc(float(k))),
                                               self.fc(eps)))

            def col_body(b: Block, jvs: list[SSAValue]) -> Block:
                (c,) = jvs
                v = self.load(b, z.name, self.affine(b, [(r, z_pitch), (c, 1)]))
                self.store(b, out.name, self.affine(b, [(r, cols), (c, 1)]),
                           self.fdiv(b, v, denom))
                return b

            return self.nest(tail, [cols], col_body)

        return self.nest(cur, [rows], row_body)

    # -- softmax ---------------------------------------------------------------
    def _exp(self, blk: Block, v: SSAValue) -> SSAValue:
        """``e**v``, expanded from base arithmetic (the bare-metal harness links no math library)."""
        from .fmath import FloatBuilder

        return FloatBuilder(blk).exp(v)

    def _row_weights(self, blk: Block, src, r: SSAValue, k: int, pitch: int,
                     scale: float, sink: str | None = None) -> tuple[Block, SSAValue, list]:
        """Per-row softmax denominator, and the row maximum it is stabilised against.

        Returns the block to continue in plus ``(row_max, denom)``. The maximum is subtracted before
        the exponential -- the standard stabilisation, and the form the corpus's own linalg spelling
        of softmax uses (a `maximumf` reduce, a subtract, `math.exp`, an `addf` reduce, a divide).

        When ``sink`` names an f32 row buffer, each UNNORMALISED ``exp(scale*x - row_max)`` is stored
        into it as the denominator accumulates. The denominator pass already evaluates exactly that
        expression, so a caller that needs the weights themselves divides the buffer through by the
        denominator afterwards instead of evaluating a second exponential per element.
        """
        def rmax(b: Block, t: SSAValue, acc: SSAValue):
            v = self.load(b, src.name, self.affine(b, [(r, pitch), (t, 1)]))
            gt = llvm.FCmpOp(v, acc, "ogt")
            b.add_op(gt)
            sel = llvm.SelectOp(gt.res, v, acc)
            b.add_op(sel)
            return b, sel.results[0]

        tail, mx = self.reduce(blk, k, self.fc(-3.4028234663852886e38), rmax)
        mx = self.fmul(tail, mx, self.fc(scale))

        def dsum(b: Block, t: SSAValue, acc: SSAValue):
            v = self.load(b, src.name, self.affine(b, [(r, pitch), (t, 1)]))
            e = self._exp(b, self.sub_f(b, self.fmul(b, v, self.fc(scale)), mx))
            if sink is not None:
                self.store(b, sink, self.affine(b, [(t, 1)]), e)
            return b, self.fadd(b, acc, e)

        tail, den = self.reduce(tail, k, self.fc(0.0), dsum)
        return tail, mx, den

    def sub_f(self, blk, a, b):
        op = llvm.FSubOp(a, b)
        blk.add_op(op)
        return op.results[0]

    def softmax(self, cur: Block, op) -> Block:
        """``y[i, j] = exp(scale*x[i, j] - max_j) / sum_j exp(...)`` over the trailing axis."""
        src = self.wl.tensors[op.operands["src"]]
        out = self.wl.tensors[op.out]
        if tuple(out.shape) != tuple(src.shape):
            raise ScalarLaneError(
                f"softmax maps a tensor onto its own extent; got {tuple(src.shape)} -> "
                f"{tuple(out.shape)}"
            )
        scale = float(op.attrs.get("scale", 1.0))
        k = int(src.shape[-1])
        rows = layout.row_count(src.shape)
        s_pitch = layout.row_pitch(src.shape)
        o_pitch = layout.row_pitch(out.shape)

        def row_body(blk: Block, ivs: list[SSAValue]) -> Block:
            (r,) = ivs
            tail, mx, den = self._row_weights(blk, src, r, k, s_pitch, scale)

            def col(b: Block, jvs: list[SSAValue]) -> Block:
                (c,) = jvs
                v = self.load(b, src.name, self.affine(b, [(r, s_pitch), (c, 1)]))
                e = self._exp(b, self.sub_f(b, self.fmul(b, v, self.fc(scale)), mx))
                self.store(b, out.name, self.affine(b, [(r, o_pitch), (c, 1)]),
                           self.fdiv(b, e, den))
                return b

            return self.nest(tail, [k], col)

        return self.nest(cur, [rows], row_body)

    def softmax_weighted_sum(self, cur: Block, op) -> Block:
        """``y[i, n] = sum_j softmax(x)[i, j] * value[j, n]`` -- the weights never rounded.

        The softmax and the contraction that consumes it are one stage, so the weights are used at
        the precision they were computed in. Putting them through the interface's declared integer
        container first is a different function; see :mod:`..lowering.softmaxfuse`.
        """
        src = self.wl.tensors[op.operands["src"]]
        val = self.wl.tensors[op.operands["value"]]
        out = self.wl.tensors[op.out]
        scale = float(op.attrs.get("scale", 1.0))
        transposed = bool(op.attrs.get("transposed", False))
        k = int(src.shape[-1])
        rows = layout.row_count(src.shape)
        n = int(val.shape[-2] if transposed else val.shape[-1])
        if int(val.shape[-1] if transposed else val.shape[-2]) != k:
            raise ScalarLaneError(
                f"the weighted sum contracts {k} weights against {tuple(val.shape)}, which does "
                "not span that axis"
            )
        s_pitch = layout.row_pitch(src.shape)
        v_pitch = layout.row_pitch(val.shape)
        o_pitch = layout.row_pitch(out.shape)

        # The weights of one row are the same for every output column, so they are derived once into
        # a kernel-frame buffer and read back k times instead of being re-derived n*k times. See
        # `lowering.softmaxfuse.apply`: the stored value is the identical f32, so this is a cost
        # change only.
        #
        # The exponential is the expensive term (it is expanded from base arithmetic -- the
        # bare-metal harness links no math library), so it is evaluated ONCE per weight: the
        # denominator pass stores each unnormalised `exp(scale*x - mx)` into the row buffer as it
        # accumulates, and the buffer is then divided through by the denominator in place. The
        # stored value is the identical f32 that pass already computed, and the divide is the same
        # `fdiv` against the same denominator, so each weight is bit-for-bit what re-evaluating the
        # exponential would have produced.
        row_buf = op.operands.get("weights")

        def weight(b: Block, r: SSAValue, t: SSAValue, mx: SSAValue, den: SSAValue) -> SSAValue:
            x = self.load(b, src.name, self.affine(b, [(r, s_pitch), (t, 1)]))
            return self.fdiv(
                b, self._exp(b, self.sub_f(b, self.fmul(b, x, self.fc(scale)), mx)), den)

        def row_body(blk: Block, ivs: list[SSAValue]) -> Block:
            (r,) = ivs
            tail, mx, den = self._row_weights(blk, src, r, k, s_pitch, scale, sink=row_buf)

            if row_buf is not None:
                def normalise(b: Block, tvs: list[SSAValue]) -> Block:
                    (t,) = tvs
                    at = self.affine(b, [(t, 1)])
                    self.store(b, row_buf, at,
                               self.fdiv(b, self.load(b, row_buf, at), den))
                    return b

                tail = self.nest(tail, [k], normalise)

            def col(b: Block, jvs: list[SSAValue]) -> Block:
                (c,) = jvs

                def acc(b2: Block, t: SSAValue, sofar: SSAValue):
                    if row_buf is None:
                        w = weight(b2, r, t, mx, den)
                    else:
                        w = self.load(b2, row_buf, self.affine(b2, [(t, 1)]))
                    terms = [(c, v_pitch), (t, 1)] if transposed else [(t, v_pitch), (c, 1)]
                    v = self.load(b2, val.name, self.affine(b2, terms))
                    return b2, self.fadd(b2, sofar, self.fmul(b2, w, v))

                end, total = self.reduce(b, k, self.fc(0.0), acc)
                self.store(end, out.name, self.affine(end, [(r, o_pitch), (c, 1)]), total)
                return end

            return self.nest(tail, [n], col)

        return self.nest(cur, [rows], row_body)

    def conv2d(self, cur: Block, wl: Workload, op) -> Block:
        """im2col convolution, with the taps that fall outside the activation contributing zero."""
        ifm = wl.tensors[op.operands["ifm"]]
        weight = wl.tensors[wl.residents.get(op.operands["weight"], op.operands["weight"])]
        g = geometry_from(op.attrs, ifm.shape, weight.shape)
        out = op.out
        ep = op.epilogue

        def body(blk: Block, ivs: list[SSAValue]) -> Block:
            row, co = ivs
            # row -> (batch, oh, ow): the output pixel this committed row carries
            pix = self.ic(g.ho * g.wo)
            nb_ = llvm.SDivOp(row, pix)
            blk.add_op(nb_)
            rem = llvm.SRemOp(row, pix)
            blk.add_op(rem)
            oh = llvm.SDivOp(rem.results[0], self.ic(g.wo))
            blk.add_op(oh)
            ow = llvm.SRemOp(rem.results[0], self.ic(g.wo))
            blk.add_op(ow)
            b_i, oh_i, ow_i = nb_.results[0], oh.results[0], ow.results[0]

            def over_kh(bh: Block, kh_i: SSAValue, acc_h: SSAValue):
                ih = self.affine(bh, [(oh_i, g.sh), (kh_i, g.dh)], -g.pad_t)
                in_h = self.iand(
                    bh,
                    self.icmp(bh, ih, self.ic(0), 5),        # ih >= 0
                    self.icmp(bh, ih, self.ic(g.h), 2),      # ih <  H
                )

                def over_kw(bw: Block, kw_i: SSAValue, acc_w: SSAValue):
                    iw = self.affine(bw, [(ow_i, g.sw), (kw_i, g.dw)], -g.pad_l)
                    ok = self.iand(
                        bw,
                        self.iand(
                            bw,
                            self.icmp(bw, iw, self.ic(0), 5),
                            self.icmp(bw, iw, self.ic(g.w), 2),
                        ),
                        in_h,
                    )

                    def tap(bt: Block):
                        def over_c(bc: Block, c: SSAValue, acc_c: SSAValue):
                            a_off = self.elem_off(bc, ifm.name, [b_i, ih, iw], c)
                            k_row = self.affine(
                                bc, [(kh_i, g.kw * g.ci), (kw_i, g.ci), (c, 1)]
                            )
                            w_off = self.elem_off(bc, weight.name, [k_row], co)
                            prod = self.fmul(
                                bc, self.load(bc, ifm.name, a_off),
                                self.load(bc, weight.name, w_off),
                            )
                            return bc, self.fadd(bc, acc_c, prod)

                        return self.reduce(bt, g.ci, acc_w, over_c)

                    return self.guard(bw, ok, acc_w, tap)

                return self.reduce(bh, g.kw, acc_h, over_kw)

            tail, acc = self.reduce(blk, g.kh, self.fc(0.0), over_kh)
            tail, val = self.epilogue(tail, acc, ep, co)
            self.store(tail, out, self.elem_off(tail, out, [row], co), val)
            return tail

        return self.nest(cur, [g.m, g.co], body)


    def emit_ops(self, cur: Block, ops: list) -> tuple[Block, int]:
        """Compile ``ops`` into blocks appended after ``cur``; returns (last block, count).

        The same dispatch serves a whole-program lane route and a single off-mesh STAGE of a hybrid
        kernel -- one operation model, one set of loop nests, whichever route asked for it.
        """
        commit_of = {op.operands["src"]: op for op in self.wl.ops if op.kind == "commit"}
        emitted = 0
        for op in ops:
            if op.kind in ("resident_pack", "evict", "matmul"):
                continue
            if op.kind == "commit":
                mm = next((o for o in self.wl.ops if o.kind == "matmul" and o.out == op.operands["src"]),
                          None)
                if mm is None:
                    raise ScalarLaneError("host lane: a commit consumes an accumulator no matmul made")
                rhs = self.wl.residents.get(mm.operands["rhs"], mm.operands["rhs"])
                cur = self.contraction(
                    cur, op.out, [], mm.shape["m"], mm.shape["n"], mm.shape["k"],
                    mm.operands["lhs"], rhs, ep=op.epilogue,
                )
            elif op.kind == "matmul_batched":
                a, w = self.wl.tensors[op.operands["a"]], self.wl.tensors[op.operands["w"]]
                batch = [int(d) for d in a.shape[:-2]]
                cur = self.contraction(
                    cur, op.out, batch, int(a.shape[-2]), int(w.shape[-1]), int(a.shape[-1]),
                    a.name, w.name, ep=op.epilogue,
                )
            elif op.kind in ("attention_qk", "attention_pv"):
                qk = op.kind == "attention_qk"
                lhs = self.wl.tensors[op.operands["q" if qk else "p"]]
                rhs = self.wl.tensors[op.operands["k" if qk else "v"]]
                # attention_qk contracts the TRAILING head dim of both operands; attention_pv does not
                n = int(rhs.shape[-2]) if qk else int(rhs.shape[-1])
                cur = self.contraction(
                    cur, op.out, [], int(lhs.shape[-2]), n, int(lhs.shape[-1]),
                    lhs.name, rhs.name, rhs_transposed=qk, ep=op.epilogue,
                )
            elif op.kind == "movement":
                cur = self.elementwise(cur, op.out, op.operands["src"])
            elif op.kind == "bias_add":
                cur = self.elementwise(cur, op.out, op.operands["src"], bias=op.operands["bias"])
            elif op.kind == "conv2d":
                cur = self.conv2d(cur, wl, op)
            elif op.kind == "rmsnorm":
                cur = self.rmsnorm(cur, op)
            elif op.kind == "rope":
                cur = self.rope(cur, op)
            elif op.kind == "softmax":
                cur = self.softmax(cur, op)
            elif op.kind == "softmax_weighted_sum":
                cur = self.softmax_weighted_sum(cur, op)
            elif op.kind == "row_weight":
                cur = self.row_weight(cur, op)
            elif op.kind == "row_normalize":
                cur = self.row_normalize(cur, op)
            elif op.kind == "int_digits":
                cur = self.int_digits(cur, op)
            elif op.kind == "digit_combine":
                cur = self.digit_combine(cur, op)
            else:
                raise ScalarLaneError(f"host lane: no scalar lowering for {op.kind!r}")
            emitted += 1
        return cur, emitted


def compile_workload(wl: Workload, args: list[str]) -> ModuleOp:
    """Compile every region of ``wl`` onto the scalar lane and return the LLVM-dialect module."""
    lane = ScalarLane(wl, args)
    cur, emitted = lane.emit_ops(lane.entry, list(wl.ops))
    if not emitted:
        raise ScalarLaneError("host lane: the program carries no region to compute")
    cur.add_op(llvm.ReturnOp())
    for op in reversed(lane.prelude):
        first = lane.entry.first_op
        if first is None:
            lane.entry.add_op(op)
        else:
            lane.entry.insert_op_before(op, first)
    fn = llvm.FuncOp(
        KERNEL_SYMBOL,
        llvm.LLVMFunctionType([lane.ptr] * len(args)),
        linkage=llvm.LinkageAttr("external"),
        body=Region(lane.blocks),
    )
    module = ModuleOp([*lane.tables.values(), *lane.helpers.values(), fn])
    module.verify()
    return module


def stage_tensors(wl: Workload, ops: list) -> list[str]:
    """The DRAM tensors one off-mesh stage touches, in a stable order.

    These become the stage function's pointer parameters, so the stage reaches exactly the buffers
    its own regions name and nothing else.
    """
    names: list[str] = []

    def take(ref: str | None) -> None:
        if not ref:
            return
        name = wl.residents.get(ref, ref)
        if name in wl.tensors and name not in names:
            names.append(name)

    for op in ops:
        if op.kind in ("resident_pack", "evict"):
            continue
        for ref in op.operands.values():
            take(ref)
        if op.epilogue is not None:
            take(op.epilogue.bias)
        for name in op.attrs.get("outs", [op.out]):
            take(name)
    return names


def compile_stage(wl: Workload, ops: list, name: str, helpers: dict,
                  tables: dict) -> tuple[llvm.FuncOp, list[str]]:
    """Compile ONE off-mesh stage into its own function of the host module.

    Route X keeps the stage OUT of ``gemmini_kernel``'s own body and calls it from the point the
    command stream marks. That is not cosmetic: the kernel's body is the artifact a consumer decodes
    into an instruction trace, and it can only expand a tiled command loop into the commands it
    issues while every value it steps through is one the loop itself defines. Scalar arithmetic on
    payload the consumer cannot know -- a load, a call, a trig table -- inlined into the same body
    leaves it no such footing, and it falls back to reading the body statically, which reports a
    5-tile store loop as ONE store. A call is opaque and costs it nothing.
    """
    args = stage_tensors(wl, ops)
    lane = ScalarLane(wl, args, helpers=helpers, tables=tables)
    cur, emitted = lane.emit_ops(lane.entry, list(ops))
    if not emitted:
        raise ScalarLaneError("lane stage: the stage carries no region to compute")
    cur.add_op(llvm.ReturnOp())
    for op in reversed(lane.prelude):
        first = lane.entry.first_op
        if first is None:
            lane.entry.add_op(op)
        else:
            lane.entry.insert_op_before(op, first)
    fn = llvm.FuncOp(
        name, llvm.LLVMFunctionType([lane.ptr] * len(args)),
        linkage=llvm.LinkageAttr("internal"), body=Region(lane.blocks),
    )
    return fn, args
