"""Hand scheduled, CPU-looped Gemmini GEMM using only primitive RoCC commands.

The public capsule backend expands every tile into an instruction. This separate
model-scale kernel keeps a tuned output block resident in the accumulator and
uses ordinary RISC-V branches for M/N/K repetition. It emits an xDSL Gemmini
dialect schedule that `golden_device_lower` converts to LLVM inline asm. It
has no path to a Gemmini LOOP_* command.

ABI: ``void gemmini_golden_gemm(i8 *A, i8 *B, <i8|i32> *C[, i32 *bias])``
with dense row-major A[M,K], B[K,N], C[M,N].  The output is initialized from
zero or the optional bias, and optionally narrowed with scale/ReLU through
Gemmini's store path.  Callers must match the model's exact semantics.

An explicit segmented input contract may borrow a typed dense i8 owner through
a noncontiguous read-only matrix view. The caller closes owner storage/lifetime
and output nonoverlap; this option changes input DMA addresses only.
"""

from __future__ import annotations

import argparse
import io
import math
import json
from typing import Callable
from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import Float32Type, FloatAttr, IntegerAttr, ModuleOp, StringAttr, i8, i64
from xdsl.ir import SSAValue
from xdsl.printer import Printer

from merlin.llvmlower.segmented_matrix_view import SegmentedRows

from .codegen.builder import FnBuilder, PTR
from .ir import gemmini_dialect as G
from .tables import isa, rtl_facts as F


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    output_dtype: str = "i8"
    bm: int = 4
    bn: int = 4
    bias: bool = False
    scale: float = 1.0
    relu: bool = False
    wide_store: bool = False
    reuse_b: bool = False
    cache_b: bool = False
    cache_a: bool = False
    pipeline_m: bool = False
    prefetch_m: bool = False
    banked_m: bool = False
    wide_a: bool = False
    wide_b: bool = False
    separate_b_bank: bool = False
    prefetch_b: bool = False

    def validate(self, *, prefetch_b_rows: tuple[int, int] | None = None,
                 cached_b_resource_capacity: bool = False) -> None:
        if type(cached_b_resource_capacity) is not bool:
            raise ValueError("cached B capacity selection must be boolean")
        if cached_b_resource_capacity and not self.cache_b:
            raise ValueError("cached B capacity selection requires cached B")
        if min(self.m, self.n, self.k, self.bm, self.bn) <= 0:
            raise ValueError("all extents and block sizes must be positive")
        if self.output_dtype not in ("i8", "i32"):
            raise ValueError("output dtype must be i8 or i32")
        if self.scale <= 0 or not math.isfinite(self.scale):
            raise ValueError("store scale must be finite and positive")
        if self.output_dtype == "i32" and (self.scale != 1.0 or self.relu):
            raise ValueError("i32 readout does not apply scale or ReLU")
        if self.wide_a and (self.k <= F.DIM or self.k % F.DIM):
            raise ValueError("wide A panels require K to be a positive tile multiple larger than one tile")
        # Wide B groups up to four adjacent tiles within each output block.
        # Partial channel groups and multiple N blocks use their exact extents.
        if self.cache_a and (_ceil_div(self.m, F.DIM) > self.bm or
                             self.wide_a or self.pipeline_m or self.cache_b):
            raise ValueError("cached A needs one complete M block and no competing A schedule")
        if self.banked_m and not (self.pipeline_m and self.wide_a and self.cache_b):
            raise ValueError("banked M needs alternating slots, wide A, and cached B")
        if self.banked_m and self.bm * _ceil_div(self.k, F.DIM) * F.DIM > F.SPAD_BANK_ROWS:
            raise ValueError("A block does not fit one stock scratchpad bank")
        if self.prefetch_m and not (self.pipeline_m and self.wide_a and self.cache_b):
            raise ValueError("M prefetch needs alternating slots, wide A, and cached B")
        if prefetch_b_rows is not None and not self.prefetch_b:
            raise ValueError("explicit B slots require B prefetch")
        if self.prefetch_b:
            if not self.cache_a or self.k <= F.DIM or self.separate_b_bank:
                raise ValueError("B prefetch needs cached A, multiple K panels, and its own bank placement")
            if self.bn * F.DIM > F.SPAD_BANK_ROWS:
                raise ValueError("one prefetched B panel must fit one scratchpad bank")
            a_end = self.bm * _ceil_div(self.k, F.DIM) * F.DIM
            if prefetch_b_rows is None:
                if a_end > 2 * F.SPAD_BANK_ROWS:
                    raise ValueError("prefetched B requires cached A to fit the lower two banks")
            else:
                if (type(prefetch_b_rows) is not tuple or len(prefetch_b_rows) != 2
                        or any(type(row) is not int or row < 0 or row % F.DIM
                               for row in prefetch_b_rows)):
                    raise ValueError("B slots require two nonnegative DIM-aligned row bases")
                span = self.bn * F.DIM
                for row in prefetch_b_rows:
                    if row < a_end or row + span > F.SPAD_ROWS:
                        raise ValueError("B slot overlaps reserved A or exceeds the scratchpad")
                    if row // F.SPAD_BANK_ROWS != (row + span - 1) // F.SPAD_BANK_ROWS:
                        raise ValueError("each B slot must fit within one scratchpad bank")
                first, second = sorted(prefetch_b_rows)
                if first + span > second:
                    raise ValueError("prefetched B slots overlap")
        slots = 2 if self.pipeline_m else 1
        a_panel_tiles = _ceil_div(self.k, F.DIM) if self.wide_a or self.cache_a else 1
        a_rows = slots * self.bm * a_panel_tiles
        if slots * self.bm * self.bn * F.DIM > F.ACC_ROWS:
            raise ValueError("output block exceeds the accumulator")
        if (a_rows + self.bn) * F.DIM > F.SPAD_ROWS:
            raise ValueError("operand block exceeds the scratchpad")
        if self.cache_b:
            nt, kt = _ceil_div(self.n, F.DIM), _ceil_div(self.k, F.DIM)
            if nt > self.bn:
                raise ValueError("cached B needs one output-channel block")
            # Keep the established compile-size budget unless explicitly
            # selecting a capacity-proved complete-weight schedule. The
            # latter still proves the actual placement, including bank gaps.
            if ((not cached_b_resource_capacity and kt * nt > 128)
                    or (a_rows + kt * nt) * F.DIM > F.SPAD_ROWS):
                raise ValueError("cached B panels exceed the static or scratchpad budget")
            b_base = (2 * F.SPAD_BANK_ROWS if self.banked_m or self.separate_b_bank
                      else a_rows * F.DIM)
            if b_base + kt * nt * F.DIM > F.SPAD_ROWS:
                raise ValueError("placed cached B panels exceed the scratchpad")
        if self.separate_b_bank:
            if self.banked_m:
                raise ValueError("separate B placement is redundant with banked M")
            b_tiles = (_ceil_div(self.k, F.DIM) * _ceil_div(self.n, F.DIM)
                       if self.cache_b else self.bn)
            if a_rows * F.DIM > 2 * F.SPAD_BANK_ROWS:
                raise ValueError("A panels exceed the lower two scratchpad banks")
            if b_tiles * F.DIM > F.SPAD_ROWS - 2 * F.SPAD_BANK_ROWS:
                raise ValueError("B panels exceed the upper two scratchpad banks")
        if self.k > 0xFFFFFFFF or self.n > 0xFFFFFFFF:
            raise ValueError("row strides exceed the Gemmini configuration field")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _groups(extent: int, block: int) -> list[tuple[int, int, int, tuple[int, ...]]]:
    """Generate full and edge groups; a group's row widths are compile-time facts."""
    tiles = _ceil_div(extent, F.DIM)
    # Only tiles with all DIM elements belong to the full-width loop.  The
    # final partial tile must remain in the edge group even when block=1.
    full = (extent // F.DIM) // block * block
    out = []
    if full:
        out.append((0, full, block, (F.DIM,) * block))
    if full < tiles:
        widths = tuple(min(F.DIM, extent - (full + i) * F.DIM)
                       for i in range(tiles - full))
        out.append((full, tiles, tiles - full, widths))
    return out


class GoldenGemm:
    def __init__(self, shape: Shape, *, prefetch_b_rows: tuple[int, int] | None = None,
            resident_a_load_tiles: int = 1, input_view: SegmentedRows | None = None,
            cached_b_resource_capacity: bool = False,
            stationary_b_tail_before_last_full: bool = False):
        shape.validate(prefetch_b_rows=prefetch_b_rows,
                       cached_b_resource_capacity=cached_b_resource_capacity)
        if type(resident_a_load_tiles) is not int or not 1 <= resident_a_load_tiles <= 4:
            raise ValueError('resident A DMA grouping must be an integer in 1..4')
        if resident_a_load_tiles != 1 and not shape.cache_a:
            raise ValueError('resident A DMA grouping requires the complete cached A layout')
        if input_view is not None:
            if not isinstance(input_view, SegmentedRows):
                raise ValueError('input view requires the typed segmented-row contract')
            if not shape.cache_a:
                raise ValueError('segmented input requires complete cached A residency')
            if (input_view.rows, input_view.cols) != (shape.m, shape.k):
                raise ValueError('input view logical shape differs from GEMM')
            if input_view.dtype != 'i8' or input_view.element_bytes != 1:
                raise ValueError('segmented input requires physical i8 source storage')
            if input_view.row_stride > 0xFFFFFFFF:
                raise ValueError('input view row stride exceeds the configuration field')
            if input_view.source_elements >= 1 << 63:
                raise ValueError('input view source extent exceeds signed pointer indexing')
        if type(stationary_b_tail_before_last_full) is not bool:
            raise ValueError("stationary B tail selection must be boolean")
        if stationary_b_tail_before_last_full and (
                not shape.cache_a or not shape.reuse_b or shape.bias
                or shape.m // F.DIM < 2 or shape.m % F.DIM == 0):
            raise ValueError("stationary B tail placement requires complete cached A, reuse B, two full tiles, one short tail and no bias")
        self.stationary_b_tail_before_last_full = stationary_b_tail_before_last_full
        self.shape = shape
        self.cached_b_resource_capacity = cached_b_resource_capacity
        self.input_view = input_view
        self.resident_a_load_tiles = resident_a_load_tiles
        # Placement is a target schedule fact, separate from source dimensions
        # and numeric semantics. The default retains the established banks2/3.
        self.prefetch_b_rows = prefetch_b_rows
        self.fb = FnBuilder([PTR] * (4 if shape.bias else 3))
        self.a, self.b, self.c = self.fb.entry.args[:3]
        self.bias = self.fb.entry.args[3] if shape.bias else None

    def _rocc(self, kind: str, attrs: dict, pointer: SSAValue | None = None, *,
              operands: tuple[SSAValue, ...] | None = None) -> None:
        if operands is not None and pointer is not None:
            raise ValueError("command operands and pointer are mutually exclusive")
        values = list(operands) if operands is not None else ([pointer] if pointer is not None else [])
        cls = {
            "flush": G.FlushOp, "config_ex": G.ConfigExOp,
            "config_ld": G.ConfigLdOp, "config_st": G.ConfigStOp,
            "mvin": G.MvinOp, "mvout": G.MvoutOp,
            "preload": G.PreloadOp, "compute": G.ComputeOp,
            "fence": G.FenceOp,
        }[kind]
        op = cls(operands=[values], result_types=[[]])
        op.attributes.update({k: IntegerAttr(int(v), i64) if isinstance(v, int)
                              else FloatAttr(float(v), Float32Type()) if isinstance(v, float)
                              else StringAttr(str(v)) for k, v in attrs.items()})
        self.fb.add(op)

    def _ptr(self, base: SSAValue, row: SSAValue, row_pitch: int,
             col: SSAValue, element_bytes: int = 1) -> SSAValue:
        off = self.fb.add_i(self.fb.mul_i(row, self.fb.const(row_pitch)), col)
        if element_bytes != 1:
            off = self.fb.mul_i(off, self.fb.const(element_bytes))
        return self.fb.add(llvm.GEPOp(base, [llvm.GEP_USE_SSA_VAL], i8,
                                      ssa_indices=[off])).results[0]

    def _tile(self, index: SSAValue, delta: int) -> SSAValue:
        return self.fb.mul_i(self.fb.add_i(index, self.fb.const(delta)),
                             self.fb.const(F.DIM))

    def _load_b_panel(self, n0: SSAValue, k0: SSAValue,
                      nr: tuple[int, ...], kr: int, b_base: int) -> None:
        s = self.shape
        krow = self._tile(k0, 0)
        if s.wide_b:
            for d in range(0, len(nr), 4):
                ptr = self._ptr(self.b, krow, s.n, self._tile(n0, d))
                self._rocc("mvin", {"local": (b_base + d) * F.DIM,
                    "rows": kr, "cols": sum(nr[d:d + 4]), "load_id": 1}, ptr)
        else:
            for d, cols in enumerate(nr):
                ptr = self._ptr(self.b, krow, s.n, self._tile(n0, d))
                self._rocc("mvin", {"local": (b_base + d) * F.DIM,
                    "rows": kr, "cols": cols, "load_id": 1}, ptr)

    def _k_tile(self, m0: SSAValue, n0: SSAValue, k0: SSAValue,
                mr: tuple[int, ...], nr: tuple[int, ...], kr: int,
                first: bool, cached_k: int | None = None, slot: int = 0,
                wide_k: int | None = None, b_slot: int = 0) -> None:
        s = self.shape
        kt = _ceil_div(s.k, F.DIM)
        a_panel_tiles = kt if s.wide_a or s.cache_a else 1
        a_base = slot * s.bm * a_panel_tiles if s.pipeline_m else 0
        b_base = (2 if s.pipeline_m else 1) * s.bm * a_panel_tiles
        if s.separate_b_bank:
            b_base = 2 * F.SPAD_BANK_ROWS // F.DIM
        if s.banked_m:
            a_base = slot * (F.SPAD_BANK_ROWS // F.DIM)
            b_base = 2 * F.SPAD_BANK_ROWS // F.DIM
        if s.prefetch_b:
            b_base = self._prefetched_b_base(b_slot)
        acc_base = slot * (F.ACC_BANK_ROWS // F.DIM if s.banked_m else s.bm * s.bn) if s.pipeline_m else 0
        krow = self._tile(k0, 0)
        if not s.wide_a and not s.cache_a:
            for a, rows in enumerate(mr):
                mrow = self._tile(m0, a)
                ptr = self._ptr(self.a, mrow, s.k, krow)
                self._rocc("mvin", {"local": (a_base + a) * F.DIM, "rows": rows,
                                      "cols": kr, "load_id": 0}, ptr)
        if cached_k is None and not s.prefetch_b:
            self._load_b_panel(n0, k0, nr, kr, b_base)
        def b_addr(d: int) -> int:
            if cached_k is None:
                return (b_base + d) * F.DIM
            return (b_base + cached_k * _ceil_div(s.n, F.DIM) + d) * F.DIM
        def a_addr(a: int) -> int:
            return (a_base + a * a_panel_tiles + (wide_k or 0)) * F.DIM
        dynamic_a = self.fb.mul_i(k0, self.fb.const(F.DIM)) if s.cache_a else None

        def compute(rows: int, accumulate: bool, a: int = 0) -> None:
            attrs = {"a_cols": kr, "a_rows": rows, "accumulate": accumulate}
            if dynamic_a is None:
                attrs["a"] = a_addr(0)
                self._rocc("compute", attrs)
            else:
                attrs["a_max"] = (a * kt + kt - 1) * F.DIM
                attrs["a_reserved_rows"] = s.bm * kt * F.DIM
                address=dynamic_a if a == 0 else self.fb.add_i(dynamic_a,self.fb.const(a * kt * F.DIM))
                self._rocc("compute", attrs, address)

        if s.reuse_b:
            # Keep a weight tile in the array while varying the A row tile.
            # A garbage BD address on subsequent PRELOADs preserves the
            # stationary operand; COMPUTE_ACCUMULATE is the matching opcode.
            row_order = list(range(len(mr)))
            if self.stationary_b_tail_before_last_full:
                # The first tile loads real B. A short independent output tile
                # now executes before retained full tiles, so its next PRELOAD
                # keeps the stationary weight and has an effective garbage D.
                # Physical A/C addresses and each output's increasing K stay.
                assert len(mr) >= 3 and mr[-1] < F.DIM
                assert all(rows == F.DIM for rows in mr[:-1])
                row_order = [0, len(mr) - 1, *range(1, len(mr) - 1)]
            for d, cols in enumerate(nr):
                for a in row_order:
                    rows = mr[a]
                    acc_row = (acc_base + a * s.bn + d) * F.DIM
                    self._rocc("preload", {
                        "bd": b_addr(d) if a == 0 else isa.GARBAGE_ADDR,
                        "c": isa.acc_addr(acc_row, accumulate=not first),
                        "bd_cols": cols, "bd_rows": kr,
                        "c_cols": cols, "c_rows": rows})
                    if dynamic_a is None:
                        self._rocc("compute", {"a": a_addr(a),
                                                "a_cols": kr, "a_rows": rows,
                                                "accumulate": a != 0})
                    else:
                        compute(rows, a != 0, a)
        else:
            for a, rows in enumerate(mr):
                for d, cols in enumerate(nr):
                    acc_row = (acc_base + a * s.bn + d) * F.DIM
                    self._rocc("preload", {
                        "bd": b_addr(d),
                        "c": isa.acc_addr(acc_row, accumulate=not first),
                        "bd_cols": cols, "bd_rows": kr,
                        "c_cols": cols, "c_rows": rows})
                    if dynamic_a is None:
                        self._rocc("compute", {"a": a_addr(a),
                                                "a_cols": kr, "a_rows": rows})
                    else:
                        compute(rows, False, a)

    def _prepare_output_block(self, m0: SSAValue, n0: SSAValue,
                      mr: tuple[int, ...], nr: tuple[int, ...],
                      slot: int = 0) -> None:
        s = self.shape
        acc_base = slot * (F.ACC_BANK_ROWS // F.DIM if s.banked_m else s.bm * s.bn) if s.pipeline_m else 0
        if s.wide_a:
            kt = _ceil_div(s.k, F.DIM)
            a_base = slot * (F.SPAD_BANK_ROWS // F.DIM if s.banked_m else s.bm * kt) if s.pipeline_m else 0
            for a, rows in enumerate(mr):
                mrow = self._tile(m0, a)
                # Retain the full K panel, issuing only the already supported
                # at-most-four-tile MVIN width. Each group occupies disjoint
                # local K tiles; resource validation covers the complete panel.
                for ki in range(0, kt, 4):
                    ptr = self._ptr(self.a, mrow, s.k, self.fb.const(ki * F.DIM))
                    self._rocc("mvin", {"local": (a_base + a * kt + ki) * F.DIM,
                                          "rows": rows, "cols": min(4 * F.DIM, s.k-ki * F.DIM),
                                          "load_id": 0}, ptr)
        if self.bias is not None:
            for a, rows in enumerate(mr):
                for d, cols in enumerate(nr):
                    ncol = self._tile(n0, d)
                    ptr = self._ptr(self.bias, self.fb.const(0), s.n,
                                    ncol, 4)
                    self._rocc("mvin", {"local": isa.acc_addr((acc_base + a * s.bn + d) * F.DIM),
                                          "rows": rows, "cols": cols,
                                          "load_id": 2}, ptr)

    def _prefetched_b_base(self, slot: int) -> int:
        """Return a proved scratchpad row base in DIM-tile units."""
        if self.prefetch_b_rows is not None:
            return self.prefetch_b_rows[slot] // F.DIM
        return (2 + slot) * F.SPAD_BANK_ROWS // F.DIM

    def _reduce_output_block(self, m0, n0, mr, nr, slot=0, prefetch=None):
        s = self.shape
        if s.prefetch_b:
            panels = _ceil_div(s.k, F.DIM)
            def load(ki, bank, kr=F.DIM):
                self._load_b_panel(n0, ki, nr, kr, self._prefetched_b_base(bank))
            def compute(ki, bank, kr=F.DIM, first=False):
                self._k_tile(m0, n0, ki, mr, nr, kr, first,
                             slot=slot, b_slot=bank)
            load(self.fb.const(0), 0)
            load(self.fb.const(1), 1, min(F.DIM, s.k - F.DIM))
            compute(self.fb.const(0), 0, first=self.bias is None)
            # Every pair prefetches a full next panel before current mesh work.
            # Reserve the final one to three panels for a static exact tail drain.
            pairs = max(0, (s.k // F.DIM - 2) // 2)
            def pair(ki):
                next_ki = self.fb.add_i(ki, self.fb.const(1))
                load(next_ki, 0)
                compute(ki, 1)
                load(self.fb.add_i(ki, self.fb.const(2)), 1)
                compute(next_ki, 0)
            self.fb.for_loop(1, 1 + 2 * pairs, 2, pair)
            for ki in range(1 + 2 * pairs, panels):
                if ki + 1 < panels:
                    load(self.fb.const(ki + 1), (ki + 1) % 2,
                         min(F.DIM, s.k - (ki + 1) * F.DIM))
                compute(self.fb.const(ki), ki % 2,
                        min(F.DIM, s.k - ki * F.DIM))
        elif s.cache_b or s.wide_a:
            for ki in range(_ceil_div(s.k, F.DIM)):
                self._k_tile(m0, n0, self.fb.const(ki), mr, nr,
                             min(F.DIM, s.k - ki * F.DIM),
                             self.bias is None and ki == 0,
                             ki if s.cache_b else None, slot,
                             ki if s.wide_a or s.cache_a else None)
                if prefetch is not None and ki == max(0, _ceil_div(s.k, F.DIM) // 2 - 1):
                    prefetch()
        else:
            first_kr = min(F.DIM, s.k)
            self._k_tile(m0, n0, self.fb.const(0), mr, nr, first_kr,
                         self.bias is None, slot=slot)
            full_k = s.k // F.DIM
            if full_k > 1:
                self.fb.for_loop(1, full_k, 1,
                                 lambda k: self._k_tile(m0, n0, k, mr, nr, F.DIM,
                                                        False, slot=slot))
            if s.k % F.DIM and s.k > F.DIM:
                self._k_tile(m0, n0, self.fb.const(full_k), mr, nr,
                             s.k % F.DIM, False, slot=slot)

    def _output_block(self, m0: SSAValue, n0: SSAValue,
                      mr: tuple[int, ...], nr: tuple[int, ...],
                      slot: int = 0, *, prepared: bool = False,
                      prefetch: Callable[[], None] | None = None) -> None:
        s = self.shape
        acc_base = slot * (F.ACC_BANK_ROWS // F.DIM if s.banked_m else s.bm * s.bn) if s.pipeline_m else 0
        if not prepared:
            self._prepare_output_block(m0, n0, mr, nr, slot)
        self._reduce_output_block(m0, n0, mr, nr, slot, prefetch)
        out_bytes = 4 if s.output_dtype == "i32" else 1
        for a, rows in enumerate(mr):
            mrow = self._tile(m0, a)
            d = 0
            while d < len(nr):
                # Stock Gemmini can commit adjacent accumulator tiles in one
                # 16x64 store.  Mirror the reference's wide MVOUT for full i8
                # tiles; keep partial edges and i32 readout conservative.
                width = 1
                if s.wide_store and s.output_dtype == "i8" and nr[d] == F.DIM:
                    while (width < 4 and d + width < len(nr) and
                           nr[d + width] == F.DIM):
                        width += 1
                ncol = self._tile(n0, d)
                ptr = self._ptr(self.c, mrow, s.n, ncol, out_bytes)
                self._rocc("mvout", {
                    "local": isa.acc_addr((acc_base + a * s.bn + d) * F.DIM,
                                           full_row=s.output_dtype == "i32"),
                    "rows": rows, "cols": sum(nr[d:d + width])}, ptr)
                d += width

    def _for_groups(self, groups, body, *, retain_loop=False) -> None:
        for start, stop, step, widths in groups:
            if stop - start == step:
                body(self.fb.const(start), widths)
            else:
                self.fb.for_loop(start, stop, step,
                                 lambda iv, w=widths: body(iv, w),retain_loop=retain_loop)

    def _emit_config(self) -> None:
        s = self.shape
        self._rocc("fence", {})
        self._rocc("flush", {})
        self._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY})
        self._rocc("config_ld", {"stride": self.input_view.row_stride if self.input_view else s.k,
                                  "load_id": 0})
        self._rocc("config_ld", {"stride": s.n, "load_id": 1})
        if self.bias is not None:
            self._rocc("config_ld", {"stride": 0, "load_id": 2})
        self._rocc("config_st", {"stride": s.n * (4 if s.output_dtype == "i32" else 1),
                                  "acc_act": isa.RELU if s.relu else isa.NO_ACTIVATION,
                                  "acc_scale": s.scale})

    def _emit_work(self) -> None:
        s = self.shape
        if s.cache_a:
            kt=_ceil_div(s.k,F.DIM)
            for a in range(_ceil_div(s.m,F.DIM)):
                rows=min(F.DIM,s.m-a * F.DIM)
                # CONFIG_LD uses DIM block stride: widening the DRAM packet
                # fills the same consecutive K tiles in the proved complete
                # resident A allocation. Partial K packets preserve exact
                # extents; later execute addresses and reduction order stay.
                for ki in range(0,kt,self.resident_a_load_tiles):
                    kr = min(F.DIM*self.resident_a_load_tiles, s.k - ki * F.DIM)
                    if self.input_view is not None:
                        first_row=a * F.DIM
                        for row,count,offset in self.input_view.split_rows(first_row,rows):
                            ptr=self._ptr(self.a,self.fb.const(0),1,
                                          self.fb.const(offset+ki * F.DIM))
                            self._rocc("mvin", {"local": (a * kt + ki) * F.DIM + row-first_row,
                                "rows": count, "cols": kr, "load_id": 0},ptr)
                        continue
                    ptr = self._ptr(self.a, self.fb.const(a * F.DIM), s.k,
                                    self.fb.const(ki * F.DIM))
                    self._rocc("mvin", {"local": (a * kt + ki) * F.DIM,
                                          "rows": rows, "cols": kr, "load_id": 0}, ptr)
        if s.cache_b:
            nt = _ceil_div(s.n, F.DIM)
            a_panel_tiles = _ceil_div(s.k, F.DIM) if s.wide_a or s.cache_a else 1
            b_base = 2 * F.SPAD_BANK_ROWS // F.DIM if s.banked_m or s.separate_b_bank else (2 if s.pipeline_m else 1) * s.bm * a_panel_tiles
            for ki in range(_ceil_div(s.k, F.DIM)):
                kr = min(F.DIM, s.k - ki * F.DIM)
                for d in range(0, nt, 4 if s.wide_b else 1):
                    cols = min(4*F.DIM if s.wide_b else F.DIM, s.n-d*F.DIM)
                    ptr = self._ptr(self.b, self.fb.const(ki * F.DIM),
                                    s.n, self.fb.const(d * F.DIM))
                    self._rocc("mvin", {
                        "local": (b_base + ki * nt + d) * F.DIM,
                        "rows": kr, "cols": cols, "load_id": 1}, ptr)

        def m_body(m0: SSAValue, mr: tuple[int, ...], slot: int = 0) -> None:
            def n_body(n0: SSAValue, nr: tuple[int, ...]) -> None:
                self._output_block(m0, n0, mr, nr, slot)
            self._for_groups(_groups(s.n, s.bn), n_body)

        if s.prefetch_m:
            nr = tuple(min(F.DIM, s.n - d * F.DIM)
                       for d in range(_ceil_div(s.n, F.DIM)))
            n0 = self.fb.const(0)
            for start, stop, step, widths in _groups(s.m, s.bm):
                count = (stop - start) // step
                self._prepare_output_block(self.fb.const(start), n0, widths, nr, 0)
                def block(iv, slot, next_slot=None):
                    callback = None
                    if next_slot is not None:
                        def callback():
                            self._prepare_output_block(
                                self.fb.add_i(iv, self.fb.const(step)), n0,
                                widths, nr, next_slot)
                    self._output_block(iv, n0, widths, nr, slot,
                                       prepared=True, prefetch=callback)
                pairs = count // 2
                if pairs > 1:
                    def pair(iv):
                        block(iv, 0, 1)
                        block(self.fb.add_i(iv, self.fb.const(step)), 1, 0)
                    self.fb.for_loop(start, start + (pairs - 1) * 2 * step,
                                     2 * step, pair)
                if pairs:
                    last = start + (pairs - 1) * 2 * step
                    block(self.fb.const(last), 0, 1)
                    block(self.fb.const(last + step), 1, 0 if count % 2 else None)
                if count % 2:
                    block(self.fb.const(start + pairs * 2 * step), 0)
        elif s.pipeline_m:
            for start, stop, step, widths in _groups(s.m, s.bm):
                count = (stop - start) // step
                pairs = count // 2
                if pairs:
                    def pair_body(iv, w=widths):
                        m_body(iv, w, 0)
                        m_body(self.fb.add_i(iv, self.fb.const(step)), w, 1)
                    self.fb.for_loop(start, start + pairs * 2 * step,
                                     2 * step, pair_body)
                if count % 2:
                    m_body(self.fb.const(start + pairs * 2 * step), widths, 0)
        else:
            self._for_groups(_groups(s.m, s.bm), m_body)

    def _finish(self, symbol: str, batch: int = 1) -> ModuleOp:
        s = self.shape
        self._rocc("fence", {})
        fn = llvm.FuncOp(symbol, llvm.LLVMFunctionType([arg.type for arg in self.fb.entry.args]),
                         linkage=llvm.LinkageAttr("external"), body=self.fb.finish())
        module = ModuleOp([fn])
        module.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
        module.attributes["gemmini.golden_shape"] = StringAttr(
            f"{s.m}x{s.k}x{s.n}:{s.output_dtype}:bm{s.bm}:bn{s.bn}:"
            f"bias{int(s.bias)}:scale{s.scale}:relu{int(s.relu)}:"
            f"wide_store{int(s.wide_store)}:reuse_b{int(s.reuse_b)}:"
            f"cache_b{int(s.cache_b)}:pipeline_m{int(s.pipeline_m)}:"
            f"cache_a{int(s.cache_a)}:prefetch_m{int(s.prefetch_m)}:banked_m{int(s.banked_m)}:"
            f"wide_a{int(s.wide_a)}:wide_b{int(s.wide_b)}")
        if s.separate_b_bank:
            module.attributes["gemmini.separate_b_bank"] = IntegerAttr(1, i64)
        if s.prefetch_b:
            module.attributes["gemmini.prefetch_b"] = IntegerAttr(1, i64)
        if self.prefetch_b_rows is not None:
            module.attributes["gemmini.prefetch_b_rows"] = StringAttr(
                ",".join(str(row) for row in self.prefetch_b_rows))
        if self.resident_a_load_tiles != 1:
            module.attributes["gemmini.resident_a_load_tiles"] = IntegerAttr(self.resident_a_load_tiles,i64)
        if self.input_view is not None:
            module.attributes["gemmini.segmented_input_contract"] = StringAttr(
                json.dumps(self.input_view.__dict__,sort_keys=True))
        if self.stationary_b_tail_before_last_full:
            module.attributes["gemmini.stationary_b_tail_before_last_full"] = IntegerAttr(1, i64)
        module.attributes["gemmini.golden_batch"] = IntegerAttr(batch, i64)
        module.verify()
        return module

    def build(self) -> ModuleOp:
        self._emit_config()
        self._emit_work()
        return self._finish("gemmini_golden_gemm")

    def build_batched(self, batch: int) -> ModuleOp:
        """Configure once, then repeat the primitive tile schedule per batch."""
        if batch <= 0 or self.shape.bias:
            raise ValueError("batch must be positive and batched bias is not supported")
        if self.input_view is not None:
            raise ValueError("segmented input batch storage requires a separate ABI contract")
        s = self.shape
        base_a, base_b, base_c = self.a, self.b, self.c
        self._emit_config()

        def offset(base: SSAValue, iv: SSAValue, elements: int, elem_bytes: int = 1) -> SSAValue:
            delta = self.fb.mul_i(iv, self.fb.const(elements * elem_bytes))
            return self.fb.add(llvm.GEPOp(base, [llvm.GEP_USE_SSA_VAL], i8,
                                          ssa_indices=[delta])).results[0]

        def body(iv: SSAValue) -> None:
            self.a = offset(base_a, iv, s.m * s.k)
            self.b = offset(base_b, iv, s.k * s.n)
            self.c = offset(base_c, iv, s.m * s.n,
                            4 if s.output_dtype == "i32" else 1)
            self._emit_work()

        self.fb.for_loop(0, batch, 1, body)
        return self._finish("gemmini_golden_batched_gemm", batch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("m", "n", "k"):
        ap.add_argument("--" + name, type=int, required=True)
    ap.add_argument("--output-dtype", choices=("i8", "i32"), default="i8")
    ap.add_argument("--bm", type=int, default=4)
    ap.add_argument("--bn", type=int, default=4)
    ap.add_argument("--tune", action="store_true",
                    help="choose a legal block by the analytical command/traffic model")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--wide-store", action="store_true",
                    help="commit up to four adjacent i8 accumulator tiles per MVOUT")
    ap.add_argument("--reuse-b", action="store_true",
                    help="reuse stationary B across output-row tiles")
    ap.add_argument("--cache-b", action="store_true",
                    help="preload a small whole B matrix once into scratchpad")
    ap.add_argument("--cache-a", action="store_true",
                    help="preload one short output-row A tile across N blocks")
    ap.add_argument("--pipeline-m", action="store_true",
                    help="alternate A and accumulator slots across output blocks")
    ap.add_argument("--prefetch-m", action="store_true")
    ap.add_argument("--banked-m", action="store_true")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    ap.add_argument("--separate-b-bank", action="store_true")
    ap.add_argument("--emit", choices=("target", "llvm"), required=True)
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    shape = Shape(args.m, args.n, args.k, output_dtype=args.output_dtype,
                  bm=args.bm, bn=args.bn, bias=args.bias, scale=args.scale,
                  relu=args.relu, wide_store=args.wide_store,
                  reuse_b=args.reuse_b, cache_b=args.cache_b,
                  cache_a=args.cache_a,
                  pipeline_m=args.pipeline_m, prefetch_m=args.prefetch_m, banked_m=args.banked_m, wide_a=args.wide_a,
                  wide_b=args.wide_b, separate_b_bank=args.separate_b_bank)
    if args.tune:
        if (args.bm, args.bn) != (4, 4):
            ap.error("--tune cannot be combined with manual --bm/--bn")
        from .golden_tuning import tune
        shape, _ = tune(shape)
    module = GoldenGemm(shape).build()
    if args.emit == "llvm":
        from .golden_device_lower import lower
        module = lower(module)
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    out = stream.getvalue() + "\n"
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(out)
    else:
        print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
