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
"""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import Float32Type, FloatAttr, IntegerAttr, ModuleOp, StringAttr, i8, i64
from xdsl.ir import SSAValue
from xdsl.printer import Printer

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
    pipeline_m: bool = False
    wide_a: bool = False
    wide_b: bool = False

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.bm, self.bn) <= 0:
            raise ValueError("all extents and block sizes must be positive")
        if self.output_dtype not in ("i8", "i32"):
            raise ValueError("output dtype must be i8 or i32")
        if self.scale <= 0 or not math.isfinite(self.scale):
            raise ValueError("store scale must be finite and positive")
        if self.output_dtype == "i32" and (self.scale != 1.0 or self.relu):
            raise ValueError("i32 readout does not apply scale or ReLU")
        if self.wide_a and (self.k <= F.DIM or self.k > F.DIM * 4 or
                            self.k % F.DIM):
            raise ValueError("wide A load needs K in {32,48,64}")
        if self.wide_b and (self.n <= F.DIM or self.n > F.DIM * 4 or
                            self.n % F.DIM or _ceil_div(self.n, F.DIM) > self.bn):
            raise ValueError("wide B load needs one N block of 32, 48 or 64")
        slots = 2 if self.pipeline_m else 1
        a_panel_tiles = _ceil_div(self.k, F.DIM) if self.wide_a else 1
        a_rows = slots * self.bm * a_panel_tiles
        if slots * self.bm * self.bn * F.DIM > F.ACC_ROWS:
            raise ValueError("output block exceeds the accumulator")
        if (a_rows + self.bn) * F.DIM > F.SPAD_ROWS:
            raise ValueError("operand block exceeds the scratchpad")
        if self.cache_b:
            nt, kt = _ceil_div(self.n, F.DIM), _ceil_div(self.k, F.DIM)
            if nt > self.bn:
                raise ValueError("cached B needs one output-channel block")
            if kt * nt > 128 or (a_rows + kt * nt) * F.DIM > F.SPAD_ROWS:
                raise ValueError("cached B panels exceed the static or scratchpad budget")
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
    def __init__(self, shape: Shape):
        shape.validate()
        self.shape = shape
        self.fb = FnBuilder([PTR] * (4 if shape.bias else 3))
        self.a, self.b, self.c = self.fb.entry.args[:3]
        self.bias = self.fb.entry.args[3] if shape.bias else None

    def _rocc(self, kind: str, attrs: dict, pointer: SSAValue | None = None) -> None:
        cls = {
            "flush": G.FlushOp, "config_ex": G.ConfigExOp,
            "config_ld": G.ConfigLdOp, "config_st": G.ConfigStOp,
            "mvin": G.MvinOp, "mvout": G.MvoutOp,
            "preload": G.PreloadOp, "compute": G.ComputeOp,
            "fence": G.FenceOp,
        }[kind]
        op = cls(operands=[[pointer] if pointer is not None else []], result_types=[[]])
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

    def _k_tile(self, m0: SSAValue, n0: SSAValue, k0: SSAValue,
                mr: tuple[int, ...], nr: tuple[int, ...], kr: int,
                first: bool, cached_k: int | None = None, slot: int = 0,
                wide_k: int | None = None) -> None:
        s = self.shape
        kt = _ceil_div(s.k, F.DIM)
        a_panel_tiles = kt if s.wide_a else 1
        a_base = slot * s.bm * a_panel_tiles if s.pipeline_m else 0
        b_base = (2 if s.pipeline_m else 1) * s.bm * a_panel_tiles
        acc_base = slot * s.bm * s.bn if s.pipeline_m else 0
        krow = self._tile(k0, 0)
        if not s.wide_a:
            for a, rows in enumerate(mr):
                mrow = self._tile(m0, a)
                ptr = self._ptr(self.a, mrow, s.k, krow)
                self._rocc("mvin", {"local": (a_base + a) * F.DIM, "rows": rows,
                                      "cols": kr, "load_id": 0}, ptr)
        if cached_k is None:
            if s.wide_b:
                ptr = self._ptr(self.b, krow, s.n, self.fb.const(0))
                self._rocc("mvin", {"local": b_base * F.DIM,
                                      "rows": kr, "cols": s.n, "load_id": 1}, ptr)
            else:
                for d, cols in enumerate(nr):
                    ncol = self._tile(n0, d)
                    ptr = self._ptr(self.b, krow, s.n, ncol)
                    self._rocc("mvin", {"local": (b_base + d) * F.DIM,
                                          "rows": kr, "cols": cols, "load_id": 1}, ptr)
        def b_addr(d: int) -> int:
            if cached_k is None:
                return (b_base + d) * F.DIM
            return (b_base + cached_k * _ceil_div(s.n, F.DIM) + d) * F.DIM
        def a_addr(a: int) -> int:
            return (a_base + a * a_panel_tiles + (wide_k or 0)) * F.DIM
        if s.reuse_b:
            # Keep a weight tile in the array while varying the A row tile.
            # A garbage BD address on subsequent PRELOADs preserves the
            # stationary operand; COMPUTE_ACCUMULATE is the matching opcode.
            for d, cols in enumerate(nr):
                for a, rows in enumerate(mr):
                    acc_row = (acc_base + a * s.bn + d) * F.DIM
                    self._rocc("preload", {
                        "bd": b_addr(d) if a == 0 else isa.GARBAGE_ADDR,
                        "c": isa.acc_addr(acc_row, accumulate=not first),
                        "bd_cols": cols, "bd_rows": kr,
                        "c_cols": cols, "c_rows": rows})
                    self._rocc("compute", {"a": a_addr(a),
                                            "a_cols": kr, "a_rows": rows,
                                            "accumulate": a != 0})
        else:
            for a, rows in enumerate(mr):
                for d, cols in enumerate(nr):
                    acc_row = (acc_base + a * s.bn + d) * F.DIM
                    self._rocc("preload", {
                        "bd": b_addr(d),
                        "c": isa.acc_addr(acc_row, accumulate=not first),
                        "bd_cols": cols, "bd_rows": kr,
                        "c_cols": cols, "c_rows": rows})
                    self._rocc("compute", {"a": a_addr(a),
                                            "a_cols": kr, "a_rows": rows})

    def _output_block(self, m0: SSAValue, n0: SSAValue,
                      mr: tuple[int, ...], nr: tuple[int, ...],
                      slot: int = 0) -> None:
        s = self.shape
        acc_base = slot * s.bm * s.bn if s.pipeline_m else 0
        if s.wide_a:
            kt = _ceil_div(s.k, F.DIM)
            a_base = slot * s.bm * kt if s.pipeline_m else 0
            for a, rows in enumerate(mr):
                mrow = self._tile(m0, a)
                ptr = self._ptr(self.a, mrow, s.k, self.fb.const(0))
                self._rocc("mvin", {"local": (a_base + a * kt) * F.DIM,
                                      "rows": rows, "cols": s.k,
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
        if s.cache_b or s.wide_a:
            for ki in range(_ceil_div(s.k, F.DIM)):
                self._k_tile(m0, n0, self.fb.const(ki), mr, nr,
                             min(F.DIM, s.k - ki * F.DIM),
                             self.bias is None and ki == 0,
                             ki if s.cache_b else None, slot,
                             ki if s.wide_a else None)
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

    def _for_groups(self, groups, body) -> None:
        for start, stop, step, widths in groups:
            if stop - start == step:
                body(self.fb.const(start), widths)
            else:
                self.fb.for_loop(start, stop, step,
                                 lambda iv, w=widths: body(iv, w))

    def build(self) -> ModuleOp:
        s = self.shape
        self._rocc("fence", {})
        self._rocc("flush", {})
        self._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY})
        self._rocc("config_ld", {"stride": s.k, "load_id": 0})
        self._rocc("config_ld", {"stride": s.n, "load_id": 1})
        if self.bias is not None:
            self._rocc("config_ld", {"stride": 0, "load_id": 2})
        self._rocc("config_st", {"stride": s.n * (4 if s.output_dtype == "i32" else 1),
                                  "acc_act": isa.RELU if s.relu else isa.NO_ACTIVATION,
                                  "acc_scale": s.scale})
        if s.cache_b:
            nt = _ceil_div(s.n, F.DIM)
            a_panel_tiles = _ceil_div(s.k, F.DIM) if s.wide_a else 1
            b_base = (2 if s.pipeline_m else 1) * s.bm * a_panel_tiles
            for ki in range(_ceil_div(s.k, F.DIM)):
                kr = min(F.DIM, s.k - ki * F.DIM)
                for d in range(1 if s.wide_b else nt):
                    cols = s.n if s.wide_b else min(F.DIM, s.n - d * F.DIM)
                    ptr = self._ptr(self.b, self.fb.const(ki * F.DIM),
                                    s.n, self.fb.const(d * F.DIM))
                    self._rocc("mvin", {
                        "local": (b_base + ki * nt + d) * F.DIM,
                        "rows": kr, "cols": cols, "load_id": 1}, ptr)

        def m_body(m0: SSAValue, mr: tuple[int, ...], slot: int = 0) -> None:
            def n_body(n0: SSAValue, nr: tuple[int, ...]) -> None:
                self._output_block(m0, n0, mr, nr, slot)
            self._for_groups(_groups(s.n, s.bn), n_body)

        if s.pipeline_m:
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
        self._rocc("fence", {})
        fn = llvm.FuncOp("gemmini_golden_gemm", llvm.LLVMFunctionType([PTR] * (4 if s.bias else 3)),
                         linkage=llvm.LinkageAttr("external"), body=self.fb.finish())
        module = ModuleOp([fn])
        module.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
        module.attributes["gemmini.golden_shape"] = StringAttr(
            f"{s.m}x{s.k}x{s.n}:{s.output_dtype}:bm{s.bm}:bn{s.bn}:"
            f"bias{int(s.bias)}:scale{s.scale}:relu{int(s.relu)}:"
            f"wide_store{int(s.wide_store)}:reuse_b{int(s.reuse_b)}:"
            f"cache_b{int(s.cache_b)}:pipeline_m{int(s.pipeline_m)}:"
            f"wide_a{int(s.wide_a)}:wide_b{int(s.wide_b)}")
        module.verify()
        return module


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
    ap.add_argument("--pipeline-m", action="store_true",
                    help="alternate A and accumulator slots across output blocks")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    ap.add_argument("--emit", choices=("target", "llvm"), required=True)
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    shape = Shape(args.m, args.n, args.k, output_dtype=args.output_dtype,
                  bm=args.bm, bn=args.bn, bias=args.bias, scale=args.scale,
                  relu=args.relu, wide_store=args.wide_store,
                  reuse_b=args.reuse_b, cache_b=args.cache_b,
                  pipeline_m=args.pipeline_m, wide_a=args.wide_a,
                  wide_b=args.wide_b)
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
