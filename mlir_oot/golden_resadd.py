"""Stock-Gemmini residual add as an identity matmul with the skip on D.

This is the large q1013 residual improvement: A times a resident identity plus
B as the preloaded D operand, with scale/ReLU on Gemmini's load/store paths.
It requires stock FireSimGemminiRocketConfig, whose WS dataflow accepts a real D
address.  LeanGemminiRocketConfig forces D to garbage and cannot use this form.
No hardware-loop command is emitted.

ABI: void gemmini_golden_resadd(i8 *A, i8 *B, i8 *C,
                                i8 *identity16x16, i8 *scratch1024)
The 1024-byte, 64-byte-aligned scratch stages partial output tiles before an
ordinary CPU copy into dense C.  A partial DMA may touch bytes beyond the
logical tensor when aimed directly at C.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.printer import Printer

from .codegen.builder import FnBuilder, PTR
from .golden_gemm import GoldenGemm, Shape, _groups
from .tables import isa, rtl_facts as F


def build(m: int, n: int, *, lhs_scale: float = 1.0,
          rhs_scale: float = 1.0, output_scale: float = 1.0,
          relu: bool = False) -> ModuleOp:
    if min(m, n) <= 0 or any(not math.isfinite(s) or s <= 0 for s in
                             (lhs_scale, rhs_scale, output_scale)):
        raise ValueError("extents and scales must be positive and finite")
    e = GoldenGemm(Shape(m, n, F.DIM, bias=True, bn=4, wide_store=True))
    e.fb = FnBuilder([PTR] * 5)
    e.a, e.b, e.c, identity, scratch = e.fb.entry.args
    e._rocc("fence", {})
    e._rocc("flush", {})
    e._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY})
    e._rocc("config_ld", {"stride": n, "scale": lhs_scale, "load_id": 0})
    e._rocc("config_ld", {"stride": n, "scale": rhs_scale, "load_id": 1})
    e._rocc("config_ld", {"stride": F.DIM, "load_id": 2})
    e._rocc("config_st", {"stride": n,
                           "acc_act": isa.RELU if relu else isa.NO_ACTIVATION,
                           "acc_scale": output_scale})
    e._rocc("mvin", {"local": 32, "rows": F.DIM, "cols": F.DIM,
                     "load_id": 2}, identity)

    def m_body(m0, mr):
        rows = mr[0]
        mrow = e._tile(m0, 0)

        def n_body(n0, nr):
            for d, cols in enumerate(nr):
                ncol = e._tile(n0, d)
                a_ptr = e._ptr(e.a, mrow, n, ncol)
                b_ptr = e._ptr(e.b, mrow, n, ncol)
                e._rocc("mvin", {"local": 0, "rows": rows, "cols": cols,
                                 "load_id": 0}, a_ptr)
                e._rocc("mvin", {"local": 16, "rows": rows, "cols": cols,
                                 "load_id": 1}, b_ptr)
                e._rocc("preload", {"bd": 32, "c": isa.acc_addr(d * F.DIM),
                                    "bd_cols": F.DIM, "bd_rows": F.DIM,
                                    "c_cols": cols, "c_rows": rows})
                e._rocc("compute", {"a": 0, "bd": 16,
                                    "a_cols": cols, "a_rows": rows,
                                    "bd_cols": cols, "bd_rows": rows})
            d = 0
            while d < len(nr):
                width = 1
                if nr[d] == F.DIM:
                    while (width < 4 and d + width < len(nr) and
                           nr[d + width] == F.DIM):
                        width += 1
                ncol = e._tile(n0, d)
                cols = sum(nr[d:d + width])
                # DMA writes occur in bus bursts, so a dense row pitch below
                # 64 bytes can spill into the next allocation even for a full
                # tile.  ResNet's channel pitches are multiples of 64 and use
                # the direct path; other pitches stage every store.
                partial = (rows != F.DIM or cols != width * F.DIM or n % 64 != 0)
                if partial:
                    e._rocc("fence", {})
                    e._rocc("config_st", {"stride": 64,
                                           "acc_act": isa.RELU if relu else isa.NO_ACTIVATION,
                                           "acc_scale": output_scale})
                    c_ptr = scratch
                else:
                    c_ptr = e._ptr(e.c, mrow, n, ncol)
                e._rocc("mvout", {"local": isa.acc_addr(d * F.DIM),
                                  "rows": rows, "cols": cols},
                        c_ptr)
                if partial:
                    e._rocc("fence", {})

                    def copy_row(r):
                        def copy_col(c):
                            src = e.fb.add_i(e.fb.mul_i(r, e.fb.const(64)), c)
                            dst_row = e.fb.add_i(mrow, r)
                            dst_col = e.fb.add_i(ncol, c)
                            dst = e.fb.add_i(e.fb.mul_i(dst_row, e.fb.const(n)), dst_col)
                            val = e.fb.load_i64(scratch, src, "i8")
                            e.fb.store_i64(val, e.c, dst, "i8")
                        e.fb.for_loop(0, cols, 1, copy_col)

                    e.fb.for_loop(0, rows, 1, copy_row)
                    e._rocc("config_st", {"stride": n,
                                           "acc_act": isa.RELU if relu else isa.NO_ACTIVATION,
                                           "acc_scale": output_scale})
                d += width

        e._for_groups(_groups(n, 4), n_body)

    e._for_groups(_groups(m, 1), m_body)
    e._rocc("fence", {})
    fn = llvm.FuncOp("gemmini_golden_resadd", llvm.LLVMFunctionType([PTR] * 5),
                     linkage=llvm.LinkageAttr("external"), body=e.fb.finish())
    module = ModuleOp([fn])
    module.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
    module.attributes["gemmini.golden_resadd"] = StringAttr(
        f"{m}x{n}:lhs{lhs_scale}:rhs{rhs_scale}:out{output_scale}:relu{int(relu)}")
    module.verify()
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lhs-scale", type=float, default=1.0)
    ap.add_argument("--rhs-scale", type=float, default=1.0)
    ap.add_argument("--output-scale", type=float, default=1.0)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--emit", choices=("target", "llvm"), required=True)
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()
    op = build(args.m, args.n,
               lhs_scale=args.lhs_scale, rhs_scale=args.rhs_scale,
               output_scale=args.output_scale, relu=args.relu)
    if args.emit == "llvm":
        from .golden_device_lower import lower
        op = lower(op)
    stream = io.StringIO()
    Printer(stream=stream).print_op(op)
    result = stream.getvalue() + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
