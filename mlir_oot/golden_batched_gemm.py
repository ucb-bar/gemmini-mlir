"""Batched dense GEMM with ordinary CPU batch repetition and primitive Gemmini ops.

ABI: void gemmini_golden_batched_gemm(i8 *A, i8 *B, <i8|i32> *C).
The tensors are dense [batch,M,K], [batch,K,N], [batch,M,N].  Every batch calls
the same xDSL-generated Gemmini kernel; no hardware LOOP instruction is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i8, i64

from .codegen.builder import FnBuilder, PTR
from .golden_device_compile import compile_module
from .golden_gemm import GoldenGemm, Shape


def build(batch: int, shape: Shape):
    if batch <= 0 or shape.bias:
        raise ValueError("batch must be positive and batched bias is not supported")
    module = GoldenGemm(shape).build()
    fb = FnBuilder([PTR] * 3)
    a, b, c = fb.entry.args

    def offset(base, iv, elements: int, element_bytes: int = 1):
        off = fb.mul_i(iv, fb.const(elements * element_bytes))
        return fb.add(llvm.GEPOp(base, [llvm.GEP_USE_SSA_VAL], i8,
                                 ssa_indices=[off])).results[0]

    def body(iv):
        ap = offset(a, iv, shape.m * shape.k)
        bp = offset(b, iv, shape.k * shape.n)
        cp = offset(c, iv, shape.m * shape.n,
                    4 if shape.output_dtype == "i32" else 1)
        fb.add(llvm.CallOp("gemmini_golden_gemm", ap, bp, cp))

    fb.for_loop(0, batch, 1, body)
    fn = llvm.FuncOp("gemmini_golden_batched_gemm",
                     llvm.LLVMFunctionType([PTR] * 3),
                     linkage=llvm.LinkageAttr("external"), body=fb.finish())
    module.body.blocks[0].add_op(fn)
    module.attributes["gemmini.golden_batch"] = IntegerAttr(batch, i64)
    module.attributes["gemmini.golden_batch_shape"] = StringAttr(
        json.dumps(asdict(shape), sort_keys=True))
    module.verify()
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "m", "n", "k"):
        ap.add_argument("--" + name, type=int, required=True)
    ap.add_argument("--output-dtype", choices=("i8", "i32"), default="i32")
    ap.add_argument("--bm", type=int, default=4)
    ap.add_argument("--bn", type=int, default=4)
    ap.add_argument("--reuse-b", action="store_true")
    ap.add_argument("--cache-b", action="store_true")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    shape = Shape(args.m, args.n, args.k, output_dtype=args.output_dtype,
                  bm=args.bm, bn=args.bn, reuse_b=args.reuse_b,
                  cache_b=args.cache_b, wide_a=args.wide_a,
                  wide_b=args.wide_b)
    print(json.dumps(compile_module(build(args.batch, shape),
                                    args.llvm_bin, args.workdir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
