"""Batched dense GEMM with one configuration and primitive Gemmini tile loops.

ABI: void gemmini_golden_batched_gemm(i8 *A, i8 *B, <i8|i32> *C).
Tensors are dense [batch,M,K], [batch,K,N], [batch,M,N].  The repeated batch
body uses ordinary CPU branches, with no Gemmini hardware LOOP instruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .golden_device_compile import compile_module
from .golden_gemm import GoldenGemm, Shape


def build(batch: int, shape: Shape):
    return GoldenGemm(shape).build_batched(batch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "m", "n", "k"):
        ap.add_argument("--" + name, type=int, required=True)
    ap.add_argument("--output-dtype", choices=("i8", "i32"), default="i32")
    ap.add_argument("--bm", type=int, default=4)
    ap.add_argument("--bn", type=int, default=4)
    ap.add_argument("--reuse-b", action="store_true")
    ap.add_argument("--cache-b", action="store_true")
    ap.add_argument("--cache-a", action="store_true")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    shape = Shape(args.m, args.n, args.k, output_dtype=args.output_dtype,
                  bm=args.bm, bn=args.bn, reuse_b=args.reuse_b,
                  cache_b=args.cache_b, cache_a=args.cache_a,
                  wide_a=args.wide_a,
                  wide_b=args.wide_b)
    print(json.dumps(compile_module(build(args.batch, shape),
                                    args.llvm_bin, args.workdir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
