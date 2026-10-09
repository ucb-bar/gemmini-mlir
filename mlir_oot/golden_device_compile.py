"""Reproducible xDSL Gemmini device compilation for the golden kernels.

The target dialect is the source artifact.  Compilation verifies it, lowers
primitive commands to LLVM/RoCC, translates to LLVM IR, and assembles a RISC-V
object.  A linked program must be audited again after runtime libraries enter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .golden_device_lower import lower
from .golden_gemm import GoldenGemm, Shape
from .golden_resadd import build as build_resadd
from .no_fsm_audit import audit_elf


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_module(module, llvm_bin: Path, workdir: Path) -> dict:
    """Compile one verified target module and return hashes plus object audit."""
    workdir.mkdir(parents=True, exist_ok=True)
    target_ir = workdir / "kernel.gemmini.mlir"
    llvm_mlir = workdir / "kernel.llvm.mlir"
    llvm_ir = workdir / "kernel.ll"
    obj = workdir / "kernel.o"
    target_ir.write_text(str(module) + "\n")
    llvm_mlir.write_text(str(lower(module)) + "\n")
    llvm_bin = llvm_bin.resolve(strict=True)
    commands = [
        [str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir),
         "-o", str(llvm_ir)],
        [str(llvm_bin / "clang"), "--target=riscv64-unknown-elf",
         "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-c", str(llvm_ir),
         "-o", str(obj)],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    object_audit = audit_elf(obj.read_bytes())
    (workdir / "object_nofsm_audit.json").write_text(
        json.dumps(object_audit, indent=2) + "\n")
    if object_audit["status"] != "pass":
        raise ValueError("device object contains a forbidden Gemmini command")
    receipt = {
        "schema": "gemmini_golden_device_compile_v1",
        "target_ir_sha256": _sha(target_ir),
        "llvm_mlir_sha256": _sha(llvm_mlir),
        "llvm_ir_sha256": _sha(llvm_ir),
        "object_sha256": _sha(obj),
        "object_nofsm_status": object_audit["status"],
        "compiler_sha256": {
            "mlir-translate": _sha(llvm_bin / "mlir-translate"),
            "clang": _sha(llvm_bin / "clang"),
        },
        "compiler_argv": commands,
    }
    (workdir / "device_compile.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", choices=("gemm", "resadd"), required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int)
    ap.add_argument("--output-dtype", choices=("i8", "i32"), default="i8")
    ap.add_argument("--bm", type=int, default=4)
    ap.add_argument("--bn", type=int, default=4)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--wide-store", action="store_true")
    ap.add_argument("--reuse-b", action="store_true")
    ap.add_argument("--cache-b", action="store_true")
    ap.add_argument("--cache-a", action="store_true")
    ap.add_argument("--pipeline-m", action="store_true")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    if args.kernel == "gemm":
        if args.k is None:
            ap.error("gemm requires --k")
        module = GoldenGemm(Shape(
            args.m, args.n, args.k, output_dtype=args.output_dtype,
            bm=args.bm, bn=args.bn, bias=args.bias, scale=args.scale,
            relu=args.relu, wide_store=args.wide_store, reuse_b=args.reuse_b,
            cache_b=args.cache_b, cache_a=args.cache_a,
            pipeline_m=args.pipeline_m,
            wide_a=args.wide_a, wide_b=args.wide_b)).build()
    else:
        module = build_resadd(args.m, args.n, output_scale=args.scale,
                              relu=args.relu)
    print(json.dumps(compile_module(module, args.llvm_bin, args.workdir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
