#!/usr/bin/env python3
"""Compile, link, audit and numerically test one batched Gemmini kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from merlin.perf.layer_bench import build_program, run_on_gsim

from mlir_oot.golden_batched_gemm import build
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import Shape
from mlir_oot.no_fsm_audit import audit_elf


def _write_expected(path: Path, batch: int, m: int, n: int, k: int) -> None:
    lines = ["static const int32_t expected_values[BATCH][M][N] = {"]
    for q in range(batch):
        lines.append("  {")
        for i in range(m):
            values = []
            for j in range(n):
                values.append(str(sum((((q * 3 + i * 7 + p * 2) % 11) - 5) *
                                      (((q * 5 + p * 3 + j * 2) % 13) - 6)
                                      for p in range(k))))
            lines.append("    {" + ", ".join(values) + "},")
        lines.append("  },")
    lines.append("};")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "m", "n", "k"):
        ap.add_argument("--" + name, type=int, required=True)
    ap.add_argument("--bm", type=int, default=2)
    ap.add_argument("--bn", type=int, default=2)
    ap.add_argument("--reuse-b", action="store_true")
    ap.add_argument("--cache-b", action="store_true")
    ap.add_argument("--cache-a", action="store_true")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--prebuilt-object", type=Path)
    ap.add_argument("--kernel-symbol", default="gemmini_golden_batched_gemm")
    ap.add_argument("--embed-expected", action="store_true")
    ap.add_argument("--max-cycles", type=int, default=3_000_000)
    ap.add_argument("--timeout-s", type=int, default=180)
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.kernel_symbol):
        ap.error("--kernel-symbol must be a C identifier")
    wd = args.workdir.resolve()
    wd.mkdir(parents=True, exist_ok=False)
    shape = Shape(args.m, args.n, args.k, output_dtype="i32",
                  bm=args.bm, bn=args.bn, reuse_b=args.reuse_b,
                  cache_b=args.cache_b, cache_a=args.cache_a,
                  wide_a=args.wide_a,
                  wide_b=args.wide_b)
    if args.prebuilt_object:
        obj = args.prebuilt_object.resolve(strict=True)
        compilation = json.loads((obj.parent / "device_compile.json").read_text())
        if hashlib.sha256(obj.read_bytes()).hexdigest() != compilation["object_sha256"]:
            raise ValueError("prebuilt device object disagrees with its compilation receipt")
        binding = obj.parent / "upstream_contraction_binding.json"
        if binding.exists():
            record = json.loads(binding.read_text())
            if (record["dimensions"] != {"batch": args.batch, "m": args.m,
                                          "n": args.n, "k": args.k} or
                    record["schedule"] != vars(shape) or
                    record["binding"]["abi"] != "gemmini_golden_batched_gemm"):
                raise ValueError("probe geometry disagrees with upstream compiled object")
        catalog = obj.parent / "device_catalog.json"
        if catalog.exists():
            record = json.loads(catalog.read_text())
            kernel = next((x for x in record["kernels"]
                           if x["symbol"] == args.kernel_symbol), None)
            if (kernel is None or kernel["dimensions"] !=
                    {"batch": args.batch, "m": args.m, "n": args.n, "k": args.k} or
                    kernel["schedule"] != vars(shape) or not kernel["batched"]):
                raise ValueError("probe symbol or geometry disagrees with device catalog")
    else:
        obj = wd / "kernel.o"
        compilation = compile_module(build(args.batch, shape), args.llvm_bin, wd)
    source = Path(__file__).with_name("batched_gemm_probe.c")
    cflags = [f"-DBATCH={args.batch}", f"-DM={args.m}",
              f"-DN={args.n}", f"-DK={args.k}",
              f"-DKERNEL_SYMBOL={args.kernel_symbol}"]
    if args.embed_expected:
        _write_expected(wd / "batched_expected.inc", args.batch, args.m, args.n, args.k)
        cflags += ["-DEMBED_EXPECTED", f"-I{wd}"]
    program = build_program([source, obj], wd, target="gemmini",
                            extra_cflags=cflags,
                            max_loaded_bytes=None)
    audit = audit_elf(program.elf.read_bytes())
    (wd / "nofsm_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("linked ELF fails the zero-FSM audit")
    run = run_on_gsim(program.elf, target="gemmini", max_cycles=args.max_cycles,
                      timeout_s=args.timeout_s, backdoor=True,
                      stdout_path=wd / "gsim.stdout")
    match = re.search(r"^GOLDEN_BATCHED_CYCLES (\d+)$", run.stdout_tail, re.MULTILINE)
    passed = (run.completed and run.returncode == 0 and match is not None and
              f"GOLDEN_BATCHED PASS B={args.batch} M={args.m} N={args.n} K={args.k}"
              in run.stdout_tail)
    result = {"status": "pass" if passed else "fail",
              "shape": [args.batch, args.m, args.n, args.k],
              "kernel_symbol": args.kernel_symbol,
              "kernel_cycles": int(match.group(1)) if match else None,
              "elf_sha256": program.elf_sha256,
              "object_sha256": compilation["object_sha256"],
              "gsim_engine_sha256": run.engine.get("binary_sha256"),
              "stdout_tail": run.stdout_tail,
              "stderr_tail": run.stderr_tail if not passed else ""}
    (wd / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
