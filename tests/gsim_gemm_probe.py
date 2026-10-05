#!/usr/bin/env python3
"""Build, audit and numerically check one xDSL golden GEMM on pinned Gemmini GSIM.

Run from a Merlin checkout with its Gemmini target selected in MERLIN_TARGET_PATH.
The user's environment supplies the target's toolchain and GSIM pin.  This probe
checks arithmetic and ELF legality; FireSim model timing needs a separate workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path
from dataclasses import asdict

from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_tuning import tune
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program, run_on_gsim


def _write_expected(path: Path, shape: Shape, amplitude: int = 1) -> None:
    """Move the expensive O(MNK) oracle off the cycle-accurate simulated CPU."""
    import numpy as np
    a = amplitude * ((np.arange(shape.m,dtype=np.int64)[:,None]*7 + np.arange(shape.k)*3)%11-5)
    b = amplitude * ((np.arange(shape.k,dtype=np.int64)[:,None]*5 + np.arange(shape.n)*2)%13-6)
    expected = a @ b
    if shape.bias:
        expected += np.arange(shape.n)%5-2
    if shape.output_dtype == 'i8':
        expected = np.rint(expected.astype(np.float32)*np.float32(shape.scale))
        expected = np.clip(expected,0 if shape.relu else -128,127).astype(np.int64)
    lines = ['static const int32_t expected_values[M][N] = {']
    lines.extend('    {' + ', '.join(str(x) for x in row) + '},' for row in expected)
    lines.append('};')
    path.write_text('\n'.join(lines)+'\n')


def _static_inputs(workdir, llvm_bin, shape, amplitude):
    """Embed fixture bytes using data-only assembly; no simulated setup loops."""
    import numpy as np
    a = amplitude * ((np.arange(shape.m,dtype=np.int64)[:,None]*7 + np.arange(shape.k)*3)%11-5)
    b = amplitude * ((np.arange(shape.k,dtype=np.int64)[:,None]*5 + np.arange(shape.n)*2)%13-6)
    text=[]
    for name,values in [('a',a),('b',b)]:
        path=workdir/(name+'.bin');values.astype(np.int8).tofile(path)
        text.append(f'.section .data\n.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n')
    asm=workdir/'input_data.S';asm.write_text(''.join(text));obj=workdir/'input_data.o'
    subprocess.run([str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-c',str(asm),'-o',str(obj)],check=True,capture_output=True)
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("m", "n", "k"):
        ap.add_argument("--" + name, type=int, required=True)
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--prebuilt-object", type=Path,
                    help="test an object compiled from an upstream capsule")
    ap.add_argument("--kernel-symbol", default="gemmini_golden_gemm")
    ap.add_argument("--output-dtype", choices=("i8", "i32"), default="i32")
    ap.add_argument("--bm", type=int, default=4)
    ap.add_argument("--bn", type=int, default=4)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--build-only", action="store_true",
                    help="build and audit for a separate hardware numeric run")
    ap.add_argument("--embed-expected", action="store_true",
                    help="compute oracle on the host and link expected values into the probe")
    ap.add_argument("--static-inputs", action="store_true")
    ap.add_argument("--wide-store", action="store_true")
    ap.add_argument("--reuse-b", action="store_true")
    ap.add_argument("--cache-b", action="store_true")
    ap.add_argument("--cache-a", action="store_true")
    ap.add_argument("--pipeline-m", action="store_true")
    ap.add_argument("--prefetch-m", action="store_true")
    ap.add_argument("--banked-m", action="store_true")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    ap.add_argument("--separate-b-bank", action="store_true")
    ap.add_argument("--prefetch-b", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--input-amplitude", type=int, default=1, choices=range(1,22))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=180)
    ap.add_argument("--max-cycles", type=int, default=3_000_000)
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.kernel_symbol):
        ap.error("--kernel-symbol must be a C identifier")

    shape = Shape(args.m, args.n, args.k, output_dtype=args.output_dtype,
                  bm=args.bm, bn=args.bn, bias=args.bias, scale=args.scale,
                  relu=args.relu, wide_store=args.wide_store,
                  reuse_b=args.reuse_b, cache_b=args.cache_b,
                  cache_a=args.cache_a,
                  pipeline_m=args.pipeline_m, prefetch_m=args.prefetch_m, banked_m=args.banked_m, wide_a=args.wide_a,
                  wide_b=args.wide_b, separate_b_bank=args.separate_b_bank,
                  prefetch_b=args.prefetch_b)
    if args.tune:
        if (args.bm, args.bn) != (4, 4):
            ap.error("--tune cannot be combined with manual --bm/--bn")
        shape, _ = tune(shape)
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    if args.prebuilt_object:
        obj = args.prebuilt_object.resolve(strict=True)
        compilation = json.loads((obj.parent / "device_compile.json").read_text())
        if hashlib.sha256(obj.read_bytes()).hexdigest() != compilation["object_sha256"]:
            raise ValueError("prebuilt device object disagrees with its compilation receipt")
        binding = obj.parent / "upstream_binding.json"
        if binding.exists() and json.loads(binding.read_text())["shape"] != asdict(shape):
            raise ValueError("probe shape disagrees with upstream compiled shape")
        catalog = obj.parent / "device_catalog.json"
        if catalog.exists():
            record = json.loads(catalog.read_text())
            kernel = next((x for x in record["kernels"]
                           if x["symbol"] == args.kernel_symbol), None)
            if (kernel is None or kernel["dimensions"] !=
                    {"batch": 1, "m": args.m, "n": args.n, "k": args.k} or
                    kernel["schedule"] != asdict(shape) or kernel["batched"]):
                raise ValueError("probe symbol or geometry disagrees with device catalog")
    else:
        obj = workdir / "kernel.o"
        compilation = compile_module(GoldenGemm(shape).build(), args.llvm_bin, workdir)

    source = Path(__file__).with_name("gemm_probe.c")
    cflags = [f"-DM={args.m}", f"-DN={args.n}", f"-DK={args.k}",
              f"-DKERNEL_SYMBOL={args.kernel_symbol}", f"-DINPUT_AMPLITUDE={args.input_amplitude}"]
    if args.output_dtype == "i8":
        cflags.append("-DOUT_I8")
    if args.bias:
        cflags.append("-DUSE_BIAS")
    if args.relu:
        cflags.append("-DUSE_RELU")
    if args.embed_expected:
        _write_expected(workdir / "gemm_expected.inc", shape, args.input_amplitude)
        cflags.extend(["-DEMBED_EXPECTED", f"-I{workdir}"])
    cflags.append(f"-DSCALE={args.scale}f")
    inputs=[]
    if args.static_inputs:
        inputs.append(_static_inputs(workdir,args.llvm_bin,shape,args.input_amplitude))
        cflags.append('-DSTATIC_INPUTS')
    cflags.append('-march=rv64gc')
    built = build_program([source, obj, *inputs], workdir, target="gemmini",
                          extra_cflags=cflags,
                          max_loaded_bytes=None)
    audit = audit_elf(built.elf.read_bytes())
    (workdir / "nofsm_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("final linked ELF contains a forbidden Gemmini instruction")

    if args.build_only:
        receipt = {"status": "built_not_executed", "shape": asdict(shape), "input_amplitude": args.input_amplitude,
                   "elf_sha256": built.elf_sha256, "nofsm_audit": audit}
        (workdir / "build_only.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 0

    run = run_on_gsim(built.elf, target="gemmini", max_cycles=args.max_cycles,
                      timeout_s=args.timeout_s, backdoor=True,
                      stdout_path=workdir / "gsim.stdout")
    match = re.search(r"^GOLDEN_GEMM_CYCLES (\d+)$", run.stdout_tail, re.MULTILINE)
    passed = (run.completed and run.returncode == 0 and
              f"GOLDEN_GEMM PASS M={args.m} N={args.n} K={args.k}" in run.stdout_tail and
              match is not None)
    result = {
        "schema": "gemmini_golden_gemm_gsim_probe_v1",
        "shape": [args.m, args.n, args.k],
        "kernel_symbol": args.kernel_symbol,
        "block": [shape.bm, shape.bn],
        "selected_schedule": asdict(shape),
        "output_dtype": args.output_dtype,
        "bias": args.bias,
        "input_amplitude": args.input_amplitude,
        "scale": args.scale,
        "relu": args.relu,
        "embedded_expected": args.embed_expected,
        "static_inputs": args.static_inputs,
        "wide_store": args.wide_store,
        "reuse_b": args.reuse_b,
        "cache_b": args.cache_b,
        "cache_a": args.cache_a,
        "pipeline_m": args.pipeline_m,
        "prefetch_m": args.prefetch_m,
        "banked_m": args.banked_m,
        "wide_a": args.wide_a,
        "wide_b": args.wide_b,
        "separate_b_bank": args.separate_b_bank,
        "prefetch_b": args.prefetch_b,
        "status": "pass" if passed else "fail",
        "kernel_cycles": int(match.group(1)) if match else None,
        "engine_cycles_including_harness": run.finish.cycles if run.finish else None,
        "elf_sha256": built.elf_sha256,
        "target_ir_sha256": compilation["target_ir_sha256"],
        "llvm_ir_sha256": compilation["llvm_mlir_sha256"],
        "gsim_engine_sha256": run.engine.get("binary_sha256"),
        "nofsm_audit": audit,
        "stdout_tail": run.stdout_tail,
        "stderr_tail": run.stderr_tail if not passed else "",
    }
    (workdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("nofsm_audit", "stdout_tail", "stderr_tail")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
