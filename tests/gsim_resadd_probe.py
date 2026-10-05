#!/usr/bin/env python3
"""Build, audit and numerically test stock-Gemmini residual-as-matmul."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlir_oot.golden_resadd import build
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program, run_on_gsim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    wd = args.workdir.resolve()
    wd.mkdir(parents=True, exist_ok=False)
    compile_module(build(args.m, args.n), args.llvm_bin, wd)
    source = Path(__file__).with_name("resadd_probe.c")
    program = build_program([source, wd / "kernel.o"], wd, target="gemmini",
                            extra_cflags=[f"-DM={args.m}", f"-DN={args.n}"],
                            max_loaded_bytes=None)
    audit = audit_elf(program.elf.read_bytes())
    (wd / "nofsm_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("linked ELF fails no-FSM audit")
    run = run_on_gsim(program.elf, target="gemmini", max_cycles=1_000_000,
                      timeout_s=150, backdoor=True,
                      stdout_path=wd / "gsim.stdout")
    match = re.search(r"^GOLDEN_RESADD_CYCLES (\d+)$", run.stdout_tail, re.MULTILINE)
    passed = (run.completed and run.returncode == 0 and match and
              f"GOLDEN_RESADD PASS M={args.m} N={args.n}" in run.stdout_tail)
    result = {"status": "pass" if passed else "fail", "m": args.m, "n": args.n,
              "kernel_cycles": int(match.group(1)) if match else None,
              "elf_sha256": program.elf_sha256,
              "gsim_engine_sha256": run.engine.get("binary_sha256"),
              "stdout_tail": run.stdout_tail,
              "stderr_tail": run.stderr_tail if not passed else ""}
    (wd / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
