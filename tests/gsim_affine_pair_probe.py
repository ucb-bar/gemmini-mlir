"""Measure a proved affine predictor, or its complete exact correction path.

The predictor-only arm closes target arithmetic for every pair independently.
Corrected timings include target commands, transfers and the complete CPU scan.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from merlin.llvmlower.quantized_affine_pair import (
    derive,
    emit_correction,
    predictor_table,
    source_table,
)
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_wide_resadd import build, tables
from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--certificate", type=Path, required=True)
    ap.add_argument(
        "--mode", choices=("predictor", "corrected", "control"), required=True
    )
    ap.add_argument("--fixture-dir", type=Path)
    ap.add_argument(
        "--correction-schedule",
        choices=("scalar", "packed_prefix"),
        default="packed_prefix",
    )
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--max-cycles", type=int, default=15000000)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()
    w = args.workdir.resolve()
    w.mkdir(parents=True, exist_ok=False)
    proof = json.loads(args.certificate.read_text())
    if proof != derive(**proof["source"], **proof["predictor"]):
        raise ValueError("pair certificate changed")
    if args.mode == "control" and proof["mismatched_pairs"]:
        raise ValueError("control must be exact without correction")
    if args.fixture_dir:
        a = np.fromfile(args.fixture_dir / "lhs.bin", np.int8)
        b = np.fromfile(args.fixture_dir / "rhs.bin", np.int8)
    else:
        a = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
        b = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    if len(a) != len(b) or len(a) <= 0 or len(a) % 1024:
        raise ValueError("equal positive input extents divisible by1024 required")
    a.tofile(w / "lhs.bin")
    b.tofile(w / "rhs.bin")
    n = len(a)
    table = (
        predictor_table(**proof["predictor"], relu=proof["source"]["relu"])
        if args.mode == "predictor"
        else source_table(**proof["source"])
    )
    table[a.astype(np.int16) + 128, b.astype(np.int16) + 128].tofile(w / "expected.bin")
    p, q, scale = (proof["predictor"][k] for k in ("p", "q", "scale"))
    device = build(
        n // 64,
        p,
        q,
        scale,
        relu=proof["source"]["relu"],
        prefetch_m=True,
        banked_accumulators=True,
    )
    device.body.block.first_op.properties["sym_name"] = StringAttr(
        "gemmini_pair_predictor"
    )
    compilation = compile_module(device, args.llvm_bin, w)
    (w / "tables.h").write_text(
        "static const int8_t coefficients[768] __attribute__((aligned(64)))={"
        + ",".join(map(str, tables(p, q)))
        + "};\n"
    )
    assembly = w / "inputs.S"
    assembly.write_text(
        "\n".join(
            ".section .rodata\n.balign 64\n.global "
            + name
            + "\n"
            + name
            + ':\n.incbin "'
            + str(w / file)
            + '"\n'
            for name, file in [
                ("input_a", "lhs.bin"),
                ("input_b", "rhs.bin"),
                ("expected", "expected.bin"),
            ]
        )
    )
    subprocess.run(
        [
            str(args.llvm_bin / "clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-c",
            str(assembly),
            "-o",
            str(w / "inputs.o"),
        ],
        check=True,
        capture_output=True,
    )
    correction = (
        emit_correction(
            proof,
            "correct_pairs",
            packed_prefix=args.correction_schedule == "packed_prefix",
        )
        if args.mode == "corrected"
        else ""
    )
    call = (
        "correct_pairs(input_a,input_b,c.output,N);" if args.mode == "corrected" else ""
    )
    source = w / "probe.c"
    source.write_text(
        correction
        + f"""#include <stdint.h>
#include <stdio.h>
#include "tables.h"
#define N {n}
extern const int8_t input_a[N],input_b[N],expected[N];
static struct {{int8_t output[N];uint8_t guard[2048];}} c __attribute__((aligned(64)));
extern void gemmini_pair_predictor(const int8_t*,const int8_t*,int8_t*,const int8_t*);
int main(void) {{
 for(int i=0;i<N;i++)c.output[i]=-37;
 for(int i=0;i<2048;i++)c.guard[i]=0x5a;
 uint64_t begin,end;__asm__ volatile("csrr %0, mcycle":"=r"(begin)::"memory");
 gemmini_pair_predictor(input_a,input_b,c.output,coefficients);
 {call}
 __asm__ volatile("csrr %0, mcycle":"=r"(end)::"memory");
 printf("AFFINE_PAIR_CYCLES %d\\n",(int)(end-begin));
 for(int i=0;i<N;i++)if(c.output[i]!=expected[i]){{printf("FAIL %d got%d want%d\\n",i,c.output[i],expected[i]);return 1;}}
 for(int i=0;i<2048;i++)if(c.guard[i]!=0x5a){{printf("FAIL guard%d\\n",i);return 2;}}
 printf("AFFINE_PAIR_PASS N={n} MODE={args.mode}\\n");return 0;
}}
"""
    )
    built = build_program(
        [source, w / "kernel.o", w / "inputs.o"],
        w,
        target="gemmini",
        extra_cflags=[f"-I{w}", "-ffp-contract=off"],
        max_loaded_bytes=None,
    )
    audit = audit_elf(built.elf.read_bytes())
    (w / "nofsm_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "pass":
        raise ValueError("affine capsule gained FSM instruction")
    run = run_on_gsim(
        built.elf,
        target="gemmini",
        max_cycles=args.max_cycles,
        timeout_s=args.timeout_s,
        backdoor=True,
        stdout_path=w / "gsim.stdout",
    )
    marker = f"AFFINE_PAIR_PASS N={n} MODE={args.mode}"
    passed = bool(run.completed and run.returncode == 0 and marker in run.stdout_tail)
    cycles = re.findall(r"^AFFINE_PAIR_CYCLES (\d+)$", run.stdout_tail, re.MULTILINE)
    result = {
        "schema": "gemmini_affine_pair_capsule_v1",
        "status": "pass" if passed else "fail",
        "mode": args.mode,
        "elements": n,
        "full_pair_domain": args.fixture_dir is None,
        "correction_schedule": args.correction_schedule,
        "certificate_path": str(args.certificate.resolve()),
        "certificate_sha256": sha(args.certificate),
        "predictor": proof["predictor"],
        "source": proof["source"],
        "kernel_cycles": int(cycles[0]) if len(cycles) == 1 else None,
        "metric_scope": "Complete predictor plus CPU correction/traffic when corrected; no whole-model cost transfer",
        "fixture_pins": {
            name: sha(w / name) for name in ("lhs.bin", "rhs.bin", "expected.bin")
        },
        "compilation": compilation,
        "elf_sha256": built.elf_sha256,
        "engine_sha256": run.engine.get("binary_sha256"),
        "source_c_sha256": sha(source),
        "coefficient_header_sha256": sha(w / "tables.h"),
        "correction_code_sha256": hashlib.sha256(correction.encode()).hexdigest()
        if correction
        else None,
        "run_completed": run.completed,
        "run_returncode": run.returncode,
        "guards_expected": 2048,
        "guards_checked": 2048 if passed else None,
        "correctness_scope": "All outputs and guards checked only after AFFINE_PAIR_PASS; timed-out kernel counters are unqualified observations",
        "caller_obligations": {
            "original_inputs_preserved_after_prediction": True,
            "prediction_output_inputs_nonoverlapping": True,
            "correction_rn_even_gradual_underflow": True,
            "status": "Capsule establishes distinct static objects; generic emitter requires caller ABI proof",
        },
        "nofsm_status": audit["status"],
        "stdout": run.stdout_tail,
        "stderr": run.stderr_tail,
    }
    (w / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
