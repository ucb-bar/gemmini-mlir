"""Complete source-exact residual correction with common-address target arms.

The control either reproduces a supplied source-bound raw kernel or uses a
complete-pair-proved diagonal producer. Every arm includes input/coefficient
DMA, output store/fence and required CPU correction. Checks are outside ROI.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.llvmlower.quantized_affine_pair import derive, predictor_table, source_table
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_streamed_resadd import build as streamed
from mlir_oot.golden_wide_resadd import build, tables
from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rename(module, name):
    fn = next(
        op
        for op in module.body.block.ops
        if isinstance(op, llvm.FuncOp) and op.body.blocks
    )
    fn.properties["sym_name"] = StringAttr(name)
    module.verify()
    return module


def section_offset(elf, name):
    raw = elf.read_bytes()
    assert raw[:6] == b"\x7fELF\x02\x01"
    start = int.from_bytes(raw[40:48], "little")
    size = int.from_bytes(raw[58:60], "little")
    count = int.from_bytes(raw[60:62], "little")
    strings = int.from_bytes(raw[62:64], "little")
    section = raw[start + strings * size : start + (strings + 1) * size]
    base = int.from_bytes(section[24:32], "little")
    for i in range(count):
        row = raw[start + i * size : start + (i + 1) * size]
        label = base + int.from_bytes(row[:4], "little")
        if raw[label : raw.index(b"\0", label)].decode() == name:
            return int.from_bytes(row[24:32], "little")
    raise ValueError("selector section absent")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--certificate", type=Path, required=True)
    p.add_argument("--control-p", type=int, required=True)
    p.add_argument("--control-q", type=int, required=True)
    p.add_argument("--control-scale", type=float, required=True)
    p.add_argument("--control-object", type=Path)
    p.add_argument("--control-symbol")
    p.add_argument("--control-object-sha256")
    p.add_argument("--original-a", type=Path)
    p.add_argument("--original-b", type=Path)
    p.add_argument("--source-capture-receipt", type=Path)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--benchmark-header", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    proof = json.loads(args.certificate.read_text())
    assert proof == derive(**proof["source"], **proof["predictor"])
    control = {"p": args.control_p, "q": args.control_q, "scale": args.control_scale}
    control_proof = derive(**proof["source"], **control)
    assert control_proof["mismatched_pairs"] == 0
    count = args.m * 64
    if args.original_a is not None:
        assert args.original_b and args.source_capture_receipt
        a = np.frombuffer(args.original_a.read_bytes(), dtype=np.int8)
        b = np.frombuffer(args.original_b.read_bytes(), dtype=np.int8)
        source_capture = json.loads(args.source_capture_receipt.read_text())
        assert a.size == b.size == count
        assert source_capture["all1000_original_f32_bits_exact"]
        assert sha(args.original_a) == source_capture["lhs_sha256"]
        assert sha(args.original_b) == source_capture["rhs_sha256"]
    else:
        assert args.original_b is None and args.source_capture_receipt is None
        pair = np.arange(count, dtype=np.int32) % 65536
        if count < 65536:
            pair = (pair * 40503 + 32768) % 65536
        a = (pair // 256 - 128).astype(np.int8)
        b = (pair % 256 - 128).astype(np.int8)
        source_capture = None
    index_a, index_b = a.astype(np.int16) + 128, b.astype(np.int16) + 128
    expected = source_table(**proof["source"])[index_a, index_b]
    predicted = predictor_table(**proof["predictor"], relu=proof["source"]["relu"])[
        index_a, index_b
    ]
    fixture = ""
    for name, data in [
        ("a", a),
        ("b", b),
        ("check_a", a),
        ("check_b", b),
        ("expected", expected),
        ("predicted", predicted),
    ]:
        path = work / (name + ".bin")
        path.write_bytes(data.tobytes())
        fixture += (
            f'.balign 64\n.global fixture_{name}\nfixture_{name}:\n.incbin "{path}"\n'
        )
    fixture_path = work / "fixture.S"
    fixture_path.write_text(".section .rodata\n" + fixture)
    (work / "benchmark_buffer.h").write_bytes(args.benchmark_header.read_bytes())
    coefficients = []
    for i, data in enumerate(
        [
            tables(control["p"], control["q"]),
            tables(proof["predictor"]["p"], proof["predictor"]["q"]),
        ]
    ):
        coefficients.append(
            f"static const int8_t coefficients{i}[768] __attribute__((aligned(64)))={{"
            + ",".join(map(str, data))
            + "};\n"
        )
    (work / "coefficients.h").write_text("".join(coefficients))
    objects, compilations = [], []
    if args.control_object:
        assert args.control_symbol and args.control_object_sha256
        assert sha(args.control_object) == args.control_object_sha256
        control_symbol = args.control_symbol
        objects.append(args.control_object)
    else:
        control_symbol = "exact_control"
        d = work / "control"
        compilations.append(
            compile_module(
                rename(
                    build(
                        args.m,
                        **control,
                        relu=proof["source"]["relu"],
                        prefetch_m=True,
                        banked_accumulators=True,
                    ),
                    control_symbol,
                ),
                args.llvm_bin,
                d,
            )
        )
        objects.append(d / "kernel.o")
    d = work / "serial"
    compilations.append(
        compile_module(
            rename(
                build(
                    args.m,
                    **proof["predictor"],
                    relu=proof["source"]["relu"],
                    prefetch_m=True,
                    banked_accumulators=True,
                ),
                "serial_predictor",
            ),
            args.llvm_bin,
            d,
        )
    )
    objects.append(d / "kernel.o")
    module, code, _ = streamed(args.m, proof, correction_symbol="merlin_correct")
    d = work / "streamed"
    compilations.append(
        compile_module(rename(module, "streamed_predictor"), args.llvm_bin, d)
    )
    objects.append(d / "kernel.o")
    c = work / "correction.c"
    c.write_text(code)
    command = [
        str(args.llvm_bin / "clang"),
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffp-contract=off",
        "-frounding-math",
        "-fno-fast-math",
        "-c",
        str(c),
        "-o",
        str(work / "correction.o"),
    ]
    subprocess.run(command, check=True, capture_output=True)
    objects.append(work / "correction.o")
    program = work / "probe.c"
    program.write_text(
        """#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "benchmark_buffer.h"
#include "coefficients.h"
#define N COUNT
extern const int8_t fixture_a[N],fixture_b[N],fixture_check_a[N],fixture_check_b[N],fixture_expected[N],fixture_predicted[N];
extern void CONTROL(const int8_t*,const int8_t*,int8_t*,const int8_t*);
extern void serial_predictor(const int8_t*,const int8_t*,int8_t*,const int8_t*);
extern void streamed_predictor(const int8_t*,const int8_t*,int8_t*,const int8_t*);
extern void merlin_correct(const int8_t*,const int8_t*,int8_t*,size_t);
volatile uint64_t arm_selector __attribute__((section(".selector")))=0;
static struct {uint8_t before[2048];int8_t output[N];uint8_t after[2048];} result __attribute__((aligned(64)));
static void poison(void){merlin_benchmark_fill(result.before,0x5a,2048);merlin_benchmark_fill(result.output,0x7d,N);merlin_benchmark_fill(result.after,0xa5,2048);}
static int guards(void){for(size_t i=0;i<2048;i++)if(result.before[i]!=0x5a||result.after[i]!=0xa5)return 1;return 0;}
static int equal(const void*p,const void*q,size_t n){return merlin_benchmark_first_difference(p,q,n)==n;}
int main(void){
 poison();CONTROL(fixture_a,fixture_b,result.output,coefficients0);
 if(!equal(result.output,fixture_expected,N)||guards()){printf("%s\\n","FAIL control");return 1;}
 poison();serial_predictor(fixture_a,fixture_b,result.output,coefficients1);
 if(!equal(result.output,fixture_predicted,N)||guards()){printf("%s\\n","FAIL predictor");return 2;}
 merlin_correct(fixture_a,fixture_b,result.output,N);
 if(!equal(result.output,fixture_expected,N)||guards()){printf("%s\\n","FAIL serial correction");return 3;}
 poison();streamed_predictor(fixture_a,fixture_b,result.output,coefficients1);
 if(!equal(result.output,fixture_expected,N)||guards()){printf("%s\\n","FAIL streamed correction");return 4;}
 poison();uint64_t begin,end,arm=arm_selector;
 __asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 if(arm==0)CONTROL(fixture_a,fixture_b,result.output,coefficients0);
 else if(arm==1){serial_predictor(fixture_a,fixture_b,result.output,coefficients1);merlin_correct(fixture_a,fixture_b,result.output,N);}
 else streamed_predictor(fixture_a,fixture_b,result.output,coefficients1);
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
 printf("STREAMED_RESIDUAL_CYCLES %lu %lu\\n",(unsigned long)arm,(unsigned long)(end-begin));
 if(!equal(result.output,fixture_expected,N)||guards()){printf("%s\\n","FAIL timed result");return 5;}
 if(!equal(fixture_a,fixture_check_a,N)||!equal(fixture_b,fixture_check_b,N)){printf("%s\\n","FAIL immutable inputs");return 6;}
 printf("STREAMED_RESIDUAL_PASS arm%lu all%d guards4096 inputs%d\\n",(unsigned long)arm,N,2*N);return 0;
}
""".replace("COUNT", str(count)).replace("CONTROL", control_symbol)
    )
    built = build_program(
        [program, fixture_path, *objects],
        work / "arm0",
        target="gemmini",
        extra_cflags=[
            f"-I{work}",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-frounding-math",
        ],
        max_loaded_bytes=None,
    )
    raw = built.elf.read_bytes()
    offset = section_offset(built.elf, ".selector")
    assert raw[offset : offset + 8] == bytes(8)
    records = []
    for arm in range(3):
        directory = work / f"arm{arm}"
        directory.mkdir(exist_ok=True)
        elf = built.elf if arm == 0 else directory / "layer.elf"
        if arm:
            changed = bytearray(raw)
            changed[offset] = arm
            elf.write_bytes(changed)
            assert [i for i, (a, b) in enumerate(zip(raw, changed)) if a != b] == [
                offset
            ]
        audit = audit_elf(elf.read_bytes())
        assert audit["status"] == "pass"
        (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        run = subprocess.run(
            [
                "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike",
                "-g",
                "--extension=gemmini",
                "--isa=RV64GC",
                "-m0x80000000:0x80000000",
                str(elf),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        output = run.stdout + run.stderr
        (directory / "spike.stdout").write_text(output)
        marker = (
            f"STREAMED_RESIDUAL_PASS arm{arm} all{count} guards4096 inputs{2 * count}"
        )
        assert run.returncode == 0 and marker in output
        target = run_on_gsim(
            elf,
            target="gemmini",
            max_cycles=30000000,
            timeout_s=args.timeout,
            backdoor=True,
            stdout_path=directory / "gsim.stdout",
        )
        lines = [
            line.split()[-1]
            for line in target.stdout_tail.splitlines()
            if line.startswith(f"STREAMED_RESIDUAL_CYCLES {arm} ")
        ]
        passed = (
            target.completed
            and target.returncode == 0
            and marker in target.stdout_tail
            and len(lines) == 1
        )
        record = {
            "arm": arm,
            "elf": str(elf),
            "elf_sha256": sha(elf),
            "marker": marker,
            "strict_RV64GC": True,
            "completed": target.completed,
            "returncode": target.returncode,
            "passed": passed,
            "cycles": int(lines[0]) if len(lines) == 1 else None,
            "engine": target.engine,
            "stdout": target.stdout_tail,
            "stderr": target.stderr_tail,
        }
        (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print("ARM", arm, "PASS", passed, "CYCLES", record["cycles"], flush=True)
        if not passed:
            break
    result = {
        "schema": "gemmini_streamed_residual_complete_capsule_v1",
        "m": args.m,
        "elements": count,
        "full_pair_domain": count == 65536 and source_capture is None,
        "source_capture": source_capture,
        "certificate": proof,
        "control_certificate": control_proof,
        "compilations": compilations,
        "correction_compiler_argv": command,
        "selector_offset": offset,
        "same_addresses": True,
        "metric_scope": "Complete producer including DMA/stores/fences plus every required exact CPU correction; checks and identical warmup outside ROI",
        "records": records,
        "passed": len(records) == 3 and all(r["passed"] for r in records),
        "whole_cycles": "UNKNOWN",
        "physical_DRAM": "UNKNOWN",
        "asynchronous_overlap": "UNKNOWN; elapsed complete cost measured",
        "default_enabled": False,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
