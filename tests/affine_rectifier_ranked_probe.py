"""Exact current ranked residual versus complete finite rectifier pipeline.

This research fixture binds an immutable source route and captured original
operands. The two ELFs differ only in a data selector; all code, operand/output
addresses, warmups, descriptors and outside-ROI checks are common.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from affine_rectifier_capability_probe import pin
from merlin.llvmlower.quantized_affine_pair import derive, source_table
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize
from merlin.perf.layer_bench import build_program, run_on_gsim

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_rectified_resadd import Capabilities, Plan, build
from mlir_oot.no_fsm_audit import audit_elf


def command(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--control-bundle", type=Path, required=True)
    cli.add_argument("--route", type=int, default=0)
    cli.add_argument("--predictor-certificate", type=Path, required=True)
    cli.add_argument("--capture-receipt", type=Path, required=True)
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--llvm-bin", type=Path, required=True)
    cli.add_argument("--core", type=Path, required=True)
    cli.add_argument("--workdir", type=Path, required=True)
    cli.add_argument("--timeout", type=int, default=1800)
    cli.add_argument("--build-only", action="store_true")
    args = cli.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    route = json.loads(args.control_bundle.read_text())["routes"][args.route]
    directory = args.control_bundle.resolve().parent / route["symbol"]
    original_kernel, original_adapter, original_source = [
        directory / name for name in ("kernel.o", "adapter.o", "adapter.c")
    ]
    assert pin(original_kernel)["sha256"] == route["compilation"]["object_sha256"]
    assert (
        pin(original_adapter)["sha256"] == route["adapter_compilation"]["object_sha256"]
    )
    assert (
        pin(original_source)["sha256"] == route["adapter_compilation"]["source_sha256"]
    )
    predictor = json.loads(args.predictor_certificate.read_text())
    if "source_certificate" in predictor:
        predictor = predictor["source_certificate"]
    assert predictor["source"] == route["proof"]["source"]
    pair = derive(**predictor["source"], **predictor["predictor"])
    certificate = synthesize(pair, max_pairs=1, indicator_family="axis_offsets")
    control_proof = derive(**route["proof"]["source"], **route["proof"]["coefficients"])
    assert not control_proof["mismatched_pairs"]
    m, n = route["m"], route["n"]
    plan = Plan(m, n, certificate, Capabilities(*([True] * 6)))
    count = m * n
    capture = json.loads(args.capture_receipt.read_text())
    assert capture["all1000_original_f32_bits_exact"] and capture["elements"] == count
    a_path, b_path = args.inputs / "a.bin", args.inputs / "b.bin"
    assert pin(a_path)["sha256"] == capture["lhs_sha256"]
    assert pin(b_path)["sha256"] == capture["rhs_sha256"]
    a, b = [np.frombuffer(path.read_bytes(), np.int8) for path in (a_path, b_path)]
    assert a.size == b.size == count
    expected = source_table(**pair["source"])[
        a.astype(np.int16) + 128, b.astype(np.int16) + 128
    ]
    compilation = compile_module(build(plan), args.llvm_bin, work / "candidate")
    compiler = [
        str(args.llvm_bin / "clang"),
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]
    table_values = ",".join(map(str, np.frombuffer(plan.tables(), np.int8)))
    candidate_source = work / "candidate_adapter.c"
    candidate_source.write_text(f"""#include <stdint.h>
typedef struct{{void *allocated,*aligned;intptr_t offset,sizes[2],strides[2];}} mem2;
static const int8_t tables[{len(plan.tables())}] __attribute__((aligned(64)))={{{table_values}}};
extern void gemmini_golden_rectified_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*);
void rectified_residual(mem2*r,mem2*a,mem2*b,mem2*c){{
 if(!a->aligned||a->offset<0||a->sizes[0]!={m}||a->sizes[1]!={n}||a->strides[0]!={n}||a->strides[1]!=1||!b->aligned||b->offset<0||b->sizes[0]!={m}||b->sizes[1]!={n}||b->strides[0]!={n}||b->strides[1]!=1||!c->aligned||c->offset<0||c->sizes[0]!={m}||c->sizes[1]!={n}||c->strides[0]!={n}||c->strides[1]!=1)__builtin_trap();
 gemmini_golden_rectified_resadd((const int8_t*)a->aligned+a->offset,(const int8_t*)b->aligned+b->offset,(int8_t*)c->aligned+c->offset,tables);*r=*c;
}}
""")
    candidate_object = work / "candidate_adapter.o"
    command([*compiler, "-c", str(candidate_source), "-o", str(candidate_object)])
    tools = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin")
    objcopy, readelf = (
        str(tools / "riscv64-unknown-elf-objcopy"),
        str(tools / "riscv64-unknown-elf-readelf"),
    )
    rebound = work / "control_adapter_rebound.o"
    command(
        [
            objcopy,
            "--redefine-sym",
            "_mlir_ciface_" + route["symbol"] + "=exact_residual",
            str(original_adapter),
            str(rebound),
        ]
    )
    text_pins = []
    for index, obj in enumerate((original_adapter, rebound)):
        before = pin(obj)
        text = work / f"adapter_text{index}.bin"
        command(
            [
                objcopy,
                "--dump-section",
                ".text=" + str(text),
                str(obj),
                str(work / f"reader{index}.o"),
            ]
        )
        assert pin(obj) == before
        text_pins.append(pin(text))
    assert text_pins[0]["sha256"] == text_pins[1]["sha256"]
    fixture_text = ".section .rodata\n"
    for name, data in (
        ("a", a),
        ("b", b),
        ("check_a", a),
        ("check_b", b),
        ("expected", expected),
    ):
        path = work / (name + ".bin")
        path.write_bytes(data.tobytes())
        fixture_text += (
            f'.balign 64\n.global fixture_{name}\nfixture_{name}:\n.incbin "{path}"\n'
        )
    fixture = work / "fixture.S"
    fixture.write_text(fixture_text)
    header = args.core / "merlin/runtime/c/benchmark_buffer.h"
    (work / "benchmark_buffer.h").write_bytes(header.read_bytes())
    probe = work / "probe.c"
    probe.write_text(
        r"""#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "benchmark_buffer.h"
#define N ELEMENT_COUNT
#define M ROW_COUNT
#define K COLUMN_COUNT
typedef struct{void*allocated,*aligned;intptr_t offset,sizes[2],strides[2];}mem2;
extern const int8_t fixture_a[N],fixture_b[N],fixture_check_a[N],fixture_check_b[N],fixture_expected[N];
extern void exact_residual(mem2*,mem2*,mem2*,mem2*),rectified_residual(mem2*,mem2*,mem2*,mem2*);
volatile uint64_t arm_selector __attribute__((section(".selector")))=0;
static struct{uint8_t before[2048];int8_t output[N];uint8_t after[2048];}result __attribute__((aligned(64)));
static int eq(const void*a,const void*b,size_t n){return merlin_benchmark_first_difference(a,b,n)==n;}
static int guards(void){for(size_t i=0;i<2048;i++)if(result.before[i]!=0x5a||result.after[i]!=0xa5)return 0;return 1;}
static void poison(void){merlin_benchmark_fill(result.before,0x5a,2048);merlin_benchmark_fill(result.output,0x7d,N);merlin_benchmark_fill(result.after,0xa5,2048);}
static uint64_t cycle(void){uint64_t x;asm volatile("csrr %0,mcycle":"=r"(x)::"memory");return x;}
static uint64_t inst(void){uint64_t x;asm volatile("csrr %0,minstret":"=r"(x)::"memory");return x;}
static unsigned flags(void){unsigned x;asm volatile("csrr %0,fflags":"=r"(x)::"memory");return x;}
int main(void){
 mem2 a={(void*)fixture_a,(void*)fixture_a,0,{M,K},{K,1}},b={(void*)fixture_b,(void*)fixture_b,0,{M,K},{K,1}},c={result.output,result.output,0,{M,K},{K,1}},r={0},old_a=a,old_b=b,old_c=c;
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");
 for(unsigned warm=0;warm<2;warm++){
  poison();if(warm)rectified_residual(&r,&a,&b,&c);else exact_residual(&r,&a,&b,&c);
  if(!eq(result.output,fixture_expected,N)||!eq(&r,&c,sizeof(c))||!guards()){printf("RECTIFIED_RANKED_FAIL warm%u\n",warm);return 11;}
 }
 poison();uint64_t arm=arm_selector;unsigned f0=flags();uint64_t c0=cycle(),i0=inst();
 if(arm)rectified_residual(&r,&a,&b,&c);else exact_residual(&r,&a,&b,&c);
 uint64_t i1=inst(),c1=cycle();unsigned f1=flags();
 printf("RECTIFIED_RANKED_COUNTER arm=%lu cycles=%lu instructions=%lu before=%u after=%u\n",(unsigned long)arm,(unsigned long)(c1-c0),(unsigned long)(i1-i0),f0,f1);
 if(!eq(result.output,fixture_expected,N)||!eq(&r,&c,sizeof(c))||!guards()||f0!=f1){printf("RECTIFIED_RANKED_FAIL timed\n");return 12;}
 if(!eq(fixture_a,fixture_check_a,N)||!eq(fixture_b,fixture_check_b,N)||!eq(&a,&old_a,sizeof(a))||!eq(&b,&old_b,sizeof(b))||!eq(&c,&old_c,sizeof(c)))return 13;
 printf("RECTIFIED_RANKED_PASS arm%lu all%d guards4096 inputs%d descriptors flags\n",(unsigned long)arm,N,2*N);return 0;
}
""".replace("ELEMENT_COUNT", str(count))
        .replace("ROW_COUNT", str(m))
        .replace("COLUMN_COUNT", str(n))
    )
    built = build_program(
        [
            probe,
            fixture,
            rebound,
            original_kernel,
            candidate_object,
            work / "candidate/kernel.o",
        ],
        work / "arm0",
        target="gemmini",
        extra_cflags=["-I" + str(work), "-O2", "-fno-fast-math", "-ffp-contract=off"],
        max_loaded_bytes=None,
    )
    raw = built.elf.read_bytes()
    section = next(
        line.split("]", 1)[1].split()
        for line in command([readelf, "-SW", str(built.elf)]).splitlines()
        if "]" in line and line.split("]", 1)[1].split()[:1] == [".selector"]
    )
    offset = int(section[3], 16)
    assert raw[offset : offset + 8] == bytes(8)
    other = work / "arm1/layer.elf"
    other.parent.mkdir()
    changed = bytearray(raw)
    changed[offset] = 1
    other.write_bytes(changed)
    assert [i for i, (x, y) in enumerate(zip(raw, changed)) if x != y] == [offset]
    elfs = [built.elf, other]
    audits = [audit_elf(path.read_bytes()) for path in elfs]
    assert all(row["status"] == "pass" for row in audits)
    recipe = {
        "schema": "source_bound_affine_rectifier_ranked_pair_v1",
        "route": route,
        "certificate": certificate,
        "plan": plan.attributes(),
        "candidate_compilation": compilation,
        "source_capture": capture,
        "adapter_text_identity": text_pins,
        "selector_offset": offset,
        "ELFs_differ_one_byte": True,
        "audits": audits,
        "scope": "Complete actual ranked ABI/descriptor checks, current39 or rectifier14 producer, all config/seed/DMA/store/reload/compute/fence and result descriptor. Common addresses and identical warmups; poisoning/exact outputs/input immutability/guards/flags outsideROI. No CPU correction.",
        "whole_model_route_enabled": False,
        "whole_cycles": "UNKNOWN",
        "physical_DRAM": "UNKNOWN",
        "source_numeric_contract": "all65536 original signed-byte pairs EXACT; whole all1000 f32 bits gate unchanged",
        "originals": [
            pin(p)
            for p in (
                args.control_bundle,
                args.predictor_certificate,
                args.capture_receipt,
                a_path,
                b_path,
                original_kernel,
                original_adapter,
                original_source,
                header,
            )
        ],
        "generated": [
            pin(p) for p in (candidate_source, candidate_object, probe, fixture, *elfs)
        ],
        "token_usage_available": False,
    }
    (work / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    for arm, elf in enumerate(elfs):
        run = subprocess.run(
            [
                str(tools / "spike"),
                "-g",
                "--extension=gemmini",
                "--isa=rv64gc",
                "-m0x80000000:0x400000000",
                str(elf),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        (elf.parent / "spike.stdout").write_text(run.stdout)
        (elf.parent / "spike.stderr").write_text(run.stderr)
        assert (
            run.returncode == 0
            and f"RECTIFIED_RANKED_PASS arm{arm} all{count}" in run.stdout
        )
        print("STRICT_RANKED_PASS", arm, count, flush=True)
    if args.build_only:
        return 0

    def measure(arm):
        elf = elfs[arm]
        run = run_on_gsim(
            elf,
            target="gemmini",
            max_cycles=20000000,
            timeout_s=args.timeout,
            backdoor=True,
            stdout_path=elf.parent / "gsim.stdout",
        )
        lines = [
            line
            for line in run.stdout_tail.splitlines()
            if line.startswith(f"RECTIFIED_RANKED_COUNTER arm={arm} ")
        ]
        counters = (
            {k: int(v) for k, v in (field.split("=") for field in lines[0].split()[1:])}
            if len(lines) == 1
            else None
        )
        passed = (
            run.completed
            and run.returncode == 0
            and f"RECTIFIED_RANKED_PASS arm{arm} all{count}" in run.stdout_tail
            and counters is not None
        )
        record = {
            "arm": arm,
            "passed": passed,
            "completed": run.completed,
            "returncode": run.returncode,
            "counters": counters,
            "engine": run.engine,
            "elf": pin(elf),
            "stdout": pin(elf.parent / "gsim.stdout"),
            "stdout_tail": run.stdout_tail,
            "stderr_tail": run.stderr_tail,
            "scope": recipe["scope"],
        }
        (elf.parent / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print("GSIM_RANKED", arm, passed, counters, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(measure, range(2)))
    (work / "result.json").write_text(
        json.dumps(
            {
                "schema": "affine_rectifier_ranked_result_v1",
                "recipe": pin(work / "recipe.json"),
                "passed": all(r["passed"] for r in records),
                "records": records,
                "whole_cycles": "UNKNOWN",
                "default_enabled": False,
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
