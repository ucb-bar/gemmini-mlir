"""Complete producer/transfer/correction cost, immutable all-pair or owned inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from affine_rectifier_capability_probe import pin
from merlin.llvmlower.quantized_affine_pair import derive, source_table
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize
from merlin.perf.layer_bench.build import build_program

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_rectified_resadd import Capabilities, Plan, build
from mlir_oot.golden_wide_resadd import build as control_build
from mlir_oot.golden_wide_resadd import tables as control_tables
from mlir_oot.no_fsm_audit import audit_elf


def generate(args):
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if args.source_certificate:
        bound = json.loads(args.source_certificate.read_text())
        source = bound["source"]
        predictor = bound["predictor"]
    else:
        # Explicit research fixture binds original scalar constants. No production
        # source/name matching occurs; the reusable compiler consumes the proof.
        source = {
            "lhs_scale": 0.011258588172495365,
            "rhs_scale": 0.00940733402967453,
            "output_scale": 0.011643771082162857,
            "relu": True,
        }
        predictor = {"p": 298, "q": 249, "scale": 0.0032446938566863537}
    original = derive(**source, **predictor)
    certificate = synthesize(original, max_pairs=1, indicator_family="axis_offsets")
    plan = Plan(args.m, args.n, certificate, Capabilities(*([True] * 6)))
    count = args.m * args.n
    if args.inputs:
        a = np.frombuffer((args.inputs / "a.bin").read_bytes(), dtype=np.int8).copy()
        b = np.frombuffer((args.inputs / "b.bin").read_bytes(), dtype=np.int8).copy()
        if len(a) != count or len(b) != count:
            raise ValueError("owned captured input shape changed")
    else:
        index = np.arange(count, dtype=np.int64) % 65536
        a = (index // 256 - 128).astype(np.int8)
        b = (index % 256 - 128).astype(np.int8)
        if count < 65536:
            for i, row in enumerate(certificate["relation"]):
                a[i], b[i] = row["lhs"], row["rhs"]
    expected = source_table(**source)[
        a.astype(np.int16) + 128, b.astype(np.int16) + 128
    ]
    control_proof = derive(
        **source, p=args.control_p, q=args.control_q, scale=args.control_scale
    )
    if control_proof["mismatched_pairs"]:
        raise ValueError("selected current control no longer equals original source")
    if args.n != 64:
        raise ValueError(
            "matched legacy control currently has fixed64-column ABI; independentN uses standalone build tests"
        )
    candidate = compile_module(build(plan), args.llvm_bin, work / "candidate")
    control = compile_module(
        control_build(
            args.m,
            args.control_p,
            args.control_q,
            args.control_scale,
            relu=source["relu"],
            prefetch_m=True,
            banked_accumulators=True,
        ),
        args.llvm_bin,
        work / "control",
    )
    coefficient = plan.tables()
    old_coefficient = control_tables(args.control_p, args.control_q)
    for name, value in (
        ("a.bin", a.tobytes()),
        ("b.bin", b.tobytes()),
        ("expected.bin", expected.tobytes()),
        ("tables.bin", coefficient),
        ("control_tables.bin", old_coefficient),
    ):
        (work / name).write_bytes(value)
    c = r"""#include <stdint.h>
#include <stdio.h>
#include "benchmark_buffer.h"
#define COUNT ELEMENT_COUNT
extern void gemmini_golden_rectified_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*);
extern void gemmini_golden_wide_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*);
struct box{uint8_t before[64];int8_t data[COUNT];uint8_t after[64];};
static struct box A __attribute__((aligned(64))),B __attribute__((aligned(64))),C __attribute__((aligned(64)));
static const int8_t original_a[COUNT] __attribute__((aligned(64)))={A_VALUES};
static const int8_t original_b[COUNT] __attribute__((aligned(64)))={B_VALUES};
static const int8_t expected[COUNT] __attribute__((aligned(64)))={EXPECTED_VALUES};
static const int8_t tables[TABLE_COUNT] __attribute__((aligned(64)))={TABLE_VALUES};
static const int8_t old_tables[OLD_TABLE_COUNT] __attribute__((aligned(64)))={OLD_TABLE_VALUES};
static uint64_t cycle(void){uint64_t v;asm volatile("csrr %0,mcycle":"=r"(v)::"memory");return v;}
static uint64_t inst(void){uint64_t v;asm volatile("csrr %0,minstret":"=r"(v)::"memory");return v;}
static unsigned flags(void){unsigned v;asm volatile("csrr %0,fflags":"=r"(v)::"memory");return v;}
static int guards(const struct box*b){for(unsigned i=0;i<64;i++)if(b->before[i]!=0xdb||b->after[i]!=0xdb)return 0;return 1;}
int main(void){
 merlin_benchmark_fill(&A,0xdb,sizeof(A));merlin_benchmark_fill(&B,0xdb,sizeof(B));
 for(unsigned i=0;i<COUNT;i++){A.data[i]=original_a[i];B.data[i]=original_b[i];}
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");
 for(unsigned repeat=0;repeat<2;repeat++)for(unsigned arm=0;arm<2;arm++){
  merlin_benchmark_fill(&C,0xdb,sizeof(C));unsigned before=flags();uint64_t c0=cycle(),i0=inst();
  if(arm)gemmini_golden_rectified_resadd(A.data,B.data,C.data,tables);else gemmini_golden_wide_resadd(A.data,B.data,C.data,old_tables);
  uint64_t i1=inst(),c1=cycle();unsigned after=flags();size_t first=merlin_benchmark_first_difference(C.data,expected,COUNT);
  if(first!=COUNT){printf("RECTIFIER_FAIL arm=%u index=%lu A=%d B=%d expected=%d actual=%d\n",arm,(unsigned long)first,A.data[first],B.data[first],expected[first],C.data[first]);return 11;}
  if(!guards(&A)||!guards(&B)||!guards(&C))return 12;
  for(unsigned i=0;i<COUNT;i++)if(((volatile const int8_t*)A.data)[i]!=original_a[i]||((volatile const int8_t*)B.data)[i]!=original_b[i])return 13;
  if(before!=after)return 14;
  printf("RECTIFIER_COUNTER arm=%u repeat=%u cycles=%lu instructions=%lu before=%u after=%u\n",arm,repeat,(unsigned long)(c1-c0),(unsigned long)(i1-i0),before,after);
 }
 printf("RECTIFIER_PASS elements=%u repeats=2 full_outputs=exact inputs=immutable guards=pass\n",COUNT);return 0;
}
"""

    def values(data):
        return ",".join(map(str, np.frombuffer(data, dtype=np.int8)))

    replacements = {
        "ELEMENT_COUNT": str(count),
        "A_VALUES": values(a.tobytes()),
        "B_VALUES": values(b.tobytes()),
        "EXPECTED_VALUES": values(expected.tobytes()),
        "TABLE_COUNT": str(len(coefficient)),
        "OLD_TABLE_COUNT": str(len(old_coefficient)),
        "TABLE_VALUES": values(coefficient),
        "OLD_TABLE_VALUES": values(old_coefficient),
    }
    for key in sorted(replacements, key=len, reverse=True):
        c = c.replace(key, replacements[key])
    (work / "probe.c").write_text(c)
    header = args.core.resolve() / "merlin/runtime/c/benchmark_buffer.h"
    built = build_program(
        [work / "probe.c", work / "candidate/kernel.o", work / "control/kernel.o"],
        work / "build",
        target="gemmini",
        extra_cflags=[
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I",
            str(header.parent),
        ],
        max_loaded_bytes=None,
    )
    audit = audit_elf(built.elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("final executable uses forbidden target instruction")
    record = {
        "schema": "complete_affine_rectifier_pipeline_v1",
        "source_certificate": certificate,
        "control_certificate": control_proof,
        "plan": plan.attributes(),
        "candidate_compilation": candidate,
        "control_compilation": control,
        "elf": pin(built.elf),
        "loaded_bytes": built.loaded_bytes,
        "audit": audit,
        "source": pin(work / "probe.c"),
        "inputs": {
            name: pin(work / name)
            for name in (
                "a.bin",
                "b.bin",
                "expected.bin",
                "tables.bin",
                "control_tables.bin",
            )
        },
        "scope": "Both kernels complete producer, configs, readonly seed loads, actual predictor store/reload and exact correction to the same C addresses. Setup/fullchecks/guards outsideROI; no CPU correction. Independent numeric/capability proof required; no performance gate or whole-model selection enabled.",
    }
    (work / "built.json").write_text(json.dumps(record, indent=2) + "\n")
    print(built.elf, built.elf_sha256, built.loaded_bytes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--source-certificate", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--control-p", type=int, default=2609)
    p.add_argument("--control-q", type=int, default=2180)
    p.add_argument("--control-scale", type=float, default=0.00037060913746245205)
    generate(p.parse_args())
