"""Independent nonsquare, multi-column-panel exact rectifier qualification."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
from affine_rectifier_capability_probe import pin
from merlin.llvmlower.quantized_affine_pair import derive, source_table
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize
from merlin.perf.layer_bench import build_program, run_on_gsim

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_rectified_resadd import Capabilities, Plan, build
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.spad_fence_coalescing import OrderingContract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--ordering-source", type=Path)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    bound = json.loads(args.certificate.read_text())["source_certificate"]
    proof = derive(**bound["source"], **bound["predictor"])
    certificate = synthesize(proof, max_pairs=1, indicator_family="axis_offsets")
    plan = Plan(48, 128, certificate, Capabilities(*([True] * 6)))
    count = plan.m * plan.n
    index = np.arange(count, dtype=np.int32)
    a = ((index * 73 + 3) % 256 - 128).astype(np.int8)
    b = ((index * 151 + 19) % 256 - 128).astype(np.int8)
    for i, row in enumerate(certificate["relation"]):
        a[i], b[i] = row["lhs"], row["rhs"]
    expected = source_table(**proof["source"])[
        a.astype(np.int16) + 128, b.astype(np.int16) + 128
    ]
    compilation = compile_module(
        build(
            plan,
            coalesce_internal_spad=bool(args.ordering_source),
            ordering_contract=OrderingContract(str(args.ordering_source))
            if args.ordering_source
            else None,
        ),
        args.llvm_bin,
        work / "device",
    )
    values = lambda data: ",".join(map(str, np.frombuffer(data, dtype=np.int8)))
    source = work / "probe.c"
    source.write_text(f"""#include <stdint.h>
#include <stdio.h>
#include "benchmark_buffer.h"
#define N {count}
extern void gemmini_golden_rectified_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*);
static const int8_t a[N] __attribute__((aligned(64)))={{{values(a.tobytes())}}};
static const int8_t b[N] __attribute__((aligned(64)))={{{values(b.tobytes())}}};
static const int8_t expected[N] __attribute__((aligned(64)))={{{values(expected.tobytes())}}};
static const int8_t tables[{len(plan.tables())}] __attribute__((aligned(64)))={{{values(plan.tables())}}};
static struct{{uint8_t before[64];int8_t output[N];uint8_t after[64];}}r __attribute__((aligned(64)));
int main(void){{merlin_benchmark_fill(&r,0xdb,sizeof(r));gemmini_golden_rectified_resadd(a,b,r.output,tables);
 size_t first=merlin_benchmark_first_difference(r.output,expected,N);
 if(first!=N){{printf("INDEPENDENT_FAIL index%lu expected%d actual%d\\n",(unsigned long)first,expected[first],r.output[first]);return 1;}}
 for(unsigned i=0;i<64;i++)if(r.before[i]!=0xdb||r.after[i]!=0xdb)return 2;
 printf("RECTIFIER_INDEPENDENT_PASS M48 N128 all%d guards128\\n",N);return 0;}}
""")
    header = args.core / "merlin/runtime/c/benchmark_buffer.h"
    built = build_program(
        [source, work / "device/kernel.o"],
        work / "build",
        target="gemmini",
        extra_cflags=[
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(header.parent),
        ],
        max_loaded_bytes=None,
    )
    audit = audit_elf(built.elf.read_bytes())
    assert audit["status"] == "pass"
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    run = subprocess.run(
        [str(spike), "--extension=gemmini", "--isa=rv64gc", str(built.elf)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    (work / "spike.stdout").write_text(run.stdout)
    (work / "spike.stderr").write_text(run.stderr)
    assert (
        run.returncode == 0
        and "RECTIFIER_INDEPENDENT_PASS M48 N128 all6144 guards128" in run.stdout
    )
    gsim = run_on_gsim(
        built.elf,
        target="gemmini",
        max_cycles=1000000,
        timeout_s=160,
        stdout_path=work / "gsim.stdout",
    )
    (work / "gsim.json").write_text(
        json.dumps(dataclasses.asdict(gsim), indent=2, default=str) + "\n"
    )
    passed = (
        gsim.completed
        and gsim.returncode == 0
        and "RECTIFIER_INDEPENDENT_PASS M48 N128 all6144 guards128" in gsim.stdout_tail
    )
    record = {
        "schema": "affine_rectifier_independent_shape_v1",
        "passed": passed,
        "plan": plan.attributes(),
        "certificate": certificate,
        "compilation": compilation,
        "audit": audit,
        "elf": pin(built.elf),
        "inputs": [pin(p) for p in (args.certificate, source, header, spike)],
        "scope": "Complete nonsquare multi-column-panel source-exact output and all output guards; no cycle ranking or whole-model gate.",
    }
    (work / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print("INDEPENDENT_COMPLETE", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
