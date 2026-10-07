"""Matched complete axis-offset versus predictor-key exact residual cost."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from merlin.llvmlower import quantized_affine_pair as pair
from merlin.llvmlower import quantized_affine_rectifier as rectifier
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_key_rectified_resadd import Capabilities, Plan, build
from mlir_oot.golden_rectified_resadd import Capabilities as ControlCapabilities
from mlir_oot.golden_rectified_resadd import Plan as ControlPlan
from mlir_oot.golden_rectified_resadd import build as control_build
from mlir_oot.golden_wide_resadd import build as wide_control_build
from mlir_oot.golden_wide_resadd import tables as wide_tables
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.spad_fence_coalescing import OrderingContract


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def generate(args):
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if args.source_certificate:
        bound = json.loads(args.source_certificate.read_text())
        bound = bound.get("source_certificate", bound)
        proof = pair.derive(**bound["source"], **bound["predictor"])
        max_pairs = bound["max_pairs"]
    else:
        # Explicit research fixture; production derivation consumes a complete
        # source proof and never matches these values or the fixture identity.
        proof = pair.derive(
            0.011258588172495365,
            0.00940733402967453,
            0.011643771082162857,
            p=298,
            q=249,
            scale=0.0032446938566863537,
            relu=True,
        )
        max_pairs = 1
    contract = OrderingContract(str(args.ordering_source.resolve()))
    base_cap = ControlCapabilities(*([True] * 6))
    if args.control_manifest:
        manifest = json.loads(args.control_manifest.read_text())
        routes = [
            r for r in manifest["routes"] if r["proof"]["source"] == proof["source"]
        ]
        if len(routes) != 1 or routes[0]["numeric_policy"]["max_output_lsb"] != 0:
            raise ValueError("one exact source-semantic control binding required")
        route = routes[0]
        coefficients = route["proof"]["coefficients"]
        exact = pair.derive(**proof["source"], **coefficients)
        if exact["mismatched_pairs"] != 0:
            raise ValueError(
                "control coefficients are not source-exact over the complete domain"
            )
        old = SimpleNamespace(
            tables=lambda: wide_tables(coefficients["p"], coefficients["q"]),
            attributes=lambda: route,
            certificate=exact,
        )
        old_module = wide_control_build(
            args.m * args.n // 64,
            coefficients["p"],
            coefficients["q"],
            coefficients["scale"],
            relu=proof["source"]["relu"],
            prefetch_m=route["device_schedule"]["prefetch_m"],
            banked_accumulators=route["device_schedule"]["banked_accumulators"],
        )
        next(iter(old_module.body.block.ops)).properties["sym_name"] = StringAttr(
            "gemmini_golden_rectified_resadd"
        )
    else:
        old = ControlPlan(
            args.m,
            args.n,
            rectifier.derive(
                proof, max_pairs=max_pairs, indicator_family="axis_offsets"
            ),
            base_cap,
        )
        old_module = control_build(
            old,
            coalesce_internal_spad=True,
            ordering_contract=contract,
            panel_batch=args.panel_batch,
        )
    new = Plan(
        args.m,
        args.n,
        rectifier.derive(proof, max_pairs=max_pairs, indicator_family="predictor_key"),
        Capabilities(base_cap, True),
        panel_batch=args.panel_batch,
        max_key_fibres=args.max_key_fibres,
    )
    count = args.m * args.n
    if args.inputs:
        a = np.frombuffer((args.inputs / "a.bin").read_bytes(), np.int8).copy()
        b = np.frombuffer((args.inputs / "b.bin").read_bytes(), np.int8).copy()
        if len(a) != count or len(b) != count:
            raise ValueError(
                "original input physical shape differs from declared geometry"
            )
    else:
        index = np.arange(count, dtype=np.int64) % 65536
        if count < 65536:
            index = (index * 73) % 65536
        a, b = (index // 256 - 128).astype(np.int8), (index % 256 - 128).astype(np.int8)
        if count < 65536:
            for i, relation in enumerate(new.certificate["relation"]):
                a[i], b[i] = relation["lhs"], relation["rhs"]
    expected = pair.source_table(**proof["source"])[
        a.astype(np.int16) + 128, b.astype(np.int16) + 128
    ]
    old_compile = compile_module(
        old_module,
        args.llvm_bin,
        work / "control",
    )
    new_compile = compile_module(
        build(new, ordering_contract=contract), args.llvm_bin, work / "candidate"
    )
    values = lambda data: ",".join(map(str, np.frombuffer(data, np.int8)))
    for name, data in (
        ("a.bin", a.tobytes()),
        ("b.bin", b.tobytes()),
        ("expected.bin", expected.tobytes()),
        ("old_tables.bin", old.tables()),
        ("new_tables.bin", new.tables()),
    ):
        (work / name).write_bytes(data)
    source = r"""#include <stdint.h>
#include <stdio.h>
#include "benchmark_buffer.h"
#define N ELEMENTS
extern void gemmini_golden_rectified_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*);
extern void gemmini_golden_key_rectified_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*,int8_t*);
static const int8_t original_a[N] __attribute__((aligned(64)))={AV};
static const int8_t original_b[N] __attribute__((aligned(64)))={BV};
static const int8_t expected[N] __attribute__((aligned(64)))={EV};
static const int8_t old_tables[OLDLEN] __attribute__((aligned(64)))={OV};
static const int8_t new_tables[NEWLEN] __attribute__((aligned(64)))={NV};
struct box{uint8_t before[64];int8_t data[N];uint8_t after[64];};
static struct box A __attribute__((aligned(64))),B __attribute__((aligned(64))),C __attribute__((aligned(64)));
static struct{uint8_t before[64];int8_t data[SCRATCH];uint8_t after[64];}S __attribute__((aligned(64)));
static uint64_t cycles(void){uint64_t x;asm volatile("csrr %0,mcycle":"=r"(x)::"memory");return x;}
static uint64_t instructions(void){uint64_t x;asm volatile("csrr %0,minstret":"=r"(x)::"memory");return x;}
static unsigned flags(void){unsigned x;asm volatile("csrr %0,fflags":"=r"(x)::"memory");return x;}
static int guards(const struct box*p){for(unsigned i=0;i<64;i++)if(p->before[i]!=0xa5||p->after[i]!=0xa5)return 0;return 1;}
int main(void){
 merlin_benchmark_fill(&A,0xa5,sizeof(A));merlin_benchmark_fill(&B,0xa5,sizeof(B));
 for(unsigned i=0;i<N;i++){A.data[i]=original_a[i];B.data[i]=original_b[i];}
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");
 for(unsigned repeat=0;repeat<2;repeat++)for(unsigned arm=0;arm<2;arm++){
  merlin_benchmark_fill(&C,0xa5,sizeof(C));merlin_benchmark_fill(&S,0xa5,sizeof(S));
  unsigned f0=flags();uint64_t c0=cycles(),i0=instructions();
  if(arm)gemmini_golden_key_rectified_resadd(A.data,B.data,C.data,new_tables,S.data);
  else gemmini_golden_rectified_resadd(A.data,B.data,C.data,old_tables);
  uint64_t i1=instructions(),c1=cycles();unsigned f1=flags();
  size_t diff=merlin_benchmark_first_difference(C.data,expected,N);
  if(diff!=N){printf("KEY_FAIL arm=%u index=%lu a=%d b=%d expected=%d actual=%d\n",arm,(unsigned long)diff,A.data[diff],B.data[diff],expected[diff],C.data[diff]);return 11;}
  if(!guards(&A)||!guards(&B)||!guards(&C))return 12;
  for(unsigned i=0;i<64;i++)if(S.before[i]!=0xa5||S.after[i]!=0xa5)return 13;
  if(merlin_benchmark_first_difference(A.data,original_a,N)!=N||merlin_benchmark_first_difference(B.data,original_b,N)!=N)return 14;
  if(f0!=f1)return 15;
  printf("KEY_COUNTER arm=%u repeat=%u cycles=%lu instructions=%lu before=%u after=%u\n",arm,repeat,(unsigned long)(c1-c0),(unsigned long)(i1-i0),f0,f1);
 }
 printf("KEY_PASS values=%u repeats=2 outputs=exact inputs=immutable guards=512\n",N);return 0;
}
"""
    for key, value in {
        "ELEMENTS": count,
        "AV": values(a.tobytes()),
        "BV": values(b.tobytes()),
        "EV": values(expected.tobytes()),
        "OLDLEN": len(old.tables()),
        "OV": values(old.tables()),
        "NEWLEN": len(new.tables()),
        "NV": values(new.tables()),
        "SCRATCH": new.scratch_bytes,
    }.items():
        source = source.replace(key, str(value))
    (work / "probe.c").write_text(source)
    header = args.core.resolve() / "merlin/runtime/c/benchmark_buffer.h"
    program = build_program(
        [work / "probe.c", work / "control/kernel.o", work / "candidate/kernel.o"],
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
    audit = audit_elf(program.elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("final executable no-FSM audit refused")
    record = {
        "schema": "predictor_key_rectifier_complete_pair_built_v1",
        "elf": pin(program.elf),
        "source": pin(work / "probe.c"),
        "driver": pin(__file__),
        "header": pin(header),
        "candidate": new.attributes(),
        "control": old.attributes(),
        "candidate_certificate": new.certificate,
        "control_certificate": old.certificate,
        "candidate_compile": new_compile,
        "control_compile": old_compile,
        "audit": audit,
        "stock_capability": "UNKNOWN pending actual hardware probe",
        "scope": "Both complete producer+stores+reloads+seed/config/fence/final publication at common A/B/C addresses; scratch private, all guards and every output checked outsideROI",
        "inputs": {
            name: pin(work / name)
            for name in (
                "a.bin",
                "b.bin",
                "expected.bin",
                "new_tables.bin",
                "old_tables.bin",
            )
        },
    }
    (work / "built.json").write_text(json.dumps(record, indent=2) + "\n")
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    run = subprocess.run(
        [
            str(spike),
            "--extension=gemmini",
            "--isa=rv64gc",
            "-m0x80000000:0x80000000",
            str(program.elf),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    (work / "spike.stdout").write_text(run.stdout)
    (work / "spike.stderr").write_text(run.stderr)
    marker = (
        f"KEY_PASS values={count} repeats=2 outputs=exact inputs=immutable guards=512"
    )
    if run.returncode != 0 or marker not in run.stdout:
        raise ValueError(
            "complete original ordered source/guards failed production Spike"
        )
    print("SPIKE_PASS", program.elf, flush=True)
    gsim = run_on_gsim(
        program.elf,
        target="gemmini",
        max_cycles=args.max_cycles,
        timeout_s=args.timeout,
        stdout_path=work / "gsim.stdout",
    )
    (work / "gsim.json").write_text(
        json.dumps(dataclasses.asdict(gsim), indent=2, default=str) + "\n"
    )
    print("GSIM_COMPLETE", gsim.completed, gsim.stdout_tail, flush=True)
    return 0 if gsim.completed and marker in gsim.stdout_tail else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workdir", "core", "llvm-bin", "ordering-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-certificate", type=Path)
    parser.add_argument("--control-manifest", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--panel-batch", type=int, choices=(1, 4), default=4)
    parser.add_argument("--max-key-fibres", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-cycles", type=int, default=3000000)
    parser.add_argument("--timeout", type=int, default=240)
    raise SystemExit(generate(parser.parse_args()))
