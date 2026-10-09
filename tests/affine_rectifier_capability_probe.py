"""Tiny source-pinned execute-ACC-read/SPAD-write and seeded-ACC probe.

All intermediate destinations are private and disjoint. This first capability
experiment is separate from a residual implementation and contains no FSM op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from merlin.perf.layer_bench.build import build_program
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i64

from mlir_oot.codegen.builder import PTR, FnBuilder
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def pin(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def module():
    if F.DIM != 16 or F.OPERAND_DTYPE != "i8" or F.ACCUMULATOR_DTYPE != "i32":
        raise ValueError("probe requires pinned 16x16 signed-i8/i32 target")
    e = GoldenGemm(Shape(16, 16, 16))
    e.fb = FnBuilder([PTR] * 6)
    seed, identity, increment, positive, negative, full = e.fb.entry.args
    e._rocc("fence", {})
    e._rocc("flush", {})
    e._rocc("config_ex", {"dataflow": isa.WEIGHT_STATIONARY})
    e._rocc("config_ld", {"stride": 16, "load_id": 0})
    e._rocc("mvin", {"local": 4096, "rows": 16, "cols": 16, "load_id": 0}, identity)
    e._rocc("mvin", {"local": 0, "rows": 16, "cols": 16, "load_id": 0}, increment)
    e._rocc("config_ld", {"stride": 64, "load_id": 1, "shrunk": 0})
    e._rocc(
        "mvin", {"local": isa.acc_addr(0), "rows": 16, "cols": 16, "load_id": 1}, seed
    )
    e._rocc("config_st", {"stride": 64, "acc_act": isa.NO_ACTIVATION, "acc_scale": 1.0})
    e._rocc("fence", {})
    for scale, destination, pointer in ((1.0, 16, positive), (-1.0, 32, negative)):
        e._rocc(
            "config_ex",
            {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.RELU, "acc_scale": scale},
        )
        e._rocc(
            "preload",
            {
                "bd": 4096,
                "c": destination,
                "bd_cols": 16,
                "bd_rows": 16,
                "c_cols": 16,
                "c_rows": 16,
            },
        )
        e._rocc("compute", {"a": isa.acc_addr(0), "a_cols": 16, "a_rows": 16})
        e._rocc("fence", {})
        e._rocc("mvout", {"local": destination, "rows": 16, "cols": 16}, pointer)
        e._rocc("fence", {})
    e._rocc(
        "config_ex",
        {"dataflow": isa.WEIGHT_STATIONARY, "act": isa.NO_ACTIVATION, "acc_scale": 1.0},
    )
    e._rocc(
        "preload",
        {
            "bd": 4096,
            "c": isa.acc_addr(0, accumulate=True),
            "bd_cols": 16,
            "bd_rows": 16,
            "c_cols": 16,
            "c_rows": 16,
        },
    )
    e._rocc("compute", {"a": 0, "a_cols": 16, "a_rows": 16})
    e._rocc(
        "mvout", {"local": isa.acc_addr(0, full_row=True), "rows": 16, "cols": 16}, full
    )
    e._rocc("fence", {})
    fn = llvm.FuncOp(
        "affine_rectifier_capability",
        llvm.LLVMFunctionType([PTR] * 6),
        linkage=llvm.LinkageAttr("external"),
        body=e.fb.finish(),
    )
    result = ModuleOp([fn])
    result.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
    result.verify()
    return result


def generate(work):
    work.mkdir(parents=True, exist_ok=False)
    seed = [
        (
            -70000,
            -129,
            -128,
            -1,
            0,
            1,
            126,
            127,
            128,
            129,
            70000,
            27397,
            -27397,
            65535,
            -65535,
            33,
        )[i % 16]
        for i in range(256)
    ]
    increment = [(i * 7) % 19 - 9 for i in range(256)]
    source = r"""#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "benchmark_buffer.h"
extern void affine_rectifier_capability(const int32_t*,const int8_t*,const int8_t*,int8_t*,int8_t*,int32_t*);
static const int32_t seed[256] __attribute__((aligned(64)))={SEED};
static const int8_t identity[256] __attribute__((aligned(64)))={IDENTITY};
static const int8_t increment[256] __attribute__((aligned(64)))={INCREMENT};
struct box {uint8_t before[64];int8_t data[1024];uint8_t after[64];};
static struct box positive __attribute__((aligned(64))),negative __attribute__((aligned(64)));
struct fullbox {uint8_t before[64];int32_t data[256];uint8_t after[64];};
static struct fullbox full __attribute__((aligned(64)));
static uint64_t cycle(void){uint64_t v;asm volatile("csrr %0,mcycle":"=r"(v)::"memory");return v;}
static uint64_t inst(void){uint64_t v;asm volatile("csrr %0,minstret":"=r"(v)::"memory");return v;}
static int clipped(int32_t x){return x<0?0:x>127?127:x;}
int main(void){
 for(unsigned repeat=0;repeat<2;repeat++){
  merlin_benchmark_fill(&positive,0xdb,sizeof(positive));merlin_benchmark_fill(&negative,0xdb,sizeof(negative));merlin_benchmark_fill(&full,0xdb,sizeof(full));
  uint64_t c=cycle(),i=inst();affine_rectifier_capability(seed,identity,increment,positive.data,negative.data,full.data);uint64_t ie=inst(),ce=cycle();
  for(unsigned r=0;r<16;r++)for(unsigned k=0;k<64;k++){
   unsigned offset=r*64+k;
   if(k<16){unsigned j=r*16+k;if(positive.data[offset]!=clipped(seed[j])||negative.data[offset]!=clipped(-seed[j])||full.data[j]!=seed[j]+increment[j]){printf("CAP_FAIL r=%u k=%u pos=%d neg=%d full=%d\n",r,k,positive.data[offset],negative.data[offset],full.data[j]);return 11;}}
   else if((uint8_t)positive.data[offset]!=0xdb||(uint8_t)negative.data[offset]!=0xdb)return 12;
  }
  const struct box *boxes[]={&positive,&negative};for(unsigned b=0;b<2;b++)for(unsigned j=0;j<64;j++)if(boxes[b]->before[j]!=0xdb||boxes[b]->after[j]!=0xdb)return 13;
  for(unsigned j=0;j<64;j++)if(full.before[j]!=0xdb||full.after[j]!=0xdb)return 14;
  printf("CAP_COUNTER repeat=%u cycles=%lu instructions=%lu\n",repeat,(unsigned long)(ce-c),(unsigned long)(ie-i));
 }
 printf("CAP_PASS pairs=256 repeats=2 scaled_acc_reads=2 spad_writes=2 seeded_acc_add=1\n");return 0;
}
"""
    replacements = {
        "SEED": ",".join(map(str, seed)),
        "INCREMENT": ",".join(map(str, increment)),
        "IDENTITY": ",".join(
            "1" if r == c else "0" for r in range(16) for c in range(16)
        ),
    }
    for key, value in replacements.items():
        source = source.replace(key, value)
    (work / "probe.c").write_text(source)
    return seed, increment


def build(args):
    work = args.workdir.resolve()
    seed, increment = generate(work)
    core = args.core.resolve()
    header = core / "merlin/runtime/c/benchmark_buffer.h"
    stock = args.stock.resolve()
    relative = (
        "generators/firechip/chip/src/main/scala/TargetConfigs.scala",
        "generators/gemmini/chipyard/GemminiConfigs.scala",
        "generators/gemmini/src/main/scala/gemmini/Configs.scala",
        "generators/gemmini/src/main/scala/gemmini/GemminiConfigs.scala",
        "generators/gemmini/src/main/scala/gemmini/ExecuteController.scala",
        "generators/gemmini/src/main/scala/gemmini/Scratchpad.scala",
        "generators/gemmini/src/main/scala/gemmini/ReservationStation.scala",
    )
    capability = {
        "schema": "source_pinned_affine_rectifier_capability_v1",
        "config_chain": [
            "FireSimGemminiRocketConfig",
            "chipyard.GemminiRocketConfig",
            "gemmini.DefaultGemminiConfig",
            "GemminiConfigs.defaultConfig",
        ],
        "stock_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=stock, text=True
        ).strip(),
        "fields": {
            "acc_read_small_width": True,
            "ex_read_from_acc": True,
            "ex_write_to_spad": True,
            "ex_write_to_acc": True,
        },
        "path": "Execute ACC reads have full=false, CONFIG_EX acc_scale and activation; identity compute writes clipped/ReLU SPAD output",
        "store_spad_used": False,
        "source_pins": [pin(stock / path) for path in relative],
        "scope": "Source configuration chain and implementation witness; actual target path must pass this independent runtime probe. Bitstream capability closure still requires stock probe.",
    }
    (work / "capability_sources.json").write_text(
        json.dumps(capability, indent=2) + "\n"
    )
    compilation = compile_module(module(), args.llvm_bin, work / "device")
    built = build_program(
        [work / "probe.c", work / "device/kernel.o"],
        work / "build",
        target="gemmini",
        extra_cflags=[
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I",
            str(header.parent),
        ],
    )
    audit = audit_elf(built.elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("final zero-FSM audit failed")
    record = {
        "elf": pin(built.elf),
        "audit": audit,
        "device": compilation,
        "header": pin(header),
        "source": pin(work / "probe.c"),
        "capability": pin(work / "capability_sources.json"),
        "expected": {"seed": seed, "increment": increment},
        "numeric": "Exact source-derived integer seed + input; +/-1 scaled ACC reads and ReLU narrowing; every valid byte and guard checked",
    }
    (work / "built.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "elf": str(built.elf),
                "sha256": built.elf_sha256,
                "audit": audit["status"],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--stock", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    build(p.parse_args())
