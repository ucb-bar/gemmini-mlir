"""Capability screen for exact identity-scale signed-i32 ACC DMA accumulation.

No ACC execute read, nonunit accumulator input scale, SPAD store or FSM is used.
This does not grant a model rewrite or price a correction network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def module():
    if F.DIM != 16 or F.ACCUMULATOR_DTYPE != "i32":
        raise ValueError("pinned DIM16/i32 accumulator required")
    emitter = GoldenGemm(Shape(16, 16, 16))
    emitter.fb = FnBuilder([PTR] * 3)
    seed, delta, output = emitter.fb.entry.args
    emitter._rocc("fence", {})
    emitter._rocc("flush", {})
    emitter._rocc("config_ld", {"stride": 64, "load_id": 1, "shrunk": 0, "scale": 1.0})
    emitter._rocc("mvin", {"local": isa.acc_addr(0), "rows": 16, "cols": 16, "load_id": 1}, seed)
    emitter._rocc("fence", {})
    emitter._rocc("mvin", {"local": isa.acc_addr(0, accumulate=True), "rows": 16, "cols": 16, "load_id": 1}, delta)
    emitter._rocc("fence", {})
    emitter._rocc("config_st", {"stride": 64, "acc_act": isa.NO_ACTIVATION, "acc_scale": 1.0})
    emitter._rocc("mvout", {"local": isa.acc_addr(0, full_row=True), "rows": 16, "cols": 16}, output)
    emitter._rocc("fence", {})
    function = llvm.FuncOp("acc_identity_rmw", llvm.LLVMFunctionType([PTR] * 3), linkage=llvm.LinkageAttr("external"), body=emitter.fb.finish())
    result = ModuleOp([function])
    result.attributes["gemmini.dim"] = IntegerAttr(16, i64)
    result.verify()
    return result


def generate(args):
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    seeds = [-70000, -33131, -129, -128, -1, 0, 1, 126, 127, 128, 129, 70000, 27397, -27397, 65535, -65535]
    left = [seeds[index % 16] for index in range(256)]
    right = [seeds[(index * 7 + index // 16 + 3) % 16] for index in range(256)]
    source = r'''#include <stdint.h>
#include <stdio.h>
#include "benchmark_buffer.h"
extern void acc_identity_rmw(const int32_t*,const int32_t*,int32_t*);
static const int32_t a[256] __attribute__((aligned(64)))={LEFT};
static const int32_t b[256] __attribute__((aligned(64)))={RIGHT};
struct box {uint8_t prefix[64];int32_t data[256];uint8_t suffix[64];};
static struct box output __attribute__((aligned(64)));
static uint64_t counter(void){uint64_t value;__asm__ volatile("csrr %0,mcycle":"=r"(value)::"memory");return value;}
int main(void){
 for(unsigned repeat=0;repeat<2;repeat++){
  merlin_benchmark_fill(&output,0xa5,sizeof(output));
  uint64_t before=counter();acc_identity_rmw(a,b,output.data);uint64_t after=counter();
  for(unsigned i=0;i<256;i++)if(output.data[i]!=a[i]+b[i]){printf("ACC_RMW_FAIL index=%u actual=%d expected=%d\n",i,output.data[i],a[i]+b[i]);return 11;}
  for(unsigned i=0;i<64;i++)if(output.prefix[i]!=0xa5||output.suffix[i]!=0xa5)return 12;
  printf("ACC_RMW_COUNTER repeat=%u cycles=%lu\n",repeat,(unsigned long)(after-before));
 }
 printf("ACC_RMW_PASS values=256 repeats=2 identity_scale=1 accumulator_rmw=1\n");return 0;
}
'''.replace("LEFT", ",".join(map(str, left))).replace("RIGHT", ",".join(map(str, right)))
    (work / "probe.c").write_text(source)
    compilation = compile_module(module(), args.llvm_bin, work / "device")
    header = args.core / "merlin/runtime/c/benchmark_buffer.h"
    program = build_program([work / "probe.c", work / "device/kernel.o"], work / "build", target="gemmini", extra_cflags=["-O2", "-fno-fast-math", "-ffp-contract=off", "-I", str(header.parent)])
    audit = audit_elf(program.elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("final no-FSM audit failed")
    result = {"schema": "acc_identity_rmw_capability_built_v1", "elf": pin(program.elf), "source": pin(work / "probe.c"), "driver": pin(__file__), "header": pin(header), "compilation": compilation, "audit": audit, "expected": [a + b for a, b in zip(left, right, strict=True)], "scope": "identity-scale DMA ACC read-modify-write, not ACC execute read or scaled MVIN; actual engine qualification required", "stock_support": "UNKNOWN"}
    (work / "built.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"elf": result["elf"], "audit": audit["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workdir", "core", "llvm-bin"):
        parser.add_argument("--" + name, type=Path, required=True)
    generate(parser.parse_args())
