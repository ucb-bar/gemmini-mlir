"""Real grouped xDSL products plus complete source PV/i8 cost comparison."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import StringAttr

from merlin.perf.layer_bench import build_program
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.no_fsm_audit import audit_elf
from source_integer_pv_native import HERE, OUT as NATIVE, CLANG, OLD, CORE, PROBES, sha, save

OUT = HERE / "out/artifacts/probes/source-integer-pv-target-v4-20261007"
GCC = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc")
ALLOCATOR = Path("/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/malloc.o")
CURRENT = PROBES / "closed-i8-interval-normal-2062-20261007/selected/target/host_llvm/expanded.ll"
OBS = PROBES / "dynamic-i8-attention-quant-observations-20261007"
LOCAL = PROBES / "dynamic-i8-attention-original-attribution-20261007"
CAP = PROBES / "tiny-attention-original-v-projection-capture-v2-20261007"


def source_body(text, symbol, public):
    start = text.index("define internal void @" + symbol + "(")
    end = text.index("\n}", start) + 2
    return text[start:end].replace(f"define internal void @{symbol}(", f"define void @{public}(", 1)


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    assert json.loads((NATIVE / "qualification.json").read_text())["status"] == "pass"
    directory = NATIVE / "context_00"
    # Immutable source context, exact producer constants and full original
    # compiled consumer are inherited from the independently qualified native.
    arrays = {name: np.load(directory / f"{name}.npy") for name in ("p", "code", "beta")}
    arrays.update(v=np.load(CAP / "context_00/v.npy"), original=np.load(OBS / "context_00/original.npy"),
                  candidate=np.load(directory / "candidate_i8.npy"))
    arrays.update({f"partial{g}": np.load(directory / f"group_{g}_reference.npy") for g in range(6)})
    data = ["#include <stdint.h>\n"]
    for name, array in arrays.items():
        if array.dtype == np.dtype("f4"):
            kind, words = "float", [float(x).hex()+"f" for x in array.reshape(-1)]
        elif array.dtype == np.dtype("i4"):
            kind, words = "int32_t", [str(int(x)) for x in array.reshape(-1)]
        else:
            assert array.dtype == np.dtype("i1")
            kind, words = "int8_t", [str(int(x)) for x in array.reshape(-1)]
        data.append(f"_Alignas(64) const {kind} data_{name}[{array.size}]={{"+",".join(words)+"};\n")
    (OUT / "data.c").write_text("".join(data))
    current = CURRENT.read_text()
    bodies = [source_body(current, "forward.extracted.112", "original_pv"),
              source_body(current, "forward.extracted.537", "original_quant")]
    tail = [line for line in current.splitlines() if line.startswith("attributes #") or line.startswith("!")
            or (line.startswith("declare ") and "@llvm." in line)]
    (OUT / "source.ll").write_text("\n".join(bodies+tail)+"\n")
    devices = []
    for reduction in (8, 16, 24):
        module = GoldenGemm(Shape(64, 64, reduction, output_dtype="i32", bm=4, bn=4,
                                 cache_a=True, wide_b=True)).build_batched(4)
        function = next(iter(module.ops))
        function.properties["sym_name"] = StringAttr(f"integer_product_{reduction}")
        compile_module(module, CLANG.parent, OUT / f"product_{reduction}")
        devices.append(OUT / f"product_{reduction}/kernel.o")
    # Target capability belongs here: the source observer's RNE clamp value is
    # separately qualified, with nontrapping/unobserved flags. No CPU instruction
    # spelling enters the generic executor or selects its source policy.
    cap = r'''#include <stdint.h>
static inline int8_t observer_rne(float x){float lo=-128.0f,hi=127.0f;long n;
 __asm__ volatile("fmax.s %0,%0,%2\n\tfmin.s %0,%0,%3\n\tfcvt.w.s %1,%0,rne":"+&f"(x),"=r"(n):"f"(lo),"f"(hi));return(int8_t)n;}
#define MERLIN_CLOSED_RNE_I8 observer_rne
#define MERLIN_SOURCE_BITCAST_COPY __builtin_memcpy
'''
    (OUT / "target_capability.h").write_text(cap)
    (OUT / "executor.c").write_bytes((directory / "executor.c").read_bytes())
    driver = r'''#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern const float data_p[2048],data_beta[256],data_v[16384];extern const int32_t data_code[2048];
extern const int32_t data_partial0[16384],data_partial1[16384],data_partial2[16384],data_partial3[16384],data_partial4[16384],data_partial5[16384];
extern const int8_t data_original[16384],data_candidate[16384];
extern void integer_product_8(const int8_t*,const int8_t*,int32_t*),integer_product_16(const int8_t*,const int8_t*,int32_t*),integer_product_24(const int8_t*,const int8_t*,int32_t*);
extern void original_pv(float*,const float*,const float*),original_quant(const float*,int8_t*);
typedef int(*product_callback)(void*,unsigned,size_t,size_t,size_t,size_t,const int8_t*,const int8_t*,int32_t*);
typedef struct{uint64_t certified,replayed,source_pairs,unadmitted_rows,product_calls,readout_words;}Counts;
extern size_t integer_observer_workspace_bytes(void),integer_observer_workspace_alignment(void);
extern int integer_observer(const float*,const int32_t*,const float*,int8_t*,void*,size_t,product_callback,void*,Counts*);
void htif_puts(const char*p){printf("%s",p);}void htif_puthex(unsigned long long v){printf("%lx",(unsigned long)v);}
void htif_putd(long v){printf("%ld",v);}void htif_putc(int v){printf("%c",v);}void htif_exit(int v){exit(v);}
static uint64_t ticks(void){uint64_t v;__asm__ volatile("csrr %0,mcycle":"=r"(v)::"memory");return v;}
static uint64_t instructions(void){uint64_t v;__asm__ volatile("csrr %0,minstret":"=r"(v)::"memory");return v;}
static void fence(void){__asm__ volatile("fence rw,rw":::"memory");}
static void mode(unsigned m,unsigned sticky){unsigned f=(m<<5)|sticky;__asm__ volatile("csrw fcsr,%0"::"r"(f):"memory");}
static uint64_t hash(const void*p,size_t n){const unsigned char*x=p;uint64_t h=1469598103934665603ULL;for(size_t i=0;i<n;i++)h=(h^x[i])*1099511628211ULL;return h;}
static uint64_t inputs(void){return hash(data_p,8192)^hash(data_code,8192)^hash(data_beta,1024)^hash(data_v,65536);}
static int check_partials=0,partial_failed=0;static uint64_t actual_readouts=0;
static int product(void*opaque,unsigned g,size_t h,size_t m,size_t k,size_t n,const int8_t*a,const int8_t*b,int32_t*c){
 if(h!=4||m!=64||n!=64)return 0;
 if(k==8)integer_product_8(a,b,c);else if(k==16)integer_product_16(a,b,c);else if(k==24)integer_product_24(a,b,c);else return 0;
 actual_readouts+=16384;
 if(check_partials){const int32_t*ref[6]={data_partial0,data_partial1,data_partial2,data_partial3,data_partial4,data_partial5};for(size_t i=0;i<16384;i++)if(c[i]!=ref[g][i]){partial_failed=1;return 0;}}
 return 1;
}
static _Alignas(64) float endpoint[16384];static _Alignas(64) int8_t output[16384+128];
static Counts count;static void*workspace;static size_t capacity;
static void control(void){for(size_t i=0;i<16384;i++)endpoint[i]=0.0f;original_pv(endpoint,data_p,data_v);original_quant(endpoint,output+64);}
static int candidate(void){return integer_observer(data_p,data_code,data_beta,output+64,workspace,capacity,product,0,&count);}
static int compare_bytes(const void*a,const void*b,size_t n){const unsigned char*x=a,*y=b;for(size_t i=0;i<n;i++)if(x[i]!=y[i])return 1;return 0;}
static int gate(void){return !compare_bytes(output+64,data_original,16384)&&!compare_bytes(output+64,data_candidate,16384);}
static int guards(void){for(size_t i=0;i<64;i++)if(output[i]!=73||output[16448+i]!=73)return 0;return 1;}
int main(void){capacity=integer_observer_workspace_bytes();workspace=malloc(capacity+64);if(!workspace||(uintptr_t)workspace%integer_observer_workspace_alignment())return 8;
 uint64_t pin=inputs();memset(output,73,sizeof(output));memset(workspace,0xa5,capacity);mode(0,0);check_partials=1;
 if(!candidate()||partial_failed||actual_readouts!=98304||!gate()||!guards()||inputs()!=pin)return 1;
 printf("INTEGER_PV_ORIGINAL_GATE i8=16384 readout=98304 dirtyguards PASS\n");
 check_partials=0;for(unsigned m=0;m<5;m++)for(unsigned sticky=0;sticky<2;sticky++){
  mode(m,sticky?31:0);control();static int8_t ref[16384];memcpy(ref,output+64,16384);
  mode(m,sticky?31:0);if(m==0){if(!candidate())return 2;}else control();
  if(compare_bytes(ref,output+64,16384)||!guards()||inputs()!=pin)return 3;
  printf("INTEGER_PV_MODE mode=%u sticky=%u same16384 PASS\n",m,sticky);
 }
 const unsigned order[4]={0,1,1,0};for(unsigned sample=0;sample<4;sample++){
  memset(output+64,73,16384);memset(workspace,0xa5,capacity);mode(0,0);fence();uint64_t t=ticks(),i=instructions();
  if(order[sample]){if(!candidate())return 4;}else control();fence();i=instructions()-i;t=ticks()-t;
  if(!gate()||!guards()||inputs()!=pin)return 5;
  printf("INTEGER_PV_ROW id=%u sample=%u cycles=%lu instructions=%lu digest=%lx\n",order[sample],sample,t,i,(unsigned long)hash(output+64,16384));
 }
 printf("INTEGER_PV_COUNTS certified=%lu replayed=%lu sourcepairs=%lu rows=%lu calls=%lu readouts=%lu workspace=%lu\n",count.certified,count.replayed,count.source_pairs,count.unadmitted_rows,count.product_calls,count.readout_words,capacity);
 mode(0,0);printf("INTEGER_PV_COMPLETE original16384i8 actual98304readouts inputguards all5modes PASS\n");return 0;}
'''
    (OUT / "main.c").write_text(driver)
    sysroot = subprocess.check_output([str(GCC), "-print-sysroot"], text=True).strip()
    flags = ["--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-builtin", "-ffp-contract=off", "--sysroot="+sysroot, "-isystem", sysroot+"/include",
             "-I", str(CORE / "merlin/runtime/c")]
    commands = []
    for name in ("executor.c", "data.c", "source.ll", "main.c"):
        source, obj = OUT / name, OUT / (Path(name).stem+".o")
        extra = ["-include", str(OUT / "target_capability.h")] if name == "executor.c" else []
        argv = [str(CLANG), *flags, *extra, "-MD", "-MF", str(obj.with_suffix(".d")), "-c", str(source), "-o", str(obj)]
        commands.append(argv)
        complete = subprocess.run(argv, capture_output=True, text=True)
        (OUT / (source.stem+".compile.stdout")).write_text(complete.stdout)
        (OUT / (source.stem+".compile.stderr")).write_text(complete.stderr)
        assert complete.returncode == 0, complete.stderr
    program = build_program([OUT / "executor.o", OUT / "data.o", OUT / "source.o", OUT / "main.o", *devices, ALLOCATOR],
                            OUT / "build", target="gemmini", max_loaded_bytes=None)
    audit = audit_elf(program.elf.read_bytes());save(OUT / "nofsm.json", audit);assert audit["status"] == "pass"
    argv = [str(GCC.with_name("spike")), "--isa=rv64gc", "--extension=gemmini", "-m0x80000000:0x400000000", str(program.elf)]
    commands.append(argv)
    complete = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    (OUT / "spike.stdout").write_text(complete.stdout);(OUT / "spike.stderr").write_text(complete.stderr)
    record = {"schema": "exact_source_integer_pv_complete_target_v1", "status": "pass" if complete.returncode==0 and "INTEGER_PV_COMPLETE" in complete.stdout else "fail",
              "elf_path": str(program.elf), "elf_sha256": sha(program.elf), "zeroFSM": True, "commands": commands,
              "native_all22_compiled_i8_reference": str(NATIVE / "qualification.json"), "source_context": 0,
              "actual_device_integer_readouts_exact": 98304, "original_i8_words_exact": 16384,
              "cost_scope": "Both completePV/quantizer output paths; candidate includes lattice+V prep, packing,6batched products/24KVproducts,384KiB readout,i64reconstruction,bounds/refusal/replay/finish/frame/store. Original P/floatV and integerV source views already exist; preceding QK/softmax/projection/GQA views outside. Workspace allocation/dirty poison outside both windows; reuse owned storage is explicit.",
              "cycles": "UNKNOWN: strict Spike only", "whole_model": "No route installed/no whole gate", "token_usage_available": False,
              "pins": {str(path):sha(path) for path in [Path(__file__), NATIVE / "qualification.json", CURRENT, CLANG, GCC, GCC.with_name("spike"), ALLOCATOR,
                         *[path for path in OUT.rglob("*") if path.is_file()]]}}
    save(OUT / "qualification.json", record)
    print(complete.stdout, flush=True)
    assert record["status"] == "pass", complete.stderr


if __name__ == "__main__":main()
