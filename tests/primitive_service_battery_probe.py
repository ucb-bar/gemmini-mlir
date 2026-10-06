"""Generate legal operational Gemmini primitive-service calibration streams."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlir_oot.tables import rtl_facts as F

DIM = F.DIM
N = K = 64
MAXM = 128
PANELS = 256
BBASE = 2 * F.SPAD_BANK_ROWS
MASK64 = (1 << 64) - 1


def aval(m, k):
    return (m * 7 + k * 3) % 13 - 6


def bval(k, n):
    return (k * 5 + n * 11) % 17 - 8


def generate(work):
    work.mkdir(parents=True, exist_ok=False)
    assert DIM == 16 and BBASE + K * N // DIM <= F.SPAD_ROWS
    assert MAXM * K // DIM <= BBASE and MAXM * N // DIM <= F.ACC_ROWS
    assert PANELS * DIM <= F.SPAD_ROWS
    expected = [
        sum(aval(m, k) * bval(k, n) for k in range(K))
        for m in range(MAXM)
        for n in range(N)
    ]
    tile = [
        sum(aval(m, k) * bval(k, n) for k in range(DIM))
        for m in range(DIM)
        for n in range(DIM)
    ]
    assert max(map(abs, expected)) <= K * 6 * 8 < 1 << 31
    loadbytes = bytes((i * 17 + 11) & 255 for i in range(PANELS * DIM * DIM))
    cases = []
    for m in (32, 64, 128):
        commands = (m // DIM) * (N // DIM) * (K // DIM)
        cases.append(
            {
                "id": len(cases),
                "family": "resident_gemm",
                "size": m,
                "kernel": "resident_gemm",
                "partition": "heldout" if m == 64 else "training",
                "M": m,
                "N": N,
                "K": K,
                "dtype": "signed_i8_to_exact_i32",
                "a_spad_rows": m * K // DIM,
                "b_spad_interval": [BBASE, BBASE + K * N // DIM],
                "acc_rows": m * N // DIM,
                "roi_primitive_commands": {
                    "PRELOAD_CMD": commands,
                    "COMPUTE_AND_FLIP_CMD": commands,
                },
                "roi_requested_dma_load_bytes": 0,
                "roi_requested_dma_store_bytes": 0,
                "array_work_padded_rows": commands * DIM,
                "preparation_load_commands": (m // DIM) * (K // DIM)
                + (K // DIM) * (N // DIM),
                "verification_store_commands": (m // DIM) * (N // DIM),
                "expected_fflags": 0,
            }
        )
    for family in ("requested_load", "i32_readback"):
        for count in (16, 64, 256):
            cases.append(
                {
                    "id": len(cases),
                    "family": family,
                    "size": count,
                    "kernel": family,
                    "partition": "heldout" if count == 64 else "training",
                    "roi_primitive_commands": {
                        "LOAD_CMD" if family == "requested_load" else "STORE_CMD": count
                    },
                    "roi_requested_dma_load_bytes": count * DIM * DIM
                    if family == "requested_load"
                    else 0,
                    "roi_requested_dma_store_bytes": count * DIM * DIM * 4
                    if family == "i32_readback"
                    else 0,
                    "array_work_padded_rows": 0,
                    "live_spad_interval": [0, count * DIM]
                    if family == "requested_load"
                    else None,
                    "live_acc_interval": [0, DIM] if family == "i32_readback" else None,
                    "verification_store_commands": count
                    if family == "requested_load"
                    else 0,
                    "expected_fflags": 0,
                }
            )
    checksum = 14695981039346656037

    def mix(value):
        nonlocal checksum
        checksum = ((checksum ^ value) * 1099511628211) & MASK64

    for case in cases:
        for _ in range(2):
            if case["family"] == "resident_gemm":
                for value in expected[: case["M"] * N]:
                    mix(value & 0xFFFFFFFF)
            elif case["family"] == "requested_load":
                data = loadbytes[: case["size"] * DIM * DIM]
                for i in range(0, len(data), 8):
                    mix(int.from_bytes(data[i : i + 8], "little"))
            else:
                for _ in range(case["size"]):
                    for value in tile:
                        mix(value & 0xFFFFFFFF)
    manifest = {
        "schema": "gemmini_primitive_service_battery_v1",
        "cases": cases,
        "repetitions": 2,
        "empty_windows": 3,
        "expected_checksum": f"{checksum:016x}",
        "resources": {
            "DIM": DIM,
            "SPAD_ROWS": F.SPAD_ROWS,
            "SPAD_BANK_ROWS": F.SPAD_BANK_ROWS,
            "ACC_ROWS": F.ACC_ROWS,
            "BBASE": BBASE,
        },
        "scope": "Operational complete-finish primitive streams. Configs and actual completed DMA input/ACC preparation outsideROI; verification/readback outsidecompute and loadROIs. Shared fenced mcycle/minstret helper includes issue/call/completion stalls. Not pure rates.",
        "partition": "MiddleM64 and64DMAcommands heldout with all repeats. No model or Jack timing fitted.",
        "physical_dram_bytes": "UNKNOWN; requested payload counts only",
        "overlap": "No DMA in resident computeROI; CPUissue and accelerator service overlap UNKNOWN",
        "expected_checksum_policy": "word-wise FNV64 fold supplements independent full byte comparisons and source/guard/tail checks",
        "dependency_proof": "All A/B tiles disjoint and resident before computeROI; firstK overwrites privateACC tile, subsequent increasingK accumulates; noACC reuse through completion. ReadbackACC computed/fenced beforeROI and immutable across stores. Loadpanels have disjointSPAD rows; disjoint resident identityB prepared/fenced outsideROI; exactidentityGEMM/rawi32/store/fence verification follows completion for everypanel.",
        "input_validation": "Every source byte checked once after all windows through volatile CPU loads; full outputs/guards/logical tails checked each repetition; primitive producers only read immutable inputs and write disjoint private outputs",
        "address_contract": "Bare-metal physical DMA pointers; globals separately64B aligned; all destination ranges private, inputs immutable; no buffers overlap.",
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    table = ",".join(
        "{"
        + ",".join(
            (
                case["kernel"],
                str(
                    0
                    if case["family"] == "resident_gemm"
                    else 1
                    if case["family"] == "requested_load"
                    else 2
                ),
                str(case["size"]),
            )
        )
        + "}"
        for case in cases
    )
    source = r"""#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini.h"
#include "benchmark_buffer.h"
#define NOINLINE __attribute__((noinline,noclone))
#define NN 64
#define KK 64
#define MAXM 128
#define PANELS 256
#define BBASE B_BASE
#define ACCBASE UINT32_C(0x80000000)
#define ACCUM UINT32_C(0x40000000)
#define FULL UINT32_C(0x20000000)
_Static_assert(DIM==16 && ADDR_LEN==32 && BANK_NUM==4 && BANK_ROWS==4096 && ACC_ROWS==1024,"pinned target geometry required");
_Static_assert(sizeof(elem_t)==1 && sizeof(acc_t)==4,"i8/i32 target required");
_Static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"checksum byte ABI");
static const elem_t A[MAXM*KK] __attribute__((aligned(64)))={A_VALUES};
static const elem_t B[KK*NN] __attribute__((aligned(64)))={B_VALUES};
static const int32_t expected[MAXM*NN] __attribute__((aligned(64)))={EXPECTED};
static const int32_t expected_tile[DIM*DIM] __attribute__((aligned(64)))={EXPECTED_TILE};
static const elem_t source_panels[PANELS*DIM*DIM] __attribute__((aligned(64)))={LOAD_VALUES};
static const elem_t identity[DIM*DIM] __attribute__((aligned(64)))={IDENTITY_VALUES};
struct i32box {uint64_t before[8];int32_t data[PANELS*DIM*DIM];uint64_t after[8];};
static struct i32box output __attribute__((aligned(64)));
struct context {uint32_t count;};static struct context ctx;
static uint64_t checksum=UINT64_C(14695981039346656037);
static void digest(uint64_t v){checksum^=v;checksum*=UINT64_C(1099511628211);}
static int guard_ok(uint64_t*a){for(int i=0;i<8;i++)if(a[i]!=UINT64_C(0x5a5a5a5a5a5a5a5a))return 0;return 1;}
static int sources_ok(void){const volatile elem_t*a=A,*b=B,*panels=source_panels;for(int m=0;m<MAXM;m++)for(int k=0;k<KK;k++)if(a[m*KK+k]!=(m*7+k*3)%13-6)return 0;for(int k=0;k<KK;k++)for(int n=0;n<NN;n++)if(b[k*NN+n]!=(k*5+n*11)%17-8)return 0;for(int i=0;i<PANELS*DIM*DIM;i++)if((uint8_t)panels[i]!=(uint8_t)(i*17+11))return 0;return 1;}
static void prepare_gemm(uint32_t m){
 gemmini_config_ex(WS,NO_ACTIVATION,0);gemmini_config_ld(KK);
 for(uint32_t row=0;row<m;row+=DIM)for(uint32_t k=0;k<KK;k+=DIM)gemmini_extended_mvin(A+row*KK+k,((row/DIM)*(KK/DIM)+k/DIM)*DIM,DIM,DIM);
 gemmini_config_ld(NN);for(uint32_t k=0;k<KK;k+=DIM)for(uint32_t col=0;col<NN;col+=DIM)gemmini_extended_mvin(B+k*NN+col,BBASE+((k/DIM)*(NN/DIM)+col/DIM)*DIM,DIM,DIM);
 gemmini_fence();
}
static NOINLINE void resident_gemm(struct context*c){
 for(uint32_t row=0;row<c->count;row+=DIM)for(uint32_t col=0;col<NN;col+=DIM)for(uint32_t k=0;k<KK;k+=DIM){
  uint32_t a=((row/DIM)*(KK/DIM)+k/DIM)*DIM;
  uint32_t b=BBASE+((k/DIM)*(NN/DIM)+col/DIM)*DIM;
  uint32_t acc=ACCBASE|((row/DIM)*(NN/DIM)+col/DIM)*DIM|(k?ACCUM:0);
  gemmini_preload(b,acc);gemmini_compute_preloaded(a,GARBAGE_ADDR);
 }
}
static NOINLINE void requested_load(struct context*c){for(uint32_t i=0;i<c->count;i++)gemmini_mvin(source_panels+i*DIM*DIM,i*DIM);}
static NOINLINE void i32_readback(struct context*c){for(uint32_t i=0;i<c->count;i++)gemmini_mvout(output.data+i*DIM*DIM,ACCBASE|FULL);}
static NOINLINE void empty(struct context*c){(void)c;__asm__ volatile("":::"memory");}
typedef void (*kernel_t)(struct context*);
struct counters {uint32_t cycles,instructions,frm,flags_before,flags_after,overflow;};
static NOINLINE struct counters measure(kernel_t fn,struct context*c){
 uint64_t begin,end,ibegin,iend,frm,before,after;
 __asm__ volatile("csrwi fflags,0\ncsrr %0,frm\ncsrr %1,fflags":"=r"(frm),"=r"(before)::"memory");
 __asm__ volatile("fence rw,rw\ncsrr %0,mcycle\ncsrr %1,minstret":"=r"(begin),"=r"(ibegin)::"memory");fn(c);
 __asm__ volatile("fence rw,rw\ncsrr %0,minstret\ncsrr %1,mcycle":"=r"(iend),"=r"(end)::"memory");
 __asm__ volatile("csrr %0,fflags":"=r"(after)::"memory");
 struct counters result={(uint32_t)(end-begin),(uint32_t)(iend-ibegin),(uint32_t)frm,(uint32_t)before,(uint32_t)after,(end<begin)||(iend<ibegin)||((end-begin)>UINT32_MAX)||((iend-ibegin)>UINT32_MAX)};return result;
}
static const struct {kernel_t fn;uint32_t family,size;} cases[9]={CASE_TABLE};
static int report(int id,int repeat,struct counters r){if(r.overflow||r.frm||r.flags_before||r.flags_after)return 1;printf("SERVICE_ROW id=%d repeat=%d cycles=%u instructions=%u frm=%u flags_before=%u flags_after=%u\n",id,repeat,r.cycles,r.instructions,r.frm,r.flags_before,r.flags_after);return 0;}
int main(void){
 __asm__ volatile("csrwi frm,0":::"memory");gemmini_flush(0);gemmini_fence();
 for(int repeat=0;repeat<3;repeat++)if(report(-1,repeat,measure(empty,&ctx)))return 10;
 for(int id=0;id<9;id++)for(int repeat=0;repeat<2;repeat++){
  ctx.count=cases[id].size;
  merlin_benchmark_fill(output.before,0x5a,sizeof(output.before));merlin_benchmark_fill(output.after,0x5a,sizeof(output.after));
  merlin_benchmark_fill(output.data,0xdb,sizeof(output.data));
  if(cases[id].family==0)prepare_gemm(ctx.count);
  else if(cases[id].family==1){gemmini_config_ex(WS,NO_ACTIVATION,0);gemmini_config_ld(DIM);gemmini_mvin(identity,BBASE);gemmini_fence();}
  else{gemmini_config_ex(WS,NO_ACTIVATION,0);gemmini_config_ld(KK);gemmini_mvin(A,0);gemmini_config_ld(NN);gemmini_mvin(B,BBASE);gemmini_preload(BBASE,ACCBASE);gemmini_compute_preloaded(0,GARBAGE_ADDR);gemmini_fence();gemmini_config_st(DIM*sizeof(acc_t));gemmini_fence();}
  struct counters r=measure(cases[id].fn,&ctx);
  uint32_t bytes=0;
  if(cases[id].family==0){
   gemmini_config_st(NN*sizeof(acc_t));for(uint32_t row=0;row<ctx.count;row+=DIM)for(uint32_t col=0;col<NN;col+=DIM)gemmini_mvout(output.data+row*NN+col,ACCBASE|FULL|((row/DIM)*(NN/DIM)+col/DIM)*DIM);gemmini_fence();
   bytes=ctx.count*NN*sizeof(acc_t);if(merlin_benchmark_first_difference(output.data,expected,bytes)!=bytes)return 11;
   for(uint32_t i=0;i<ctx.count*NN;i++)digest((uint32_t)output.data[i]);
  }else if(cases[id].family==1){
   gemmini_config_st(DIM*sizeof(acc_t));for(uint32_t panel=0;panel<ctx.count;panel++){gemmini_preload(BBASE,ACCBASE);gemmini_compute_preloaded(panel*DIM,GARBAGE_ADDR);gemmini_mvout(output.data+panel*DIM*DIM,ACCBASE|FULL);gemmini_fence();}
   uint32_t elements=ctx.count*DIM*DIM;bytes=elements*sizeof(acc_t);for(uint32_t i=0;i<elements;i++)if(output.data[i]!=(int32_t)source_panels[i])return 12;
   for(uint32_t i=0;i<elements;i+=8){uint64_t value=0;for(uint32_t j=0;j<8;j++)value|=(uint64_t)(uint8_t)output.data[i+j]<<(j*8);digest(value);}
  }else{bytes=ctx.count*DIM*DIM*sizeof(acc_t);for(uint32_t panel=0;panel<ctx.count;panel++)if(merlin_benchmark_first_difference(output.data+panel*DIM*DIM,expected_tile,DIM*DIM*sizeof(acc_t))!=DIM*DIM*sizeof(acc_t))return 13;for(uint32_t i=0;i<ctx.count*DIM*DIM;i++)digest((uint32_t)output.data[i]);}
  uint8_t *base=(uint8_t*)output.data;uint32_t total=sizeof(output.data);for(uint32_t i=bytes;i<total;i+=8){uint64_t value;__builtin_memcpy(&value,base+i,sizeof(value));if(value!=UINT64_C(0xdbdbdbdbdbdbdbdb))return 14;}
  if(!guard_ok(output.before)||!guard_ok(output.after))return 15;
  if(report(id,repeat,r))return 16;
 }
 if(!sources_ok())return 18;
 if(checksum!=UINT64_C(EXPECTED_CHECKSUM))return 17;
 printf("SERVICE_PASS cases=9 repeats=2 checksum=%08x%08x\n",(uint32_t)(checksum>>32),(uint32_t)checksum);return 0;
}
"""

    def values(seq):
        return ",".join(map(str, seq))

    replacements = {}
    for key, value in {
        "B_BASE": str(BBASE),
        "IDENTITY_VALUES": values(
            1 if row == col else 0 for row in range(DIM) for col in range(DIM)
        ),
        "A_VALUES": values(aval(m, k) for m in range(MAXM) for k in range(K)),
        "B_VALUES": values(bval(k, n) for k in range(K) for n in range(N)),
        "EXPECTED_TILE": values(tile),
        "EXPECTED": values(expected),
        "LOAD_VALUES": values(b if b < 128 else b - 256 for b in loadbytes),
        "CASE_TABLE": table,
        "EXPECTED_CHECKSUM": str(checksum),
    }.items():
        replacements[key] = value
    for key in sorted(replacements, key=len, reverse=True):
        source = source.replace(key, replacements[key])
    (work / "battery.c").write_text(source)
    shutil.copyfile(
        "/scratch/agustin/tmp/merlin-calibration-battery-frozen-main-20261006/merlin/runtime/c/benchmark_buffer.h",
        work / "benchmark_buffer.h",
    )
    print("PRIMITIVE_GENERATED", len(cases), f"{checksum:016x}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    args = p.parse_args()
    generate(args.workdir.resolve())
