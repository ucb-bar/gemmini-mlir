"""Generate a workload-independent RV64GC service calibration battery."""

import argparse
import ctypes
import hashlib
import json
import math
import shutil
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def generate(work):
    work.mkdir(parents=True, exist_ok=False)
    seed = [np.float32(1.1234567 + i * 0.0314159) for i in range(8)]
    coefficient = np.float32(1 + 2**-14)
    bias = np.float32(0.00012345679)
    libm = ctypes.CDLL("libm.so.6")
    libm.fmaf.argtypes = [ctypes.c_float] * 3
    libm.fmaf.restype = ctypes.c_float
    cases = []
    for op in ("div", "fma"):
        for independent in (False, True):
            for count in (1024, 4096, 16384):
                output = list(seed)
                inexact_operations = 0
                min_value, max_value = min(map(float, seed)), max(map(float, seed))
                for index in range(count):
                    lane = index % 8 if independent else 0
                    previous = output[lane]
                    exact = (
                        Fraction(float(previous)) / Fraction(float(coefficient))
                        if op == "div"
                        else Fraction(float(previous)) * Fraction(float(coefficient))
                        + Fraction(float(bias))
                    )
                    if op == "div":
                        output[lane] = np.float32(output[lane] / coefficient)
                    else:
                        output[lane] = np.float32(
                            libm.fmaf(
                                float(output[lane]), float(coefficient), float(bias)
                            )
                        )
                    assert (
                        math.isfinite(float(output[lane]))
                        and 2**-126 < float(output[lane]) < 2**127
                    )
                    assert output[lane] != previous, (
                        "recurrence must not settle to a fixed point"
                    )
                    encoding = bits(output[lane])
                    center = Fraction(float(output[lane]))
                    lower = Fraction(
                        struct.unpack("<f", struct.pack("<I", encoding - 1))[0]
                    )
                    upper = Fraction(
                        struct.unpack("<f", struct.pack("<I", encoding + 1))[0]
                    )
                    lower_midpoint, upper_midpoint = (
                        (lower + center) / 2,
                        (upper + center) / 2,
                    )
                    assert lower_midpoint <= exact <= upper_midpoint
                    assert (
                        exact not in (lower_midpoint, upper_midpoint)
                        or encoding % 2 == 0
                    )
                    inexact_operations += Fraction(float(output[lane])) != exact
                    min_value = min(min_value, float(output[lane]), float(exact))
                    max_value = max(max_value, float(output[lane]), float(exact))
                assert inexact_operations > 0
                cases.append(
                    {
                        "id": len(cases),
                        "family": op,
                        "independent_lanes": 8 if independent else 1,
                        "size": count,
                        "partition": "heldout" if count == 4096 else "training",
                        "initial_words": list(map(bits, seed)),
                        "expected_words": list(map(bits, output)),
                        "coefficient_word": bits(coefficient),
                        "bias_word": bits(bias),
                        "kernel": op + ("_ind" if independent else "_dep"),
                        "expected_fflags": 1,
                        "exact_rational_inexact_operations": inexact_operations,
                        "exact_and_rounded_value_range": [min_value, max_value],
                        "no_fixed_point": True,
                    }
                )
    for stride in (1, 17):
        for count in (128, 512, 2048):
            cases.append(
                {
                    "id": len(cases),
                    "family": "memory",
                    "size": count,
                    "stride": stride,
                    "partition": "heldout" if count == 512 else "training",
                    "kernel": "memory",
                    "requested_cpu_load_bytes": count * 8,
                    "requested_cpu_store_bytes": count * 8,
                    "source_and_destination_extent_bytes": ((count - 1) * stride + 1)
                    * 8,
                }
            )
    for body_bytes in (4096, 16384, 65536):
        width = body_bytes // 4
        delta = sum((i * 5) % 31 - 15 for i in range(width)) * (16384 // width)
        cases.append(
            {
                "id": len(cases),
                "family": "footprint",
                "size": body_bytes,
                "partition": "heldout" if body_bytes == 16384 else "training",
                "kernel": f"footprint_{width}",
                "body_instructions": width,
                "total_add_instructions": 16384,
                "repeats": 16384 // width,
                "expected_word": (0x123456789ABCDEF0 + delta) & ((1 << 64) - 1),
            }
        )
    manifest = {
        "schema": "rv64gc_service_battery_v1",
        "cases": cases,
        "repetitions": 2,
        "empty_windows": 3,
        "independent_expected": "binary32 NumPy DIV and native libm fmaf, exact word comparisons; independent integer formula and full memory/guards",
        "partition_rule": "middle sizes held out together with both lane arms and repeats; small and large train",
        "timing": "Same fenced mcycle/minstret window; setup/validation/checksum/UART outside ROI",
        "memory_regime": "CPU initializes complete extents before each repeat. Cache warmth/misses unknown; no cold-cache claim.",
        "flags": "Explicit RNE per FP instruction; actual FRM/fflags bound in every row. Exact rational recurrences prove inexact flag and finite normal nonconstant values; expected flags are NX=1 for FP and zero otherwise.",
        "overlap": "No accelerator activity; CPU and accelerator overlap UNKNOWN",
        "checksum": "FNV64 of every checked FP result word, memory result word, and footprint word, in fixed case/repeat order; checksum supplements full exact checks",
        "fitting": "No Jack or model timing labels used",
    }
    digest_value = 14695981039346656037

    def mix(value):
        nonlocal digest_value
        for i in range(8):
            digest_value = (
                (digest_value ^ ((value >> (8 * i)) & 255)) * 1099511628211
            ) & ((1 << 64) - 1)

    for case in cases:
        for _ in range(2):
            if case["family"] in ("div", "fma"):
                for value in case["expected_words"]:
                    mix(value)
            elif case["family"] == "memory":
                extent = (case["size"] - 1) * case["stride"] + 1
                for i in range(extent + 8):
                    value = 0xDBDBDBDBDBDBDBDB
                    if i % case["stride"] == 0 and i // case["stride"] < case["size"]:
                        value = (
                            ((i * 2862933555777941757 + 3037000493) & ((1 << 64) - 1))
                            ^ 0x9E3779B97F4A7C15
                        ) + 0x3141592653589793
                    mix(value & ((1 << 64) - 1))
            else:
                mix(case["expected_word"])
    manifest["expected_checksum"] = f"{digest_value:016x}"
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    seed_table = ",".join(hex(bits(x)) + "u" for x in seed)
    expected_fp = ",\n".join(
        "{" + ",".join(hex(v) + "u" for v in c["expected_words"]) + "}"
        for c in cases
        if c["family"] in ("div", "fma")
    )
    source = r"""#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include "benchmark_buffer.h"
#define MAXWORDS (2048*17+8)
#define NOINLINE __attribute__((noinline,noclone))
struct fpbox { uint64_t before[8]; float words[8]; uint64_t after[8]; };
struct context { uint32_t count,stride,repeats; uint64_t value; float initial[8]; struct fpbox fp __attribute__((aligned(64))); };
struct memorybox { uint64_t before[8],data[MAXWORDS],after[8]; };
static struct context ctx __attribute__((aligned(64)));
static struct memorybox input_box __attribute__((aligned(64))),output_box __attribute__((aligned(64)));
static const uint32_t seed_words[8]={SEED_TABLE};
static const uint32_t expected_fp[12][8]={EXPECTED_FP};
static uint64_t checksum=UINT64_C(14695981039346656037);
static uint32_t word(float v){union {float f;uint32_t u;} x={.f=v};return x.u;}
static float number(uint32_t v){union {float f;uint32_t u;} x={.u=v};return x.f;}
static void digest(uint64_t v){for(int i=0;i<8;i++){checksum^=(v>>(8*i))&255;checksum*=UINT64_C(1099511628211);}}
static inline float divide(float x,float y){float z;__asm__ volatile("fdiv.s %0,%1,%2,rne":"=f"(z):"f"(x),"f"(y):"memory");return z;}
static inline float fma_once(float x,float y,float a){float z;__asm__ volatile("fmadd.s %0,%1,%2,%3,rne":"=f"(z):"f"(x),"f"(y),"f"(a):"memory");return z;}
static NOINLINE void empty(struct context*c){(void)c;__asm__ volatile("":::"memory");}
FP_KERNELS
FOOTPRINT_KERNELS
static NOINLINE void memory(struct context*c){
 volatile uint64_t *a=input_box.data,*b=output_box.data;
 for(uint32_t i=0;i<c->count;i++){uint32_t j=i*c->stride;b[j]=(a[j]^UINT64_C(0x9e3779b97f4a7c15))+UINT64_C(0x3141592653589793);}
}
typedef void (*kernel_t)(struct context*);
struct specification {kernel_t fn;uint32_t family,size,stride,repeats;uint64_t expected;};
static const struct specification cases[21]={CASE_TABLE};
struct counters {uint32_t cycles,instructions,frm,flags_before,flags_after,overflow;};
static NOINLINE struct counters measure(kernel_t fn,struct context*c){
 uint64_t begin,end,ibegin,iend,frm,before,after;
 __asm__ volatile("csrwi fflags,0\ncsrr %0,frm\ncsrr %1,fflags":"=r"(frm),"=r"(before)::"memory");
 __asm__ volatile("fence rw,rw\ncsrr %0,mcycle\ncsrr %1,minstret":"=r"(begin),"=r"(ibegin)::"memory");
 fn(c);
 __asm__ volatile("fence rw,rw\ncsrr %0,minstret\ncsrr %1,mcycle":"=r"(iend),"=r"(end)::"memory");
 __asm__ volatile("csrr %0,fflags":"=r"(after)::"memory");
 struct counters result={(uint32_t)(end-begin),(uint32_t)(iend-ibegin),(uint32_t)frm,(uint32_t)before,(uint32_t)after,(end<begin)||(iend<ibegin)||((end-begin)>UINT32_MAX)||((iend-ibegin)>UINT32_MAX)};return result;
}
static uint64_t pattern(uint32_t i){return (uint64_t)i*UINT64_C(2862933555777941757)+UINT64_C(3037000493);}
static int check_guards(uint64_t*a){for(int i=0;i<8;i++)if(a[i]!=UINT64_C(0x5a5a5a5a5a5a5a5a))return 0;return 1;}
int main(void){
 __asm__ volatile("csrwi frm,0":::"memory");
 for(int repeat=0;repeat<3;repeat++){struct counters r=measure(empty,&ctx);if(r.overflow||r.frm||r.flags_before||r.flags_after)return 10;
  printf("SERVICE_ROW id=-1 repeat=%d cycles=%u instructions=%u frm=%u flags_before=%u flags_after=%u\n",repeat,r.cycles,r.instructions,r.frm,r.flags_before,r.flags_after);}
 for(int id=0;id<21;id++)for(int repeat=0;repeat<2;repeat++){
  const struct specification*s=&cases[id];ctx.count=s->size;ctx.stride=s->stride;ctx.repeats=s->repeats;ctx.value=UINT64_C(0x123456789abcdef0);
  for(int i=0;i<8;i++)ctx.initial[i]=number(seed_words[i]);
  merlin_benchmark_fill(ctx.fp.before,0x5a,sizeof(ctx.fp.before));merlin_benchmark_fill(ctx.fp.words,0xdb,sizeof(ctx.fp.words));merlin_benchmark_fill(ctx.fp.after,0x5a,sizeof(ctx.fp.after));
  uint32_t extent=0;
  if(s->family==1){extent=(s->size-1)*s->stride+1;
   merlin_benchmark_fill(input_box.before,0x5a,sizeof(input_box.before));merlin_benchmark_fill(input_box.after,0x5a,sizeof(input_box.after));
   merlin_benchmark_fill(output_box.before,0x5a,sizeof(output_box.before));merlin_benchmark_fill(output_box.after,0x5a,sizeof(output_box.after));
   for(uint32_t i=0;i<extent+8;i++){input_box.data[i]=pattern(i);output_box.data[i]=UINT64_C(0xdbdbdbdbdbdbdbdb);}}
  struct counters r=measure(s->fn,&ctx);if(r.overflow||r.frm||r.flags_before)return 11;
  if(s->family==0){if(r.flags_after!=1)return 18;if(!check_guards(ctx.fp.before)||!check_guards(ctx.fp.after))return 12;
   for(int i=0;i<8;i++){uint32_t got=word(ctx.fp.words[i]);if(got!=expected_fp[id][i]){printf("SERVICE_FP_FAIL id=%d lane=%d got=%08x want=%08x\n",id,i,got,expected_fp[id][i]);return 13;}digest(got);}
   printf("SERVICE_FP id=%d repeat=%d words=",id,repeat);for(int i=0;i<8;i++)printf("%08x%s",word(ctx.fp.words[i]),i==7?"\n":",");
  }else if(s->family==1){if(r.flags_after||!check_guards(input_box.before)||!check_guards(input_box.after)||!check_guards(output_box.before)||!check_guards(output_box.after))return 14;
   for(uint32_t i=0;i<extent+8;i++){uint64_t p=pattern(i),want=(i%s->stride==0&&i/s->stride<s->size)?(p^UINT64_C(0x9e3779b97f4a7c15))+UINT64_C(0x3141592653589793):UINT64_C(0xdbdbdbdbdbdbdbdb);
    if(input_box.data[i]!=p||output_box.data[i]!=want){printf("SERVICE_MEMORY_FAIL id=%d index=%u\n",id,i);return 15;}digest(output_box.data[i]);}
  }else{if(r.flags_after||ctx.value!=s->expected)return 16;digest(ctx.value);}
  printf("SERVICE_ROW id=%d repeat=%d cycles=%u instructions=%u frm=%u flags_before=%u flags_after=%u\n",id,repeat,r.cycles,r.instructions,r.frm,r.flags_before,r.flags_after);
 }
 if(checksum!=UINT64_C(EXPECTED_CHECKSUM))return 17;
 printf("SERVICE_PASS cases=21 repeats=2 checksum=%08x%08x\n",(uint32_t)(checksum>>32),(uint32_t)checksum);return 0;
}
"""
    kernels = []
    for op in ("div", "fma"):
        for independent in (False, True):
            name = op + ("_ind" if independent else "_dep")
            declarations = (
                ";".join(f"float a{i}=c->initial[{i}]" for i in range(8)) + ";"
            )
            expressions = []
            for i in range(8):
                lane = i if independent else 0
                expression = (
                    f"divide(a{lane},scale)"
                    if op == "div"
                    else f"fma_once(a{lane},scale,bias)"
                )
                expressions.append(f"a{lane}={expression};")
            stores = "".join(f"c->fp.words[{i}]=a{i};" for i in range(8))
            kernels.append(
                f"static NOINLINE void {name}(struct context*c){{{declarations}float scale=number({hex(bits(coefficient))}u),bias=number({hex(bits(bias))}u);for(uint32_t n=0;n<c->count/8;n++){{"
                + "".join(expressions)
                + "}"
                + stores
                + "}"
            )
    footprints = []
    for case in cases:
        if case["family"] != "footprint":
            continue
        assembly = (
            ".option push\\n.option norvc\\n"
            + "".join(
                f"addi %0,%0,{(i * 5) % 31 - 15}\\n"
                for i in range(case["body_instructions"])
            )
            + ".option pop\\n"
        )
        footprints.append(
            f'static NOINLINE void {case["kernel"]}(struct context*c){{uint64_t x=c->value;for(uint32_t n=0;n<c->repeats;n++){{__asm__ volatile("{assembly}":"+r"(x)::"memory");}}c->value=x;}}'
        )
    table = []
    for case in cases:
        family = (
            0
            if case["family"] in ("div", "fma")
            else 1
            if case["family"] == "memory"
            else 2
        )
        table.append(
            "{"
            + ",".join(
                (
                    case["kernel"],
                    str(family),
                    str(case["size"]),
                    str(case.get("stride", 0)),
                    str(case.get("repeats", 0)),
                    f"UINT64_C({case.get('expected_word', 0)})",
                )
            )
            + "}"
        )
    source = (
        source.replace("EXPECTED_CHECKSUM", str(digest_value))
        .replace("SEED_TABLE", seed_table)
        .replace("EXPECTED_FP", expected_fp)
        .replace("FP_KERNELS", "\n".join(kernels))
        .replace("FOOTPRINT_KERNELS", "\n".join(footprints))
        .replace("CASE_TABLE", ",\n".join(table))
    )
    (work / "battery.c").write_text(source)
    header = Path(
        "/scratch/agustin/tmp/merlin-calibration-battery-frozen-main-20261006/merlin/runtime/c/benchmark_buffer.h"
    )
    shutil.copyfile(header, work / "benchmark_buffer.h")
    print("BATTERY_GENERATED", len(cases), hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.workdir.resolve())
