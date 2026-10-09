"""Generate a separate fixed-read-count RV64GC gather service battery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MASK64 = (1 << 64) - 1
SEED = 0x123456789ABCDEF0
READS = 4096


def pattern(index):
    return (index * 2862933555777941757 + 3037000493) & MASK64


def raw_stream():
    x = 0x31415927
    stream = []
    for _ in range(READS):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        stream.append(x)
    return stream


def generate(work):
    work.mkdir(parents=True, exist_ok=False)
    stream = raw_stream()
    cases = []
    for working_set in (64 * 1024, 512 * 1024, 8 * 1024 * 1024):
        words = working_set // 8
        indices = [value & (words - 1) for value in stream]
        accumulator = SEED
        for index in indices:
            accumulator = (
                ((accumulator << 7) | (accumulator >> 57)) & MASK64
            ) ^ pattern(index)
        cases.append(
            {
                "id": len(cases),
                "family": "gather",
                "size": READS,
                "kernel": "gather",
                "partition": "heldout" if working_set == 512 * 1024 else "training",
                "working_set_bytes": working_set,
                "gather_value_load_bytes": READS * 8,
                "index_stream_load_bytes": READS * 4,
                "requested_cpu_load_bytes": READS * 12,
                "requested_cpu_store_bytes": 8,
                "source_and_destination_extent_bytes": working_set,
                "unique_requested_words": len(set(indices)),
                "unique_requested_64B_regions": len({index // 8 for index in indices}),
                "requested_region_granule_bytes": 64,
                "requested_region_granule_is_hardware_cache_line": "UNKNOWN: geometry not pinned here",
                "indices": indices,
                "expected_word": accumulator,
                "expected_fflags": 0,
            }
        )
    checksum = 14695981039346656037
    for case in cases:
        for _ in range(2):
            for i in range(8):
                checksum = (
                    (checksum ^ ((case["expected_word"] >> (i * 8)) & 255))
                    * 1099511628211
                ) & MASK64
    manifest = {
        "schema": "rv64gc_gather_service_battery_v1",
        "cases": cases,
        "repetitions": 2,
        "empty_windows": 3,
        "expected_checksum": f"{checksum:016x}",
        "index_stream": "Fixed4096 xorshift32 draws from0x31415927; same raw stream masked by typed power-of-two word extent",
        "partition_rule": "512KiB middle working set held out including both repeats;64KiB and8MiB training",
        "dependency_policy": "One rotate/XOR accumulating register; load indices independent of accumulator. Operational gather stream, not pure memory bandwidth.",
        "setup": "Entire8MiB table initialized once before all windows; index stream initialized before each window. Full immutable table checked after all windows, indices and exterior guards checked after each window. No cold-cache claim.",
        "timing": "Same fenced mcycle/minstret kernel-call windows; initialization, checks, UART and checksum outside",
        "fitting": "No model or Jack labels fitted",
        "physical_memory_traffic": "UNKNOWN",
        "cache_misses": "UNKNOWN",
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    table = ",".join(
        "{"
        + ",".join(
            (str(case["working_set_bytes"] // 8), f"UINT64_C({case['expected_word']})")
        )
        + "}"
        for case in cases
    )
    source = r"""#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#define READS 4096
#define MAXWORDS (8*1024*1024/8)
#define NOINLINE __attribute__((noinline,noclone))
struct tablebox {uint64_t before[8],data[MAXWORDS],after[8];};
struct indexbox {uint64_t before[8];uint32_t data[READS];uint64_t after[8];};
struct context {uint32_t count;uint64_t value;};
struct valuebox {uint64_t before[8];struct context context;uint64_t after[8];};
static struct tablebox table __attribute__((aligned(64)));
static struct indexbox indices __attribute__((aligned(64)));
static struct valuebox output __attribute__((aligned(64)));
static const struct {uint32_t words;uint64_t expected;} cases[3]={CASE_TABLE};
static uint64_t checksum=UINT64_C(14695981039346656037);
static uint64_t pattern(uint32_t i){return (uint64_t)i*UINT64_C(2862933555777941757)+UINT64_C(3037000493);}
static void digest(uint64_t v){for(int i=0;i<8;i++){checksum^=(v>>(i*8))&255;checksum*=UINT64_C(1099511628211);}}
static void guard(uint64_t *a){for(int i=0;i<8;i++)a[i]=UINT64_C(0x5a5a5a5a5a5a5a5a);}
static int guard_ok(uint64_t*a){for(int i=0;i<8;i++)if(a[i]!=UINT64_C(0x5a5a5a5a5a5a5a5a))return 0;return 1;}
static uint32_t next(uint32_t x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
static NOINLINE void empty(struct context*c){(void)c;__asm__ volatile("":::"memory");}
static NOINLINE void gather(struct context*c){
 volatile uint64_t *a=table.data;volatile uint32_t *index=indices.data;
 uint64_t value=c->value;for(uint32_t i=0;i<c->count;i++){uint64_t element=a[index[i]];value=((value<<7)|(value>>57))^element;}c->value=value;
}
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
static int report(int id,int repeat,struct counters r){if(r.overflow||r.frm||r.flags_before||r.flags_after)return 1;
 printf("SERVICE_ROW id=%d repeat=%d cycles=%u instructions=%u frm=%u flags_before=%u flags_after=%u\n",id,repeat,r.cycles,r.instructions,r.frm,r.flags_before,r.flags_after);return 0;}
int main(void){
 __asm__ volatile("csrwi frm,0":::"memory");guard(table.before);guard(table.after);guard(indices.before);guard(indices.after);guard(output.before);guard(output.after);
 for(uint32_t i=0;i<MAXWORDS;i++)table.data[i]=pattern(i);
 for(int repeat=0;repeat<3;repeat++)if(report(-1,repeat,measure(empty,&output.context)))return 10;
 for(int id=0;id<3;id++)for(int repeat=0;repeat<2;repeat++){
  uint32_t x=0x31415927;for(uint32_t i=0;i<READS;i++){x=next(x);indices.data[i]=x&(cases[id].words-1);}
  output.context.count=READS;output.context.value=UINT64_C(0x123456789abcdef0);
  struct counters r=measure(gather,&output.context);
  if(output.context.value!=cases[id].expected||!guard_ok(table.before)||!guard_ok(table.after)||!guard_ok(indices.before)||!guard_ok(indices.after)||!guard_ok(output.before)||!guard_ok(output.after))return 11;
  x=0x31415927;for(uint32_t i=0;i<READS;i++){x=next(x);if(indices.data[i]!=(x&(cases[id].words-1)))return 12;}
  digest(output.context.value);if(report(id,repeat,r))return 13;
 }
 for(uint32_t i=0;i<MAXWORDS;i++)if(table.data[i]!=pattern(i))return 14;
 if(checksum!=UINT64_C(EXPECTED_CHECKSUM))return 15;
 printf("SERVICE_PASS cases=3 repeats=2 checksum=%08x%08x\n",(uint32_t)(checksum>>32),(uint32_t)checksum);return 0;
}
""".replace("CASE_TABLE", table).replace("EXPECTED_CHECKSUM", str(checksum))
    (work / "battery.c").write_text(source)
    print("GATHER_GENERATED", len(cases), f"{checksum:016x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.workdir.resolve())
