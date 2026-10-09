"""Complete original sibling GEMMs and unchanged closed source observer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.llvmlower.source_expression_interval import IntervalEffectContract
from merlin.llvmlower.streamed_pointwise_pair import emit_streamed_pair_coordinator
from merlin.perf.layer_bench import build_program
from paired_pointwise_prepare import CONTEXT, LLVM, ROOT, rename, sha
from paired_pointwise_prepare import OUT as CONSUMER

from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from mlir_oot.no_fsm_audit import audit_elf

OUT = ROOT / "out/artifacts/probes/paired-pointwise-compound-20261007"
PRIMITIVE = ROOT / "out/artifacts/probes/paired-gemm-original-source-20261007"
CONTROL = Path(
    "/scratch/agustin/tmp/gemmini-rne-observer-cells-20261007/out/artifacts/probes/rne-zero-observer-M8-20261007"
)
GCC = Path(
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
)
BASE = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build"
)

MAIN = r"""#include <stddef.h>
#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t input_a[],weight_g[],weight_u[],observer_expected[];
extern const int32_t expected_g[],expected_u[];
extern const float scale_g[],scale_u[],table_b16[][2];
extern void paired_dense_begin(const int8_t*);
extern void paired_dense_issue(const int8_t*,const int8_t*,int32_t*,int32_t*,uint64_t);
extern void paired_dense_wait(void);
extern void gemmini_golden_a5705ab56e324ba1(const int8_t*,const int8_t*,int32_t*);
extern void cells_M8(int32_t*,float*,int32_t*,float*,int8_t*);
extern void tile_current(void*,void*,void*,void*,void*);
extern void tile_source(void*,void*,void*,void*,void*);
typedef int(*Prep)(void*);
typedef int(*Issue)(void*,size_t,size_t,int32_t*,int32_t*);
typedef int(*Consume)(void*,size_t,size_t,const int32_t*,const int32_t*,int8_t*);
typedef void(*Finish)(void*);
typedef int(*Fallback)(void*,int8_t*);
extern int stream_pair(size_t,size_t,size_t,void*,size_t,int8_t*,void*,Prep,Issue,Prep,Consume,Finish,Fallback,int);
static int32_t arena[4*8*512+32] __attribute__((aligned(64)));
static int32_t full_g[8*5632+32] __attribute__((aligned(64)));
static int32_t full_u[8*5632+32] __attribute__((aligned(64)));
static int8_t tile[8*512+128] __attribute__((aligned(64)));
static int8_t reference[8*512+128] __attribute__((aligned(64)));
static int8_t output[8*5632+128] __attribute__((aligned(64)));
struct Context {int pending,validate,failed,issues,waits,consumes;size_t offset;int32_t*g,*u;};
static struct Context context;
static uint64_t hash(const void*p,size_t n){const unsigned char*x=p;uint64_t h=1469598103934665603ull;for(size_t i=0;i<n;i++)h=(h^x[i])*1099511628211ull;return h;}
static unsigned flags(void){unsigned f;__asm__ volatile("csrr %0,fflags":"=r"(f)::"memory");return f;}
static unsigned mode_now(void){unsigned f;__asm__ volatile("csrr %0,frm":"=r"(f)::"memory");return f;}
static void mode(unsigned frm,unsigned f){__asm__ volatile("csrw frm,%0\ncsrw fflags,%1"::"r"(frm),"r"(f):"memory");}
static uint64_t ticks(void){uint64_t x;__asm__ volatile("csrr %0,mcycle":"=r"(x)::"memory");return x;}
static uint64_t instructions(void){uint64_t x;__asm__ volatile("csrr %0,minstret":"=r"(x)::"memory");return x;}
static void fence(void){__asm__ volatile("fence":::"memory");}
static void clear(void){
 for(unsigned i=0;i<4*8*512+32;i++)arena[i]=0x53535353;
 for(unsigned i=0;i<8*512+128;i++)tile[i]=reference[i]=73;
 for(unsigned i=0;i<8*5632+128;i++)output[i]=73;
 context.pending=context.failed=context.issues=context.waits=context.consumes=0;
}
static int prepare(void*opaque){struct Context*c=opaque;if(c->pending)return 0;paired_dense_begin(input_a);return 1;}
static int issue(void*opaque,size_t offset,size_t count,int32_t*g,int32_t*u){struct Context*c=opaque;
 if(c->pending||count!=512||offset%512||offset>5632-512)return 0;
 paired_dense_issue(weight_g,weight_u,g,u,offset);
 c->pending=1;c->offset=offset;c->g=g;c->u=u;c->issues++;return 1;
}
static int wait(void*opaque){struct Context*c=opaque;if(!c->pending)return 0;
 paired_dense_wait();c->pending=0;c->waits++;
 if(c->validate)for(unsigned r=0;r<8;r++)for(unsigned n=0;n<512;n++){
  size_t src=r*5632+c->offset+n,t=r*512+n;
  if(c->g[t]!=expected_g[src]||c->u[t]!=expected_u[src]){c->failed=1;return 0;}
 }return 1;
}
static int consume(void*opaque,size_t offset,size_t count,const int32_t*g,const int32_t*u,int8_t*out){struct Context*c=opaque;
 if(count!=512||offset%512||offset>5632-512)return 0;
 // This callback has no device command, scratchpad/accumulator access, hidden
 // allocation or pointer escape. Its exact source helper reads only ready
 // private i32 words and immutable broadcast parameter slices.
 tile_current((void*)g,(void*)(scale_g+offset),(void*)u,(void*)(scale_u+offset),tile+64);
 for(unsigned r=0;r<8;r++)for(unsigned n=0;n<512;n++)out[r*5632+offset+n]=tile[64+r*512+n];
 c->consumes++;return 1;
}
static void finish(void*opaque){struct Context*c=opaque;paired_dense_wait();c->pending=0;}
static int original(void*opaque,int8_t*out){(void)opaque;
 gemmini_golden_a5705ab56e324ba1(input_a,weight_g,full_g);
 gemmini_golden_a5705ab56e324ba1(input_a,weight_u,full_u);
 cells_M8(full_g,(float*)scale_g,full_u,(float*)scale_u,out);return 1;
}
static int run(unsigned id){
 if(id==0)return original(&context,output+64);
 return stream_pair(8,5632,512,arena,4*8*512*4,output+64,&context,prepare,issue,wait,consume,finish,original,id==2);
}
static int valid(void){
 for(unsigned i=0;i<8*5632+128;i++)if((i<64||i>=8*5632+64)?output[i]!=73:output[i]!=observer_expected[i-64])return 0;
 for(unsigned i=0;i<64;i++)if(tile[i]!=73||tile[8*512+64+i]!=73)return 0;
 for(unsigned i=4*8*512;i<4*8*512+32;i++)if(arena[i]!=0x53535353)return 0;
 return !context.failed&&!context.pending;
}
static int observer_modes(void){
 const unsigned presets[]={0,1,2,4,8,16,31};
 for(unsigned frm=0;frm<5;frm++)for(unsigned s=0;s<7;s++)for(unsigned offset=0;offset<5632;offset+=512){
  for(unsigned r=0;r<8;r++)for(unsigned c=0;c<512;c++){arena[r*512+c]=expected_g[r*5632+offset+c];arena[8*512+r*512+c]=expected_u[r*5632+offset+c];}
  for(unsigned i=0;i<8*512+128;i++)tile[i]=reference[i]=73;
  mode(frm,presets[s]);tile_source(arena,(void*)(scale_g+offset),arena+8*512,(void*)(scale_u+offset),reference+64);
  if(mode_now()!=frm||(flags()&presets[s])!=presets[s])return 0;
  mode(frm,presets[s]);tile_current(arena,(void*)(scale_g+offset),arena+8*512,(void*)(scale_u+offset),tile+64);
  if(mode_now()!=frm||(flags()&presets[s])!=presets[s])return 0;
  for(unsigned i=0;i<8*512+128;i++)if(tile[i]!=reference[i])return 0;
 }return 1;
}
int main(void){
 uint64_t pins[]={hash(input_a,8*2048),hash(weight_g,2048*5632),hash(weight_u,2048*5632),hash(scale_g,5632*4),hash(scale_u,5632*4),hash(table_b16,512*1024)};
 if(!observer_modes()){printf("PAIRED_FAIL observer_modes\n");return 1;}
 printf("PAIRED_SOURCE_GATE original45056i8 all5frm sticky7 guards sourcecontinuation\n");
 for(unsigned id=0;id<3;id++){clear();context.validate=1;mode(0,31);
  if(!run(id)||!valid()||mode_now()!=0||(flags()&31)!=31){printf("PAIRED_FAIL initial %u\n",id);return 2;}
  if(id==0)for(unsigned i=0;i<8*5632;i++)if(full_g[i]!=expected_g[i]||full_u[i]!=expected_u[i])return 3;
  printf("PAIRED_INITIAL id=%u issues=%u waits=%u consumes=%u outputdigest=%lx\n",id,context.issues,context.waits,context.consumes,hash(output+64,8*5632));
 }
 printf("PAIRED_POINTERS a=%lx wg=%lx wu=%lx sg=%lx su=%lx table=%lx arena=%lx out=%lx tile=%lx fg=%lx fu=%lx\n",(unsigned long)input_a,(unsigned long)weight_g,(unsigned long)weight_u,(unsigned long)scale_g,(unsigned long)scale_u,(unsigned long)table_b16,(unsigned long)arena,(unsigned long)(output+64),(unsigned long)(tile+64),(unsigned long)full_g,(unsigned long)full_u);
 const unsigned order[]={0,1,2,2,1,0};
 for(unsigned sample=0;sample<6;sample++){
  unsigned id=order[sample];context.validate=0;mode(0,0);fence();uint64_t t=ticks(),i=instructions();
  // Complete preparation, initialization, command issue/wait, source observer,
  // compact-tile scatter, final fence and public output publication are timed.
  clear();int ok=run(id);fence();i=instructions()-i;t=ticks()-t;
  if(!ok||!valid()||mode_now()!=0){printf("PAIRED_FAIL timed %u\n",id);return 4;}
  printf("PAIRED_ROW id=%u sample=%u cycles=%lu instructions=%lu digest=%lx issues=%u waits=%u consumes=%u\n",id,sample,t,i,hash(output+64,8*5632),context.issues,context.waits,context.consumes);
 }
 if(pins[0]!=hash(input_a,8*2048)||pins[1]!=hash(weight_g,2048*5632)||pins[2]!=hash(weight_u,2048*5632)||pins[3]!=hash(scale_g,5632*4)||pins[4]!=hash(scale_u,5632*4)||pins[5]!=hash(table_b16,512*1024)){printf("PAIRED_FAIL inputs\n");return 5;}
 printf("PAIRED_PASS original90112i32 original45056i8 allguards immutableinputs rank0 DONE\n");return 0;
}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "driver.py").write_bytes(Path(__file__).read_bytes())
    commands = []

    def run(argv):
        command = list(map(str, argv))
        commands.append(command)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    # Re-link the SAME source-cloned helper against the existing immutable
    # table owner. Both schedules and the existing full helper read one VMA.
    # New semantic table sharing requires complete canonical expression/cell
    # equality, already closed by the native preparation receipt.
    (OUT / "lookup_shared.ll").write_text(
        rename(
            (CONSUMER / "target/lookup.ll").read_text(), {"@tile_table": "@table_b16"}
        )
    )
    run(
        [
            LLVM / "llvm-link",
            "-S",
            *[
                CONSUMER / name
                for name in ("source.ll", "selected.ll", "activation.ll", "quantize.ll")
            ],
            OUT / "lookup_shared.ll",
            CONSUMER / "target/scanner.ll",
            CONSUMER / "target/guard.ll",
            "-o",
            OUT / "linked.ll",
        ]
    )
    run(
        [
            LLVM / "opt",
            "-S",
            "-passes=always-inline",
            OUT / "linked.ll",
            "-o",
            OUT / "inlined.ll",
        ]
    )
    merlin_host_llvm_transform(LLVM, combine_clamp=True)(
        OUT / "inlined.ll", OUT / "late_rne"
    )
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O3",
        "-ffreestanding",
        "-fno-builtin",
        "-ffp-contract=off",
    ]
    run(
        [
            LLVM / "clang",
            *flags,
            "-c",
            OUT / "late_rne/model.ll",
            "-o",
            OUT / "consumer.o",
        ]
    )
    (OUT / "consumer.dump").write_text(
        run([LLVM / "llvm-objdump", "-dr", OUT / "consumer.o"]).stdout
    )
    undefined = run([LLVM / "llvm-nm", "-u", OUT / "consumer.o"]).stdout.split()
    assert undefined == ["U", "table_b16"]
    (OUT / "coordinator.c").write_text(
        emit_streamed_pair_coordinator(
            symbol="stream_pair",
            effects=IntervalEffectContract(True, True, True, True, True),
        )
    )
    data = {}
    for name, source in (
        ("scale_g", "scale_a.npy"),
        ("scale_u", "scale_b.npy"),
        ("observer_expected", "expected.npy"),
    ):
        path = OUT / (name + ".bin")
        path.write_bytes(np.load(CONTEXT / source).tobytes())
        data[name] = path
    (OUT / "observer_data.S").write_text(
        ".section .rodata\n"
        + "".join(
            f'.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n'
            for name, path in data.items()
        )
    )
    (OUT / "main.c").write_text(MAIN)
    for name in ("main.c", "coordinator.c", "observer_data.S"):
        run([LLVM / "clang", *flags, "-c", OUT / name, "-o", OUT / (name + ".o")])
    previous = json.loads((CONTROL / "timing/qualification.json").read_text())
    wanted = {"helpers.o", "tables.o"}
    old = [
        Path(p) for p, digest in previous["objects"].items() if Path(p).name in wanted
    ]
    assert len(old) == 2 and all(sha(p) == previous["objects"][str(p)] for p in old)
    objects = [
        BASE / "device_catalog/kernel.o",
        PRIMITIVE / "device/kernel.o",
        PRIMITIVE / "data.S.o",
        *old,
        CONTROL / "candidate.o",
        CONTROL / "target_guard.o",
        OUT / "consumer.o",
        OUT / "coordinator.c.o",
        OUT / "observer_data.S.o",
        OUT / "main.c.o",
    ]
    before = {str(p): sha(p) for p in objects}
    build = build_program(
        objects, OUT / "build", target="gemmini", max_loaded_bytes=None
    )
    assert before == {str(p): sha(p) for p in objects}
    audit = audit_elf(build.elf.read_bytes())
    assert audit["status"] == "pass"
    (OUT / "nofsm.json").write_text(json.dumps(audit, indent=2) + "\n")
    command = [GCC.with_name("spike"), "--isa=rv64gc", "--extension=gemmini", build.elf]
    result = run(command)
    (OUT / "spike.stdout").write_text(result.stdout)
    (OUT / "spike.stderr").write_text(result.stderr)
    assert "PAIRED_PASS" in result.stdout and "PAIRED_FAIL" not in result.stdout
    rows = [
        dict(piece.split("=", 1) for piece in line.split()[1:])
        for line in result.stdout.splitlines()
        if line.startswith("PAIRED_ROW ")
    ]
    assert [int(r["id"]) for r in rows] == [0, 1, 2, 2, 1, 0]
    for row in rows:
        assert int(row["digest"], 16) == 0xB702909CA80DFE2A
        assert (int(row["issues"]), int(row["waits"]), int(row["consumes"])) == (
            (0, 0, 0) if row["id"] == "0" else (11, 11, 11)
        )
    pins = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            *OUT.rglob("*"),
            *objects,
            CONSUMER / "qualification.json",
            PRIMITIVE / "qualification.json",
            *CONSUMER.rglob("*"),
            *CONTEXT.glob("*.npy"),
        ]
        if p.is_file()
    }
    (OUT / "qualification.json").write_text(
        json.dumps(
            {
                "schema": "complete_source_sibling_pointwise_compound_v1",
                "status": "pass",
                "scope": "Original context21 complete8x2048x5632 sibling gate/up and45056-byte observer, one common table/input/output/arena VMA; old-full, serial-tile, overlap-tile schedules in SAME ELF",
                "original_i32_words": 90112,
                "original_i8_words": 45056,
                "rounding_modes": "all5target/7sticky observer fixture; complete initial and timed integer/observer path RNE with stableFRM",
                "rows": rows,
                "objects": before,
                "elf": {"path": str(build.elf), "sha256": sha(build.elf)},
                "consumer_undefined_symbols": undefined,
                "consumer_resource_effects": "No target/device call, hidden allocation, input mutation or pointer escape; only immutable table import",
                "roi": "Allbegin/issue/wait/config/load/packing/init/lookup/sourcefallback/observer/frame/scatter/finalfence and alloutputpublication; validation only outside",
                "overlap_limit": "Issue executes CPU command loop before return; only still-pending target work can overlap consumer. Queue occupancy/physicalmemory/cycles UNKNOWN",
                "hardware_cycles": "UNKNOWN",
                "whole_performance": "UNKNOWN; no extrapolation from oldprofile",
                "commands": commands,
                "pins": pins,
                "token_usage_available": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("PAIRED_COMPOUND_STRICT_PASS", sha(build.elf), flush=True)


if __name__ == "__main__":
    main()
