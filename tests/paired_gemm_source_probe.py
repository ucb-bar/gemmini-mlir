"""Actual original sibling operands and complete primitive issue readout gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from merlin.llvmlower import quant_hoist
from merlin.perf.layer_bench import build_program

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_gemm_phases import GoldenGemmPhases
from mlir_oot.no_fsm_audit import audit_elf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/artifacts/probes/paired-gemm-original-source-20261007"
CAPTURE = Path(
    "/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-packed-rhs-current-20261006/capture"
)
CONTEXT = Path(
    "/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006/out/artifacts/probes/source-continuation-contexts-v3-20261007/context_21"
)
BUILD = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build"
)
LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
GCC = Path(
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    plan = quant_hoist.read_plan(BUILD)
    values = quant_hoist.read_values(BUILD)
    original_g = np.load(CONTEXT / "a.npy").reshape(8, 5632)
    original_u = np.load(CONTEXT / "b.npy").reshape(8, 5632)
    a = np.fromfile(CAPTURE / "a.bin", dtype=np.int8).reshape(8, 2048)
    g_weight = np.fromfile(CAPTURE / "b.bin", dtype=np.int8).reshape(2048, 5632)
    u_weight = np.ascontiguousarray(values[plan[5].key])
    assert np.array_equal(g_weight, values[plan[4].key])
    assert np.array_equal(
        original_g.ravel(), np.fromfile(CAPTURE / "expected.bin", dtype="<i4")
    )
    assert np.array_equal(a.astype(np.int32) @ g_weight.astype(np.int32), original_g)
    assert np.array_equal(a.astype(np.int32) @ u_weight.astype(np.int32), original_u)
    (OUT / "up_weight.bin").write_bytes(u_weight.tobytes())
    (OUT / "up_expected.bin").write_bytes(original_u.astype("<i4").tobytes())
    shape = Shape(
        8, 5632, 2048, "i32", bm=1, bn=32, cache_a=True, wide_b=True, prefetch_b=True
    )
    provider = GoldenGemmPhases(shape, columns=512)
    module = provider.build(prefix="paired_dense")
    device = compile_module(module, LLVM, OUT / "device")
    files = {
        "input_a": CAPTURE / "a.bin",
        "weight_g": CAPTURE / "b.bin",
        "weight_u": OUT / "up_weight.bin",
        "expected_g": CAPTURE / "expected.bin",
        "expected_u": OUT / "up_expected.bin",
    }
    assembly = ".section .rodata\n" + "".join(
        f'.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n'
        for name, path in files.items()
    )
    (OUT / "data.S").write_text(assembly)
    text = r"""#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t input_a[],weight_g[],weight_u[];
extern const int32_t expected_g[],expected_u[];
extern void paired_dense_begin(const int8_t*);
extern void paired_dense_issue(const int8_t*,const int8_t*,int32_t*,int32_t*,uint64_t);
extern void paired_dense_wait(void);
extern void gemmini_golden_a5705ab56e324ba1(const int8_t*,const int8_t*,int32_t*);
static int32_t g[8*512+32] __attribute__((aligned(64)));
static int32_t u[8*512+32] __attribute__((aligned(64)));
static int32_t full[8*5632+32] __attribute__((aligned(64)));
static uint64_t hash(const void*p,uint64_t n){const unsigned char*x=p;uint64_t h=1469598103934665603ull;for(uint64_t i=0;i<n;i++)h=(h^x[i])*1099511628211ull;return h;}
int main(void){
uint64_t ah=hash(input_a,8*2048),gh=hash(weight_g,2048*5632),uh=hash(weight_u,2048*5632);
for(unsigned b=0;b<2;b++){const int8_t*w=b?weight_u:weight_g;const int32_t*e=b?expected_u:expected_g;
 for(unsigned i=0;i<8*5632+32;i++)full[i]=0x53535353;
 gemmini_golden_a5705ab56e324ba1(input_a,w,full);
 for(unsigned i=0;i<8*5632;i++)if(full[i]!=e[i]){printf("BASE_FAIL %u %u %d %d\n",b,i,full[i],e[i]);return 1;}
 for(unsigned i=8*5632;i<8*5632+32;i++)if(full[i]!=0x53535353)return 2;
}
for(unsigned repeat=0;repeat<2;repeat++){
 paired_dense_begin(input_a);
 for(unsigned offset=0;offset<5632;offset+=512){
  for(unsigned i=0;i<8*512+32;i++)g[i]=u[i]=0x53535353;
  paired_dense_issue(weight_g,weight_u,g,u,offset);paired_dense_wait();
  for(unsigned r=0;r<8;r++)for(unsigned c=0;c<512;c++){
   unsigned local=r*512+c,source=r*5632+offset+c;
   if(g[local]!=expected_g[source]||u[local]!=expected_u[source]){printf("PHASE_FAIL %u %u %d %d %d %d\n",offset,local,g[local],expected_g[source],u[local],expected_u[source]);return 3;}
  }
  for(unsigned i=8*512;i<8*512+32;i++)if(g[i]!=0x53535353||u[i]!=0x53535353)return 4;
 }
 paired_dense_wait();
}
if(ah!=hash(input_a,8*2048)||gh!=hash(weight_g,2048*5632)||uh!=hash(weight_u,2048*5632))return 5;
printf("PAIRED_PHASE_PASS original90112i32 repeats2 dirtyguards unchangedinputs\n");return 0;
}"""
    (OUT / "main.c").write_text(text)
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O3",
        "-ffreestanding",
        "-fno-builtin",
    ]
    commands = []
    for source in ("main.c", "data.S"):
        command = [
            str(LLVM / "clang"),
            *flags,
            "-c",
            str(OUT / source),
            "-o",
            str(OUT / (source + ".o")),
        ]
        subprocess.run(command, check=True, capture_output=True)
        commands.append(command)
    objects = [
        BUILD / "device_catalog/kernel.o",
        OUT / "device/kernel.o",
        OUT / "main.c.o",
        OUT / "data.S.o",
    ]
    pins = {str(path): sha(path) for path in objects}
    built = build_program(
        objects, OUT / "build", target="gemmini", max_loaded_bytes=None
    )
    assert pins == {str(path): sha(path) for path in objects}
    audit = audit_elf(built.elf.read_bytes())
    assert audit["status"] == "pass"
    (OUT / "nofsm.json").write_text(json.dumps(audit, indent=2) + "\n")
    command = [
        str(GCC.with_name("spike")),
        "--isa=rv64gc",
        "--extension=gemmini",
        str(built.elf),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    (OUT / "spike.stdout").write_text(result.stdout)
    (OUT / "spike.stderr").write_text(result.stderr)
    assert result.returncode == 0 and "PAIRED_PHASE_PASS" in result.stdout, (
        result.stdout + result.stderr
    )
    pins.update({str(path): sha(path) for path in files.values()})
    for path in (
        *OUT.rglob("*"),
        Path(__file__),
        CAPTURE / "validation.json",
        CONTEXT / "a.npy",
        CONTEXT / "b.npy",
        BUILD / "quant_hoist_args.json",
        BUILD / "device_catalog/device_catalog.json",
    ):
        if path.is_file():
            pins[str(path)] = sha(path)
    (OUT / "qualification.json").write_text(
        json.dumps(
            {
                "schema": "paired_cached_input_original_source_readout_v1",
                "status": "pass",
                "scope": "Original first executed sibling gate/up projections, exact source-context21 arrays, sameA and original parameter plan entries4/5; original whole gate capture remains frozen.",
                "independent_integer_reference": "Both full 90112 readout words match independent i32 source products and previously compiled original source captured tensors.",
                "resources": asdict(provider.resources),
                "device": device,
                "commands": commands + [command],
                "elf": {"path": str(built.elf), "sha256": sha(built.elf)},
                "host_consumer_and_overlap_qualification": "PENDING; this capsule proves complete primitive readout only, not full compound correctness or profitability",
                "hardware_cycles": "UNKNOWN",
                "pins": pins,
                "token_usage_available": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("PAIRED_ORIGINAL_READOUT_PASS", sha(built.elf), flush=True)


if __name__ == "__main__":
    main()
