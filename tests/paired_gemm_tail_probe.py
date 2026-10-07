"""Actual independent signed nonzero M/K tails and repeated dirty readouts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.perf.layer_bench import build_program
from paired_pointwise_prepare import LLVM, ROOT, sha
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_gemm_phases import GoldenGemmPhases
from mlir_oot.no_fsm_audit import audit_elf


def main():
    out = ROOT / "out/artifacts/probes/paired-gemm-independent-MK-tail-20261007"
    out.mkdir(parents=True, exist_ok=False)
    shape = Shape(
        7, 64, 73, "i32", bm=1, bn=2, cache_a=True, wide_b=True, prefetch_b=True
    )
    provider = GoldenGemmPhases(shape, columns=32)
    module = provider.build(prefix="tail_pair")
    original = GoldenGemm(shape).build()
    for function in tuple(original.body.block.ops):
        function.properties["sym_name"] = StringAttr("tail_original")
        function.detach()
        module.body.block.add_op(function)
    compiled = compile_module(module, LLVM, out / "device")
    rng = np.random.default_rng(7320647)
    a = rng.integers(-128, 128, (7, 73), dtype=np.int8)
    weights = [rng.integers(-128, 128, (73, 64), dtype=np.int8) for _ in range(2)]
    gold = [a.astype(np.int32) @ w.astype(np.int32) for w in weights]
    arrays = {
        "input": a,
        "wg": weights[0],
        "wu": weights[1],
        "gg": gold[0],
        "gu": gold[1],
    }
    for name, array in arrays.items():
        (out / (name + ".bin")).write_bytes(array.tobytes())
    (out / "data.S").write_text(
        ".section .rodata\n"
        + "".join(
            f'.balign 64\n.global {name}\n{name}:\n.incbin "{out / (name + ".bin")}"\n'
            for name in arrays
        )
    )
    (out / "main.c").write_text(r"""#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t input[],wg[],wu[];extern const int32_t gg[],gu[];
extern void tail_original(const int8_t*,const int8_t*,int32_t*);
extern void tail_pair_begin(const int8_t*);
extern void tail_pair_issue(const int8_t*,const int8_t*,int32_t*,int32_t*,uint64_t);
extern void tail_pair_wait(void);
static int32_t full[7*64+32] __attribute__((aligned(64)));
static int32_t g[7*32+32] __attribute__((aligned(64))),u[7*32+32] __attribute__((aligned(64)));
int main(void){
 for(unsigned b=0;b<2;b++){for(unsigned i=0;i<7*64+32;i++)full[i]=0x53535353;
  tail_original(input,b?wu:wg,full);const int32_t*e=b?gu:gg;
  for(unsigned i=0;i<7*64+32;i++)if(full[i]!=(i<7*64?e[i]:0x53535353))return 1;
 }
 for(unsigned repeat=0;repeat<2;repeat++){tail_pair_begin(input);
  for(unsigned offset=0;offset<64;offset+=32){for(unsigned i=0;i<7*32+32;i++)g[i]=u[i]=0x53535353;
   tail_pair_issue(wg,wu,g,u,offset);tail_pair_wait();
   for(unsigned r=0;r<7;r++)for(unsigned c=0;c<32;c++)if(g[r*32+c]!=gg[r*64+offset+c]||u[r*32+c]!=gu[r*64+offset+c])return 2;
   for(unsigned i=7*32;i<7*32+32;i++)if(g[i]!=0x53535353||u[i]!=0x53535353)return 3;
  }tail_pair_wait();
 }printf("PAIRED_TAIL_PASS M7 N64 K73 signed_nonzero repeats2 dirtyguards original896i32\n");return 0;
}""")
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
    for name in ("main.c", "data.S"):
        command = list(
            map(
                str,
                [LLVM / "clang", *flags, "-c", out / name, "-o", out / (name + ".o")],
            )
        )
        subprocess.run(command, check=True, capture_output=True)
        commands.append(command)
    build = build_program(
        [out / "device/kernel.o", out / "main.c.o", out / "data.S.o"],
        out / "build",
        target="gemmini",
        max_loaded_bytes=None,
    )
    audit = audit_elf(build.elf.read_bytes())
    assert audit["status"] == "pass"
    gcc = Path(
        "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
    )
    command = [
        str(gcc.with_name("spike")),
        "--isa=rv64gc",
        "--extension=gemmini",
        str(build.elf),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    commands.append(command)
    (out / "spike.stdout").write_text(result.stdout)
    (out / "spike.stderr").write_text(result.stderr)
    assert result.returncode == 0 and "PAIRED_TAIL_PASS" in result.stdout
    (out / "qualification.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "scope": "Independent signed nonzero source products with M/K tails and repeated dirty destinations; no model fixture or performance claim",
                "compilation": compiled,
                "nofsm": audit,
                "elf_sha256": sha(build.elf),
                "commands": commands,
                "pins": {
                    str(p): sha(p)
                    for p in [Path(__file__), *out.rglob("*")]
                    if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("INDEPENDENT_PAIRED_TAIL_PASS", sha(build.elf))


if __name__ == "__main__":
    main()
