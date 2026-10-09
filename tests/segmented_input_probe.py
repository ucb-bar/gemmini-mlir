"""Independent strict target numeric/guard probe for segmented input residency."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.llvmlower.segmented_matrix_view import SegmentedRows
from merlin.perf.layer_bench import build_program, run_on_gsim

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.no_fsm_audit import audit_elf


def pin(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    m, n, k = 33, 73, 65
    view = SegmentedRows(m, k, 7, 131, 2000, 10000, 1, "i8", 5)
    source = ((np.arange(view.source_elements) * 37 + 11) % 43 - 21).astype(np.int8)
    matrix = np.array(
        [[source[view.offset(row, col)] for col in range(k)] for row in range(m)],
        dtype=np.int32,
    )
    right = ((np.arange(k * n) * 53 + 19) % 41 - 20).astype(np.int8)
    expected = (matrix @ right.astype(np.int32).reshape(k, n)).astype("<i4")
    for name, array, ctype in [
        ("source", source, "int8_t"),
        ("right", right, "int8_t"),
        ("expected", expected.ravel(), "int32_t"),
    ]:
        (work / (name + ".bin")).write_bytes(array.tobytes())
        (work / (name + ".h")).write_text(
            f"static const {ctype} {name}[{array.size}] __attribute__((aligned(64)))={{"
            + ",".join(map(str, array.tolist()))
            + "};\n"
        )
    program = work / "probe.c"
    program.write_text("""#include <stdint.h>
#include <stdio.h>
#include "source.h"
#include "right.h"
#include "expected.h"
static struct {uint8_t before[2048];int32_t output[2409];uint8_t after[2048];} c __attribute__((aligned(64)));
extern void gemmini_golden_gemm(const int8_t*,const int8_t*,int32_t*);
int main(void){
 for(int i=0;i<2409;i++)c.output[i]=-37;
 for(int i=0;i<2048;i++){c.before[i]=0x5a;c.after[i]=0xa5;}
 uint64_t begin,end;__asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 gemmini_golden_gemm(source,right,c.output);
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
 printf("SEGMENTED_INPUT_CYCLES %lu\\n",(unsigned long)(end-begin));
 for(int i=0;i<2409;i++)if(c.output[i]!=expected[i]){printf("FAIL output%d got%d expected%d\\n",i,c.output[i],expected[i]);return 1;}
 for(int i=0;i<2048;i++)if(c.before[i]!=0x5a||c.after[i]!=0xa5){printf("FAIL guard%d\\n",i);return 2;}
 const volatile int8_t* source_after=source;
 const volatile int8_t* right_after=right;
 for(int i=0;i<10000;i++)if(source_after[i]!=(int8_t)((i*37+11)%43-21)){printf("FAIL source%d\\n",i);return 3;}
 for(int i=0;i<4745;i++)if(right_after[i]!=(int8_t)((i*53+19)%41-20)){printf("FAIL right%d\\n",i);return 4;}
 printf("SEGMENTED_INPUT PASS all2409 guards4096 inputs14745\\n");return 0;
}
""")
    shape = Shape(
        m,
        n,
        k,
        output_dtype="i32",
        bm=3,
        bn=4,
        cache_a=True,
        wide_b=True,
        reuse_b=True,
        prefetch_b=True,
    )
    module = GoldenGemm(shape, resident_a_load_tiles=4, input_view=view).build()
    compilation = compile_module(module, args.llvm_bin, work)
    built = build_program(
        [program, work / "kernel.o"],
        work,
        target="gemmini",
        extra_cflags=[f"-I{work}", "-march=rv64gc", "-mabi=lp64d"],
        max_loaded_bytes=None,
    )
    audit = audit_elf(built.elf.read_bytes())
    assert audit["status"] == "pass"
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    argv = [
        str(spike),
        "-g",
        "--extension=gemmini",
        "--isa=rv64gc",
        "-m0x80000000:0x80000000",
        str(built.elf),
    ]
    replay = subprocess.run(
        argv, capture_output=True, text=True, timeout=120, check=False
    )
    (work / "spike.stdout").write_text(replay.stdout)
    (work / "spike.stderr").write_text(replay.stderr)
    marker = "SEGMENTED_INPUT PASS all2409 guards4096 inputs14745"
    assert replay.returncode == 0 and marker in replay.stdout
    run = run_on_gsim(
        built.elf,
        target="gemmini",
        max_cycles=1000000,
        timeout_s=300,
        backdoor=True,
        stdout_path=work / "gsim.stdout",
    )
    lines = [
        line.split()[-1]
        for line in run.stdout_tail.splitlines()
        if line.startswith("SEGMENTED_INPUT_CYCLES ")
    ]
    passed = run.completed and run.returncode == 0 and marker in run.stdout_tail
    result = {
        "schema": "segmented_input_numeric_capsule_v1",
        "status": "pass" if passed else "fail",
        "shape": shape.__dict__,
        "view": view.__dict__,
        "outputs_checked": 2409,
        "guards_checked": 4096,
        "input_bytes_unchanged": 14745,
        "elf": pin(built.elf),
        "audit": audit,
        "compilation": compilation,
        "pins": {
            name: pin(work / (name + ".bin"))
            for name in ("source", "right", "expected")
        },
        "driver": pin(Path(__file__)),
        "spike_engine": pin(spike),
        "spike_argv": argv,
        "gsim_engine": run.engine,
        "kernel_gsim_cycles": int(lines[-1]) if len(lines) == 1 else None,
        "gsim_scope": "Independent synthetic i32 shape/tail correctness; no stock or original-model performance claim",
    }
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("status", "kernel_gsim_cycles", "elf")},
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
