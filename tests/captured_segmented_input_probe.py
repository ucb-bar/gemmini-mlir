"""Same-address original-consumed-value copy/borrowed-input comparison.

Both linked arms include both exact kernels and the same fixture/storage. Their
only final-ELF byte difference is the explicit volatile arm selector. Unread
owner cells are synthetic zeros; all consumed matrix values retain their source
capture pins. This is a capsule, not a whole-model host schedule or stock claim.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.llvmlower.segmented_matrix_view import SegmentedRows
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

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
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--view-proofs", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    capture = json.loads((args.fixture / "receipt.json").read_text())
    assert (
        capture["kernel_symbol"] == args.kernel
        and capture["all1000_original_f32_bits_exact"]
    )
    for name in ("a", "b", "expected"):
        assert (
            pin(args.fixture / (name + ".bin"))["sha256"] == capture[name + "_sha256"]
        )
    matches = json.loads(args.view_proofs.read_text())["matches"]
    entry = next(v for v in matches if v["source_callee"] + "_kernel" == args.kernel)
    assert (
        entry["shape"] == capture["shape"]
        and entry["source_numeric_proof"] == capture["source_numeric_proof"]
    )
    shape = Shape(**capture["shape"])
    view = SegmentedRows(**entry["source_view"]["address"])
    assert shape.output_dtype == "i8" and not shape.bias
    matrix = np.frombuffer(
        (args.fixture / "a.bin").read_bytes(), dtype=np.int8
    ).reshape(shape.m, shape.k)
    owner = np.zeros(view.source_elements, dtype=np.int8)
    for row in range(shape.m):
        owner[view.offset(row) : view.offset(row) + shape.k] = matrix[row]
    owner_path = work / "owner.bin"
    owner_path.write_bytes(owner.tobytes())
    for row in range(shape.m):
        assert np.array_equal(
            owner[view.offset(row) : view.offset(row) + shape.k], matrix[row]
        )
    objects = []
    compilations = {}
    for name, contract in [("control", None), ("borrowed", view)]:
        generator = GoldenGemm(shape, input_view=contract)
        module = generator.build()
        module.body.block.first_op.properties["sym_name"] = StringAttr(
            "capsule_" + name
        )
        directory = work / name
        compilations[name] = compile_module(module, args.llvm_bin, directory)
        objects.append(directory / "kernel.o")
    assembly = work / "fixture.S"
    code = ".section .rodata\n"
    for name, path in [
        ("fixture_owner", owner_path),
        ("fixture_b", args.fixture / "b.bin"),
        ("fixture_expected", args.fixture / "expected.bin"),
    ]:
        code += f'.balign 64\n.globl {name}\n{name}:\n.incbin "{path.resolve()}"\n'
    assembly.write_text(code)
    m, n, k = shape.m, shape.n, shape.k
    count = m * n
    assert count % 8 == 0
    program = work / "probe.c"
    program.write_text(f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern const int8_t fixture_owner[{view.source_elements}],fixture_b[{k * n}],fixture_expected[{count}];
extern void capsule_control(const int8_t*,const int8_t*,int8_t*);
extern void capsule_borrowed(const int8_t*,const int8_t*,int8_t*);
volatile uint64_t arm_selector __attribute__((section(".selector")))=ARM_SELECTOR;
static int8_t packed_a[{m * k}] __attribute__((aligned(64)));
static struct {{uint8_t before[2048];int8_t data[{count}];uint8_t after[2048];}} out __attribute__((aligned(64)));
int main(void){{
 for(int i=0;i<{count};i++)out.data[i]=-37;
 for(int i=0;i<2048;i++){{out.before[i]=0x5a;out.after[i]=0xa5;}}
 uint64_t begin,end,arm=arm_selector;
 __asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 if(arm==0){{
  for(int row=0;row<{m};row++){{
   intptr_t off={view.origin}+(row/{view.segment_rows})*{view.segment_stride}+(row%{view.segment_rows})*{view.row_stride};
   memcpy(packed_a+row*{k},fixture_owner+off,{k});
  }}
  capsule_control(packed_a,fixture_b,out.data);
 }}else capsule_borrowed(fixture_owner,fixture_b,out.data);
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
 printf("SEGMENTED_CAPTURE_CYCLES %lu %lu\\n",(unsigned long)arm,(unsigned long)(end-begin));
 for(int i=0;i<{count};i+=8){{
  uint64_t got,want;memcpy(&got,out.data+i,8);memcpy(&want,fixture_expected+i,8);
  if(got!=want){{printf("FAIL output%d\\n",i);return 1;}}
 }}
 for(int i=0;i<2048;i++)if(out.before[i]!=0x5a||out.after[i]!=0xa5){{printf("FAIL guard%d\\n",i);return 2;}}
 printf("SEGMENTED_CAPTURE PASS arm%lu all{count} guards4096\\n",(unsigned long)arm);return 0;
}}
""")
    built = {}
    for arm in (0, 1):
        directory = work / ("arm" + str(arm))
        built[arm] = build_program(
            [*objects, program, assembly],
            directory,
            target="gemmini",
            extra_cflags=["-march=rv64gc", "-mabi=lp64d", f"-DARM_SELECTOR={arm}"],
            max_loaded_bytes=None,
        )
        assert audit_elf(built[arm].elf.read_bytes())["status"] == "pass"
    a, b = [built[i].elf.read_bytes() for i in (0, 1)]
    assert len(a) == len(b)
    changes = [i for i, (lhs, rhs) in enumerate(zip(a, b, strict=True)) if lhs != rhs]
    assert len(changes) == 1 and a[changes[0]] == 0 and b[changes[0]] == 1
    gate = {}
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    for arm in (0, 1):
        argv = [
            str(spike),
            "-g",
            "--extension=gemmini",
            "--isa=rv64gc",
            "-m0x80000000:0x80000000",
            str(built[arm].elf),
        ]
        replay = subprocess.run(
            argv, capture_output=True, text=True, timeout=120, check=False
        )
        directory = work / ("arm" + str(arm))
        (directory / "spike.stdout").write_text(replay.stdout)
        (directory / "spike.stderr").write_text(replay.stderr)
        marker = f"SEGMENTED_CAPTURE PASS arm{arm} all{count} guards4096"
        assert replay.returncode == 0 and marker in replay.stdout
        gate[arm] = {
            "elf": pin(built[arm].elf),
            "spike_argv": argv,
            "outputs_checked": count,
            "guards_checked": 4096,
        }
    recipe = {
        "schema": "captured_segmented_input_same_address_recipe_v1",
        "pins": {
            "source_fixture_receipt": pin(args.fixture / "receipt.json"),
            "source_view_proofs": pin(args.view_proofs),
            "owner_fixture": pin(owner_path),
            "driver": pin(Path(__file__)),
        },
        "shape": shape.__dict__,
        "view": view.__dict__,
        "same_elf_layout": True,
        "elf_byte_change_ledger": [
            {"offset": i, "before": a[i], "after": b[i]} for i in changes
        ],
        "numeric_gates": gate,
        "compilations": compilations,
        "fixture_scope": "Every consumed matrix value is the original captured value; unread owner cells are zeros, not the complete original physical producer allocation",
        "timing_scope": "Portable generated CPU memcpy materialization plus the unchanged dense device versus borrowed segmented device; not exact current whole host code or stock cycles",
    }
    (work / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    results = {}
    for arm in (0, 1):
        directory = work / ("arm" + str(arm))
        run = run_on_gsim(
            built[arm].elf,
            target="gemmini",
            max_cycles=5000000,
            timeout_s=600,
            backdoor=True,
            stdout_path=directory / "gsim.stdout",
        )
        marker = f"SEGMENTED_CAPTURE PASS arm{arm} all{count} guards4096"
        lines = [
            line.split()[-1]
            for line in run.stdout_tail.splitlines()
            if line.startswith(f"SEGMENTED_CAPTURE_CYCLES {arm} ")
        ]
        passed = run.completed and run.returncode == 0 and marker in run.stdout_tail
        results[arm] = {
            "status": "pass" if passed else "fail",
            "engine": run.engine,
            "cycles": int(lines[-1]) if len(lines) == 1 else None,
            "stdout": run.stdout_tail,
            "stderr": run.stderr_tail,
            "elf": pin(built[arm].elf),
        }
        (directory / "result.json").write_text(
            json.dumps(results[arm], indent=2) + "\n"
        )
        assert passed
    result = {
        "recipe": pin(work / "recipe.json"),
        "arms": results,
        "delta_cycles": results[1]["cycles"] - results[0]["cycles"],
        "improvement_percent": 100 * (1 - results[1]["cycles"] / results[0]["cycles"]),
    }
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
