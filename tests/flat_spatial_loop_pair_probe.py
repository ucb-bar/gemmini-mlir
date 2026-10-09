"""Same-address AB/BA exact convolution pair with explicit CPU command loops.

This experiment driver owns its deterministic independent fixtures. It checks
every output and guard, preserves the source reduction and complete command
stream, and does not select a production model policy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from merlin.common.digest import sha256_file
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.no_fsm_audit import audit_elf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--spike", type=Path, required=True)
    for name, value in (
        ("h", 11),
        ("w", 19),
        ("cin", 65),
        ("cout", 73),
        ("bn", 4),
        ("band-rows", 7),
        ("stride", 1),
    ):
        parser.add_argument("--" + name, type=int, default=value)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--max-cycles", type=int, default=10000000)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    shape = ConvShape(
        args.h, args.w, args.cin, args.cout, stride=args.stride, bn=args.bn, wide_b=True
    )
    options = {
        "virtual_padding": True,
        "wide_a": True,
        "separate_b_bank": True,
        "band_rows": args.band_rows,
    }
    compiled = []
    for ordinal in (0, 1):
        module = GoldenFlatConv(shape, loop_spatial=bool(ordinal), **options).build()
        module.body.block.first_op.properties["sym_name"] = StringAttr(
            f"flat_arm{ordinal}"
        )
        folder = work / f"arm{ordinal}"
        receipt = compile_module(module, args.llvm_bin, folder)
        compiled.append({"compilation": receipt, "object": str(folder / "kernel.o")})
    lhs = (
        np.arange(shape.h * shape.w * shape.cin, dtype=np.int64) % 251 - 125
    ).reshape(shape.h, shape.w, shape.cin)
    rhs = (np.arange(9 * shape.cin * shape.cout, dtype=np.int64) % 241 - 120).reshape(
        3, 3, shape.cin, shape.cout
    )
    padded = np.pad(lhs, ((1, 1), (1, 1), (0, 0)))
    expected = np.zeros((shape.oh, shape.ow, shape.cout), dtype=np.int64)
    for kh in range(3):
        for kw in range(3):
            expected += (
                padded[
                    kh : kh + shape.oh * shape.stride : shape.stride,
                    kw : kw + shape.ow * shape.stride : shape.stride,
                ]
                @ rhs[kh, kw]
            )
    assembly = []
    arrays = {
        "lhs": lhs.astype(np.int8),
        "rhs": rhs.astype(np.int8),
        "expected": expected.astype("<i4"),
    }
    for name, array in arrays.items():
        path = work / (name + ".bin")
        path.write_bytes(array.tobytes())
        assembly.append(
            f'.section .data\n.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n'
        )
    asm = work / "fixture.S"
    asm.write_text("".join(assembly))
    data = work / "fixture.o"
    subprocess.run(
        [
            str(args.llvm_bin / "clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-c",
            str(asm),
            "-o",
            str(data),
        ],
        check=True,
        capture_output=True,
    )
    count = expected.size
    source = work / "probe.c"
    source.write_text(f"""#include <stdint.h>
#include <stdio.h>
extern int8_t lhs[{lhs.size}],rhs[{rhs.size}];
extern const int32_t expected[{count}];
extern void flat_arm0(const int8_t*,const int8_t*,int32_t*);
extern void flat_arm1(const int8_t*,const int8_t*,int32_t*);
static struct {{uint8_t before[2048];int32_t output[{count}];uint8_t after[2048];}} box __attribute__((aligned(64)));
static uint64_t cycle(void){{uint64_t x;__asm__ volatile("csrr %0, mcycle":"=r"(x)::"memory");return x;}}
int main(void){{
 for(int round=0;round<2;++round)for(int slot=0;slot<2;++slot){{
  int arm=slot^round;
  for(int i=0;i<{count};++i)box.output[i]=INT32_C(-193721);
  for(int i=0;i<2048;++i){{box.before[i]=0x5a;box.after[i]=0xa5;}}
  uint64_t start=cycle();
  if(arm)flat_arm1(lhs,rhs,box.output);else flat_arm0(lhs,rhs,box.output);
  uint64_t elapsed=cycle()-start;
  printf("FLAT_PAIR %d %d %lu\\n",round,arm,(unsigned long)elapsed);
  for(int i=0;i<{count};++i)if(box.output[i]!=expected[i]){{printf("FAIL %d %d %d\\n",i,box.output[i],expected[i]);return 1;}}
  for(int i=0;i<2048;++i)if(box.before[i]!=0x5a||box.after[i]!=0xa5){{printf("%s\\n","FAIL guard");return 2;}}
 }}
  for(int i=0;i<{lhs.size};++i)if(lhs[i]!=i%251-125){{printf("%s\\n","FAIL inputA");return 3;}}
  for(int i=0;i<{rhs.size};++i)if(rhs[i]!=i%241-120){{printf("%s\\n","FAIL inputB");return 4;}}
 printf("%s\\n","FLAT_PAIR PASS");return 0;
}}
""")
    built = build_program(
        [source, data, *[Path(x["object"]) for x in compiled]],
        work,
        target="gemmini",
        extra_cflags=["-march=rv64gc", "-mabi=lp64d", "-fno-builtin"],
        max_loaded_bytes=None,
    )
    audit = audit_elf(built.elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("linked final instruction audit failed")
    recipe = {
        "schema": "flat_spatial_command_loop_pair_v1",
        "shape": asdict(shape),
        "options": options,
        "elements": count,
        "compiled": compiled,
        "elf": str(built.elf),
        "elf_sha256": sha256_file(built.elf),
        "fixture_pins": {
            str(work / (name + ".bin")): sha256_file(work / (name + ".bin"))
            for name in arrays
        },
        "driver_sha256": sha256_file(Path(__file__)),
        "source_sha256": sha256_file(source),
        "nofsm": audit,
        "scope": "Same input/output addresses and immutable fixtures; complete primitive kernel ROI; exact i32 outputs, dirty guards and input rereads after all four calls. GSIM only, no stock/whole prediction.",
    }
    (work / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    with (work / "spike.log").open("w") as stdout:
        strict = subprocess.run(
            [str(args.spike), "--isa=rv64gc", "--extension=gemmini", str(built.elf)],
            stdout=stdout,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    if strict.returncode or "FLAT_PAIR PASS" not in (work / "spike.log").read_text():
        raise ValueError("strict Spike full output/guard/input gate failed")
    print("STRICT_PAIR_PASS", built.elf_sha256, flush=True)
    run = run_on_gsim(
        built.elf,
        target="gemmini",
        max_cycles=args.max_cycles,
        timeout_s=args.timeout_s,
        backdoor=True,
        stdout_path=work / "gsim.stdout",
    )
    lines = [
        tuple(map(int, row))
        for row in re.findall(
            r"^FLAT_PAIR (\d+) (\d+) (\d+)$", run.stdout_tail, re.MULTILINE
        )
    ]
    passed = (
        run.completed
        and run.returncode == 0
        and "FLAT_PAIR PASS" in run.stdout_tail
        and len(lines) == 4
    )
    result = {
        "recipe": recipe,
        "completed": run.completed,
        "returncode": run.returncode,
        "pass_all": passed,
        "cycles": lines,
        "engine": run.engine,
        "stdout": run.stdout_tail,
        "stderr": run.stderr_tail,
    }
    if passed:
        sums = [
            sum(cycles for _, selected, cycles in lines if selected == arm)
            for arm in (0, 1)
        ]
        result["paired_mean_fraction_change"] = sums[1] / sums[0] - 1
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("recipe", "engine", "stdout", "stderr")
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
