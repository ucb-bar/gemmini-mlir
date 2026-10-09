"""Same-address exact stem/pool CPU-loop comparison with complete fixtures."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from merlin.common.digest import sha256_file
from merlin.perf.layer_bench import build_program, run_on_gsim
from test_stem_spatial_command_loops import command_stream
from xdsl.dialects.builtin import StringAttr

from mlir_oot.executed_features import census
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_stem_pool import GoldenStemPool, StemPoolShape
from mlir_oot.no_fsm_audit import audit_elf


def pin(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--h", type=int, default=17)
    parser.add_argument("--w", type=int, default=35)
    parser.add_argument("--cout", type=int, default=19)
    parser.add_argument("--max-cycles", type=int, default=6000000)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    capture = None
    if args.fixture:
        capture = json.loads((args.fixture / "receipt.json").read_text())
        assert capture["all_original_f32_bits_exact"]
        shape = StemPoolShape(**capture["shape"])
        fixture = args.fixture.resolve()
        for name in ("a", "b", "expected"):
            assert sha256_file(fixture / (name + ".bin")) == capture[name + "_sha256"]
    else:
        shape = StemPoolShape(args.h, args.w, args.cout, 0.03125)
        fixture = work / "fixture"
        fixture.mkdir()
        a = (
            (np.arange((shape.h + 6) * (shape.w + 6) * 3) % 15 - 7)
            .astype(np.int8)
            .reshape(shape.h + 6, shape.w + 6, 3)
        )
        b = (
            (np.arange(147 * shape.cout) % 11 - 5)
            .astype(np.int8)
            .reshape(7, 7, 3, shape.cout)
        )
        acc = np.zeros((shape.oh, shape.ow, shape.cout), np.int32)
        for ky in range(7):
            for kx in range(7):
                acc += a[ky : ky + 2 * shape.oh : 2, kx : kx + 2 * shape.ow : 2].astype(
                    np.int32
                ) @ b[ky, kx].astype(np.int32)
        relu = np.maximum(acc, 0)
        padded = np.pad(relu, ((1, 1), (1, 1), (0, 0)))
        pooled = np.zeros((shape.ph, shape.pw, shape.cout), np.int32)
        for ky in range(3):
            for kx in range(3):
                pooled = np.maximum(
                    pooled,
                    padded[ky : ky + 2 * shape.ph : 2, kx : kx + 2 * shape.pw : 2],
                )
        expected = np.clip(
            np.rint(pooled.astype(np.float32) * np.float32(shape.scale)), 0, 127
        ).astype(np.int8)
        for name, data in [("a", a), ("b", b), ("expected", expected)]:
            (fixture / (name + ".bin")).write_bytes(data.tobytes())
    modules = [GoldenStemPool(shape, loop_spatial=x).build() for x in (False, True)]
    streams = [command_stream(module) for module in modules]
    assert streams[0] == streams[1]
    compilations = {}
    objects = []
    for arm, module in enumerate(modules):
        module.body.block.first_op.properties["sym_name"] = StringAttr(f"stem_arm{arm}")
        directory = work / f"device{arm}"
        compilations[arm] = compile_module(module, args.llvm_bin, directory)
        objects.append(directory / "kernel.o")
    a_count = (shape.h + 6) * (shape.w + 6) * 3
    b_count = 147 * shape.cout
    count = shape.ph * shape.pw * shape.cout
    assembly = work / "fixture.S"
    assembly.write_text(
        ".section .rodata\n"
        + "".join(
            f'.balign 64\n.globl fixture_{name}\nfixture_{name}:\n.incbin "{(fixture / (source + ".bin")).resolve()}"\n'
            for name, source in [
                ("a", "a"),
                ("b", "b"),
                ("expected", "expected"),
                ("check_a", "a"),
                ("check_b", "b"),
            ]
        )
    )
    source = work / "probe.c"
    source.write_text(f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern const int8_t fixture_a[{a_count}],fixture_b[{b_count}],fixture_expected[{count}],fixture_check_a[{a_count}],fixture_check_b[{b_count}];
extern void stem_arm0(const int8_t*,const int8_t*,int8_t*);
extern void stem_arm1(const int8_t*,const int8_t*,int8_t*);
volatile uint64_t arm_selector __attribute__((section(".selector")))=ARM_SELECTOR;
static struct {{uint8_t before[2048];int8_t output[{count}];uint8_t after[2048];}} box __attribute__((aligned(64)));
static int different(const int8_t*a,const int8_t*b,int n){{for(int i=0;i<n;i++)if(a[i]!=b[i])return 1;return 0;}}
int main(void){{
 for(int i=0;i<{count};i++)box.output[i]=-37;
 for(int i=0;i<2048;i++){{box.before[i]=0x5a;box.after[i]=0xa5;}}
 uint64_t begin,end,arm=arm_selector;
 __asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 if(arm==0)stem_arm0(fixture_a,fixture_b,box.output);else stem_arm1(fixture_a,fixture_b,box.output);
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
 printf("STEM_LOOP_CYCLES %lu %lu\\n",(unsigned long)arm,(unsigned long)(end-begin));
 if(different(box.output,fixture_expected,{count})){{printf("FAIL output\\n");return 1;}}
 for(int i=0;i<2048;i++)if(box.before[i]!=0x5a||box.after[i]!=0xa5){{printf("FAIL guard\\n");return 2;}}
 if(different(fixture_a,fixture_check_a,{a_count})||different(fixture_b,fixture_check_b,{b_count})){{printf("FAIL input\\n");return 3;}}
 printf("STEM_LOOP PASS arm%lu all{count} guards4096 inputs{a_count + b_count}\\n",(unsigned long)arm);return 0;
}}
""")
    built = {}
    audits = {}
    for arm in (0, 1):
        built[arm] = build_program(
            [*objects, source, assembly],
            work / f"arm{arm}",
            target="gemmini",
            extra_cflags=["-march=rv64gc", "-mabi=lp64d", "-fno-builtin", f"-DARM_SELECTOR={arm}"],
            max_loaded_bytes=None,
        )
        audits[arm] = audit_elf(built[arm].elf.read_bytes())
        assert audits[arm]["status"] == "pass"
    lhs, rhs = [built[arm].elf.read_bytes() for arm in (0, 1)]
    assert len(lhs) == len(rhs)
    differences = [i for i, (a, b) in enumerate(zip(lhs, rhs, strict=True)) if a != b]
    assert (
        len(differences) == 1 and lhs[differences[0]] == 0 and rhs[differences[0]] == 1
    )
    recipe = {
        "schema": "same_address_stem_spatial_loop_pair_v1",
        "shape": asdict(shape),
        "source_capture": pin(args.fixture / "receipt.json") if capture else None,
        "complete_ordered_command_identity": True,
        "commands_per_call": len(streams[0]),
        "primitive_scope": "All encoded primitive operands, DMA source/destination pointer bases/offsets, pool fields and fences",
        "compilations": compilations,
        "fixture_pins": {
            name: pin(fixture / (name + ".bin")) for name in ("a", "b", "expected")
        },
        "driver": pin(Path(__file__)),
        "source": pin(source),
        "same_elf_layout": True,
        "selector_byte_ledger": [
            {"offset": i, "before": lhs[i], "after": rhs[i]} for i in differences
        ],
        "numeric_scope": "Every output byte, prefix/suffix dirty guards and unchanged A/B bytes; exact source scale and source K order",
        "timing_scope": "One complete primitive stem+pool call, GSIM only; original capture when supplied; no stock or whole-model inference",
        "token_usage_available": False,
    }
    (work / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    results = {}
    for arm in (0, 1):
        directory = work / f"arm{arm}"
        stdout, stderr = directory / "spike.stdout", directory / "spike.stderr"
        argv = [
            str(spike),
            "-g",
            "--extension=gemmini",
            "--isa=rv64gc",
            "-m0x80000000:0x80000000",
            str(built[arm].elf),
        ]
        with stdout.open("w") as out, stderr.open("w") as err:
            replay = subprocess.run(
                argv, stdout=out, stderr=err, timeout=300, check=False
            )
        marker = (
            f"STEM_LOOP PASS arm{arm} all{count} guards4096 inputs{a_count + b_count}"
        )
        assert replay.returncode == 0 and marker in stdout.read_text()
        features = census(
            built[arm].elf,
            stderr,
            symbols=[f"stem_arm{arm}"],
            scope_id=f"stem_capsule_arm{arm}",
            readelf="readelf",
        )
        (directory / "features.json").write_text(json.dumps(features, indent=2) + "\n")
        print("STRICT_STEM_PAIR_PASS", arm, built[arm].elf_sha256, flush=True)
        run = run_on_gsim(
            built[arm].elf,
            target="gemmini",
            max_cycles=args.max_cycles,
            timeout_s=900,
            backdoor=True,
            stdout_path=directory / "gsim.stdout",
        )
        lines = [
            line.split()[-1]
            for line in run.stdout_tail.splitlines()
            if line.startswith(f"STEM_LOOP_CYCLES {arm} ")
        ]
        assert (
            run.completed
            and run.returncode == 0
            and marker in run.stdout_tail
            and len(lines) == 1
        )
        results[arm] = {
            "cycles": int(lines[0]),
            "elf": pin(built[arm].elf),
            "nofsm": audits[arm],
            "spike_argv": argv,
            "spike_stdout": pin(stdout),
            "spike_stderr": pin(stderr),
            "executed_features": pin(directory / "features.json"),
            "engine": run.engine,
            "completed": run.completed,
            "returncode": run.returncode,
            "stdout": run.stdout_tail,
            "stderr": run.stderr_tail,
        }
        (directory / "result.json").write_text(
            json.dumps(results[arm], indent=2) + "\n"
        )
    result = {
        "recipe": pin(work / "recipe.json"),
        "arms": results,
        "delta_cycles": results[1]["cycles"] - results[0]["cycles"],
        "fraction_change": results[1]["cycles"] / results[0]["cycles"] - 1,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "STEM_LOOP_PAIR_CLOSED",
        result["delta_cycles"],
        result["fraction_change"],
        flush=True,
    )


if __name__ == "__main__":
    main()
