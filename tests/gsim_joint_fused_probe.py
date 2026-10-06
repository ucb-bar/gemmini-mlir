"""Close one resident-input fused pair against the same exact control capsule.

Reuse a pinned prepared separate-predictor capsule's data/control objects and
portable decoder. This changes only the two predictor calls to one explicitly
compiled target family. Same-address control/selected ELFs differ one byte.
"""

import argparse
import json
import subprocess
from pathlib import Path

from gsim_joint_affine_pair_probe import pin
from merlin.llvmlower.quantized_affine_joint import derive_joint
from merlin.perf.layer_bench import build_program, run_on_gsim

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_joint_resadd import build, tables
from mlir_oot.no_fsm_audit import audit_elf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=18000000)
    parser.add_argument("--strict-only", action="store_true")
    args = parser.parse_args()
    prepared, work = args.prepared.resolve(), args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    original = json.loads((prepared / "recipe.json").read_text())
    assert pin(original["source"]["path"]) == original["source"]
    assert pin(original["coefficients"]["path"]) == original["coefficients"]
    assert pin(original["certificate"]["path"]) == original["certificate"]
    for fixture in original["fixture_pins"].values():
        assert pin(fixture["path"]) == fixture
    proof = json.loads(Path(original["certificate"]["path"]).read_text())
    assert proof == derive_joint(**proof["source"], predictors=proof["predictors"])
    assert proof["decoder_exact_for_all_pairs"]
    n = original["elements"]
    module = build(n // 64, proof["predictors"])
    compilation = compile_module(module, args.llvm_bin, work / "fused")
    (work / "joint_tables.h").write_text(
        "static const int8_t joint_coefficients[1280] __attribute__((aligned(64)))={"
        + ",".join(map(str, tables(proof["predictors"])))
        + "};\n"
    )
    old = Path(original["source"]["path"]).read_text()
    warmup = " joint_kernel1(fixture_a,fixture_b,first.output,coefficients1);\n joint_kernel2(fixture_a,fixture_b,second.output,coefficients2);"
    timed = "joint_kernel1(fixture_a,fixture_b,first.output,coefficients1);joint_kernel2(fixture_a,fixture_b,second.output,coefficients2);"
    call = "gemmini_golden_joint_resadd(fixture_a,fixture_b,first.output,second.output,joint_coefficients);"
    assert old.count(warmup) == 1 and old.count(timed) == 1
    source = work / "probe.c"
    source.write_text(
        '#include <stdint.h>\n#include "joint_tables.h"\nextern void gemmini_golden_joint_resadd(const int8_t*,const int8_t*,int8_t*,int8_t*,const int8_t*);\n'
        + old.replace(warmup, " " + call).replace(timed, call)
    )
    built, audits = {}, {}
    objects = [prepared / f"device{index}" / "kernel.o" for index in (0, 1, 2)]
    objects.append(work / "fused" / "kernel.o")
    for arm in (0, 1):
        built[arm] = build_program(
            [*objects, source, prepared / "fixture.S"],
            work / f"arm{arm}",
            target="gemmini",
            extra_cflags=[
                "-march=rv64gc",
                "-mabi=lp64d",
                "-fno-builtin",
                "-fno-fast-math",
                "-ffp-contract=off",
                f"-I{prepared}",
                f"-I{work}",
                f"-DARM_SELECTOR={arm}",
            ],
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
        "schema": "resident_input_joint_readout_pair_v1",
        "prepared_recipe": pin(prepared / "recipe.json"),
        "certificate": original["certificate"],
        "compilation": compilation,
        "driver": pin(__file__),
        "source": pin(source),
        "coefficient_header": pin(work / "joint_tables.h"),
        "objects": [pin(path) for path in objects],
        "fixture_assembly": pin(prepared / "fixture.S"),
        "selector_byte_ledger": [
            {"offset": i, "before": lhs[i], "after": rhs[i]} for i in differences
        ],
        "metric_scope": "Complete single resident-input two-readout kernel plus full decoder scan; unchanged complete39chunk control; paired GSIM only",
        "warmup_scope": "Same control and fused pair fully execute/validate before either timed selected arm",
        "token_usage_available": False,
    }
    (work / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    results = {}
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
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
                argv, stdout=out, stderr=err, timeout=600, check=False
            )
        marker = f"JOINT_RESIDUAL PASS arm{arm} all{n} guards8192 inputs{2 * n}"
        assert replay.returncode == 0 and marker in stdout.read_text()
        results[arm] = {
            "elf": pin(built[arm].elf),
            "nofsm": audits[arm],
            "spike_argv": argv,
            "spike_engine": pin(spike),
            "spike_stdout": pin(stdout),
            "spike_stderr": pin(stderr),
        }
        print("STRICT_FUSED_JOINT_PASS", arm, flush=True)
        if not args.strict_only:
            run = run_on_gsim(
                built[arm].elf,
                target="gemmini",
                max_cycles=args.max_cycles,
                timeout_s=1200,
                backdoor=True,
                stdout_path=directory / "gsim.stdout",
            )
            cycles = [
                line.split()[-1]
                for line in run.stdout_tail.splitlines()
                if line.startswith(f"JOINT_CYCLES {arm} ")
            ]
            assert (
                run.completed
                and run.returncode == 0
                and marker in run.stdout_tail
                and len(cycles) == 1
            )
            results[arm].update(
                cycles=int(cycles[0]),
                engine=run.engine,
                completed=run.completed,
                returncode=run.returncode,
                stdout=run.stdout_tail,
                stderr=run.stderr_tail,
            )
        (directory / "result.json").write_text(
            json.dumps(results[arm], indent=2) + "\n"
        )
    result = {
        "recipe": pin(work / "recipe.json"),
        "arms": results,
        "strict_only": args.strict_only,
    }
    if not args.strict_only:
        result.update(
            delta_cycles=results[1]["cycles"] - results[0]["cycles"],
            fraction_change=results[1]["cycles"] / results[0]["cycles"] - 1,
        )
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "FUSED_JOINT_PAIR_CLOSED",
        result.get("delta_cycles"),
        result.get("fraction_change"),
        flush=True,
    )


if __name__ == "__main__":
    main()
