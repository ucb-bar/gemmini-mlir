"""Exact two-readout residual capsule, including both stores and joint decoding.

The shared certificate proves all source pairs. This probe independently closes
both target predictor tables before measuring one complete selected arm at
identical addresses. No whole-model or stock-memory performance is inferred.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.common.digest import sha256_file
from merlin.llvmlower.quantized_affine_joint import derive_joint, emit_joint_decoder
from merlin.llvmlower.quantized_affine_pair import derive, predictor_table, source_table
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_wide_resadd import build, tables
from mlir_oot.no_fsm_audit import audit_elf


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=18000000)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    proof = json.loads(args.certificate.read_text())
    assert proof == derive_joint(**proof["source"], predictors=proof["predictors"])
    assert proof["decoder_exact_for_all_pairs"]
    control = json.loads(args.control.read_text())
    assert control == derive(**control["source"], **control["predictor"])
    assert control["mismatched_pairs"] == 0 and control["source"] == proof["source"]
    capture = None
    if args.fixture:
        capture = json.loads((args.fixture / "receipt.json").read_text())
        assert capture["all1000_original_f32_bits_exact"]
        for name in ("lhs", "rhs"):
            assert (
                sha256_file(args.fixture / (name + ".bin")) == capture[name + "_sha256"]
            )
        a, b = [
            np.fromfile(args.fixture / (name + ".bin"), np.int8)
            for name in ("lhs", "rhs")
        ]
    else:
        a = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
        b = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    n = len(a)
    assert n == len(b) and n > 0 and n % 1024 == 0
    fixture = work / "fixture"
    fixture.mkdir()
    a.tofile(fixture / "a.bin")
    b.tofile(fixture / "b.bin")
    indices = a.astype(np.int16) + 128, b.astype(np.int16) + 128
    source_table(**proof["source"])[indices].tofile(fixture / "expected.bin")
    for index, predictor in enumerate(proof["predictors"]):
        predictor_table(**predictor)[indices].tofile(fixture / f"pred{index}.bin")
    contracts = [
        dict(control["predictor"], relu=proof["source"]["relu"]),
        *proof["predictors"],
    ]
    objects, compilations, chunks = [], {}, []
    coefficient_text = ""
    for index, contract in enumerate(contracts):
        module = build(n // 64, **contract, prefetch_m=True, banked_accumulators=True)
        module.body.block.first_op.properties["sym_name"] = StringAttr(
            f"joint_kernel{index}"
        )
        directory = work / f"device{index}"
        compilations[index] = compile_module(module, args.llvm_bin, directory)
        objects.append(directory / "kernel.o")
        chunks.append((contract["p"] + 126) // 127 + (contract["q"] + 126) // 127)
        coefficient_text += f"static const int8_t coefficients{index}[768] __attribute__((aligned(64)))={{"
        coefficient_text += (
            ",".join(map(str, tables(contract["p"], contract["q"]))) + "};\n"
        )
    (work / "tables.h").write_text(coefficient_text)
    assembly = work / "fixture.S"
    assembly.write_text(
        ".section .rodata\n"
        + "".join(
            f'.balign 64\n.globl fixture_{name}\nfixture_{name}:\n.incbin "{fixture / (file + ".bin")}"\n'
            for name, file in [
                ("a", "a"),
                ("b", "b"),
                ("check_a", "a"),
                ("check_b", "b"),
                ("expected", "expected"),
                ("pred0", "pred0"),
                ("pred1", "pred1"),
            ]
        )
    )
    decoder = emit_joint_decoder(proof, "decode_joint", packed=True)
    source = work / "probe.c"
    source.write_text(
        decoder
        + f"""
#include <stdio.h>
#include "tables.h"
#define N {n}
extern const int8_t fixture_a[N],fixture_b[N],fixture_check_a[N],fixture_check_b[N],fixture_expected[N],fixture_pred0[N],fixture_pred1[N];
extern void joint_kernel0(const int8_t*,const int8_t*,int8_t*,const int8_t*);
extern void joint_kernel1(const int8_t*,const int8_t*,int8_t*,const int8_t*);
extern void joint_kernel2(const int8_t*,const int8_t*,int8_t*,const int8_t*);
volatile uint64_t arm_selector __attribute__((section(".selector")))=ARM_SELECTOR;
static struct {{uint8_t before[2048];int8_t output[N];uint8_t after[2048];}} first __attribute__((aligned(64))),second __attribute__((aligned(64)));
typedef uint64_t checkword __attribute__((may_alias));
static int different(const int8_t*a,const int8_t*b,size_t n){{
 size_t i=0;for(;n-i>=8;i+=8)if(*(const checkword*)(a+i)!=*(const checkword*)(b+i))return 1;
 for(;i<n;i++)if(a[i]!=b[i])return 1;return 0;
}}
static int guards(void){{for(size_t i=0;i<2048;i++)if(first.before[i]!=0x5a||first.after[i]!=0xa5||second.before[i]!=0x5a||second.after[i]!=0xa5)return 1;return 0;}}
int main(void){{
 for(size_t i=0;i<2048;i++){{first.before[i]=second.before[i]=0x5a;first.after[i]=second.after[i]=0xa5;}}
 joint_kernel0(fixture_a,fixture_b,first.output,coefficients0);
 if(different(first.output,fixture_expected,N)||guards()){{printf("FAIL control\\n");return 1;}}
 joint_kernel1(fixture_a,fixture_b,first.output,coefficients1);
 joint_kernel2(fixture_a,fixture_b,second.output,coefficients2);
 if(different(first.output,fixture_pred0,N)||different(second.output,fixture_pred1,N)||guards()){{printf("FAIL predictors\\n");return 2;}}
 printf("JOINT_PREDICTORS PASS all{n} guards8192\\n");
 for(size_t i=0;i<N;i++){{first.output[i]=-37;second.output[i]=-53;}}
 uint64_t begin,end,arm=arm_selector;
 __asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 if(arm==0)joint_kernel0(fixture_a,fixture_b,first.output,coefficients0);
 else{{joint_kernel1(fixture_a,fixture_b,first.output,coefficients1);joint_kernel2(fixture_a,fixture_b,second.output,coefficients2);decode_joint(first.output,second.output,N);}}
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
 printf("JOINT_CYCLES %lu %lu\\n",(unsigned long)arm,(unsigned long)(end-begin));
 if(different(first.output,fixture_expected,N)||guards()){{printf("FAIL decoded\\n");return 3;}}
 if(arm&&different(second.output,fixture_pred1,N)){{printf("FAIL readonly\\n");return 4;}}
 if(different(fixture_a,fixture_check_a,N)||different(fixture_b,fixture_check_b,N)){{printf("FAIL inputs\\n");return 5;}}
 printf("JOINT_RESIDUAL PASS arm%lu all{n} guards8192 inputs{2 * n}\\n",(unsigned long)arm);return 0;
}}
"""
    )
    built, audits = {}, {}
    for arm in (0, 1):
        built[arm] = build_program(
            [*objects, source, assembly],
            work / f"arm{arm}",
            target="gemmini",
            extra_cflags=[
                "-march=rv64gc",
                "-mabi=lp64d",
                "-fno-builtin",
                "-fno-fast-math",
                "-ffp-contract=off",
                f"-I{work}",
                f"-DARM_SELECTOR={arm}",
            ],
            max_loaded_bytes=None,
        )
        audits[arm] = audit_elf(built[arm].elf.read_bytes())
        assert audits[arm]["status"] == "pass"
    lhs, rhs = [built[arm].elf.read_bytes() for arm in (0, 1)]
    assert len(lhs) == len(rhs)
    differences = [i for i, (x, y) in enumerate(zip(lhs, rhs, strict=True)) if x != y]
    assert (
        len(differences) == 1 and lhs[differences[0]] == 0 and rhs[differences[0]] == 1
    )
    recipe = {
        "schema": "gemmini_joint_affine_pair_capsule_v1",
        "elements": n,
        "full_pair_domain": args.fixture is None,
        "certificate": pin(args.certificate),
        "control_certificate": pin(args.control),
        "source_capture": pin(args.fixture / "receipt.json") if capture else None,
        "driver": pin(Path(__file__)),
        "source": pin(source),
        "coefficients": pin(work / "tables.h"),
        "compilations": compilations,
        "chunks": chunks,
        "fixture_pins": {path.name: pin(path) for path in fixture.iterdir()},
        "same_elf_layout": True,
        "selector_byte_ledger": [
            {"offset": i, "before": lhs[i], "after": rhs[i]} for i in differences
        ],
        "timing_scope": "Both complete predictor calls including repeated input DMA/config/fences/stores plus complete joint decoder scan; control one complete exact wide call; GSIM only",
        "warmup_scope": "Same control and both predictors fully executed and numerically checked before either selected timed arm",
        "caller_obligations": "Distinct guarded first/second outputs, both based on same unchanged original inputs, readonly second through decode, complete target predictors independently checked",
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
        print("STRICT_JOINT_PASS", arm, built[arm].elf_sha256, flush=True)
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
        results[arm] = {
            "cycles": int(cycles[0]),
            "elf": pin(built[arm].elf),
            "nofsm": audits[arm],
            "spike_argv": argv,
            "spike_engine": pin(spike),
            "spike_stdout": pin(stdout),
            "spike_stderr": pin(stderr),
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
        "JOINT_PAIR_CLOSED",
        result["delta_cycles"],
        result["fraction_change"],
        flush=True,
    )


if __name__ == "__main__":
    main()
