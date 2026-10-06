"""Complete ranked mean producer/finisher at common input/output addresses."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from merlin.llvmlower.guarded_quantized_mean import derive, emit_integer_sum_finish
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.device_mean_bundle import IntegerSumPlan, adapter_source
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import GoldenGemm
from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selector_offset(elf):
    raw = elf.read_bytes()
    start, width, count, strings = (
        int.from_bytes(raw[a:b], "little")
        for a, b in [(40, 48), (58, 60), (60, 62), (62, 64)]
    )
    row = raw[start + strings * width : start + (strings + 1) * width]
    labels = int.from_bytes(row[24:32], "little")
    for index in range(count):
        row = raw[start + index * width : start + (index + 1) * width]
        label = labels + int.from_bytes(row[:4], "little")
        if raw[label : raw.index(b"\0", label)] == b".selector":
            return int.from_bytes(row[24:32], "little")
    raise ValueError("selector section absent")


def original_output(input_array, proof):
    total = np.zeros((input_array.shape[0], input_array.shape[2]), dtype=np.float32)
    for index in range(proof["count"]):
        total = (
            input_array[:, index, :].astype(np.float32)
            * np.float32(proof["input_scale"])
            + total
        )
    return np.clip(
        np.rint((total / np.float32(proof["count"])) * np.float32(proof["reciprocal"])),
        -128,
        127,
    ).astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--benchmark-header", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path)
    parser.add_argument("--original-input", type=Path)
    parser.add_argument("--capture-receipt", type=Path)
    parser.add_argument("--control-bundle", type=Path)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if args.candidate_bundle:
        assert args.original_input and args.capture_receipt and args.control_bundle
        manifest = json.loads((args.candidate_bundle / "mean.json").read_text())
        control_manifest = json.loads((args.control_bundle / "mean.json").read_text())
        capture = json.loads(args.capture_receipt.read_text())
        assert len(manifest["routes"]) == len(control_manifest["routes"]) == 1
        route, old = manifest["routes"][0], control_manifest["routes"][0]
        assert (
            route["proof"] == old["proof"]
            and capture["all1000_original_f32_bits_exact"]
        )
        assert sha(args.original_input) == capture["original_mean_input_sha256"]
        plan = IntegerSumPlan(*route["output_shape"], route["proof"]["count"])
        input_array = np.frombuffer(args.original_input.read_bytes(), np.int8).reshape(
            plan.batches, plan.count, plan.channels
        )
        candidate = args.candidate_bundle / "adapter.o"
        control = args.control_bundle / "adapter.o"
        assert (
            sha(candidate) == manifest["object_sha256"]
            and sha(control) == control_manifest["object_sha256"]
        )
        objects = [candidate, control]
        candidate_symbol, control_symbol = route["symbol"], old["symbol"]
    else:
        plan = IntegerSumPlan(3, 17, 7)
        proof = derive(plan.count, 0.7, 0.2)
        route = {
            "proof": proof,
            "output_shape": [plan.batches, plan.channels],
            "input_matrix_shape": [plan.batches * plan.count, plan.channels],
        }
        input_array = np.random.default_rng(73).integers(
            -128, 128, (plan.batches, plan.count, plan.channels), dtype=np.int8
        )
        input_array[0, :, 0] = -128
        input_array[0, :, 1] = 127
        candidate_symbol, control_symbol = "device_mean", "cpu_mean"
        module = GoldenGemm(plan.shape()).build()
        module.body.block.first_op.properties["sym_name"] = StringAttr(
            candidate_symbol + "_integer_sum_kernel"
        )
        compile_module(module, args.llvm_bin, work / "producer")
        code = work / "candidate.c"
        code.write_text(adapter_source(route, candidate_symbol))
        control_code = work / "control.c"
        control_code.write_text(
            emit_integer_sum_finish(
                proof,
                "cpu_finish",
                plan.batches,
                plan.channels,
                input_layout="channel_minor",
            )
            + f"""
typedef struct {{void* allocated,*aligned;int64_t offset,size[2],stride[2];}} memref2;
void _mlir_ciface_{control_symbol}(memref2*r,memref2*a,memref2*out) {{
 int32_t sums[{plan.batches * plan.channels}];const int8_t*input=(const int8_t*)a->aligned+a->offset;
 for(int b=0;b<{plan.batches};++b)for(int c=0;c<{plan.channels};++c) {{
  int32_t sum=0;for(int k=0;k<{plan.count};++k)sum+=input[(b*{plan.count}+k)*{plan.channels}+c];
  sums[b*{plan.channels}+c]=sum;
 }}
 cpu_finish(input,sums,(int8_t*)out->aligned+out->offset);*r=*out;
}}
"""
        )
        objects = [work / "producer/kernel.o"]
        for source in (code, control_code):
            target = source.with_suffix(".o")
            subprocess.run(
                [
                    str(args.llvm_bin / "clang"),
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-O2",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                    "-c",
                    str(source),
                    "-o",
                    str(target),
                ],
                check=True,
                capture_output=True,
            )
            objects.append(target)
        capture = None
    plan.validate()
    proof = route["proof"]
    assert proof == derive(proof["count"], proof["input_scale"], proof["output_scale"])
    expected = original_output(input_array, proof)
    sums = input_array.astype(np.int32).sum(axis=1, dtype=np.int32)
    fixture = work / "fixture.S"
    assembly = ".section .rodata\n"
    for name, array in [
        ("input", input_array),
        ("check_input", input_array),
        ("expected", expected),
        ("sums", sums),
    ]:
        binary = work / (name + ".bin")
        binary.write_bytes(array.tobytes())
        assembly += (
            f'.balign 64\n.global fixture_{name}\nfixture_{name}:\n.incbin "{binary}"\n'
        )
    fixture.write_text(assembly)
    shutil.copyfile(args.benchmark_header, work / "benchmark_buffer.h")
    program = work / "probe.c"
    program.write_text(f"""#include <stdint.h>
#include <stdio.h>
#include "benchmark_buffer.h"
#define INPUTS {input_array.size}
#define OUTPUTS {expected.size}
typedef struct {{void*allocated,*aligned;int64_t offset,size[2],stride[2];}} memref2;
extern const int8_t fixture_input[INPUTS],fixture_check_input[INPUTS],fixture_expected[OUTPUTS];
extern const int32_t fixture_sums[OUTPUTS];
extern void _mlir_ciface_{control_symbol}(memref2*,memref2*,memref2*);
extern void _mlir_ciface_{candidate_symbol}(memref2*,memref2*,memref2*,memref2*);
volatile uint64_t arm_selector __attribute__((section(".selector")))=0;
static struct {{uint8_t before[2048];int8_t output[OUTPUTS];uint8_t after[2048];}} result __attribute__((aligned(64)));
static struct {{uint8_t before[2048];int32_t sums[OUTPUTS];uint8_t after[2048];}} scratch __attribute__((aligned(64)));
static void poison(void){{merlin_benchmark_fill(&result,0xa5,sizeof(result));merlin_benchmark_fill(&scratch,0x5a,sizeof(scratch));}}
static int guards(void){{for(int i=0;i<2048;++i)if(result.before[i]!=0xa5||result.after[i]!=0xa5||scratch.before[i]!=0x5a||scratch.after[i]!=0x5a)return 1;return 0;}}
static int equal(const void*a,const void*b,int n){{return merlin_benchmark_first_difference(a,b,n)==(unsigned)n;}}
static int invoke(uint64_t arm){{
 memref2 a={{(void*)fixture_input,(void*)fixture_input,0,{{{plan.batches * plan.count},{plan.channels}}},{{{plan.channels},1}}}};
 memref2 sums={{scratch.sums,scratch.sums,0,{{{plan.batches},{plan.channels}}},{{{plan.channels},1}}}};
 memref2 out={{result.output,result.output,0,{{{plan.batches},{plan.channels}}},{{{plan.channels},1}}}},ret;
 if(arm==0)_mlir_ciface_{control_symbol}(&ret,&a,&out);else _mlir_ciface_{candidate_symbol}(&ret,&a,&sums,&out);
 return ret.allocated!=out.allocated||ret.aligned!=out.aligned||ret.offset!=0||ret.size[0]!={plan.batches}||ret.size[1]!={plan.channels}||ret.stride[0]!={plan.channels}||ret.stride[1]!=1;
}}
int main(void){{
 for(uint64_t arm=0;arm<2;++arm){{poison();if(invoke(arm)||guards()||!equal(result.output,fixture_expected,OUTPUTS)||(arm&&!equal(scratch.sums,fixture_sums,OUTPUTS*4))){{printf("%s\\n","FAIL warmup");return 1;}}}}
 poison();uint64_t arm=arm_selector,begin,end;
 __asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 int bad=invoke(arm);
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
 printf("INTEGER_MEAN_CYCLES %lu %lu\\n",(unsigned long)arm,(unsigned long)(end-begin));
 if(bad||guards()||!equal(result.output,fixture_expected,OUTPUTS)||(arm&&!equal(scratch.sums,fixture_sums,OUTPUTS*4))||!equal(fixture_input,fixture_check_input,INPUTS)){{printf("%s\\n","FAIL timed result");return 2;}}
 printf("INTEGER_MEAN_PASS arm%lu all%d sums%d guards8192 inputs%d\\n",(unsigned long)arm,OUTPUTS,OUTPUTS,INPUTS);return 0;
}}
""")
    built = build_program(
        [program, fixture, *objects],
        work / "arm0",
        target="gemmini",
        extra_cflags=[
            f"-I{work}",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-frounding-math",
        ],
        max_loaded_bytes=None,
    )
    raw, offset = built.elf.read_bytes(), selector_offset(built.elf)
    assert raw[offset : offset + 8] == bytes(8)
    records = []
    for arm in range(2):
        directory = work / f"arm{arm}"
        directory.mkdir(exist_ok=True)
        elf = built.elf
        if arm:
            elf = directory / "layer.elf"
            changed = bytearray(raw)
            changed[offset] = arm
            elf.write_bytes(changed)
            assert [
                index
                for index, (old, new) in enumerate(zip(raw, changed))
                if old != new
            ] == [offset]
        audit = audit_elf(elf.read_bytes())
        assert audit["status"] == "pass"
        (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        marker = f"INTEGER_MEAN_PASS arm{arm} all{expected.size} sums{expected.size} guards8192 inputs{input_array.size}"
        spike = subprocess.run(
            [
                "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike",
                "-g",
                "--extension=gemmini",
                "--isa=RV64GC",
                "-m0x80000000:0x80000000",
                str(elf),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        (directory / "spike.stdout").write_text(spike.stdout + spike.stderr)
        assert spike.returncode == 0 and marker in spike.stdout + spike.stderr
        measured = run_on_gsim(
            elf,
            target="gemmini",
            max_cycles=10000000,
            timeout_s=1800,
            backdoor=True,
            stdout_path=directory / "gsim.stdout",
        )
        lines = [
            line.split()[-1]
            for line in measured.stdout_tail.splitlines()
            if line.startswith(f"INTEGER_MEAN_CYCLES {arm} ")
        ]
        passed = (
            measured.completed
            and measured.returncode == 0
            and marker in measured.stdout_tail
            and len(lines) == 1
        )
        record = {
            "arm": arm,
            "passed": passed,
            "cycles": int(lines[0]) if len(lines) == 1 else None,
            "elf": str(elf),
            "elf_sha256": sha(elf),
            "marker": marker,
            "engine": measured.engine,
            "stdout": measured.stdout_tail,
            "stderr": measured.stderr_tail,
        }
        (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print("ARM", arm, "PASS", passed, "CYCLES", record["cycles"], flush=True)
        if not passed:
            break
    receipt = {
        "schema": "integer_mean_complete_ranked_capsule_v1",
        "plan": plan.proof(),
        "source_certificate": proof,
        "original_capture": capture,
        "same_addresses": True,
        "selector_byte_offset": offset,
        "actual_control_object_sha256": sha(objects[1])
        if args.candidate_bundle
        else None,
        "records": records,
        "passed": len(records) == 2 and all(row["passed"] for row in records),
        "metric_scope": "Complete ranked integer producer DMA/readout/fence plus source-exact finishing; preallocated disjoint scratch/output. Qualification checks and common warmup outside ROI. Whole allocation and host boundary costs UNKNOWN.",
        "whole_cycles": "UNKNOWN",
    }
    (work / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert receipt["passed"]


if __name__ == "__main__":
    main()
