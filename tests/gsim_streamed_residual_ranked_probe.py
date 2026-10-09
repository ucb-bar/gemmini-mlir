"""Actual exact and serial ranked residual adapters versus streamed correction.

Three common-address arms retain the actual exact39 and serial5 source-bound
adapters plus the new streamed5 schedule. Original descriptors, inputs and
source floating policy remain fixed. Warmup/checks stay outside the complete
producer/adapter/correction ROI. No model route is enabled.
"""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from merlin.llvmlower.quantized_affine_pair import (
    derive,
    source_table,
)
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_streamed_resadd import build as streamed
from mlir_oot.golden_streamed_resadd import ranked_adapter
from mlir_oot.golden_wide_resadd import build
from mlir_oot.joint_residual_catalog import _single_adapter
from mlir_oot.no_fsm_audit import audit_elf


def pin(path):
    p = Path(path).resolve()
    return {
        "path": str(p),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
    }


def command(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


def extract_text_read_only(objcopy, obj, text_output, object_output):
    """GNU objcopy may rewrite its input unless an output object is explicit."""
    obj, text_output, object_output = map(Path, (obj, text_output, object_output))
    assert len({p.resolve() for p in (obj, text_output, object_output)}) == 3
    assert not text_output.exists() and not object_output.exists()
    original = pin(obj)
    command(
        [
            str(objcopy),
            "--dump-section",
            ".text=" + str(text_output),
            str(obj),
            str(object_output),
        ]
    )
    assert pin(obj) == original
    return pin(text_output)["sha256"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-bundle", type=Path, required=True)
    cli.add_argument("--source-route", type=int, default=0)
    cli.add_argument("--exact-control-bundle", type=Path, required=True)
    cli.add_argument("--storage-proof", type=Path, required=True)
    cli.add_argument("--independent-m", type=int)
    cli.add_argument("--independent-proof", type=Path)
    cli.add_argument("--original-a", type=Path)
    cli.add_argument("--original-b", type=Path)
    cli.add_argument("--source-capture-receipt", type=Path)
    cli.add_argument("--llvm-bin", type=Path, required=True)
    cli.add_argument("--benchmark-header", type=Path, required=True)
    cli.add_argument("--workdir", type=Path, required=True)
    cli.add_argument("--timeout", type=int, default=5400)
    cli.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = cli.parse_args()
    if args.independent_m is not None or args.independent_proof is not None:
        raise ValueError(
            "this normal storage probe requires the bound original source route"
        )
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source_bundle.read_text())
    route = manifest["routes"][args.source_route].copy()
    assert route["single_output_guard"] and route["n"] == 64
    gcc = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin")
    objcopy, readelf = (
        str(gcc / "riscv64-unknown-elf-objcopy"),
        str(gcc / "riscv64-unknown-elf-readelf"),
    )
    compiler = [
        str(args.llvm_bin / "clang"),
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]
    originals = [pin(args.source_bundle)]
    if args.independent_m is not None:
        assert args.independent_proof and not args.original_a and not args.original_b
        route = {
            "m": args.independent_m,
            "n": 64,
            "symbol": "independent_residual",
            "kernel": "independent_residual_kernel",
            "proof": json.loads(args.independent_proof.read_text()),
            "single_output_guard": True,
        }
        originals.append(pin(args.independent_proof))
        producer = build(
            route["m"],
            **route["proof"]["predictor"],
            relu=route["proof"]["source"]["relu"],
        )
        fn = next(op for op in producer.body.block.ops if isinstance(op, llvm.FuncOp))
        fn.properties["sym_name"] = StringAttr(route["kernel"])
        compile_module(producer, args.llvm_bin, work / "predictor")
        kernel = work / "predictor/kernel.o"
        control_source = work / "control_original.c"
        control_source.write_text(_single_adapter(route))
        control_object = work / "control_original.o"
        command([*compiler, "-c", str(control_source), "-o", str(control_object)])
        source_capture = None
    else:
        assert args.original_a and args.original_b and args.source_capture_receipt
        directory = args.source_bundle.resolve().parent / route["symbol"]
        kernel, control_source, control_object = [
            directory / name for name in ("kernel.o", "adapter.c", "adapter.o")
        ]
        assert pin(kernel)["sha256"] == route["compilation"]["object_sha256"]
        assert (
            pin(control_source)["sha256"]
            == route["adapter_compilation"]["source_sha256"]
        )
        assert (
            pin(control_object)["sha256"]
            == route["adapter_compilation"]["object_sha256"]
        )
        recorded = route["adapter_compilation"]["argv"]
        assert recorded[:-4] == compiler
        originals += [
            pin(kernel),
            pin(control_source),
            pin(control_object),
            pin(args.source_capture_receipt),
            pin(args.original_a),
            pin(args.original_b),
        ]
        source_capture = json.loads(args.source_capture_receipt.read_text())
        assert source_capture["all1000_original_f32_bits_exact"]
        assert pin(args.original_a)["sha256"] == source_capture["lhs_sha256"]
        assert pin(args.original_b)["sha256"] == source_capture["rhs_sha256"]
    proof = route["proof"]
    assert proof == derive(**proof["source"], **proof["predictor"])
    storage_proof = json.loads(args.storage_proof.read_text())
    assert storage_proof["gsim_fir_reproduced_byte_exact"]
    exact_record = json.loads(args.exact_control_bundle.read_text())
    original_contract = {
        k: v
        for k, v in route["original_route"].items()
        if k not in ("compilation", "adapter_compilation", "device_schedule")
    }
    matching = [
        r
        for r in exact_record["routes"]
        if {
            k: v
            for k, v in r.items()
            if k not in ("compilation", "adapter_compilation", "device_schedule")
        }
        == original_contract
    ]
    assert len(matching) == 1
    exact_route = matching[0]
    assert storage_proof["route"] == exact_route
    exact_dir = args.exact_control_bundle.parent / exact_route["symbol"]
    exact_kernel, exact_adapter = exact_dir / "kernel.o", exact_dir / "adapter.o"
    assert pin(exact_kernel)["sha256"] == exact_route["compilation"]["object_sha256"]
    assert (
        pin(exact_adapter)["sha256"]
        == exact_route["adapter_compilation"]["object_sha256"]
    )
    m, count = route["m"], route["m"] * 64
    selected = dict(route, symbol="streamed_residual", kernel="streamed_kernel")
    candidate_code, storage = ranked_adapter(
        selected,
        **{
            k: storage_proof["panel_storage"][k]
            for k in ("coherence_granule_bytes", "dma_max_request_bytes")
        },
    )
    module, helper_code, _ = streamed(
        m, proof, correction_symbol="streamed_residual_correct"
    )
    function = next(
        op
        for op in module.body.block.ops
        if isinstance(op, llvm.FuncOp) and op.body.blocks
    )
    function.properties["sym_name"] = StringAttr("streamed_kernel")
    candidate_kernel_dir = work / "streamed_kernel"
    streamed_compile = compile_module(module, args.llvm_bin, candidate_kernel_dir)
    helper_source, helper_object = work / "correction.c", work / "correction.o"
    helper_source.write_text(helper_code)
    helper_argv = [
        *compiler,
        "-frounding-math",
        "-c",
        str(helper_source),
        "-o",
        str(helper_object),
    ]
    command(helper_argv)
    public = "_mlir_ciface_" + route["symbol"]
    candidate_source, candidate_object = work / "candidate.c", work / "candidate.o"
    candidate_source.write_text(
        candidate_code.replace("_mlir_ciface_streamed_residual", "streamed_residual")
    )
    candidate_command = [
        *compiler,
        "-c",
        str(candidate_source),
        "-o",
        str(candidate_object),
    ]
    command(candidate_command)
    baseline_object = work / "baseline_rebound.o"
    rename = [
        objcopy,
        "--redefine-sym",
        public + "=baseline_residual",
        str(control_object),
        str(baseline_object),
    ]
    command(rename)
    text_hashes = []
    for index, obj in enumerate([control_object, baseline_object]):
        p = work / f"baseline_text{index}.bin"
        text_hashes.append(
            extract_text_read_only(objcopy, obj, p, work / f"section_reader{index}.o")
        )
    assert text_hashes[0] == text_hashes[1]
    exact_rebound = work / "exact_rebound.o"
    exact_rename = [
        objcopy,
        "--redefine-sym",
        "_mlir_ciface_" + exact_route["symbol"] + "=exact_residual",
        str(exact_adapter),
        str(exact_rebound),
    ]
    command(exact_rename)
    exact_text_hashes = []
    for index, obj in enumerate([exact_adapter, exact_rebound]):
        exact_text_hashes.append(
            extract_text_read_only(
                objcopy,
                obj,
                work / f"exact_text{index}.bin",
                work / f"exact_reader{index}.o",
            )
        )
    assert exact_text_hashes[0] == exact_text_hashes[1]
    if args.original_a:
        a, b = [
            np.frombuffer(p.read_bytes(), np.int8)
            for p in (args.original_a, args.original_b)
        ]
        assert a.size == b.size == count
    else:
        keys = np.arange(count, dtype=np.int32) % 65536
        if count < 65536:
            keys = (keys * 40503 + 32768) % 65536
        a, b = (keys // 256 - 128).astype(np.int8), (keys % 256 - 128).astype(np.int8)
    expected = source_table(**proof["source"])[
        a.astype(np.int16) + 128, b.astype(np.int16) + 128
    ]
    fixture_text = ".section .rodata\n"
    for name, data in [
        ("a", a),
        ("b", b),
        ("check_a", a),
        ("check_b", b),
        ("expected", expected),
    ]:
        p = work / (name + ".bin")
        p.write_bytes(data.tobytes())
        fixture_text += (
            f'.balign 64\n.global fixture_{name}\nfixture_{name}:\n.incbin "{p}"\n'
        )
    fixture = work / "fixture.S"
    fixture.write_text(fixture_text)
    (work / "benchmark_buffer.h").write_bytes(args.benchmark_header.read_bytes())
    probe = work / "probe.c"
    probe.write_text(
        """#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "benchmark_buffer.h"
#define N COUNT
#define M ROWS
typedef struct{void *allocated,*aligned;intptr_t offset,sizes[2],strides[2];} mem2;
extern const int8_t fixture_a[N],fixture_b[N],fixture_check_a[N],fixture_check_b[N],fixture_expected[N];
extern void baseline_residual(mem2*,mem2*,mem2*,mem2*);
extern void exact_residual(mem2*,mem2*,mem2*,mem2*);
extern void streamed_residual(mem2*,mem2*,mem2*,mem2*);
volatile uint64_t arm_selector __attribute__((section(".selector")))=0;
static struct{uint8_t before[2048];int8_t output[N];uint8_t after[2048];} result __attribute__((aligned(64)));
static int eq(const void*a,const void*b,size_t n){return merlin_benchmark_first_difference(a,b,n)==n;}
static int guards(void){for(size_t i=0;i<2048;i++)if(result.before[i]!=0x5a||result.after[i]!=0xa5)return 1;return 0;}
static void poison(void){merlin_benchmark_fill(result.before,0x5a,2048);merlin_benchmark_fill(result.output,0x7d,N);merlin_benchmark_fill(result.after,0xa5,2048);}
int main(void){
mem2 a={(void*)fixture_a,(void*)fixture_a,0,{M,64},{64,1}},b={(void*)fixture_b,(void*)fixture_b,0,{M,64},{64,1}};
mem2 c={result.output,result.output,0,{M,64},{64,1}},r={0},old_a=a,old_b=b,old_c=c;
poison();exact_residual(&r,&a,&b,&c);
if(!eq(result.output,fixture_expected,N)||!eq(&r,&c,sizeof(c))||guards()){printf("FAIL exact\\n");return 5;}
poison();baseline_residual(&r,&a,&b,&c);
if(!eq(result.output,fixture_expected,N)||!eq(&r,&c,sizeof(c))||guards()){printf("FAIL baseline\\n");return 1;}
poison();streamed_residual(&r,&a,&b,&c);
if(!eq(result.output,fixture_expected,N)||!eq(&r,&c,sizeof(c))||guards()){printf("FAIL sparse\\n");return 2;}
poison();uint64_t begin,end,arm=arm_selector;
__asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
if(arm==0)exact_residual(&r,&a,&b,&c);else if(arm==1)baseline_residual(&r,&a,&b,&c);else streamed_residual(&r,&a,&b,&c);
__asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
printf("STREAMED_ADAPTER_CYCLES %lu %lu\\n",(unsigned long)arm,(unsigned long)(end-begin));
if(!eq(result.output,fixture_expected,N)||!eq(&r,&c,sizeof(c))||guards()){printf("FAIL timed output\\n");return 3;}
if(!eq(fixture_a,fixture_check_a,N)||!eq(fixture_b,fixture_check_b,N)||!eq(&a,&old_a,sizeof(a))||!eq(&b,&old_b,sizeof(b))||!eq(&c,&old_c,sizeof(c))){printf("FAIL immutable input/descriptor\\n");return 4;}
printf("STREAMED_ADAPTER_PASS arm%lu all%d guards4096 inputs%d descriptors\\n",(unsigned long)arm,N,2*N);return 0;
}
""".replace("COUNT", str(count)).replace("ROWS", str(m))
    )
    built = build_program(
        [
            probe,
            fixture,
            baseline_object,
            candidate_object,
            kernel,
            exact_rebound,
            exact_kernel,
            candidate_kernel_dir / "kernel.o",
            helper_object,
        ],
        work / "arm0",
        target="gemmini",
        extra_cflags=[
            "-I" + str(work),
            "-march=rv64gc",
            "-mabi=lp64d",
            "-fno-fast-math",
            "-ffp-contract=off",
        ],
        max_loaded_bytes=None,
    )
    raw = built.elf.read_bytes()
    rows = command([readelf, "-SW", str(built.elf)]).splitlines()
    section = next(
        line.split("]", 1)[1].split()
        for line in rows
        if "]" in line and line.split("]", 1)[1].split()[:1] == [".selector"]
    )
    offset = int(section[3], 16)
    assert raw[offset : offset + 8] == bytes(8)
    elfs = [built.elf, work / "arm1/layer.elf", work / "arm2/layer.elf"]
    for arm in (1, 2):
        elfs[arm].parent.mkdir()
        changed = bytearray(raw)
        changed[offset] = arm
        elfs[arm].write_bytes(changed)
        assert [i for i, (a0, b0) in enumerate(zip(raw, changed)) if a0 != b0] == [
            offset
        ]
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    audits = []
    for arm, elf in enumerate(elfs):
        audit = audit_elf(elf.read_bytes())
        assert audit["status"] == "pass"
        audits.append(audit)
        marker = f"STREAMED_ADAPTER_PASS arm{arm} all{count} guards4096 inputs{count * 2} descriptors"
        argv = [
            str(spike),
            "-g",
            "--extension=gemmini",
            "--isa=RV64GC",
            "-m0x80000000:0x80000000",
            str(elf),
        ]
        run = subprocess.run(
            argv, capture_output=True, text=True, timeout=300, check=False
        )
        stdout = elf.parent / "spike.stdout"
        stdout.write_text(run.stdout + run.stderr)
        assert run.returncode == 0 and marker in run.stdout
        print("STRICT_STREAMED_ADAPTER_PASS", arm, count, flush=True)
    recipe = {
        "schema": "source_bound_streamed_ranked_residual_adapter_pair_v1",
        "source_bundle": pin(args.source_bundle),
        "route": route,
        "certificate": proof,
        "panel_storage": storage,
        "normal_storage_proof": pin(args.storage_proof),
        "exact_control_bundle": pin(args.exact_control_bundle),
        "exact_control_route": exact_route,
        "original_source_contract_equal": True,
        "implementation_only_binding_deltas": [
            key
            for key in ("compilation", "adapter_compilation", "device_schedule")
            if route["original_route"].get(key) != exact_route.get(key)
        ],
        "exact_kernel": pin(exact_kernel),
        "exact_adapter": pin(exact_adapter),
        "exact_text_identity": exact_text_hashes,
        "streamed_compilation": streamed_compile,
        "streamed_kernel": pin(candidate_kernel_dir / "kernel.o"),
        "correction_source": pin(helper_source),
        "correction_object": pin(helper_object),
        "correction_argv": helper_argv,
        "originals": originals,
        "source_capture": source_capture,
        "candidate_compiler_argv": candidate_command,
        "source": pin(candidate_source),
        "object": pin(candidate_object),
        "serial_control_kernel": pin(kernel),
        "baseline_symbol_rebind_argv": rename,
        "baseline_text_identity": text_hashes,
        "selector_offset": offset,
        "ELFs_differ_one_byte": True,
        "audits": audits,
        "workers": args.workers,
        "scope": "Complete original ranked adapter ABI checks, producer DMA/compute/store/fence, exact guard/replay and result descriptor; unchanged inputs; identical warmup/dirty poisoning/all-byte checks outsideROI",
        "hardware_overlap": "UNKNOWN",
        "physical_DRAM": "UNKNOWN",
        "whole_accuracy_gate": "UNCHANGED; original all1000 binary32 exact, atol=rtol=0",
        "whole_model_route_enabled": False,
    }
    (work / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

    def measure(arm):
        elf = elfs[arm]
        marker = f"STREAMED_ADAPTER_PASS arm{arm} all{count} guards4096 inputs{count * 2} descriptors"
        run = run_on_gsim(
            elf,
            target="gemmini",
            max_cycles=30000000,
            timeout_s=args.timeout,
            backdoor=True,
            stdout_path=elf.parent / "gsim.stdout",
        )
        lines = [
            line.split()[-1]
            for line in run.stdout_tail.splitlines()
            if line.startswith(f"STREAMED_ADAPTER_CYCLES {arm} ")
        ]
        passed = (
            run.completed
            and run.returncode == 0
            and marker in run.stdout_tail
            and len(lines) == 1
            and run.engine["binary_sha256"] == storage_proof["gsim_engine_sha256"]
        )
        record = {
            "arm": arm,
            "elf": pin(elf),
            "passed": passed,
            "completed": run.completed,
            "returncode": run.returncode,
            "cycles": int(lines[0]) if len(lines) == 1 else None,
            "engine": run.engine,
            "stdout": run.stdout_tail,
            "stderr": run.stderr_tail,
            "metric_scope": recipe["scope"],
        }
        (elf.parent / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            "GSIM_STREAMED_ADAPTER",
            arm,
            "PASS",
            passed,
            "CYCLES",
            record["cycles"],
            flush=True,
        )
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(measure, range(3)))
    result = {
        "schema": "source_bound_streamed_ranked_residual_adapter_result_v1",
        "recipe": pin(work / "recipe.json"),
        "records": records,
        "passed": all(r["passed"] for r in records),
        "whole_cycles": "UNKNOWN",
        "default_enabled": False,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
