"""Original ranked residual: immutable current14 versus exact key9 pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from merlin.llvmlower import quantized_affine_pair as pair
from merlin.llvmlower import quantized_affine_rectifier as rectifier
from merlin.perf.layer_bench import build_program, run_on_gsim

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_key_rectified_resadd import Capabilities, Plan, build
from mlir_oot.golden_rectified_resadd import Capabilities as BasicCapabilities
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.spad_fence_coalescing import OrderingContract


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def command(argv):
    return subprocess.run(
        list(map(str, argv)), capture_output=True, text=True, check=True
    ).stdout


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in (
        "workdir",
        "core",
        "llvm-bin",
        "control-joint",
        "inputs",
        "capture-record",
    ):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--route-index", type=int, default=0)
    cli.add_argument("--timeout", type=int, default=1200)
    args = cli.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    joint = json.loads(args.control_joint.read_text())
    route = joint["routes"][args.route_index]
    old_certificate = route["proof"]
    proof = pair.derive(**old_certificate["source"], **old_certificate["predictor"])
    certificate = rectifier.derive(proof, max_pairs=1, indicator_family="predictor_key")
    plan = Plan(
        route["m"],
        route["n"],
        certificate,
        Capabilities(BasicCapabilities(**route["capabilities"]), True),
        panel_batch=route["panel_batch"],
    )
    compilation = compile_module(
        build(plan, ordering_contract=OrderingContract(route["ordering_source"])),
        args.llvm_bin,
        work / "candidate",
    )
    original_kernel = Path(route["compilation"]["compiler_argv"][-1][-1])
    original_adapter = Path(route["adapter_compilation"]["compiler_argv"][-1])
    original_source = Path(route["adapter_compilation"]["compiler_argv"][-3])
    for path, digest in (
        (original_kernel, route["compilation"]["object_sha256"]),
        (original_adapter, route["adapter_compilation"]["object_sha256"]),
        (original_source, route["adapter_compilation"]["source_sha256"]),
    ):
        if pin(path)["sha256"] != digest:
            raise ValueError("current declared14 source/object pin changed")
    capture = json.loads(args.capture_record.read_text())["source_capture"]
    count = plan.m * plan.n
    if capture["elements"] != count or not capture["all1000_original_f32_bits_exact"]:
        raise ValueError("original whole-gated operand capture required")
    for name, digest in (
        ("a.bin", capture["lhs_sha256"]),
        ("b.bin", capture["rhs_sha256"]),
    ):
        if pin(args.inputs / name)["sha256"] != digest:
            raise ValueError("captured original input differs")
    a, b = (
        np.frombuffer((args.inputs / name).read_bytes(), np.int8)
        for name in ("a.bin", "b.bin")
    )
    expected = pair.source_table(**proof["source"])[
        a.astype(np.int16) + 128, b.astype(np.int16) + 128
    ]
    table = plan.tables()
    values = ",".join(str(value if value < 128 else value - 256) for value in table)
    adapter = work / "candidate_adapter.c"
    adapter.write_text(f"""#include <stdint.h>
typedef struct{{void*allocated,*aligned;intptr_t offset,sizes[2],strides[2];}}mem2;
static const int8_t table[{len(table)}] __attribute__((aligned(64)))={{{values}}};
extern void gemmini_golden_key_rectified_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*,int8_t*);
static int8_t*pointer(mem2*d){{if(!d->aligned||d->offset<0||d->sizes[0]!={plan.m}||d->sizes[1]!={plan.n}||d->strides[0]!={plan.n}||d->strides[1]!=1)__builtin_trap();uintptr_t p=(uintptr_t)d->aligned,o=(uintptr_t)d->offset;if(p>UINTPTR_MAX-o||p+o>UINTPTR_MAX-{count})__builtin_trap();return(int8_t*)(p+o);}}
static int overlaps(const int8_t*a,const int8_t*b,uintptr_t size){{uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<=y?y-x<{count}:x-y<size;}}
void rectified_residual(mem2*r,mem2*a,mem2*b,mem2*c){{const int8_t*pa=pointer(a),*pb=pointer(b);int8_t*pc=pointer(c);if(overlaps(pc,pa,{count})||overlaps(pc,pb,{count})||overlaps(pc,table,{len(table)}))__builtin_trap();
 int8_t scratch[{plan.scratch_bytes}] __attribute__((aligned(64)));
 gemmini_golden_key_rectified_resadd(pa,pb,pc,table,scratch);*r=*c;}}
""")
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
    command([*compiler, "-c", adapter, "-o", work / "candidate_adapter.o"])
    command(
        [*compiler, "-S", "-emit-llvm", adapter, "-o", work / "candidate_adapter.ll"]
    )
    tools = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin")
    objcopy = tools / "riscv64-unknown-elf-objcopy"
    command(
        [
            objcopy,
            "--redefine-sym",
            route["kernel"] + "=immutable_current14_kernel",
            original_kernel,
            work / "control_kernel.o",
        ]
    )
    command(
        [
            objcopy,
            "--redefine-sym",
            route["kernel"] + "=immutable_current14_kernel",
            "--redefine-sym",
            "_mlir_ciface_" + route["symbol"] + "=exact_residual",
            original_adapter,
            work / "control_adapter.o",
        ]
    )
    fixture = work / "fixture.S"
    assembly = ".section .rodata\n"
    for name, array in (
        ("a", a),
        ("b", b),
        ("check_a", a),
        ("check_b", b),
        ("expected", expected),
    ):
        path = work / (name + ".bin")
        path.write_bytes(array.tobytes())
        assembly += (
            f'.balign 64\n.global fixture_{name}\nfixture_{name}:\n.incbin "{path}"\n'
        )
    fixture.write_text(assembly)
    # Preserve the already qualified complete ranked harness/ROI protocol.
    harness = Path(__file__).with_name("affine_rectifier_ranked_probe.py").read_text()
    start = harness.index(
        'r"""#include <stdint.h>', harness.index('probe = work / "probe.c"')
    )
    end = harness.index('""".replace("ELEMENT_COUNT"', start)
    source = harness[start + 4 : end]
    source = (
        source.replace("ELEMENT_COUNT", str(count))
        .replace("ROW_COUNT", str(plan.m))
        .replace("COLUMN_COUNT", str(plan.n))
    )
    (work / "probe.c").write_text(source)
    header = args.core / "merlin/runtime/c/benchmark_buffer.h"
    (work / "benchmark_buffer.h").write_bytes(header.read_bytes())
    program = build_program(
        [
            work / "probe.c",
            fixture,
            work / "control_adapter.o",
            work / "control_kernel.o",
            work / "candidate_adapter.o",
            work / "candidate/kernel.o",
        ],
        work / "arm0",
        target="gemmini",
        extra_cflags=["-I" + str(work), "-O2", "-fno-fast-math", "-ffp-contract=off"],
        max_loaded_bytes=None,
    )
    raw = program.elf.read_bytes()
    section = next(
        line.split("]", 1)[1].split()
        for line in command(
            [tools / "riscv64-unknown-elf-readelf", "-SW", program.elf]
        ).splitlines()
        if "]" in line and line.split("]", 1)[1].split()[:1] == [".selector"]
    )
    offset = int(section[3], 16)
    if raw[offset : offset + 8] != bytes(8):
        raise ValueError("selector bytes differ")
    arm1 = work / "arm1/layer.elf"
    arm1.parent.mkdir()
    changed = bytearray(raw)
    changed[offset] = 1
    arm1.write_bytes(changed)
    elfs = [program.elf, arm1]
    if [i for i, (x, y) in enumerate(zip(raw, changed)) if x != y] != [offset]:
        raise ValueError("ELFs differ beyond data selector")
    audits = [audit_elf(p.read_bytes()) for p in elfs]
    if any(r["status"] != "pass" for r in audits):
        raise ValueError("final executable noFSM refused")
    for name, path in (
        ("driver_snapshot.py", Path(__file__)),
        (
            "kernel_producer_snapshot.py",
            Path(__file__).resolve().parents[1]
            / "mlir_oot/golden_key_rectified_resadd.py",
        ),
    ):
        (work / name).write_bytes(path.read_bytes())
    record = {
        "schema": "original_current14_predictor_key9_ranked_pair_v1",
        "plan": plan.attributes(),
        "certificate": certificate,
        "current_source_route": route,
        "source_capture": capture,
        "candidate_compilation": compilation,
        "selector_offset": offset,
        "ELFs_differ_one_byte": True,
        "elfs": [pin(p) for p in elfs],
        "audits": audits,
        "source_inputs": [
            pin(p)
            for p in (
                args.control_joint,
                args.capture_record,
                args.inputs / "a.bin",
                args.inputs / "b.bin",
                original_source,
                original_kernel,
                original_adapter,
                header,
            )
        ],
        "public_ABI": "originalranked3tensors+resultdescriptor; private4096B alignedstack scratch passed to5pointerprimitive; finalcompletionbeforeownerreturn",
        "scope": "Current14 frozen rankedadapter andkernel versus key9 rankedadapter; common operand/output addresses, identical warmups, complete setup/products/all seed/store/readback/reload/config/fences/publication; outputs,input,descriptor,flags and4096outputguard checks outsideROI",
        "whole_cycles": "UNKNOWN",
        "physical_DRAM": "UNKNOWN",
        "stock_capability": "narrowidentityRMW2077 UARTPASS; fullnetworkstockUNKNOWN; actualstagedbitstreamidentity pending",
        "default_enabled": False,
    }
    (work / "recipe.json").write_text(json.dumps(record, indent=2) + "\n")
    for arm, elf in enumerate(elfs):
        run = subprocess.run(
            [
                str(tools / "spike"),
                "--extension=gemmini",
                "--isa=rv64gc",
                "-m0x80000000:0x400000000",
                str(elf),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        (elf.parent / "spike.stdout").write_text(run.stdout)
        (elf.parent / "spike.stderr").write_text(run.stderr)
        if (
            run.returncode != 0
            or f"RECTIFIED_RANKED_PASS arm{arm} all{count}" not in run.stdout
        ):
            raise ValueError("actual original ranked output/guard proof failed")
        print("SPIKE_RANKED_PASS", arm, count, flush=True)

    def measure(arm):
        elf = elfs[arm]
        result = run_on_gsim(
            elf,
            target="gemmini",
            max_cycles=12000000,
            timeout_s=args.timeout,
            stdout_path=elf.parent / "gsim.stdout",
        )
        (elf.parent / "gsim.json").write_text(
            json.dumps(dataclasses.asdict(result), indent=2, default=str) + "\n"
        )
        print("GSIM_RANKED", arm, result.completed, result.stdout_tail, flush=True)
        return (
            result.completed
            and f"RECTIFIED_RANKED_PASS arm{arm} all{count}" in result.stdout_tail
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        passed = all(pool.map(measure, (0, 1)))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
