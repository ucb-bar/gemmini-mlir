"""Complete source-bound paired source-stride convolution and exact decoder capsule."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from merlin.llvmlower.enclosed_readout import emit_pair_scan
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.readout_store_plan import PairedReadoutPlan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--source-symbol", required=True)
    parser.add_argument(
        "--schedule", choices=("control", "source_stride"), required=True
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--benchmark-header", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=8000000)
    args = parser.parse_args()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    driver = Path(__file__).resolve()
    driver_sha = sha(driver)
    shutil.copyfile(driver, root / "executed_probe_driver.py")
    fixture = args.fixture.resolve()
    receipt = json.loads((fixture / "receipt.json").read_text())
    manifest = json.loads(args.paired_manifest.read_text())
    routes = [r for r in manifest["routes"] if r["symbol"] == args.source_symbol]
    if len(routes) != 1 or not routes[0]["paired_readout"]["applied"]:
        raise ValueError("one explicitly bound paired source route required")
    route = routes[0]
    p = route["paired_readout"]["proof"]
    plan = PairedReadoutPlan(
        tuple(p["source_scales"]), tuple(p["store_scales"]), p["lo"], p["hi"], p["relu"]
    )
    if plan.certificate() != p:
        raise ValueError("paired certificate changed")
    shape = ConvShape(**receipt["shape"])
    plan.require_conv_producer(shape)
    source_proof = receipt["source_numeric_proof"]
    if tuple(source_proof["source_scales"]) != plan.source_scales:
        raise ValueError("fixture source arithmetic differs")
    n = shape.oh * shape.ow * shape.cout
    arrays = {}
    for name, dtype, count in [
        ("a", np.int8, shape.h * shape.w * shape.cin),
        ("b", np.int8, 9 * shape.cin * shape.cout),
        ("expected", np.int32, n),
    ]:
        path = fixture / (name + ".bin")
        if sha(path) != receipt[name + "_sha256"]:
            raise ValueError("immutable fixture changed")
        arrays[name] = np.fromfile(path, dtype=dtype)
        if arrays[name].size != count:
            raise ValueError("fixture geometry differs")
    raw = arrays.pop("expected")
    final = raw.astype(np.float32)
    for scale in plan.source_scales:
        final = np.multiply(final, np.float32(scale), dtype=np.float32)
    low = 0 if plan.relu else -128
    arrays["expected"] = np.clip(np.rint(final), low, 127).astype(np.int8)
    second = np.multiply(
        raw.astype(np.float32), np.float32(plan.store_scales[1]), dtype=np.float32
    )
    arrays["expected_second"] = np.clip(np.rint(second), low, 127).astype(np.int8)
    arrays["check_a"], arrays["check_b"] = arrays["a"], arrays["b"]
    symbol = route["kernel"]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol):
        raise ValueError("invalid explicit kernel symbol")
    if args.schedule == "control":
        path = args.paired_manifest.parent / args.source_symbol / "kernel.o"
        if sha(path) != route["compilation"]["object_sha256"]:
            raise ValueError("actual source control object changed")
        shutil.copyfile(path, root / "kernel.o")
        compilation = {
            "status": "exact_bound_source_object",
            "path": str(path),
            "sha256": sha(path),
        }
    else:
        from mlir_oot.conv_schedule import source_stride_resource_layout

        resident, _resource_choice = source_stride_resource_layout(shape)
        generator = GoldenResidentConv(
            resident.conv,
            rows_per_tile=resident.rows_per_tile,
            source_stride=True,
            weight_base=resident.explicit_weight_base,
            row_residue=resident.row_residue,
            store_plan=plan,
        )
        module = generator.build()
        module.body.block.first_op.properties["sym_name"] = StringAttr(symbol)
        compilation = compile_module(module, args.llvm_bin, root)
    asm = []
    for index, (name, values) in enumerate(arrays.items()):
        values.tofile(root / (name + ".bin"))
        asm.append(
            f'.section .data\n.balign {1048576 if index == 0 else 64}\n.global captured_{name}\ncaptured_{name}:\n.incbin "{root / (name + ".bin")}"\n'
        )
    (root / "inputs.S").write_text("".join(asm))
    subprocess.run(
        [
            str(args.llvm_bin / "clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-c",
            str(root / "inputs.S"),
            "-o",
            str(root / "inputs.o"),
        ],
        check=True,
        capture_output=True,
    )
    shutil.copyfile(args.benchmark_header, root / "benchmark_buffer.h")
    source = root / "probe.c"
    source.write_text(
        """#include <stdint.h>
#include <stdio.h>
#include "benchmark_buffer.h"
"""
        + emit_pair_scan(plan.certificate(), "decode", copy_policy="compiler_builtin")
        + f"""
#define N {n}
extern int8_t captured_a[{len(arrays["a"])}],captured_b[{len(arrays["b"])}];
extern const int8_t captured_expected[N],captured_expected_second[N],captured_check_a[{len(arrays["a"])}],captured_check_b[{len(arrays["b"])}];
struct box {{uint8_t before[2048];int8_t output[N];uint8_t after[2048];}};
struct box captured_first __attribute__((aligned(1048576))),captured_second __attribute__((aligned(1048576)));
extern void {symbol}(int8_t*,int8_t*,int8_t*,int8_t*);
int main(void){{
 uint8_t guard[2048];merlin_benchmark_fill(guard,0x5a,sizeof(guard));
 merlin_benchmark_fill(&captured_first,0x5a,sizeof(captured_first));merlin_benchmark_fill(&captured_second,0x5a,sizeof(captured_second));
 merlin_benchmark_fill(captured_first.output,0xdb,N);merlin_benchmark_fill(captured_second.output,0xdb,N);
 uint64_t begin,end;__asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 {symbol}(captured_a,captured_b,captured_first.output,captured_second.output);
 decode((unsigned char*)captured_first.output,(const unsigned char*)captured_second.output,N);
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");printf("PAIRED_CONV_CYCLES %d\\n",(int)(end-begin));
 if(merlin_benchmark_first_difference(captured_expected,captured_first.output,N)!=N)return 1;
 if(merlin_benchmark_first_difference(captured_expected_second,captured_second.output,N)!=N)return 2;
 if(merlin_benchmark_first_difference(guard,captured_first.before,2048)!=2048||merlin_benchmark_first_difference(guard,captured_first.after,2048)!=2048||merlin_benchmark_first_difference(guard,captured_second.before,2048)!=2048||merlin_benchmark_first_difference(guard,captured_second.after,2048)!=2048)return 3;
 if(merlin_benchmark_first_difference(captured_check_a,captured_a,{len(arrays["a"])})!={len(arrays["a"])}||merlin_benchmark_first_difference(captured_check_b,captured_b,{len(arrays["b"])})!={len(arrays["b"])})return 4;
 printf("PAIRED_CONV_PASS N={n}\\n");return 0;
}}
"""
    )
    built = build_program(
        [source, root / "kernel.o", root / "inputs.o"],
        root,
        target="gemmini",
        max_loaded_bytes=None,
        extra_cflags=[
            "-march=rv64gc",
            "-fno-builtin",
            "-fno-fast-math",
            "-ffp-contract=off",
        ],
    )
    audit = audit_elf(built.elf.read_bytes())
    (root / "nofsm_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "pass":
        raise ValueError("FSM audit failed")
    nm = subprocess.check_output(
        [str(args.llvm_bin / "llvm-nm"), "--defined-only", str(built.elf)], text=True
    )
    addresses = {
        m[2]: int(m[1], 16)
        for m in re.finditer(
            r"^([0-9a-f]+) \w (captured_a|captured_b|captured_expected|captured_first|captured_second)$",
            nm,
            re.MULTILINE,
        )
    }
    if len(addresses) != 5:
        raise ValueError("missing linked operand addresses")
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    argv = [
        str(spike),
        "-g",
        "--extension=gemmini",
        "--isa=rv64gc",
        "-m0x80000000:0x80000000",
        str(built.elf),
    ]
    run = subprocess.run(argv, capture_output=True, timeout=900, check=False)
    (root / "spike.stdout").write_bytes(run.stdout)
    (root / "spike.stderr").write_bytes(run.stderr)
    marker = f"PAIRED_CONV_PASS N={n}"
    if run.returncode or marker not in run.stdout.decode():
        raise ValueError("strict complete source capsule failed")
    run = run_on_gsim(
        built.elf,
        target="gemmini",
        max_cycles=args.max_cycles,
        timeout_s=900,
        backdoor=True,
        stdout_path=root / "gsim.stdout",
    )
    passed = run.completed and run.returncode == 0 and marker in run.stdout_tail
    cycles = re.findall(r"^PAIRED_CONV_CYCLES (\d+)$", run.stdout_tail, re.MULTILINE)
    result = {
        "status": "pass" if passed else "incomplete_or_fail",
        "schedule": args.schedule,
        "shape": asdict(shape),
        "proof": p,
        "compilation": compilation,
        "common_operand_addresses": addresses,
        "cycles": int(cycles[0]) if len(cycles) == 1 else None,
        "metric_scope": "Complete convolution plus both primitive stores/fence and exact decoder; GSIM only",
        "outputs_checked": 2 * n if passed else None,
        "guards_checked": 8192 if passed else None,
        "immutable_input_bytes_checked": len(arrays["a"]) + len(arrays["b"])
        if passed
        else None,
        "nofsm_status": audit["status"],
        "elf_sha256": built.elf_sha256,
        "strict_spike": {"status": "pass", "argv": argv, "engine_sha256": sha(spike)},
        "engine_sha256": run.engine.get("binary_sha256"),
        "run_completed": run.completed,
        "run_returncode": run.returncode,
        "fixture_receipt_sha256": sha(fixture / "receipt.json"),
        "paired_manifest_sha256": sha(args.paired_manifest),
        "driver_sha256": driver_sha,
        "executed_driver_snapshot": str(root / "executed_probe_driver.py"),
        "stdout": run.stdout_tail,
        "stderr": run.stderr_tail,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
