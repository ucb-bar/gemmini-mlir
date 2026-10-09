"""Prospective cross-executable quantizer transfer, with every new case held.

Experiment driver only. It calls Merlin's general source legality, scheduling
and LLVM emission. It does not alter the original six-case packet or fit labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from current_host_quant_fixture_probe import pin
from merlin.llvmlower.bounded_rne_maps import schedule_bounded_rne_maps
from merlin.llvmlower.bounded_rne_packet_llvm import rewrite_packet_helpers
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.perf.layer_bench import build_program
from merlin.targetgen.elf_lanes import executable_sections
from merlin.xdsl_dialects._common import text
from source_host_quant_model_battery_probe import (
    c_source,
    compile_map,
    evaluate,
    rebind_extent,
)

from mlir_oot.frontend.parse import parse_module
from mlir_oot.no_fsm_audit import audit_elf


def compile_tail_map(module, arm, llvm_bin, width):
    arm.mkdir(parents=True, exist_ok=False)
    scheduled = module.clone()
    routes = schedule_bounded_rne_maps(scheduled, lanes=width)
    if len(routes) != 1 or routes[0]["lanes"] != width:
        raise ValueError("one typed source schedule required")
    (arm / "source.mlir").write_text(text(module, generic=True))
    (arm / "scheduled.mlir").write_text(text(scheduled, generic=True))
    native = lower_to_llvm_ir(
        text(scheduled, generic=True), workdir=arm / "lower", vectorize=False
    )
    (arm / "native.ll").write_text(native)
    clang = str(llvm_bin / "clang")
    subprocess.run([clang, "-O2", "-fPIC", "-c", str(arm / "native.ll"), "-o", str(arm / "native.o")], check=True)
    subprocess.run(["cc", "-fPIC", "-shared", str(arm / "native.o"), str(mlir_runtime_c()), "-lm", "-o", str(arm / "native.so")], check=True)
    target, packet = rewrite_packet_helpers(native, host_isa="rv64gc", max_lanes=width)
    # Both full and remainder helpers must be bound; no missing conversion.
    expected_widths = {int(value) for value in routes[0]["helpers"] if int(value) >= 2}
    if {row["lanes"] for row in packet["routes"]} != expected_widths:
        raise ValueError("full/remainder packet helper coverage differs")
    (arm / "target.ll").write_text(target)
    subprocess.run([clang, "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin", "-c", str(arm / "target.ll"), "-o", str(arm / "kernel.o")], check=True)
    return {"routes": routes, "packet": packet}


def contents(path):
    blob = Path(path).read_bytes()
    return [blob[offset:offset + size] for _, offset, size, _ in executable_sections(blob)]


def generate(args):
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    fixture = args.fixture.resolve()
    original = parse_module((fixture / "source.mlir").read_text())
    binding = json.loads((fixture / "receipt.json").read_text())
    shape = binding["input_shape"]
    source_input = np.fromfile(fixture / "input.bin", dtype=np.float32).reshape(shape)
    if shape != [1, 3, 224, 224]:
        raise ValueError("this experiment requires its original bound source fixture")
    shutil.copyfile(fixture / "input.bin", work / "original_input.bin")
    extents, widths = (168, 224, 336), (5, 7)
    cases, objects = [], []
    for extent in extents:
        module, proof = rebind_extent(original, input_axis=2, extent=extent)
        selected = source_input[:, :, np.arange(extent) % shape[2], :].copy()
        group = work / f"height{extent}"
        oracle_dir = group / "oracle"
        compile_map(module, oracle_dir, args.llvm_bin)
        oracle = evaluate(oracle_dir, selected, proof["output_shape"])
        if extent == shape[2] and oracle.tobytes() != (fixture / "expected.bin").read_bytes():
            raise ValueError("current source oracle changed")
        (group / "expected.bin").write_bytes(oracle.tobytes())
        (group / "input.bin").write_bytes(selected.tobytes())
        for width in widths:
            arm = group / f"lanes{width}"
            implementation = compile_tail_map(module, arm, args.llvm_bin, width)
            if not np.array_equal(evaluate(arm, selected, proof["output_shape"]), oracle):
                raise ValueError("complete scheduled source outputs differ")
            tag = f"quant_h{extent}_w{width}"
            renamed = arm / "kernel_link.o"
            command = [str(args.objcopy)]
            for symbol in ("forward", "_mlir_ciface_forward", "dealloc_helper", *implementation["routes"][0]["helpers"].values()):
                selected_symbol = "mapped_" + tag if symbol == "forward" else "_mlir_ciface_" + tag if symbol.startswith("_mlir") else symbol + "_" + tag
                command += ["--redefine-sym", symbol + "=" + selected_symbol]
            subprocess.run(command + [str(arm / "kernel.o"), str(renamed)], check=True)
            if contents(renamed) != contents(arm / "kernel.o"):
                raise ValueError("renaming changed executable bytes")
            objects.append(renamed)
            cases.append({
                "id": len(cases), "family": "host_quant", "kernel": tag,
                "source_function": "mapped_" + tag,
                "size": selected.size, "extent": extent, "independent_lanes": width,
                "partition": "heldout", "expected_fflags": 1,
                "input_shape": proof["input_shape"], "output_shape": proof["output_shape"],
                "extent_proof": proof, "source_operations": {"fmul": selected.size, "minimum": selected.size, "maximum": selected.size, "fixed_rne_convert": selected.size},
                "requested_cpu_load_bytes": selected.nbytes,
                "requested_cpu_store_bytes": selected.size,
                "input_and_output_extent_bytes": selected.nbytes + selected.size,
                "scalar_graph_sha256": binding["source_operation_sha256"],
                "implementation": implementation, "renamed_object": pin(renamed),
                "original_object": pin(arm / "kernel.o"),
                "renaming_executable_bytes_identical": True,
                "source_oracle": pin(group / "expected.bin"), "input": pin(group / "input.bin"),
                "fixture_input_map": "per-channel row prefix/repetition r%originalHeight, unchanged scalar graph/layout; source oracle does not select eligibility",
                "new_unpriced_dimensions": ["new five/seven-lane executable schedule", "five-lane four-element tail per row", "different linked footprint and aggregate invocation allocation"],
            })
    checksum = 14695981039346656037
    for case in cases:
        for _ in range(2):
            for byte in Path(case["source_oracle"]["path"]).read_bytes():
                checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    manifest = {
        "schema": "source_host_quant_cross_executable_transfer_v1",
        "cases": cases, "repetitions": 2, "empty_windows": 3,
        "expected_checksum": f"{checksum:016x}",
        "partition_rule": "every new extent/width arm and repeat held; only frozen original2057 training0/1/4/5 parameters may be evaluated",
        "memory_regime": "same private aligned source/input/output owners and reset arena protocol; same input row-prefix/repetition; preparation before timer, complete guards/verification after; physical traffic and warmth UNKNOWN",
        "timing": "same fenced common indirect ranked-map call including descriptors/allocations/copy; setup/verification/UART excluded",
        "source_binding": pin(fixture / "receipt.json"),
        "producer": pin(__file__),
        "prospective_policy": "Original ELF/manifest model domain refuses transfer. Separately stated hardware/source-signature operational extrapolation may be scored with frozen2057 coefficients, never fitted to any new labels. Tail/stack/memory effects remain explicit unknowns; no automatic export/default promotion.",
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (work / "battery.c").write_text(c_source(cases, shape, checksum))
    assembly = '.section .rodata.source,"a",@progbits\n.balign 1048576\n.global original_source\noriginal_source:\n.incbin "original_input.bin"\n'
    for extent in extents:
        assembly += f'.balign 64\n.global expected_h{extent}\nexpected_h{extent}:\n.incbin "height{extent}/expected.bin"\n'
    (work / "operands.S").write_text(assembly)
    for source in (args.benchmark_header, args.allocator, args.allocator.with_name("htif.h")):
        shutil.copyfile(source, work / source.name)
    program = build_program([work / "battery.c", work / "operands.S", *objects, work / "merlin_malloc.c"], work, target="gemmini", max_loaded_bytes=None, extra_cflags=["-O2", "-fno-fast-math", "-ffp-contract=off", "-I", str(work), f"-DMERLIN_ARENA_BASE_ADDR={args.arena_base}ULL", "-DMERLIN_ARENA_SIZE_BYTES=0x10000000ULL"], support_first=True)
    audit = audit_elf(program.elf.read_bytes())
    if audit["status"] != "pass" or audit["custom_funct_counts"]:
        raise ValueError("zero custom/FSM executable policy failed")
    (work / "built.json").write_text(json.dumps({"schema": "source_host_quant_transfer_built_v1", "elf": pin(program.elf), "manifest": pin(work / "manifest.json"), "audit": audit, "sources": [pin(p) for p in (work / "battery.c", work / "operands.S", work / "benchmark_buffer.h", work / "merlin_malloc.c", Path(__file__))], "load_bytes": program.loaded_bytes, "arena_base": args.arena_base}, indent=2) + "\n")
    print(json.dumps({"cases": 6, "original_source_bytes_checked": sum(c["size"] for c in cases), "new_target_schedules": widths, "elf": pin(program.elf), "checksum": manifest["expected_checksum"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--objcopy", type=Path, required=True)
    parser.add_argument("--benchmark-header", type=Path, required=True)
    parser.add_argument("--allocator", type=Path, required=True)
    parser.add_argument("--arena-base", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    generate(parser.parse_args())
