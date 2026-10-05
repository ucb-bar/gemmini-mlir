"""Bind one completed strict Spike run to original operands and an immutable ELF.

This diagnostic collector does not launch hardware or reinterpret Spike mcycle
as cycles. The caller supplies the observed process exit status, after waiting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(lines: list[str], prefix: str) -> list[str]:
    values = [line.split() for line in lines if line.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"expected one {prefix!r}, got {len(values)}")
    return values[0]


def collect(case: Path, process_exit_code: int) -> dict:
    build_path = case / "build_qualification.json"
    build = json.loads(build_path.read_text())
    elf = Path(build["elf"])
    if process_exit_code != 0 or sha(elf) != build["elf_sha256"]:
        raise ValueError("unsuccessful process or changed executable")
    audit = audit_elf(elf.read_bytes())
    if audit["status"] != "pass" or audit["forbidden"] or audit["unknown"]:
        raise ValueError("final executable contains unqualified custom words")
    log_path = case / "spike.log"
    lines = log_path.read_text().splitlines()
    begin = one(lines, "ATTENTION_CERT_BEGIN ")
    variant, heads = int(begin[1]), int(begin[2])
    output = one(lines, "OUT_SHA256 ")
    count, nbytes = int(output[2]), int(output[3])
    if output[1] != "f32le" or count != build["element_count"] or nbytes != count * 4:
        raise ValueError("incomplete or differently encoded output")
    if output[4] != build["native_reference_raw_f32le_sha256"]:
        raise ValueError("target/native digest mismatch")
    if one(lines, "ATTENTION_CERT_ORIGINAL_BITS ")[1] != "0":
        raise ValueError("target differs from original BF16 bits")
    if one(lines, "ATTENTION_CERT_ORIGINAL_GATE ")[1] != "0":
        raise ValueError("original immutable elementwise gate failed")
    if one(lines, "METRIC memref_rank_mismatch ")[2] != "0":
        raise ValueError("rank mismatch")
    if one(lines, "METRIC build_hash ")[2] != build["build_hash_marker"]:
        raise ValueError("build marker mismatch")
    if lines.count("DONE") != 1:
        raise ValueError("missing or duplicate terminal sentinel")
    flags = build["compiler_argv"]
    for required in ["-O2", "-fno-fast-math", "-ffp-contract=off", "-march=rv64gc", "-mabi=lp64d"]:
        if required not in flags:
            raise ValueError(f"missing numeric/ISA compilation contract: {required}")
    if f"-DVARIANT={variant}" not in flags or f"-DHEADS={heads}" not in flags:
        raise ValueError("source strategy/scope does not match actual executable")
    for path, expected in build["original_fixture_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("original operand or golden changed")
    reference_path = case / "output.npy"
    if sha(reference_path) != build["native_reference_sha256"]:
        raise ValueError("independent native reference changed")
    reference = np.load(reference_path).astype("<f4")
    if hashlib.sha256(reference.tobytes()).hexdigest() != output[4]:
        raise ValueError("independent native output encoding changed")
    original_path = next(Path(p) for p in build["original_fixture_sha256"] if p.endswith("/output.npy"))
    original = np.load(original_path)[0, :heads].astype("<f4")
    if not np.array_equal(reference.view("u4"), original.view("u4")):
        raise ValueError("native reference differs from original-source BF16 output")
    numeric_closure = {}
    historical_builder = {}
    for path, expected in build["source_closure_sha256"].items():
        if Path(path).name == "build_capsule.py":
            historical_builder[path] = {"recorded_sha256": expected, "current_sha256": sha(Path(path)),
                "frozen_build_time_copy_available": False}
        else:
            if sha(Path(path)) != expected:
                raise ValueError("executed numeric source closure changed")
            numeric_closure[path] = expected
    for name, record in build["device_records"].items():
        if sha(case / name / "kernel.o") != record["compile"]["object_sha256"]:
            raise ValueError("device object changed")
    counts = one(lines, "ATTENTION_CERT_COUNTS ")
    stage_names = ["packing_and_operand_metadata", "qk_device_readback_recombination_norms",
                   "qk_certificate_softmax_and_replay", "pv_device_readback_recombination_norms",
                   "pv_certificate_ordered_partial_adds", "final_pv_replay_denominator_output"]
    stages = {}
    for index, name in enumerate(stage_names):
        row = one(lines, f"ATTENTION_CERT_STAGE_CYCLES {index} ")
        stages[name] = int(row[2])
    instructions = int(one(lines, "METRIC cycles ")[2])
    if sum(stages.values()) > instructions:
        raise ValueError("stage attribution exceeds complete timed interval")
    strategies = ["exact_device_absnorm_original_gamma",
                  "holder_metadata_signed_prefix_half_ulp_source_parts",
                  "holder_metadata_signed_prefix_gamma_source_parts"]
    return {
        "schema": "original_attention_strict_spike_qualification_v1",
        "qualified": True,
        "scope": f"{heads} complete heads of original first vision attention; no whole-model or other-layer qualification",
        "strategy": strategies[variant],
        "strategy_selection": "Experiment flag only; no production attention transform or model/provenance selector",
        "process_exit_code": process_exit_code,
        "elf": str(elf), "elf_sha256": build["elf_sha256"],
        "build_hash_marker": build["build_hash_marker"],
        "spike_log": str(log_path), "spike_log_sha256": sha(log_path),
        "build_qualification": str(build_path), "build_qualification_sha256": sha(build_path),
        "metric_kind": "strict Spike retired instructions; not hardware cycles",
        "complete_capsule_retired_instructions": instructions,
        "stage_retired_instructions": stages,
        "timed_interval": "Original input packing, metadata, actual primitive GEMMs/readbacks, recombination, certificates, softmax, replay, denominator refinement and output stores; digest/original audit/UART excluded",
        "physical_output_dtype": "bf16", "digest_encoding": "lossless BF16 widening to f32le",
        "output_element_count": count, "output_bytes": nbytes, "output_sha256": output[4],
        "independent_reference": str(reference_path), "independent_reference_sha256": sha(reference_path),
        "original_fixture_sha256": build["original_fixture_sha256"],
        "original_gate": {"atol": 0.03125, "rtol": 0.02, "failures": 0, "bit_mismatches": 0},
        "memref_rank_mismatch": 0, "done": True,
        "qk_unique_source_replays": int(counts[1]), "pv_source_outputs_replayed": int(counts[2]),
        "int32_plane_readback_bytes": int(counts[3]), "actual_plane_kernel_invocations": int(counts[4]),
        "executable_nofsm_audit": audit,
        "numeric_source_closure_sha256": numeric_closure,
        "historical_builder_record": historical_builder,
        "device_records": build["device_records"],
        "compiler_argv": flags, "compiler_sha256": build["compiler_sha256"],
        "token_usage_available": False,
        "token_attribution": "Root owns shared campaign snapshots; exact per-agent/optimization billing unavailable",
        "promotion": "Accuracy qualified diagnostic only; await actual stock FireSim before selecting strategy",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--process-exit-code", type=int, required=True)
    args = parser.parse_args()
    receipt = collect(args.case.resolve(), args.process_exit_code)
    target = args.case / "strict_spike_qualification.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous != receipt:
            raise ValueError("refusing to overwrite a different historical qualification")
    else:
        target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(target), "elf_sha256": receipt["elf_sha256"],
                      "retired_instructions": receipt["complete_capsule_retired_instructions"],
                      "qualified": receipt["qualified"]}))
