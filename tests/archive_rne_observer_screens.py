"""Seal exact observer screens and their unpriced cost evidence for review."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rne_zero_observer_capsule import CONTROL, CORE, HERE, LLVM, OUT, TYPED, save, sha


def main():
    archive = HERE / "docs/perf_records/rne_observer_screens_20261007"
    archive.mkdir(parents=True, exist_ok=False)
    cases = {
        "full_cells": OUT.parent / "rne-observer-cells-M8-v2-20261007",
        "zero_bin": OUT,
    }
    for name, case in cases.items():
        destination = archive / name
        destination.mkdir()
        for filename in (
            "lookup.c",
            "selected.ll",
            "activation.ll",
            "quantize.ll",
            "scale_guard.c",
            "target_guard.c",
            "candidate.dump",
            "qualification.json",
        ):
            shutil.copyfile(case / filename, destination / filename)
        for sub in ("all_modes", "timing"):
            (destination / sub).mkdir()
            for filename in (
                "main.c",
                "qualification.json",
                "nofsm_audit.json",
                "spike.stdout",
                "spike.stderr",
            ):
                shutil.copyfile(case / sub / filename, destination / sub / filename)
    for filename in (
        "normalized_source_reemission.json",
        "prospective_cost_declaration.json",
        "execution_features.json",
        "fp_dependencies.json",
        "all22_zero_context_census.json",
        "measured_core_identity.json",
    ):
        shutil.copyfile(OUT / filename, archive / "zero_bin" / filename)
    shutil.copytree(OUT / "measured_core", archive / "zero_bin/measured_core")
    shutil.copyfile(OUT / "measured_driver.py", archive / "zero_bin/measured_driver.py")
    for filename in (
        "observer_cells_source_tests_final.log",
        "observer_cells_structure_final.log",
    ):
        shutil.copyfile(CORE / "out" / filename, archive / filename)
    pins = {}
    primary_resolutions = []
    for name, case in cases.items():
        primary = json.loads((case / "qualification.json").read_text())
        for p, expected in primary["pins"].items():
            path = Path(p)
            if not path.is_file() or sha(path) != expected:
                resolution = {
                    "case": name,
                    "original_path": p,
                    "historical_sha256": expected,
                    "current_sha256": sha(path) if path.is_file() else None,
                }
                snapshots = [
                    case / "measured_driver.py",
                    case / "measured_core" / path.name,
                ]
                matching = [s for s in snapshots if s.is_file() and sha(s) == expected]
                resolution["exact_snapshot"] = str(matching[0]) if matching else None
                resolution["scope"] = (
                    "Retained historical primary; exact source snapshot when available. Final source emitter reproduces both executed lookup C byte-identically, original generatedLLVM/object/ELF/log/source data remain closed. Full-cell original driver formatting drift is retained as a historical unresolved source pin, no promotion."
                )
                primary_resolutions.append(resolution)
        for p in case.rglob("*"):
            if p.is_file():
                pins[str(p)] = sha(p)
        for p in primary["pins"]:
            path = Path(p)
            if path.is_file():
                pins[p] = sha(path)
    for p in [
        *archive.rglob("*"),
        *HERE.glob("tests/rne*py"),
        Path(__file__),
        *CORE.glob("src/merlin/llvmlower/bounded_rne_word_cells.py"),
        CORE / "src/merlin/llvmlower/source_expression_interval.py",
        CORE / "merlin/tests/ir/test_bounded_rne_word_cells.py",
        CORE / "src/merlin/llvmlower/AGENT.md",
        LLVM / "clang",
        LLVM / "opt",
        LLVM / "llvm-link",
        TYPED,
    ]:
        if p.is_file():
            pins[str(p)] = sha(p)
    for p, expected in json.loads((CONTROL / "timing/qualification.json").read_text())[
        "objects"
    ].items():
        assert sha(p) == expected
        pins[p] = expected
    packet = {
        "schema": "source_exact_rne_observer_complete_screens_v1",
        "source_baseline": "Current2076 original finite-scanned complete M8 helper; same source/table/scanners/unsupported-mode continuation.",
        "full_cell_result": {
            "status": "held_negative_instruction_screen",
            "control_instructions": 1704288.5,
            "candidate_instructions": 2064620,
            "delta_percent": 100 * (2064620 / 1704288.5 - 1),
            "cycles": "UNKNOWN",
            "added_bin_table_bytes": 2048,
            "reason": "Dependent first-quantization-indexed loads, integer key conversion and two comparisons add21.14% instructions; no credible cycle ranking asserted.",
        },
        "upper_edge_only": {
            "status": "refused",
            "reason": "Signed source up factors mean final interval endpoint order is not statically fixed; finite scale proof supplies no sign restriction.",
        },
        "zero_result": {
            "status": "qualified_capsule_pending_complete_cycle_price",
            "control_instructions": 1704288.5,
            "candidate_instructions": 1726434,
            "delta_percent": 100 * (1726434 / 1704288.5 - 1),
            "removed_fp_convert_per_call": 54186,
            "removed_fp_minmax_per_call": 108372,
            "source_activation_calls_per_helper": 216,
            "new_data_bytes": 0,
            "cycles": "UNKNOWN",
            "hardware_or_whole_promotion": False,
        },
        "gate": "Original45056i8, guards, immutable input hashes, all5FRM/7sticky source fallback, noFSM;59focused independent/core source tests and structure checks. All22 numerical cheap zero predicate census452646/991232; no compiled all22 or whole successor claimed.",
        "core_commit": "bcd6d3fc4",
        "core_path": str(CORE),
        "primary_receipts_unchanged": True,
        "historical_pin_resolutions": primary_resolutions,
        "short_pair": {
            "elf": str(OUT / "timing/build/layer.elf"),
            "elf_sha256": sha(OUT / "timing/build/layer.elf"),
            "qualification": str(OUT / "timing/qualification.json"),
            "allmode_qualification": str(OUT / "all_modes/qualification.json"),
            "feature_reemission": str(OUT / "normalized_source_reemission.json"),
            "prelabel": str(OUT / "prospective_cost_declaration.json"),
            "protocol": "Unchanged INTEGER_RESULT_SOURCE_GATE, POINTERS,4ABBA INTEGER_RESULT_ROW and INTEGER_RESULT_PASS; same parser as source finite M8 stock2075.",
        },
        "pins": pins,
        "token_usage_available": False,
    }
    save(archive / "receipt.json", packet)
    (archive / "README.md").write_text("""# Exact source observer alternatives

Merlin owns typed source/effect admission and numeric preimages; OOT supplies
the existing mode/ISA wrapper and experiment binding. Both preserve source
rounded multiplies, immutable activation table, original continuation and i8
observation. Unsupported modes retain the original source. Neither is enabled
automatically or selected by model/provenance/golden labels.

The complete full-cell alternative passed strict original observations but
added21.14% retired instructions. It is held; cycles are unknown. One-sided
membership was refused because signed source up factors do not prove order.

The separate sufficient zero-bin alternative deletes two original observers
only when both computed endpoints belong to the exact ties-even zero preimage.
All other values retain the original two observers and continuation. It adds
1.30% instructions; actual PC counts show54186 fewer conversions and108372
fewer min/max operations per complete M8 call. The original216 cold activation
calls are unchanged. Stock complete-cost pricing is pending; no whole model
or cycle claim is made. All22 cheap source-input census is deliberately separate
from the compiled first-context gate and yields452646 zero admissions among
991232 words. Physical memory/chronology/latency and whole transfer are unpriced.

Primary receipts are immutable. Exact measured source snapshots resolve the
zero variant's formatting/hardening changes; final normalized source emits
both measured lookup C bodies and ordinary control byte-identically. The full
cell driver was formatted before a source snapshot was saved: that historical
source pin remains explicitly unresolved, while its actual generated code,
objects, ELF and strict logs remain preserved. It is not a promotion packet.
""")
    print(
        "RNE_OBSERVER_SCREENS_SEALED",
        archive / "receipt.json",
        sha(archive / "receipt.json"),
        len(pins),
        flush=True,
    )


if __name__ == "__main__":
    main()
