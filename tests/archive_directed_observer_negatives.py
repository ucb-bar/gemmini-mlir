"""Seal existing exact-observer negatives and their actual source-context join."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "out/artifacts/probes"
OLD = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
FIRST_CORE = Path("/scratch/agustin/tmp/merlin-directed-rne-endpoint-main-20261007")
SECOND_CORE = Path(
    "/scratch/agustin/tmp/merlin-directed-rne-endpoint-branchless-20261007"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    destination = ROOT / "docs/perf_records/directed_observer_negatives_20261007"
    destination.mkdir(parents=True, exist_ok=False)
    qualifiers = [
        PROBES / "directed-rne-endpoint-complete-M8-20261007/qualification.json",
        PROBES / "directed-rne-endpoint-branchless-M8-20261007/qualification.json",
    ]
    pins = {}
    variants = []
    for name, path, core in zip(
        ("direction_branch", "direction_complement"),
        qualifiers,
        (FIRST_CORE, SECOND_CORE),
        strict=True,
    ):
        qualification = json.loads(path.read_text())
        for filename, expected in qualification["pins"].items():
            assert sha(filename) == expected, filename
            pins[filename] = expected
        pins[str(path)] = sha(path)
        timing = qualification["cases"][1]
        counts = [int(row["instructions"]) for row in timing["rows"]]
        control = (counts[0] + counts[3]) / 2
        candidate = (counts[1] + counts[2]) / 2
        variants.append(
            {
                "variant": name,
                "core_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=core, text=True
                ).strip(),
                "core_source_sha256": sha(
                    core / "src/merlin/llvmlower/source_expression_interval.py"
                ),
                "qualification": {"path": str(path), "sha256": sha(path)},
                "timing_elf": {
                    "path": timing["elf_path"],
                    "sha256": timing["elf_sha256"],
                },
                "all_mode_elf": qualification["cases"][0]["elf_sha256"],
                "complete_ROI_retired_ABBA": counts,
                "complete_ROI_retired_means": {
                    "control": control,
                    "candidate": candidate,
                },
                "retired_percent_change": 100 * (candidate / control - 1),
                "gate": "PASS original45056 i8 / target5FRM and7sticky / input hashes / output guards / final executable noFSM",
                "disposition": "RETAINED NEGATIVE RETIRED-WORK SCREEN; no hardware, whole-model, main, or default promotion",
                "hardware_cycles": "UNKNOWN",
            }
        )
        files = {
            "core_source.py": core
            / "src/merlin/llvmlower/source_expression_interval.py",
            "core_test.py": core / "merlin/tests/ir/test_ordered_rne_endpoint.py",
            "test.log": core
            / "out/artifacts/probes/directed-rne-endpoint-validation-20261007"
            / (
                "formatted_tests.log"
                if name == "direction_branch"
                else "branchless_tests.log"
            ),
            "initial_test_diagnostic.log": FIRST_CORE
            / "out/artifacts/probes/directed-rne-endpoint-validation-20261007/initial_tests.log",
            "qualification.json": path,
        }
        child = destination / name
        child.mkdir()
        for label, source in files.items():
            shutil.copyfile(source, child / label)
            pins[str(source)] = sha(source)
            pins[str(child / label)] = sha(child / label)
    binding_path = (
        OLD
        / "out/artifacts/probes/closed-i8-interval-result-20261007/source_binding.json"
    )
    context_path = (
        OLD
        / "out/artifacts/probes/source-continuation-contexts-v3-20261007/features_v2/contexts.json"
    )
    binding = json.loads(binding_path.read_text())
    contexts = json.loads(context_path.read_text())["rows"]
    quant_words = {route["quant_factor_bits"] for route in binding["routes"]}
    assert len(quant_words) == 1
    selected = [row for row in contexts if row["quant_factor_bits"] in quant_words]
    assert len(selected) == 1 and selected[0]["context"] == 21
    row = selected[0]
    assert row["preparation_factor_bits"] == 1016741769
    assert row["source_symbol"] == "forward.extracted.531"
    capture = (
        OLD / "out/artifacts/probes/source-continuation-contexts-v3-20261007/context_21"
    )
    for path in (binding_path, context_path, *capture.glob("*.npy")):
        pins[str(path)] = sha(path)
    features_path = PROBES / "directed-rne-endpoint-execution-20261007/features.json"
    features = json.loads(features_path.read_text())
    normal = {}
    for symbol in ("cells_M8_rne", "directed_M8_rne"):
        entry = features["entries"][symbol]
        original = features["features"][symbol]["features"]
        assert all(
            value % entry == 0 for value in original["cpu_opcode_classes"].values()
        )
        normal[symbol] = {
            "actual_entries": entry,
            "instructions_per_entry": original["executed_instructions"] // entry,
            "classes_per_entry": {
                key: value // entry
                for key, value in original["cpu_opcode_classes"].items()
            },
        }
    for directory in (
        PROBES / "directed-rne-endpoint-execution-20261007",
        PROBES / "directed-rne-endpoint-driver-20261007",
    ):
        for path in directory.rglob("*"):
            if path.is_file():
                pins[str(path)] = sha(path)
    for path in (
        ROOT / "tests/directed_rne_endpoint_target.py",
        ROOT / "tests/directed_rne_endpoint_branchless.py",
        Path(__file__),
        FIRST_CORE / "merlin/tests/ir/test_ordered_rne_endpoint.py",
        SECOND_CORE / "merlin/tests/ir/test_ordered_rne_endpoint.py",
    ):
        pins[str(path)] = sha(path)
    receipt = {
        "schema": "source_directed_observer_negative_and_context_join_v1",
        "variants": variants,
        "context_correction": {
            "actual_context": 21,
            "source_symbol": row["source_symbol"],
            "original_helper_sha256": row["original_helper_sha256"],
            "preparation_factor_bits": row["preparation_factor_bits"],
            "quant_factor_bits": row["quant_factor_bits"],
            "source_expression_sha256": binding["routes"][0][
                "source_expression_sha256"
            ],
            "source_replay_elements_per_helper": row["source_replay_elements"],
            "not_context0": "Historical firstM8 means the inherited isolated timing fixture, not the first layer. Context0 census is not the binding for stock2082/2097. Original bytes/ELFs/gates and receipts remain untouched.",
            "unchanged_complete_source_binding": binding,
        },
        "executed_feature_attribution": normal,
        "direction_branch_attribution": {
            "high_endpoint_quantizers_removed": 17945,
            "minmax_instructions_removed": 35890,
            "integer_instructions_added": 125831,
            "integer_loads_added": 17945,
            "conditional_branches_added": 17945,
            "rounded_FMUL_per_entry_both": 360808,
            "source_continuation_calls_per_entry_both": 216,
            "frame_bytes_both": 224,
            "interpretation": "One endpoint conversion is removed, but its dependent key/address/edge load/branch work increases retirement. Cross-call/memory latency and hardware cycles remain UNKNOWN.",
        },
        "proof_scope": "Strict positive source quantization factor, fully rederived source-wide finite table endpoints, typed closed integer observation and explicit nontrapping/unobserved floating flags permission. Rounded finishing FMULs and original continuation retained. Default emission unchanged.",
        "source_unsupported_domain": "The original floating continuation may produce NaN (for example overflow times zero). Original fptosi then has poison/undefined conversion semantics. Native independent tests retain this diagnostic and do not invent an i8 oracle or claim universal source parity there.",
        "mode_scope": "Native four supported host modes x7sticky; target all five FRM x7sticky. These scopes are separate.",
        "storage": {
            "unchanged_source_table_bytes": 524288,
            "extra_immutable_edge_bytes": 2048,
        },
        "cost_scope": "Both source scale scans, source input casts/products, immutable table traffic, integer certificate, original cold continuation, frame and stores inside complete ROI; output validation and input hash outside ROI.",
        "no_new_whole_or_hardware_candidate": True,
        "token_usage_available": False,
        "pins": pins,
    }
    path = destination / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# Directed observer experiments\n\nBoth exact-observer variants pass the unchanged original source gates but increase complete capsule retirement. Neither was timed on hardware or promoted. The actual inherited M8 fixture is source context21, not context0. Original evidence is immutable; this receipt supplies the explicit correction and per-entry instruction attribution.\n\nHardware cycle, memory latency and whole-model costs remain unknown. The successor work targets overlap between sibling integer producers and their closed pointwise integer observer.\n"
    )
    print(
        json.dumps(
            {
                "receipt": str(path),
                "sha256": sha(path),
                "pins": len(pins),
                "variants": variants,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
