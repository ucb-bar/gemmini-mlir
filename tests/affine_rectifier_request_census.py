"""Exact per-invocation requests from a qualified compiled ranked pair.

This exports shared feature pointers from executed operands and the existing
OOT PC census. An optional frozen service law is conditional arithmetic only:
real-D/SPAD write, extra transfer/seed services and overlap are unpriced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from affine_rectifier_capability_probe import pin

from mlir_oot.executed_features import _symbol_ranges, census, parse_pc_histogram


def reclose(row):
    path = Path(row["path"])
    if pin(path)["sha256"] != row["sha256"]:
        raise ValueError("bound execution artifact changed")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    replay = json.loads(args.replay.read_text())
    assert replay["stdout_histogram_byte_identical_production"]
    for row in recipe["originals"] + recipe["generated"]:
        reclose(row)
    paths = {key: reclose(row) for key, row in replay["pins"].items()}
    assert (
        paths["stdout"].read_bytes()
        == (paths["elf"].parent / "spike.stdout").read_bytes()
    )
    assert (
        paths["histogram"].read_bytes()
        == (paths["elf"].parent / "spike.stderr").read_bytes()
    )
    telemetry = json.loads(paths["telemetry"].read_text())
    hist = parse_pc_histogram(paths["histogram"].read_text())
    rectifier_control = bool(recipe.get("immutable_rectifier_control_recipe"))
    symbols = [
        "immutable_rectifier_control_kernel"
        if rectifier_control
        else recipe["route"]["kernel"],
        "gemmini_golden_rectified_resadd",
    ]
    readelf = "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-readelf"
    ranges = _symbol_ranges(paths["elf"], symbols, readelf)
    fields = (
        "commands",
        "requested_load_bytes",
        "requested_store_bytes",
        "unknown_dma_commands",
        "padded_mac_slots",
        "padded_compute_rows",
        "preload_real_stationary_b",
        "preload_retained_stationary_b",
        "preload_unknown_mode",
    )
    rows = []
    for arm, bounds in enumerate(ranges):
        invocations = hist[bounds["start"]]
        assert invocations == 2 - arm
        selected = [
            row
            for row in telemetry["rows"]
            if bounds["start"] <= row["pc"] < bounds["end"]
        ]
        totals = {key: sum(row[key] for row in selected) for key in fields}
        assert all(value % invocations == 0 for value in totals.values())
        requests = {key: value // invocations for key, value in totals.items()}
        assert requests["unknown_dma_commands"] == requests["preload_unknown_mode"] == 0
        scope = "ranked_pair_kernel_" + str(arm)
        observed = census(
            paths["elf"],
            paths["histogram"],
            symbols=[bounds["symbol"]],
            scope_id=scope,
            readelf=readelf,
            execution_binding={
                "elf_sha256": replay["pins"]["elf"]["sha256"],
                "histogram_sha256": replay["pins"]["histogram"]["sha256"],
                "scope_id": scope,
            },
        )
        instructions = observed["features"]["executed_instructions"]
        assert instructions % invocations == 0
        features = {
            "executed_instructions": instructions // invocations,
            "array_work": requests["padded_mac_slots"],
            "array_padded_compute_rows": requests["padded_compute_rows"],
            "preload_real_stationary_b": requests["preload_real_stationary_b"],
            "preload_retained_stationary_b": requests["preload_retained_stationary_b"],
            "requested_dma_load_bytes": requests["requested_load_bytes"],
            "requested_dma_store_bytes": requests["requested_store_bytes"],
            "requested_dma_bytes": requests["requested_load_bytes"]
            + requests["requested_store_bytes"],
        }
        for value in observed["features"]["cpu_opcode_classes"].values():
            assert value % invocations == 0
        features["cpu_opcode_classes"] = {
            key: value // invocations
            for key, value in observed["features"]["cpu_opcode_classes"].items()
        }
        features["unique_executed_pcs"] = observed["features"]["unique_executed_pcs"]
        features["touched_instruction_bytes"] = observed["features"][
            "touched_instruction_bytes"
        ]
        rows.append(
            {
                "arm": arm,
                "symbol": bounds["symbol"],
                "executed_function_invocations": invocations,
                "features": features,
                "requests": requests,
                "counter_scope": "one complete kernel invocation, config/readonly tables+seeds/input loads/predictor store+reload/finalstore included; adapter/caller instructions separate",
                "feature_status": {
                    key: "observed_functional_request; physical activity/DRAM/cycles UNKNOWN"
                    for key in features
                },
            }
        )
    m, n = recipe["plan"]["m"], recipe["plan"]["n"]
    control_passes = (
        recipe["plan"]["compute_passes_per_tile"] if rectifier_control else 39
    )
    assert (
        rows[0]["features"]["array_padded_compute_rows"] == m * n * control_passes // 16
    )
    assert (
        rows[1]["features"]["array_padded_compute_rows"]
        == m * n * recipe["plan"]["compute_passes_per_tile"] // 16
    )
    record = {
        "schema": "source_bound_affine_rectifier_requests_v1",
        "status": "PASS",
        "rows": rows,
        "source_numeric_contract": "all65536 original signed-byte output pairs exact",
        "complete_scope_timing": "separate matched ranked adapter measurements; no timing inferred from census",
        "approved_cycle_prediction": {
            "status": "UNKNOWN",
            "reasons": [
                "Frozen ordinary resident-array/preload calibration does not cover real-D/SPAD-write/relu dependency services",
                "Candidate introduces predictor store/reload and seed transfers; complete memory/CPUissue/fence/overlap costs unpriced",
                "Aggregate extent outside calibration; source-correctness does not grant performance",
            ],
        },
        "pins": [
            pin(args.recipe),
            pin(args.replay),
            *replay["pins"].values(),
            pin(Path(__file__)),
        ],
        "token_usage_available": False,
    }
    if args.frozen_model:
        model = json.loads(args.frozen_model.read_text())
        coefficients = model["model_coefficients_unchanged"]
        assert len(coefficients) == 2 and model["model_training_updates"] == 0
        record["conditional_arithmetic_ignoring_domain"] = {
            "model": pin(args.frozen_model),
            "training_updates": 0,
            "coefficients": coefficients,
            "rows": [
                {
                    "arm": row["arm"],
                    "value": coefficients[0]
                    * row["features"]["array_padded_compute_rows"]
                    + coefficients[1] * row["features"]["preload_real_stationary_b"],
                }
                for row in rows
            ],
            "status": "NOT an approved cycle estimate or whole-model forecast; all unpriced obligations above remain",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
