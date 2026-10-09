"""Recheck existing timing pairs and evaluate a counter-only diagnostic screen.

Shared fitting, held-out validation and ranking live in Merlin. This experiment
joins existing queue receipts and the ZIP-reference diagnostic. Reference cycle
labels never enter coefficient fitting. This is not a production cost provider.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast
from merlin.perf import rank_validation as rank

POINTER = "/features/executed_instructions"


def verified(reference):
    path = Path(reference["path"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"evidence changed: {path}")
    return path


def fit(rows):
    return fast.fit_linear_screen(
        rows, pointers=(POINTER,), include_fixed=True, maximum_condition=1000
    )


def run(pairs_path, reference_path):
    pairs = json.loads(pairs_path.read_text())
    if pairs["schema"] != "existing_same_elf_timing_pairs_v1":
        raise ValueError("unexpected input timing-pair schema")
    groups = defaultdict(list)
    job_by_program = {}
    evidence = []
    for row in pairs["rows"]:
        refs = [
            value
            for value in row.values()
            if isinstance(value, dict) and {"path", "sha256"} <= value.keys()
        ]
        for ref in refs:
            verified(ref)
        output = row["full_output"]
        verified({"path": output["reference"], "sha256": output["reference_sha256"]})
        hardware = json.loads(Path(row["hardware_receipt"]["path"]).read_text())
        if (
            hardware["state"] != "DONE"
            or hardware["exit_code"] != 0
            or hardware["elf_sha256"] != row["elf"]["sha256"]
            or hardware["kernel_cycles"] != row["hardware_cycles"]
            or hardware["marker_kind"] != row["marker_kind"]
            or any(hardware[key] != value for key, value in row["pins"].items())
        ):
            raise ValueError(f"hardware join disagrees for {row['job']}")
        console = Path(row["spike_console"]["path"]).read_text()
        counters = [
            int(line.split()[2])
            for line in console.splitlines()
            if line.startswith("METRIC cycles ")
        ]
        if counters != [row["spike_retired_instruction_counter"]]:
            raise ValueError(
                "instruction counter does not match the pinned Spike console"
            )
        # Group immutable captured-reference variants together. This joins
        # evidence only; it never selects compiler behavior or changes inputs.
        workload = canonical_sha256(
            [row["scope"], output["reference_sha256"], output["elements"]]
        )
        domain = canonical_sha256({"hardware": row["pins"], "scope": row["scope"]})
        observation = fast.Observation(
            row["elf"]["sha256"],
            workload,
            workload,
            domain,
            {POINTER: row["spike_retired_instruction_counter"]},
            row["hardware_cycles"],
            tuple(ref["sha256"] for ref in refs),
        )
        groups[row["marker_kind"]].append(observation)
        job_by_program[observation.id] = row["job"]
        evidence.extend(refs)
    scopes = {}
    for name, rows in groups.items():
        programs = [rank.Program(r.workload, r.id, r.cycles, r.group) for r in rows]
        raw = {r.id: r.features[POINTER] for r in rows}
        scopes[name] = {
            "jobs": [job_by_program[r.id] for r in rows],
            "raw_counter_ranking": rank.agreement(
                rank.ordered_pairs(programs), raw
            ).to_dict(),
            "feature_collisions": fast.feature_collisions(rows, (POINTER,)),
            "grouped_validation": fast.cross_validate(
                rows,
                fit,
                maximum_relative_error=0.2,
                minimum_predictions=len(rows),
                minimum_rank_rate=0.8,
                minimum_decided=6,
                minimum_slice_decided=2,
                minimum_slices=2,
            ),
        }
    reference = json.loads(reference_path.read_text())
    if reference["schema"] != "gemmini_reference_heldout_layer_features_v1":
        raise ValueError("unexpected reference feature schema")
    for ref in reference["pins"].values():
        verified(ref)
    whole_rows = groups["WHOLE_MODEL"]
    hardware_pins = json.loads(pairs_path.read_text())["rows"][0]["pins"]
    if any(reference["hardware"][key] != value for key, value in hardware_pins.items()):
        raise ValueError("reference and training target pins differ")
    model = fit(whole_rows)
    whole = reference["whole_full_model_counter_scope"]
    prediction = model.predict(
        {POINTER: whole["executed_instructions"]}, domain_sha256=whole_rows[0].domain
    )
    error = (
        abs(prediction.lo - whole["hardware_cycles"]) / whole["hardware_cycles"]
        if prediction.resolved
        else None
    )
    return {
        "schema": "golden_counter_only_cycle_screen_probe_v1",
        "evidence": {
            "driver_sha256": sha256_file(Path(__file__)),
            "shared_validator_sha256": sha256_file(Path(fast.__file__)),
            "pairs": {"path": str(pairs_path), "sha256": sha256_file(pairs_path)},
            "reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            "rehash_verified_files": len(evidence) + len(reference["pins"]),
        },
        "scopes": scopes,
        "reference_check": {
            "fit_jobs": [job_by_program[r.id] for r in whole_rows],
            "reference_job": reference["hardware"]["job_id"],
            "reference_used_for_fitting": False,
            "measured": whole,
            "prediction": prediction.to_dict(),
            "relative_error": error,
            "passes_20_percent_total_check": error is not None and error <= 0.2,
            "section_check": "UNKNOWN: this diagnostic has no independently calibrated section terms",
            "scope_licence": "same stock target; whole forward ROIs, with differing source interfaces and instrumentation",
        },
        "production_enabled": False,
        "required_next_features": [
            "scoped primitive commands",
            "emitted array work",
            "requested DMA bytes",
            "dependency and overlap evidence",
        ],
        "licence": "counter-only screening diagnostic, not physical service-rate calibration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.pairs, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["reference_check"], indent=2))


if __name__ == "__main__":
    main()
