"""Evaluate observed code footprint without licensing outside-domain predictions.

Physical-function groups preserve exact alias aggregation. Reference hardware
labels are evaluation-only. CPU latencies, cache misses and overlap are unknown;
this statistical diagnostic cannot become a physical service calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast
from operand_cycle_screen_probe import check_pins

POINTERS = (
    "/features/array_padded_compute_rows",
    "/features/touched_instruction_bytes",
    "/features/invocations",
)


def features(row):
    return {
        POINTERS[0]: row["features"]["array_padded_compute_rows"],
        POINTERS[1]: row["features"]["touched_instruction_bytes"],
        POINTERS[2]: row["invocations"],
    }


def run(observed_path, heldout_path):
    observed, heldout = (
        json.loads(path.read_text()) for path in (observed_path, heldout_path)
    )
    if any(
        x["schema"] != "gemmini_scoped_cpu_operand_features_v1"
        for x in (observed, heldout)
    ):
        raise ValueError("unexpected CPU scope dataset")
    if (
        observed["role"] != "independent_historical_control_observation"
        or heldout["role"] != "heldout_evaluation_only"
    ):
        raise ValueError("independent and held-out roles must be explicit")
    verified = {}
    check_pins(observed, verified)
    observed_evidence = tuple(sorted(set(verified.values())))
    check_pins(heldout, verified)
    prior = [
        json.loads(Path(x["pins"]["prior_operand_dataset"]["path"]).read_text())
        for x in (observed, heldout)
    ]
    check_pins(prior, verified)
    hardware = json.loads(
        Path(prior[0]["pins"]["hardware_receipt"]["path"]).read_text()
    )
    keys = ("hwdb_sha256", "bitstream_archive_sha256", "bitstream_sha256")
    if any(hardware[key] != prior[1]["hardware"][key] for key in keys):
        raise ValueError("hardware domains differ")
    domain = canonical_sha256(
        {
            "hardware": {key: hardware[key] for key in keys},
            "scope": "instrumented physical-function group and its exact source call intervals",
            "regime": "empirical screening; not physical services or fetch misses",
        }
    )
    for dataset in (observed, heldout):
        rows = dataset["observations"]
        if (
            sum(row["hardware_cycles"] for row in rows)
            != dataset["conservation"]["hardware_interval_cycles"]
        ):
            raise ValueError("physical groups do not conserve measured intervals")
        for row in rows:
            if (
                sum(scope["hardware_cycles"] for scope in row["source_intervals"])
                != row["hardware_cycles"]
            ):
                raise ValueError("shared function timing is not its exact call sum")
            if len(row["source_intervals"]) != row["invocations"]:
                raise ValueError("observed invocations disagree with source scopes")
    observations = []
    for row in observed["observations"]:
        workload = canonical_sha256(
            [observed["pins"]["prior_operand_dataset"]["sha256"], row["scope_id"]]
        )
        observations.append(
            fast.Observation(
                observed["pins"]["elf"]["sha256"],
                workload,
                workload,
                domain,
                features(row),
                row["hardware_cycles"],
                observed_evidence,
            )
        )
    model = fast.fit_linear_screen(
        observations, pointers=POINTERS, include_fixed=False, maximum_condition=1000
    )
    comparisons = []
    for row in heldout["observations"]:
        prediction = model.predict(features(row), domain_sha256=domain)
        relative_error = (
            abs(prediction.lo - row["hardware_cycles"]) / row["hardware_cycles"]
            if prediction.resolved
            else None
        )
        comparisons.append(
            {
                "scope_id": row["scope_id"],
                "physical_function": row["physical_function"],
                "invocations": row["invocations"],
                "prediction": prediction.to_dict(),
                "observed_cycles": row["hardware_cycles"],
                "relative_error": relative_error,
            }
        )
    resolved = [row for row in comparisons if row["prediction"]["resolved"]]
    return {
        "schema": "golden_cpu_footprint_cycle_screen_probe_v1",
        "evidence": {
            "driver_sha256": sha256_file(Path(__file__)),
            "shared_validator_sha256": sha256_file(Path(fast.__file__)),
            "observed": {
                "path": str(observed_path),
                "sha256": sha256_file(observed_path),
            },
            "heldout": {"path": str(heldout_path), "sha256": sha256_file(heldout_path)},
            "rehash_verified_files": len(verified),
        },
        "fit": {
            "pointers": POINTERS,
            "coefficients": model.coefficients,
            "fixed_cycles": model.fixed_cycles,
            "domains": model.domains,
            "provenance_sha256": model.provenance_sha256,
            "fit_jobs": [hardware["job_id"]],
            "reference_labels_used": False,
        },
        "resolved_reference_groups": len(resolved),
        "required_reference_groups": len(comparisons),
        "maximum_resolved_relative_error": max(
            (row["relative_error"] for row in resolved), default=None
        ),
        "comparisons": comparisons,
        "whole_prediction": "UNKNOWN: outside-domain repeated groups, residual and host services are unpriced",
        "ranking": "UNKNOWN: this corpus has no matched alternatives for a held-out workload",
        "production_enabled": False,
        "scope": "Historical1874/1899 device intervals; not current1903 or whole-model costs",
        "missing_mechanism_facts": [
            "physical traffic",
            "instruction fetch chronology",
            "CPU/array overlap",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.observed, args.heldout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "resolved_reference_groups",
                    "required_reference_groups",
                    "maximum_resolved_relative_error",
                    "whole_prediction",
                    "ranking",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
