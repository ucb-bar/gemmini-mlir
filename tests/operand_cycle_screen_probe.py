"""Check one explicit additive screen against independent operand scopes.

This diagnostic deliberately refuses to replace unidentifiable terms with a
hand-selected CPI. Reference cycle labels are excluded from fitting. Hardware
service calibration and dependency composition remain shared Merlin owners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast

POINTERS = (
    "/features/executed_instructions",
    "/features/array_padded_compute_rows",
    "/features/requested_dma_load_bytes",
    "/features/requested_dma_store_bytes",
)


def check_pins(value, verified):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path, digest = value["path"], value["sha256"]
            if path in verified and verified[path] != digest:
                raise ValueError("one evidence path has conflicting pinned digests")
            if path not in verified:
                if sha256_file(Path(path)) != digest:
                    raise ValueError(f"evidence changed: {path}")
                verified[path] = digest
        for child in value.values():
            check_pins(child, verified)
    elif isinstance(value, list):
        for child in value:
            check_pins(child, verified)


def run(observed_path, heldout_path):
    observed = json.loads(observed_path.read_text())
    heldout = json.loads(heldout_path.read_text())
    if observed["schema"] != "gemmini_current_model_layer_features_v1":
        raise ValueError("unexpected observed layer schema")
    if heldout["schema"] != "gemmini_reference_heldout_layer_features_v1":
        raise ValueError("unexpected held-out layer schema")
    if heldout["role"] != "heldout_evaluation_only":
        raise ValueError("reference labels must remain held out")
    evidence = {}
    check_pins(observed, evidence)
    observed_evidence = tuple(sorted(set(evidence.values())))
    check_pins(heldout, evidence)
    hardware = json.loads(
        Path(observed["pins"]["hardware_receipt"]["path"]).read_text()
    )
    for key in ("hwdb_sha256", "bitstream_archive_sha256", "bitstream_sha256"):
        if hardware[key] != heldout["hardware"][key]:
            raise ValueError("independent and held-out hardware identities differ")
    domain = canonical_sha256(
        {
            "hardware": {
                key: hardware[key]
                for key in (
                    "hwdb_sha256",
                    "bitstream_archive_sha256",
                    "bitstream_sha256",
                )
            },
            "scope": "instrumented_source_bound_leaf_call",
            "services": "statistical additive diagnostic, not physical mechanism calibration",
        }
    )
    rows = []
    for leaf in observed["layers"]:
        # These digests identify experimental scopes. They do not select code.
        workload = canonical_sha256([observed["numeric_contract"], leaf["scope_id"]])
        group = canonical_sha256([leaf["category"], leaf["shape"]])
        rows.append(
            fast.Observation(
                observed["pins"]["elf"]["sha256"],
                workload,
                group,
                domain,
                {
                    pointer: leaf["features"][pointer.removeprefix("/features/")]
                    for pointer in POINTERS
                },
                leaf["hardware_cycles"],
                observed_evidence,
            )
        )
    if (
        sum(row.cycles for row in rows)
        != observed["hardware_profile"]["device_counter"]
    ):
        raise ValueError("observed leaf cycles do not conserve the device intervals")
    result = {
        "schema": "golden_additive_operand_cycle_screen_probe_v1",
        "evidence": {
            "driver_sha256": sha256_file(Path(__file__)),
            "shared_validator_sha256": sha256_file(Path(fast.__file__)),
            "observed": {
                "path": str(observed_path),
                "sha256": sha256_file(observed_path),
            },
            "heldout": {"path": str(heldout_path), "sha256": sha256_file(heldout_path)},
            "rehash_verified_files": len(evidence),
        },
        "fit_policy": {
            "fit_jobs": [observed["profile_job"]],
            "heldout_job": heldout["hardware"]["job_id"],
            "heldout_labels_used_for_fitting": False,
            "pointers": POINTERS,
            "include_fixed": True,
            "maximum_condition": 1000,
            "distinct_observed_calls": len(rows),
        },
        "production_enabled": False,
        "physical_dram_bytes": "UNKNOWN",
        "overlap": "UNKNOWN: geometry aggregation does not establish chronological dependencies",
        "next_observations": [
            "CPU instruction classes and serialized readout cost",
            "declared command dependencies and issue/array overlap",
            "independent service calibration and variant ranking",
        ],
    }
    try:
        model = fast.fit_linear_screen(
            rows, pointers=POINTERS, include_fixed=True, maximum_condition=1000
        )
    except ValueError as exc:
        result["fit"] = {"status": "REJECTED", "reason": str(exc)}
        result["heldout_prediction"] = "UNKNOWN: independent additive fit refused"
        return result
    result["fit"] = {
        "status": "diagnostic_only",
        "provenance_sha256": model.provenance_sha256,
    }
    predictions = []
    for leaf in heldout["layers"]:
        prediction = model.predict(
            {
                pointer: leaf["features"][pointer.removeprefix("/features/")]
                for pointer in POINTERS
            },
            domain_sha256=domain,
        )
        predictions.append(
            {
                "scope_id": leaf["scope_id"],
                "prediction": prediction.to_dict(),
                "observed_cycles": leaf["hardware_cycles"],
            }
        )
    result["heldout_predictions"] = predictions
    result["whole_prediction"] = (
        "UNKNOWN: residual and host services require independent closure"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.observed, args.heldout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["fit"], indent=2))


if __name__ == "__main__":
    main()
