"""Explore source-bound gather locality after observing the calibration labels.

Merlin owns the recurrence census and fitter. This fixture adapter reconstructs
the released kernel's request order; it does not reconstruct pre-window state,
the measurement wrapper's stack accesses, or physical cache traffic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast
from merlin.perf.address_locality import address_locality

from operational_service_cycle_screen_probe import reclose


def object_bases(elf: Path, readelf: str) -> dict[str, int]:
    text = subprocess.run([readelf, "-sW", str(elf)], check=True,
                          capture_output=True, text=True).stdout
    objects = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == "OBJECT" and fields[-1] in {
            "table", "indices", "output"
        }:
            if fields[-1] in objects or fields[6] in {"UND", "ABS"}:
                raise ValueError("ambiguous or undefined fixture object")
            objects[fields[-1]] = int(fields[1], 16)
    if set(objects) != {"table", "indices", "output"}:
        raise ValueError("released fixture object symbols missing")
    # The bound C fixture places eight u64 guard words before each payload.
    return {name: base + 8 * 8 for name, base in objects.items()}


def run(packet_path: Path, declaration_path: Path, readelf: str) -> dict:
    packet = json.loads(packet_path.read_text())
    declared = json.loads(declaration_path.read_text())
    reports, evidence = reclose(packet)
    if declared["schema"] != "gather_locality_exploration_v1":
        raise ValueError("explicit post-label exploration declaration required")
    if packet["built"]["elf_sha256"] != declared["elf_sha256"]:
        raise ValueError("fixture executable differs from declaration")
    manifest = packet["manifest"]
    if manifest["training_ids"] != declared["training_ids"] or \
            manifest["heldout_ids"] != declared["heldout_ids"]:
        raise ValueError("source partition differs from declaration")
    granule = declared["granule_bytes"]
    capacities = tuple(declared["logical_region_thresholds"])
    bases = object_bases(Path(packet["built"]["elf"]), readelf)
    rows = []
    for case in manifest["cases"]:
        indices = case["indices"]
        if len(indices) != case["size"] or any(
            type(index) is not int or not 0 <= index < case["working_set_bytes"] // 8
            for index in indices
        ):
            raise ValueError("invalid source request extent")
        table = [bases["table"] + index * 8 for index in indices]
        # Exact emitted gather order: context count/value, each index/table pair,
        # and the final context-value store. This is not the complete ROI trace.
        kernel = [bases["output"], bases["output"] + 8]
        for offset, address in enumerate(table):
            kernel.extend((bases["indices"] + offset * 4, address))
        kernel.append(bases["output"] + 8)
        views = {}
        for name, trace in (("table", table), ("kernel", kernel)):
            census = address_locality(trace, granule=granule,
                                      max_requests=declared["max_requests"],
                                      capacities=capacities)
            views[name] = asdict(census)
            views[name]["trace_sha256"] = canonical_sha256(trace)
        if views["table"]["first_touches"] != case["unique_requested_64B_regions"]:
            raise ValueError("source regions differ from bound addresses")
        rows.append({"case": case["id"], "partition": case["partition"],
                     "views": views})
    engines = {}
    for engine, (report, target) in reports.items():
        if engine == "spike":
            continue
        samples = defaultdict(list)
        for row in report["rows"]:
            if row["id"] >= 0:
                samples[row["id"]].append(row)
        domain = canonical_sha256({"engine": target, "scope": declared["scope"],
                                   "manifest": packet["manifest"]})
        diagnostics = {}
        for view in ("table", "kernel"):
            for threshold in capacities:
                observations = []
                for row in rows:
                    raw = samples[row["case"]]
                    if len(raw) != 2 or {x["repeat"] for x in raw} != {0, 1}:
                        raise ValueError("two complete repeats required")
                    census = row["views"][view]
                    recurrence = dict(census["recurrences_at_or_above"])[threshold]
                    features = {
                        "/instructions": raw[0]["instructions"],
                        "/logical_pressure": census["first_touches"] + recurrence,
                    }
                    observations.append((row["partition"], fast.Observation(
                        packet["built"]["elf_sha256"], canonical_sha256(row["case"]),
                        row["partition"], domain, features,
                        statistics.mean(x["cycles"] for x in raw), evidence,
                    )))
                try:
                    fit = fast.fit_linear_screen(
                        [r for part, r in observations if part == "training"],
                        pointers=("/instructions", "/logical_pressure"),
                        include_fixed=False, maximum_condition=1000,
                    )
                    checks = []
                    for part, row in observations:
                        if part == "heldout":
                            prediction = fit.predict(row.features, domain_sha256=domain)
                            checks.append({"workload": row.workload,
                                           "measured": row.cycles,
                                           "prediction": prediction.to_dict(),
                                           "relative_error": abs(prediction.lo - row.cycles) / row.cycles
                                           if prediction.resolved else None})
                    diagnostics[f"{view}:{threshold}"] = {
                        "coefficients": fit.coefficients,
                        "fit_sha256": fit.provenance_sha256,
                        "checks": checks,
                    }
                except ValueError as error:
                    diagnostics[f"{view}:{threshold}"] = {"refused": str(error)}
        engines[engine] = diagnostics
    import merlin.perf.address_locality as locality
    return {
        "schema": "gather_locality_exploratory_diagnostic_v1",
        "post_label_exploration": True, "independent_validation": False,
        "production_enabled": False, "object_payload_bases": bases,
        "packet": {"path": str(packet_path), "sha256": sha256_file(packet_path)},
        "declaration": {"path": str(declaration_path), "sha256": sha256_file(declaration_path)},
        "driver_sha256": sha256_file(Path(__file__)),
        "locality_producer": {"path": locality.__file__, "sha256": sha256_file(Path(locality.__file__))},
        "shared_fitter": {"path": fast.__file__, "sha256": sha256_file(Path(fast.__file__))},
        "reclosed_artifact_digests": len(evidence), "features": rows,
        "engines": engines,
        "limits": declared["limits"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet, args.declaration, args.readelf)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("GATHER_LOCALITY_EXPLORATION_COMPLETE", list(result["engines"]))
