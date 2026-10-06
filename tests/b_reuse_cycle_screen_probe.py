"""Fit the predeclared matched resident-B operational screen using Merlin."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast

from operational_service_cycle_screen_probe import reclose


def run(packet_path, declaration_path):
    packet = json.loads(packet_path.read_text())
    declaration = json.loads(declaration_path.read_text())
    reports, evidence = reclose(packet)
    if declaration["schema"] != "matched_b_reuse_model_hypotheses_v1":
        raise ValueError("explicit matched source declaration required")
    cases = packet["manifest"]["cases"]
    if len(cases) != 6 or any(c["N"] != 64 or c["K"] != 64 for c in cases):
        raise ValueError("different released resident fixture")
    if [c["id"] for c in cases if c["partition"] == "training"] != declaration["train_case_ids"] or \
            [c["id"] for c in cases if c["partition"] == "heldout"] != declaration["heldout_case_ids"]:
        raise ValueError("source partition changed")
    outcomes = {}
    for engine, (report, target) in reports.items():
        if engine == "spike":
            continue
        grouped = defaultdict(list)
        for row in report["rows"]:
            if row["id"] >= 0:
                grouped[row["id"]].append(row)
        domain = canonical_sha256({"engine": target,
                                   "manifest": packet["manifest"],
                                   "scope": declaration["family_scope"]})
        models = {}
        for names in declaration["hypotheses"] + declaration["diagnostic_only"]:
            pointers = tuple("/features/" + n for n in names)
            observations = []
            for case in cases:
                raw = grouped[case["id"]]
                if len(raw) != 2 or {r["repeat"] for r in raw} != {0, 1}:
                    raise ValueError("two complete repeats required")
                features = {"retired_instructions": raw[0]["instructions"], **case}
                observations.append((case, fast.Observation(
                    canonical_sha256([packet["built"]["elf_sha256"], case["kernel"]]),
                    canonical_sha256([case["M"], case["N"], case["K"]]),
                    "shape:" + str(case["M"]), domain,
                    {pointer: features[name] for pointer, name in zip(pointers, names, strict=True)},
                    statistics.mean(r["cycles"] for r in raw), evidence,
                )))
            try:
                fit = fast.fit_linear_screen(
                    [r for c, r in observations if c["partition"] == "training"],
                    pointers=pointers, include_fixed=declaration["include_fixed"],
                    maximum_condition=declaration["maximum_condition"],
                )
                checks = []
                for case, row in observations:
                    if case["partition"] != "heldout":
                        continue
                    prediction = fit.predict(row.features, domain_sha256=domain)
                    checks.append({"case": case["id"], "kernel": case["kernel"],
                                   "measured_cycles": row.cycles,
                                   "prediction": prediction.to_dict(),
                                   "relative_error": abs(prediction.lo-row.cycles)/row.cycles
                                   if prediction.resolved else None})
                models["+".join(names)] = {
                    "coefficients": fit.coefficients, "fit_sha256": fit.provenance_sha256,
                    "heldout_checks": checks,
                    "coverage": {"resolved": sum(c["relative_error"] is not None for c in checks),
                                 "required": 2},
                    "heldout_order": {
                        "measured": [c["kernel"] for c in sorted(checks, key=lambda c: c["measured_cycles"])],
                        "predicted": [c["kernel"] for c in sorted(checks, key=lambda c: c["prediction"]["lo"])],
                        "tie": checks[0]["prediction"]["lo"] == checks[1]["prediction"]["lo"],
                    },
                    "diagnostic_only": names in declaration["diagnostic_only"],
                }
            except ValueError as error:
                models["+".join(names)] = {"refused": str(error)}
        outcomes[engine] = models
    return {
        "schema": "matched_b_reuse_cycle_screen_diagnostic_v1",
        "packet": {"path": str(packet_path), "sha256": sha256_file(packet_path)},
        "declaration": {"path": str(declaration_path), "sha256": sha256_file(declaration_path)},
        "driver_sha256": sha256_file(Path(__file__)),
        "shared_fitter_sha256": sha256_file(Path(fast.__file__)),
        "reclosed_artifact_digests": len(evidence), "engines": outcomes,
        "raw_repeat_windows": {engine: report["rows"] for engine, (report, _) in reports.items()
                               if engine != "spike"},
        "production_enabled": False, "limits": declaration["limits"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet, args.declaration)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("MATCHED_B_REUSE_SCREEN_COMPLETE", list(result["engines"]))
