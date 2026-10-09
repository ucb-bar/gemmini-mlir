"""Predeclared operational CPU stream screens using Merlin's shared fitter.

The provider battery binds the ISA/fixture/counts. This diagnostic fits small
and large cases only, predicting the predeclared middle sizes. Dependency
counts here describe these emitted streams; they are not arbitrary source DAG
analysis or physical FP service rates. Numerical-alternative ranking and whole
program prediction remain unqualified.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast

from mlir_oot import service_battery as provider
from mlir_oot.executed_features import _symbol_ranges
from mlir_oot.no_fsm_audit import audit_elf


def verify_packet(packet):
    """Reclose emitted-code and measured-window evidence before fitting labels."""
    if packet["schema"] != "rv64gc_service_battery_qualified_v1":
        raise ValueError("unexpected qualified provider schema")
    refs = packet.get("flat_artifact_pins")
    measurements = packet.get("measurement_refs")
    if not refs or not measurements:
        raise ValueError("measurement references and complete artifact pins are required")
    pins = {}
    for ref in refs:
        path, digest = ref["path"], ref["sha256"]
        if path in pins and pins[path] != digest:
            raise ValueError("artifact has conflicting expected digests")
        if sha256_file(Path(path)) != digest:
            raise ValueError("qualified artifact changed: " + path)
        pins[path] = digest
    for ref in packet["built"]["sources"]:
        if pins.get(ref["path"]) != ref["sha256"]:
            raise ValueError("source is outside the qualified artifact closure")
    elf = Path(packet["built"]["elf"])
    if pins.get(str(elf)) != packet["built"]["elf_sha256"]:
        raise ValueError("executable is outside the qualified artifact closure")
    if audit_elf(elf.read_bytes()) != packet["nofsm_audit"] or packet["nofsm_audit"]["status"] != "pass":
        raise ValueError("final executable audit differs")
    ranges = _symbol_ranges(elf, list(packet["function_sizes"]), "readelf")
    if {r["symbol"]: r["end"] - r["start"] for r in ranges} != packet["function_sizes"]:
        raise ValueError("function footprints differ from the actual executable")
    if set(measurements) != set(packet["observations"]):
        raise ValueError("each engine requires its own measurement references")
    for engine_name, raw in packet["observations"].items():
        bindings = measurements[engine_name]
        for ref in bindings.values():
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("measurement evidence changed")
        if sha256_file(Path(provider.__file__)) != bindings["parser"]["sha256"]:
            raise ValueError("active provider differs from the frozen measurement parser")
        manifest = json.loads(Path(bindings["manifest"]["path"]).read_text())
        if manifest != packet["manifest"]:
            raise ValueError("embedded manifest changed the predeclared source or partition")
        console = Path(bindings["stdout"]["path"]).read_text().replace("\r", "")
        report = provider.parse_service_report(console, manifest)
        if report != raw["report"]:
            raise ValueError("embedded observations differ from the measured console")
        if bindings["elf"]["sha256"] != packet["built"]["elf_sha256"]:
            raise ValueError("engine observed another executable")
        first = raw["features"][0]
        provenance = first["provenance"]
        if any(provenance[k] != bindings[b]["sha256"] for k, b in (
            ("elf_sha256", "elf"), ("stdout_sha256", "stdout"), ("manifest_sha256", "manifest"),
        )):
            raise ValueError("feature provenance differs from the measurement bindings")
        expected = provider.feature_rows(
            report, manifest, engine=first["engine"], provenance=provenance,
            function_sizes=packet["function_sizes"],
        )
        if expected != raw["features"]:
            raise ValueError("feature values or counter units differ from the qualified provider")
        if (engine_name == "spike") != (first["engine"]["kind"] == "spike_functional"):
            raise ValueError("functional engine cannot supply hardware cycle labels")
    return tuple(sorted(set(pins.values()) | {
        ref["sha256"] for bindings in measurements.values() for ref in bindings.values()
    }))


def observations(packet, engine_name, family, names, evidence):
    raw = packet["observations"][engine_name]
    buckets = defaultdict(list)
    for row in raw["features"]:
        if row["family"] == family:
            buckets[row["id"]].append(row)
    domain = canonical_sha256({
        "engine": next(iter(buckets.values()))[0]["engine"],
        "family": family,
        "timer_scope": "fenced common kernel call, setup/validation/UART excluded",
        "memory_policy": packet["manifest"]["memory_regime"],
    })
    result = []
    for ident, samples in sorted(buckets.items()):
        if {s["repeat"] for s in samples} != {0, 1} or len(samples) != 2:
            raise ValueError("each case requires its two original repeats")
        first, second = samples
        if first["features"] != second["features"] or first["engine"] != second["engine"]:
            raise ValueError("repeats disagree on executable stream features/domain")
        values = first["features"]
        case = packet["manifest"]["cases"][ident]
        if any(s["partition"] != case["partition"] for s in samples):
            raise ValueError("measurement changed the predeclared partition")
        op_count = values["fp_div_ops"] + values["fp_fma_ops"]
        derived = {
            "retired_instructions": values["retired_instructions"],
            "issued_fp_ops": op_count,
            "stream_dependent_fp_ops": op_count / values["independent_lanes"] if op_count else 0,
            "requested_cpu_bytes": values["requested_cpu_load_bytes"] + values["requested_cpu_store_bytes"],
            "memory_extent_bytes": values["memory_extent_bytes"],
        }
        result.append((case["partition"], fast.Observation(
            packet["built"]["elf_sha256"], canonical_sha256([family, ident]),
            case["partition"], domain,
            {"/features/" + name: derived[name] for name in names},
            statistics.mean(s["metric"]["mcycle_delta"] for s in samples), evidence,
        )))
    return result, domain


def run(packet_path):
    packet = json.loads(packet_path.read_text())
    proof = verify_packet(packet)
    modes = (
        ("retired_instructions",),
        ("issued_fp_ops", "stream_dependent_fp_ops"),
        ("retired_instructions", "stream_dependent_fp_ops"),
    )
    engines = {}
    for engine_name in packet["observations"]:
        if engine_name == "spike":
            continue  # Its mcycle is retirement, not a timing label.
        families = {}
        for family in ("div", "fma", "memory"):
            candidates = modes if family != "memory" else (
                ("requested_cpu_bytes",), ("requested_cpu_bytes", "memory_extent_bytes"),
            )
            outcomes = {}
            for names in candidates:
                rows, domain = observations(packet, engine_name, family, names, proof)
                train = [row for part, row in rows if part == "training"]
                heldout = [row for part, row in rows if part == "heldout"]
                try:
                    model = fast.fit_linear_screen(
                        train, pointers=tuple("/features/" + name for name in names),
                        include_fixed=False, maximum_condition=1000,
                    )
                    comparisons = []
                    for row in heldout:
                        predicted = model.predict(row.features, domain_sha256=domain)
                        comparisons.append({
                            "workload": row.workload, "measured_cycles": row.cycles,
                            "prediction": predicted.to_dict(),
                            "relative_error": abs(predicted.lo - row.cycles) / row.cycles if predicted.resolved else None,
                        })
                    outcomes["+".join(names)] = {
                        "coefficients": model.coefficients, "fit_sha256": model.provenance_sha256,
                        "training_workloads": [row.workload for row in train],
                        "heldout_comparisons": comparisons,
                        "prediction_coverage": {
                            "resolved": sum(r["relative_error"] is not None for r in comparisons),
                            "required": len(heldout),
                        },
                        "maximum_relative_error": max((r["relative_error"] for r in comparisons if r["relative_error"] is not None), default=None),
                    }
                except ValueError as error:
                    outcomes["+".join(names)] = {"refused": str(error)}
            families[family] = outcomes
        engines[engine_name] = families
    return {
        "schema": "operational_cpu_stream_screen_diagnostic_v1",
        "packet": {"path": str(packet_path), "sha256": sha256_file(packet_path)},
        "driver_sha256": sha256_file(Path(__file__)),
        "shared_fitter_sha256": sha256_file(Path(fast.__file__)),
        "independently_reclosed_artifact_digest_count": len(proof),
        "repeat_cycle_ranges": {
            name: {
                str(ident): [min(r["metric"]["mcycle_delta"] for r in raw["features"] if r["id"] == ident),
                             max(r["metric"]["mcycle_delta"] for r in raw["features"] if r["id"] == ident)]
                for ident in sorted({r["id"] for r in raw["features"]})
            }
            for name, raw in packet["observations"].items() if name != "spike"
        },
        "partition": "Middle sizes and both lane/stride arms held out; repeated measurements aggregated before fit",
        "engines": engines, "reference_labels_fitted": False, "production_enabled": False,
        "limitations": [
            "FP stream shapes are different computations, not numerically equivalent program alternatives",
            "No whole program or arbitrary DAG prediction licensed",
            "Coefficient values are operational stream screens, not pure FP latency, physical traffic or cache miss measurements",
            "Two repetitions describe observed variability only",
            "Footprint-only cases are evidence, not a separately identifiable instruction cache service",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("CPU_STREAM_SCREEN_COMPLETE", list(result["engines"]))
