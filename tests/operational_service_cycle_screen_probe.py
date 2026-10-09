"""Diagnostic gather/primitive screens; all fitting remains in shared Merlin.

Hypotheses and partitions are explicit experiment inputs. Counter windows price
operational streams, including CPU issue and completion, not pure devices.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast

from mlir_oot import service_battery as provider
from mlir_oot.executed_features import _symbol_ranges
from mlir_oot.no_fsm_audit import audit_elf


def reclose(packet):
    if packet["schema"] != "operational_service_battery_qualified_v1":
        raise ValueError("expected qualified operational service packet")
    pins = {}
    for ref in packet["flat_artifact_pins"]:
        path = Path(ref["path"])
        if pins.get(str(path), ref["sha256"]) != ref["sha256"]:
            raise ValueError("conflicting artifact digests")
        if path.stat().st_size != ref["bytes"] or sha256_file(path) != ref["sha256"]:
            raise ValueError("qualified artifact changed: " + str(path))
        pins[str(path)] = ref["sha256"]
    if not pins or set(packet["measurement_refs"]) != set(packet["observations"]):
        raise ValueError("complete artifact and per-engine references required")
    elf = Path(packet["built"]["elf"])
    if pins.get(str(elf)) != packet["built"]["elf_sha256"]:
        raise ValueError("ELF is outside the qualified closure")
    audit = audit_elf(elf.read_bytes())
    if audit != packet["nofsm_audit"] or audit["status"] != "pass":
        raise ValueError("final executable policy changed")
    ranges = _symbol_ranges(elf, list(packet["function_sizes"]), "readelf")
    if {r["symbol"]: r["end"] - r["start"] for r in ranges} != packet["function_sizes"]:
        raise ValueError("actual emitted function extents differ")
    reports = {}
    for engine, raw in packet["observations"].items():
        refs = packet["measurement_refs"][engine]
        for ref in refs.values():
            if pins.get(ref["path"]) != ref["sha256"]:
                raise ValueError("engine evidence is outside the artifact closure")
        if refs["elf"]["sha256"] != packet["built"]["elf_sha256"]:
            raise ValueError("engine observed a different executable")
        if sha256_file(Path(provider.__file__)) != refs["parser"]["sha256"]:
            raise ValueError("active parser differs from the released protocol")
        manifest = json.loads(Path(refs["manifest"]["path"]).read_text())
        if manifest != packet["manifest"]:
            raise ValueError("source geometry or heldout partition changed")
        source = Path(refs["manifest"]["path"]).with_name("battery.c")
        if pins.get(str(source)) != sha256_file(source):
            raise ValueError("released fixture source is outside the artifact closure")
        console = Path(refs["stdout"]["path"]).read_text().replace("\r", "")
        parsed = provider.parse_service_report(console, manifest)
        if parsed != raw["report"]:
            raise ValueError("embedded report differs from actual measured console")
        observed_engine = raw["features"][0]["engine"]
        expected_kind = {"spike": "spike_functional", "gsim": "gsim_rtl", "firesim": "firesim_stock"}
        if observed_engine["kind"] != expected_kind[engine]:
            raise ValueError("engine counter interpretation differs")
        if engine == "firesim":
            actual, record = provider.import_stock_report(
                Path(refs["qualification"]["path"]), manifest,
                elf=elf, parser_path=Path(refs["parser"]["path"]),
            )
            if actual != parsed or record["job_id"] != observed_engine["job_id"]:
                raise ValueError("stock measurement is not the staged terminal job")
            for key in ("hw_config", "hwdb_sha256", "bitstream_sha256"):
                if observed_engine[key] != record[key]:
                    raise ValueError("stock engine domain differs from staged hardware")
        if engine == "gsim":
            record = json.loads(Path(refs["qualification"]["path"]).read_text())
            if record["returncode"] or not record["finish"]["done"]:
                raise ValueError("partial RTL attempt cannot supply timing labels")
            if record["stdout_sha256"] != refs["stdout"]["sha256"]:
                raise ValueError("RTL receipt differs from measured console")
            if observed_engine["provenance"] != record["engine"] or observed_engine["load_path"] != record["load_path"]:
                raise ValueError("RTL engine domain differs from its execution receipt")
        reports[engine] = (parsed, observed_engine)
    reference = [r["instructions"] for r in reports["spike"][0]["rows"]]
    if any([r["instructions"] for r in report["rows"]] != reference
           for report, _ in reports.values()):
        raise ValueError("engines executed different window instruction streams")
    return reports, tuple(sorted(set(pins.values())))


def run(packet_path, hypotheses_path):
    packet = json.loads(packet_path.read_text())
    declared = json.loads(hypotheses_path.read_text())
    if declared["schema"] != "operational_service_screen_hypotheses_v1":
        raise ValueError("expected explicit operational screen declaration")
    reports, evidence = reclose(packet)
    manifest = packet["manifest"]
    engines = {}
    for engine, (report, target) in reports.items():
        if engine == "spike":
            continue
        samples = defaultdict(list)
        for row in report["rows"]:
            if row["id"] >= 0:
                samples[row["id"]].append(row)
        families = defaultdict(list)
        for ident, rows in samples.items():
            if len(rows) != 2 or {r["repeat"] for r in rows} != {0, 1}:
                raise ValueError("two complete original repeats required")
            case = manifest["cases"][ident]
            extent = case.get("working_set_bytes", 0)
            features = {
                "retired_instructions": rows[0]["instructions"],
                "requested_cpu_bytes": case.get("requested_cpu_load_bytes", 0)
                                       + case.get("requested_cpu_store_bytes", 0),
                "requested_64B_region_bytes": case.get("unique_requested_64B_regions", 0) * 64,
                "memory_working_set_bytes": extent,
                "log2_working_set_bytes": math.log2(extent) if extent else 0,
                "array_work_padded_rows": case.get("array_work_padded_rows", 0),
                "requested_dma_load_bytes": case.get("roi_requested_dma_load_bytes", 0),
                "requested_dma_store_bytes": case.get("roi_requested_dma_store_bytes", 0),
            }
            if rows[0]["instructions"] != rows[1]["instructions"]:
                raise ValueError("repeats execute different instruction streams")
            domain = canonical_sha256({
                "engine": target, "family": case["family"],
                "manifest_sha256": packet["measurement_refs"][engine]["manifest"]["sha256"],
                "scope": "same complete fenced kernel-call windows",
            })
            families[case["family"]].append((case, rows, features, domain))
        outcomes = {}
        for family, cases in families.items():
            alternatives = {}
            for names in declared["hypotheses"][family]:
                pointers = tuple("/features/" + name for name in names)
                observations = [(case["partition"], fast.Observation(
                    packet["built"]["elf_sha256"], canonical_sha256([family, case["id"]]),
                    case["partition"], domain,
                    {"/features/" + name: values[name] for name in names},
                    statistics.mean(r["cycles"] for r in rows), evidence,
                )) for case, rows, values, domain in cases]
                train = [row for part, row in observations if part == "training"]
                held = [row for part, row in observations if part == "heldout"]
                try:
                    model = fast.fit_linear_screen(
                        train, pointers=pointers, include_fixed=declared["include_fixed"],
                        maximum_condition=declared["maximum_condition"],
                    )
                    comparisons = []
                    for row in held:
                        predicted = model.predict(row.features, domain_sha256=row.domain)
                        comparisons.append({
                            "workload": row.workload, "measured_cycles": row.cycles,
                            "prediction": predicted.to_dict(),
                            "relative_error": abs(predicted.lo - row.cycles) / row.cycles
                                              if predicted.resolved else None,
                        })
                    alternatives["+".join(names)] = {
                        "coefficients": model.coefficients, "fit_sha256": model.provenance_sha256,
                        "heldout_comparisons": comparisons,
                        "prediction_coverage": {"resolved": sum(r["relative_error"] is not None for r in comparisons),
                                                "required": len(held)},
                    }
                except ValueError as error:
                    alternatives["+".join(names)] = {"refused": str(error)}
            outcomes[family] = alternatives
        engines[engine] = outcomes
    return {
        "schema": "operational_service_screen_diagnostic_v1",
        "packet": {"path": str(packet_path), "sha256": sha256_file(packet_path)},
        "declaration": {"path": str(hypotheses_path), "sha256": sha256_file(hypotheses_path)},
        "driver_sha256": sha256_file(Path(__file__)),
        "shared_fitter_sha256": sha256_file(Path(fast.__file__)),
        "reclosed_artifact_digests": len(evidence), "engines": engines,
        "raw_repeat_cycles": {name: report["rows"] for name, (report, _) in reports.items() if name != "spike"},
        "reference_labels_fitted": False, "production_enabled": False,
        "limitations": declared["limitations"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet, args.hypotheses)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("OPERATIONAL_SERVICE_SCREEN_COMPLETE", list(result["engines"]))
