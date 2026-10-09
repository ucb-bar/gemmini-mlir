"""Freeze a wholly held quantizer transfer forecast and its domain refusal."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from cpu_service_cycle_screen_probe import verify_packet
from current_host_quant_fixture_probe import pin
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast
from merlin.xdsl_dialects._common import text
from source_host_quant_model_battery_probe import evaluate, rebind_extent
from source_host_quant_model_cycle_screen_probe import collect_rows
from source_host_quant_model_feature_replay_probe import replay
from source_host_quant_transfer_battery_probe import contents

from mlir_oot.frontend.parse import parse_module


def prepare(args):
    if args.output.exists():
        raise ValueError("prospective forecast requires a fresh output")
    old, new = [json.loads(p.read_text()) for p in (args.training_packet, args.new_packet)]
    old_evidence, _ = verify_packet(old), verify_packet(new)
    if set(old["observations"]) != {"spike", "firesim"} or set(new["observations"]) != {"spike"}:
        raise ValueError("new transfer labels must be absent")
    if any(c["partition"] != "heldout" for c in new["manifest"]["cases"]):
        raise ValueError("every new case must be held")
    if [(c["extent"], c["independent_lanes"]) for c in new["manifest"]["cases"]] != [(h, w) for h in (168, 224, 336) for w in (5, 7)]:
        raise ValueError("prospective crossed cases changed")
    with contextlib.redirect_stdout(io.StringIO()):
        replay(SimpleNamespace(packet=args.new_packet, census_directory=args.census_directory, encoding_header=args.encoding_header, decode_header=args.decode_header, output=None))
    source_binding = new["manifest"]["source_binding"]
    if source_binding != old["manifest"]["source_binding"] or pin(source_binding["path"]) != source_binding:
        raise ValueError("original typed source/constant binding changed")
    fixture = Path(source_binding["path"]).parent
    original = parse_module((fixture / "source.mlir").read_text())
    binding = json.loads(Path(source_binding["path"]).read_text())
    original_input = np.fromfile(fixture / "input.bin", dtype=np.float32).reshape(binding["input_shape"])
    checked = 0
    for case in new["manifest"]["cases"]:
        module, proof = rebind_extent(original, input_axis=2, extent=case["extent"])
        arm = Path(case["original_object"]["path"]).parent
        oracle = arm.parent / "oracle"
        selected = original_input[:, :, np.arange(case["extent"]) % binding["input_shape"][2], :].copy()
        expected = Path(case["source_oracle"]["path"]).read_bytes()
        # Printers may number region-local block labels differently. Compare
        # parsed typed IR with SSA binding equivalence; original bytes stay pinned.
        if proof != case["extent_proof"] or not module.is_structurally_equivalent(parse_module((oracle / "source.mlir").read_text())):
            raise ValueError("typed source extent proof differs")
        if selected.tobytes() != Path(case["input"]["path"]).read_bytes():
            raise ValueError("input row mapping differs")
        if evaluate(oracle, selected, case["output_shape"]).tobytes() != expected or evaluate(arm, selected, case["output_shape"]).tobytes() != expected:
            raise ValueError("complete original/scheduled native outputs differ")
        if contents(case["original_object"]["path"]) != contents(case["renamed_object"]["path"]):
            raise ValueError("symbol namespacing changed executable bytes")
        if case["scalar_graph_sha256"] != binding["source_operation_sha256"]:
            raise ValueError("source scalar graph binding changed")
        route = case["implementation"]["routes"][0]
        if route["packet_axis"] != 2 or route["tail"] != (4 if case["independent_lanes"] == 5 else 0):
            raise ValueError("actual tail/packet axis differs")
        checked += case["size"]
    old_score = json.loads(args.training_score.read_text())
    outcomes = old_score["engines"]["firesim"]["outcomes"]
    forecasts = []
    for hypothesis in old["manifest"]["predeclared_hypotheses"]:
        if hypothesis["name"] == "fixed_plus_work":
            continue  # Its original training feature gate refused; no new fit.
        rows, domain, _ = collect_rows(old, "firesim", hypothesis["features"], old_evidence)
        model = fast.fit_linear_screen([r for _, _, r in rows if r.group == "training"], pointers=tuple("/features/" + n for n in hypothesis["features"]), include_fixed=hypothesis["include_fixed"], maximum_condition=1000)
        frozen = outcomes[hypothesis["name"]]
        if list(model.coefficients) != frozen["coefficients"] or model.provenance_sha256 != frozen["fit_sha256"]:
            raise ValueError("qualified training-only fit does not reproduce frozen coefficients/hash")
        held = []
        for case in new["manifest"]["cases"]:
            source = new["source_features"][str(case["id"])]
            if source["aggregate_invocations"] != 2:
                raise ValueError("source invocation aggregation changed")
            samples = [s for s in new["observations"]["spike"]["features"] if s["id"] == case["id"]]
            if len(samples) != 2 or {s["repeat"] for s in samples} != {0, 1}:
                raise ValueError("strict window coverage differs")
            values = source["features"] | {"retired_instructions": sum(s["metric"]["minstret_delta"] for s in samples)}
            features = {"/features/" + n: values[n] for n in hypothesis["features"]}
            actual_domain = canonical_sha256({"new_elf":new["built"]["elf_sha256"], "new_manifest":pin(args.new_packet)["sha256"], "future_hardware": "UNKNOWN until exact stock import"})
            refusal = model.predict(features, domain_sha256=actual_domain)
            if refusal.resolved:
                raise ValueError("original domain unexpectedly admitted different executable")
            conditional = model.fixed_cycles + sum(c * features[n] for n, c in zip(model.pointers, model.coefficients, strict=True))
            held.append({"id":case["id"], "extent":case["extent"], "lanes":case["independent_lanes"], "features": features, "original_domain_prediction": refusal.to_dict(), "conditional_signature_transfer_cycles_two_windows": conditional, "new_labels": "UNKNOWN"})
        forecasts.append({"hypothesis":hypothesis, "frozen_fit_sha256":model.provenance_sha256, "frozen_coefficients":model.coefficients, "training_ids":[r.id for _, _, r in rows if r.group == "training"], "training_domain":domain, "new_held_cases":held, "conditional_pairs":[{"extent":a["extent"], "predicted_faster_lanes":a["lanes"] if a["conditional_signature_transfer_cycles_two_windows"] < b["conditional_signature_transfer_cycles_two_windows"] else b["lanes"]} for a,b in zip(held[::2],held[1::2],strict=True)]})
    result = {
        "schema":"source_host_quant_frozen_cross_executable_forecast_v1", "status":"PRELABEL_FROZEN",
        "new_source_native_bytes_freshly_checked":checked, "source_features_freshly_replayed":6,
        "new_artifact_pins_reclosed":len(new["flat_artifact_pins"]),
        "old_artifact_pins_reclosed":len(old["flat_artifact_pins"]),
        "old_held_labels_fitted":False, "new_labels_fitted":False,
        "training_score":pin(args.training_score), "training_packet":pin(args.training_packet),
        "new_packet":pin(args.new_packet), "elf":pin(new["built"]["elf"]),
        "stock_config_required":"alveo_u250_firesim_gemmini_rocket_stock",
        "stock_engine_required":old["observations"]["firesim"]["features"][0]["engine"],
        "forecasts":forecasts,
        "automatic_export":{"exposable":False,"reasons":["Original executable domain refuses transfer.","New five-lane per-row tail, stack and linked footprint costs are unpriced.","Conditional signature-transfer arithmetic is prospective diagnostics, not qualified whole/ranking model."]},
        "signature_transfer_scope":"Same source FMUL/min/max/RNE scalar graph, constant producers and transpose layout; extents/lanes interpolate original training sizes/widths. Source-derived remainder helpers and changed linked footprint explicitly exceed original training schedule coverage. All new cases remain held. Original gate untouched.",
        "drivers":{name:pin(path) for name,path in {"forecast":Path(__file__),"battery":Path(__file__).with_name("source_host_quant_transfer_battery_probe.py"),"verifier":Path(__file__).with_name("cpu_service_cycle_screen_probe.py")}.items()},
    }
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"], "new_source_bytes_checked":checked, "new_cases":6, "labels_fitted":False, "conditional_pairs":[f["conditional_pairs"] for f in forecasts], "export":False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-packet", "training-score", "new-packet", "census-directory", "encoding-header", "decode-header", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    prepare(parser.parse_args())
