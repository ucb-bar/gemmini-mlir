"""Root close the observed2057 fit and release one frozen new transfer test."""

import argparse
import json
import shutil
from pathlib import Path

from cpu_service_cycle_screen_probe import verify_packet
from current_host_quant_fixture_probe import pin


def save(path, record):
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")


def archive(args):
    root = args.root.resolve()
    docs = root / "docs/perf_records"
    old_dir = root / "out/artifacts/probes/model-terminals-20261006"
    new_dir = root / "out/artifacts/probes/source-host-quant-transfer-20261007"
    packet_path = old_dir / "root_quant_stock2057_envelope.json"
    score_path = old_dir / "root_quant_stock2057_score.json"
    old, score = [json.loads(p.read_text()) for p in (packet_path, score_path)]
    verify_packet(old)
    if pin(packet_path) != score["packet"]:
        raise ValueError("training score envelope changed")
    for key in ("driver", "verification_driver", "shared_fitter", "shared_ranking"):
        if pin(score[key]["path"]) != score[key]:
            raise ValueError("training/scoring driver changed")
    outcomes = score["engines"]["firesim"]["outcomes"]
    summary = {}
    for name, outcome in outcomes.items():
        if outcome["held_labels_fitted"]:
            raise ValueError("held labels were used in fitting")
        if name == "fixed_plus_work":
            if outcome["refused"] != "fit needs at least two distinct points per parameter":
                raise ValueError("original fit gate changed")
            summary[name] = {"refused":outcome["refused"]}
            continue
        if outcome["current_pair_rank"]["agreed"] != 1 or outcome["broad_search_approval"]["exposable"]:
            raise ValueError("current pair or search verdict differs")
        summary[name] = {k:outcome[k] for k in ("coefficients", "fit_sha256", "maximum_held_relative_error", "current_pair_rank", "broad_search_approval")}
    save(docs / "root_source_host_quant_stock2057_model_review_20261007.json", {
        "schema":"root_source_host_quant_stock2057_model_review_v1", "status":"OBSERVED_CURRENT_HELD_PAIR",
        "job_id":2057, "old_envelope_pins_reclosed":len(old["flat_artifact_pins"]),
        "original_training_cases":[0,1,4,5], "held_current_cases":[2,3], "held_labels_fitted":False,
        "models":summary, "point_estimates_are_not_uncertainty_bounds":True,
        "source_feature_identifiability":"Original six cases have inverseFPdistance=3*sourceLaneGroups; branch/loop/FP causes cannot be separately identified.",
        "scope":"Two repeats per case, complete ranked map including descriptors/allocations/copy. Two fitted hypotheses correctly rank current8vs4 with maximum held error3.73%. One pair/one slice does not approve broad ranking or whole-model predictions.",
        "pins":{str(p):pin(p)["sha256"] for p in (packet_path,score_path,root / "docs/perf_records/root_source_host_quant_crossed_model_stock_release_20261007.json",Path(__file__))},
    })
    shutil.copy2(score_path, docs / "source_host_quant_stock2057_training_only_score.json")
    new_packet_path = new_dir / "qualified_spike.json"
    forecast_path = new_dir / "frozen_forecast.json"
    new, forecast = [json.loads(p.read_text()) for p in (new_packet_path, forecast_path)]
    verify_packet(new)
    if forecast["new_packet"] != pin(new_packet_path) or forecast["training_score"] != pin(score_path) or forecast["training_packet"] != pin(packet_path):
        raise ValueError("new immutable forecast training/source packet changed")
    if forecast["status"] != "PRELABEL_FROZEN" or forecast["new_source_native_bytes_freshly_checked"] != 978432 or forecast["automatic_export"]["exposable"]:
        raise ValueError("new numerical/source/domain qualification differs")
    for ref in forecast["drivers"].values():
        if pin(ref["path"]) != ref:
            raise ValueError("frozen transfer driver changed")
    strict_path = new_dir / "strict_run.json"
    strict = json.loads(strict_path.read_text())
    if strict["exit_code"] != 0 or new["observations"]["spike"]["report"]["checksum"] != "63371034b267c555":
        raise ValueError("strict original-source run failed")
    if set(new["observations"]) != {"spike"} or any(c["partition"] != "heldout" for c in new["manifest"]["cases"]):
        raise ValueError("new labels leaked into fitting")
    engine = old["observations"]["firesim"]["features"][0]["engine"]
    save(docs / "root_source_host_quant_transfer_stock_release_20261007.json", {
        "schema":"root_source_host_quant_frozen_transfer_stock_release_v1", "status":"RELEASED_ONE_STOCK_CALIBRATION",
        "candidate":forecast["elf"], "qualified_packet":pin(new_packet_path), "frozen_forecast":pin(forecast_path),
        "source_native_bytes_freshly_checked":978432, "source_features_freshly_replayed":6,
        "new_artifact_pins_reclosed":len(new["flat_artifact_pins"]), "strict_run":strict,
        "cases":"168/224/336extents x five/seven independent lanes; all six cases and both repeats held. Five lanes include exact four-element remainder per packet row.",
        "counter_contract":{"rows":15,"fp_rows":0,"checksum":"63371034b267c555"},
        "hardware_config":"alveo_u250_firesim_gemmini_rocket_stock",
        "hardware_identity_required":{k:engine[k] for k in ("hw_config","hwdb_sha256","bitstream_sha256")},
        "new_labels_fitted":False,
        "requirements":["Exact actual staged ELF/config/stock bitstream and runworkload receipt.","Import same15 windows with original numeric checksum/flags/FRM/retirement; retain all repeats.","Score unchanged frozen2057 parameters; original domain refusal and all new unpriced tails/stack/memory remain.","Any postlabel model update is a separate new hypothesis, cannot replace the frozen forecast."],
        "automatic_export":forecast["automatic_export"],
        "pins":{str(p):pin(p)["sha256"] for p in (new_packet_path,forecast_path,strict_path,Path(new["built"]["elf"]),new_dir / "manifest.json",new_dir / "spike.stdout",new_dir / "spike.stderr",Path(__file__))},
    })
    shutil.copy2(forecast_path, docs / "source_host_quant_frozen_transfer_forecast_20261007.json")
    print(json.dumps({"status":"PASS", "stock2057_max_held_error":max(v.get("maximum_held_relative_error",0) for v in summary.values()), "new_transfer_one_stock_released":forecast["elf"]["sha256"], "new_labels_fitted":False}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    archive(parser.parse_args())
