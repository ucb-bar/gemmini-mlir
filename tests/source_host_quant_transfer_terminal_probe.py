"""Score original frozen transfer predictions; no fitting reads new labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpu_service_cycle_screen_probe import verify_packet
from current_host_quant_fixture_probe import pin
from merlin.common.jsonio import canonical_sha256
from merlin.perf import rank_validation as rank


def close(args):
    if args.output.exists():
        raise ValueError("terminal review requires a fresh output")
    packet, forecast, release = [json.loads(p.read_text()) for p in (args.packet, args.forecast, args.release)]
    verify_packet(packet)
    if pin(args.forecast) != release["frozen_forecast"] or packet["original_prospective_packet"] != forecast["new_packet"]:
        raise ValueError("immutable prelabel forecast/prospective envelope changed")
    if packet["source_features"] != json.loads(Path(forecast["new_packet"]["path"]).read_text())["source_features"]:
        raise ValueError("stock import changed source features")
    raw = packet["observations"]["firesim"]
    engine = raw["features"][0]["engine"]
    if engine["job_id"] != args.job_id or any(engine[k] != v for k,v in release["hardware_identity_required"].items()):
        raise ValueError("released job/stock training hardware identity differs")
    outcomes = []
    for frozen in forecast["forecasts"]:
        comparisons, programs, intervals, slices = [], [], {}, {}
        for case, prediction in zip(packet["manifest"]["cases"], frozen["new_held_cases"], strict=True):
            if case["partition"] != "heldout" or case["id"] != prediction["id"]:
                raise ValueError("new cases changed partition or order")
            samples = [s for s in raw["features"] if s["id"] == case["id"]]
            if len(samples) != 2 or {s["repeat"] for s in samples} != {0,1}:
                raise ValueError("actual hardware repeat coverage differs")
            values = packet["source_features"][str(case["id"])]["features"] | {"retired_instructions":sum(s["metric"]["minstret_delta"] for s in samples)}
            observed_features = {"/features/"+name:values[name] for name in frozen["hypothesis"]["features"]}
            if observed_features != prediction["features"]:
                raise ValueError("actual stock retirement/source feature differs from forecast")
            if prediction["original_domain_prediction"]["resolved"]:
                raise ValueError("original domain refusal was removed")
            guessed = prediction["conditional_signature_transfer_cycles_two_windows"]
            # Reproduce immutable arithmetic only; no fitter sees new labels.
            recomputed = sum(c * observed_features["/features/"+name] for name,c in zip(frozen["hypothesis"]["features"],frozen["frozen_coefficients"],strict=True))
            if guessed != recomputed:
                raise ValueError("prelabel arithmetic changed")
            cycles = [s["metric"]["mcycle_delta"] for s in samples]
            measured = sum(cycles)
            ident = canonical_sha256({"elf":packet["built"]["elf_sha256"],"case_id":case["id"]})
            workload = canonical_sha256({"source":case["scalar_graph_sha256"],"input_shape":case["input_shape"],"output_shape":case["output_shape"]})
            slice_name = "extent_"+str(case["extent"])
            program = rank.Program(workload,ident,measured,slice_name)
            programs.append(program)
            slices.setdefault(slice_name,[]).append(program)
            intervals[ident] = (guessed,guessed)
            comparisons.append({"id":case["id"],"extent":case["extent"],"lanes":case["independent_lanes"],"raw_repeat_cycles":cycles,"mean_cycles":measured/2,"measured_cycles_two_windows":measured,"frozen_conditional_cycles_two_windows":guessed,"relative_error":abs(guessed-measured)/measured,"original_domain_prediction":prediction["original_domain_prediction"]})
        agreement = rank.interval_agreement(rank.ordered_pairs(programs),intervals)
        slice_agreements = {k:rank.interval_agreement(rank.ordered_pairs(v),intervals) for k,v in slices.items()}
        diagnostic_gate = rank.verdict(agreement,slice_agreements,minimum_rate=.8,minimum_decided=2,minimum_slice_decided=1,minimum_slices=2)
        if agreement.decided != 3 or agreement.agreed != 3:
            raise ValueError("reported new ordering result differs")
        outcomes.append({"name":frozen["hypothesis"]["name"],"frozen_fit_sha256":frozen["frozen_fit_sha256"],"coefficients_unchanged":frozen["frozen_coefficients"],"new_labels_fitted":False,"comparisons":comparisons,"maximum_held_relative_error":max(c["relative_error"] for c in comparisons),"prospective_signature_pair_agreement":agreement.to_dict(),"conditional_pair_gate":diagnostic_gate,"pair_gate_scope":"Three new sizes of one source graph/one schedule family, conditional declared signature transfer; this does not remove original executable-domain refusal or qualify whole-model/other-source ranking."})
    result={
        "schema":"root_source_host_quant_frozen_transfer_terminal_v1","status":"OBSERVED_PROSPECTIVE_TRANSFER","job_id":args.job_id,
        "source_original_bytes_exact":978432,"counter_windows":len(raw["report"]["rows"]),"checksum":raw["report"]["checksum"],
        "stock_artifact_pins_reclosed":len(packet["flat_artifact_pins"]),"held_cases":6,"raw_repeats":12,
        "forecasts":outcomes,"new_labels_fitted":False,
        "original_domain_all_new_predictions":"UNKNOWN; immutable refusals retained",
        "automatic_export":{"exposable":False,"reasons":["Original ELF/manifest model domain refuses this transfer.","Good ordering on three sizes of one source/schedule family does not qualify other numeric graphs, memory regimes or whole-model composition.","Point estimates have no calibrated uncertainty or separate tail/stack/cache/overlap prices."]},
        "pins":{str(p):pin(p)["sha256"] for p in (args.packet,args.forecast,args.release,Path(__file__))},
    }
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"job":args.job_id,"max_errors":{x["name"]:x["maximum_held_relative_error"] for x in outcomes},"correct_pairs_each":3,"new_fit":False,"automatic_export":False}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet","forecast","release","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--job-id",type=int,required=True)
    close(parser.parse_args())
