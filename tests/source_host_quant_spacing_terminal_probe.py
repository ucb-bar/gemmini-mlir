"""Close fixed-packet stock spacing labels, frozen transfers and train-only fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpu_service_cycle_screen_probe import verify_packet
from current_host_quant_fixture_probe import pin
from mlir_oot.no_fsm_audit import audit_elf
from source_host_quant_model_cycle_screen_probe import collect_rows, score_rows


def close(args):
    if args.output.exists():
        raise ValueError("fresh review required")
    packet, release, terminal = [json.loads(p.read_text()) for p in (args.packet,args.release,args.terminal)]
    evidence=verify_packet(packet)
    if packet["original_prospective_packet"] != release["qualified_packet"]:
        raise ValueError("prospective packet changed")
    original=json.loads(Path(release["qualified_packet"]["path"]).read_text())
    if packet["manifest"] != original["manifest"] or packet["source_features"] != original["source_features"]:
        raise ValueError("source/partition/feature changed during import")
    for p,h in terminal["pins"].items():
        if pin(p)["sha256"] != h:
            raise ValueError("terminal evidence changed: "+p)
    v=terminal["record"]
    raw=packet["observations"]["firesim"]
    if v["job_id"] != 2063 or v["state"] != "DONE" or v["phase"] != "DONE" or v["exit_code"] != 0:
        raise ValueError("stock job not successful")
    if raw["report"] != v["report"] or v["elf_sha256"] != release["candidate"]["sha256"]:
        raise ValueError("actual stock counters/ELF differ")
    engine=raw["features"][0]["engine"]
    if any(engine[k] != value for k,value in release["hardware_identity_required"].items()):
        raise ValueError("stock hardware differs from original model")
    elf=Path(v["elf"])
    if pin(elf)["sha256"] != v["elf_sha256"]:
        raise ValueError("final ELF changed")
    audit=audit_elf(elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("executable noFSM audit failed")
    outcomes=[]
    for frozen in release["original_frozen_transfer_forecasts"]:
        comparisons=[]
        for c,predicted in zip(packet["manifest"]["cases"],frozen["predictions"],strict=True):
            samples=[s for s in raw["features"] if s["id"]==c["id"]]
            if len(samples)!=2 or {s["repeat"] for s in samples}!={0,1} or c["id"]!=predicted["id"]:
                raise ValueError("held transfer repeat/order differs")
            values=packet["source_features"][str(c["id"])]["features"]|{"retired_instructions":sum(s["metric"]["minstret_delta"] for s in samples)}
            actual_features={k:values[k.removeprefix("/features/")] for k in predicted["features"]}
            if actual_features != predicted["features"] or predicted["original_domain_prediction"]["resolved"]:
                raise ValueError("prelabel features/refusal changed")
            expected=sum(a*b for a,b in zip(frozen["coefficients"],actual_features.values(),strict=True))
            if expected != predicted["conditional_cycles_two_windows"]:
                raise ValueError("immutable forecast arithmetic differs")
            cycles=[s["metric"]["mcycle_delta"] for s in samples]
            measured=sum(cycles)
            comparisons.append({"id":c["id"],"extent":c["extent"],"spacing_group":c["consumer_spacing_group"],"raw_cycles":cycles,"measured_cycles_two_windows":measured,"conditional_prediction_two_windows":expected,"relative_error":abs(expected-measured)/measured})
        pairs=[]
        for a,b in zip(comparisons[::2],comparisons[1::2],strict=True):
            measured=1-b["measured_cycles_two_windows"]/a["measured_cycles_two_windows"]
            forecasted=1-b["conditional_prediction_two_windows"]/a["conditional_prediction_two_windows"]
            pairs.append({"extent":a["extent"],"actual_group8_lower_fraction":measured,"frozen_group8_lower_fraction":forecasted,"fraction_error":forecasted-measured,"ordering_correct":(forecasted>0)==(measured>0)})
        outcomes.append({"name":frozen["name"],"fit_sha256_unchanged":frozen["fit_sha256"],"all_six_new_labels_held":True,"new_labels_fitted":False,"maximum_relative_error":max(c["relative_error"] for c in comparisons),"comparisons":comparisons,"pairs":pairs})
    fits={}
    for hypothesis in release["new_calibration_hypotheses"]:
        rows,domain,_=collect_rows(packet,"firesim",hypothesis["features"],evidence)
        if [i for i,_,r in rows if r.group=="training"] != release["new_training_ids"] or [i for i,_,r in rows if r.group=="heldout"] != release["new_held_ids"]:
            raise ValueError("crossed train/held partition changed")
        fits[hypothesis["name"]]=score_rows(rows,domain,hypothesis)
    result={"schema":"root_source_host_quant_spacing_terminal_v1","status":"STOCK_SPACING_CALIBRATION_OBSERVED","job_id":2063,"source_original_bytes_checked":1053696,"windows":15,"checksum":raw["report"]["checksum"],"packet_pins_reclosed":len(packet["flat_artifact_pins"]),"terminal_pins_reclosed":len(terminal["pins"]),"stock_hardware":release["hardware_identity_required"],"noFSM":audit,"old_models_unchanged":outcomes,"separate_new_training_only_models":fits,"new_training_ids":release["new_training_ids"],"new_held_ids":release["new_held_ids"],"held_labels_fitted":False,"automatic_export":False,"limitations":["One typed numeric source family, one stock hardware and memory regime; not whole-model qualification.","One held spacing pair and source slice cannot qualify broad candidate ranking.","Nonlinear extent effects, linked footprint, GPR dependence and physical cache service remain unpriced.","Point predictions have no calibrated uncertainty; retain original executable-domain refusals."],"pins":{str(p):pin(p)["sha256"] for p in (args.packet,args.release,args.terminal,elf,Path(__file__))}}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"old_max_errors":{o["name"]:o["maximum_relative_error"] for o in outcomes},"new_held_max_errors":{k:f.get("maximum_held_relative_error") for k,f in fits.items()},"actual_fractions":[p["actual_group8_lower_fraction"] for p in outcomes[0]["pairs"]],"held_labels_fitted":False,"automatic_export":False}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet","release","terminal","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    close(parser.parse_args())
