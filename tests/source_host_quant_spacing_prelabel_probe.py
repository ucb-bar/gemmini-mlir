"""Freeze spacing calibration hypotheses and original-model transfer arithmetic."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from cpu_service_cycle_screen_probe import verify_packet
from current_host_quant_fixture_probe import pin
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast
from source_host_quant_model_cycle_screen_probe import collect_rows
from source_host_quant_model_feature_replay_probe import replay


def freeze(args):
    if args.output.exists():
        raise ValueError("prelabel output must be fresh")
    old,new=[json.loads(p.read_text()) for p in (args.training_packet,args.new_packet)]
    evidence=verify_packet(old)
    verify_packet(new)
    if set(new["observations"]) != {"spike"}:
        raise ValueError("new hardware labels must be absent")
    if old["manifest"]["source_binding"] != new["manifest"]["source_binding"]:
        raise ValueError("original typed scalar source changed")
    with contextlib.redirect_stdout(io.StringIO()):
        replay(SimpleNamespace(packet=args.new_packet,census_directory=args.census_directory,encoding_header=args.encoding_header,decode_header=args.decode_header,output=None))
    cases=new["manifest"]["cases"]
    if [(c["extent"],c["independent_lanes"],c["consumer_spacing_group"],c["partition"]) for c in cases] != [(h,8,s,"heldout" if h==224 else "training") for h in (112,224,448) for s in (1,8)]:
        raise ValueError("frozen crossed calibration partition differs")
    equalities=[]
    for first,second in zip(cases[::2],cases[1::2],strict=True):
        a,b=[new["source_features"][str(c["id"])] for c in (first,second)]
        for key in ("source_elements","source_lane_groups","source_scoped_retired_instructions","source_scoped_conditional_branches","source_scoped_link_register_branches","source_sp_relative_loads","source_sp_relative_stores"):
            if a["features"][key] != b["features"][key]:
                raise ValueError("unexpected compiled confound: "+key)
        if a["features"]["source_fp_inverse_distance"] <= b["features"]["source_fp_inverse_distance"]:
            raise ValueError("emitted spacing feature did not change")
        equalities.append({"extent":first["extent"],"paired_ids":[first["id"],second["id"]],"same_elements_packets_scoped_instructions_branches_calls_stack_requests":True,"FP_inverse_distance":[a["features"]["source_fp_inverse_distance"],b["features"]["source_fp_inverse_distance"]],"instruction_footprint_bytes":[a["features"]["touched_instruction_bytes"],b["features"]["touched_instruction_bytes"]],"physical_cache_traffic":"UNKNOWN"})
    old_score=json.loads(args.training_score.read_text())
    forecasts=[]
    for hypothesis in old["manifest"]["predeclared_hypotheses"]:
        if hypothesis["name"]=="fixed_plus_work":
            continue
        rows,domain,_=collect_rows(old,"firesim",hypothesis["features"],evidence)
        model=fast.fit_linear_screen([r for _,_,r in rows if r.group=="training"],pointers=tuple("/features/"+k for k in hypothesis["features"]),include_fixed=hypothesis["include_fixed"],maximum_condition=1000)
        frozen=old_score["engines"]["firesim"]["outcomes"][hypothesis["name"]]
        if list(model.coefficients)!=frozen["coefficients"] or model.provenance_sha256!=frozen["fit_sha256"]:
            raise ValueError("old training-only fit/hash did not reproduce")
        predictions=[]
        for c in cases:
            observed=new["source_features"][str(c["id"])]
            samples=[s for s in new["observations"]["spike"]["features"] if s["id"]==c["id"]]
            if len(samples)!=2 or {s["repeat"] for s in samples}!={0,1}:
                raise ValueError("strict source repeat coverage differs")
            values=observed["features"]|{"retired_instructions":sum(s["metric"]["minstret_delta"] for s in samples)}
            features={"/features/"+name:values[name] for name in hypothesis["features"]}
            refusal=model.predict(features,domain_sha256=canonical_sha256({"new_elf":new["built"]["elf_sha256"],"manifest":pin(args.new_packet)["sha256"],"future_hardware":"UNKNOWN"}))
            if refusal.resolved:
                raise ValueError("original model silently widened its domain")
            conditional=model.fixed_cycles+sum(v*features[k] for k,v in zip(model.pointers,model.coefficients,strict=True))
            predictions.append({"id":c["id"],"extent":c["extent"],"spacing_group":c["consumer_spacing_group"],"features":features,"original_domain_prediction":refusal.to_dict(),"conditional_cycles_two_windows":conditional,"hardware_label":"UNKNOWN"})
        forecasts.append({"name":hypothesis["name"],"fit_sha256":model.provenance_sha256,"coefficients":model.coefficients,"all_new_cases_held_for_original_transfer":True,"predictions":predictions,"pair_arithmetic":[{"extent":a["extent"],"group1_cycles_two_windows":a["conditional_cycles_two_windows"],"group8_cycles_two_windows":b["conditional_cycles_two_windows"],"conditional_fraction_group8_lower":1-b["conditional_cycles_two_windows"]/a["conditional_cycles_two_windows"]} for a,b in zip(predictions[::2],predictions[1::2],strict=True)]})
    engine=old["observations"]["firesim"]["features"][0]["engine"]
    strict_path=args.new_packet.parent/"strict_run.json"
    strict=json.loads(strict_path.read_text())
    if strict["exit_code"]!=0 or strict["elf_sha256"]!=new["built"]["elf_sha256"]:
        raise ValueError("strict execution failed/bound a different ELF")
    result={"schema":"root_source_host_quant_spacing_prelabel_v1","status":"RELEASED_ONE_STOCK_CAUSAL_CALIBRATION","packet_pins_reclosed":len(new["flat_artifact_pins"]),"candidate":pin(new["built"]["elf"]),"qualified_packet":pin(args.new_packet),"strict_run":strict,"source_bytes_original_native_checked":1053696,"counter_contract":{"rows":15,"fp_rows":0,"checksum":"98e0beb22396e0f5"},"compiled_pair_invariants":equalities,"new_calibration_hypotheses":new["manifest"]["predeclared_hypotheses"],"new_training_ids":[0,1,4,5],"new_held_ids":[2,3],"original_frozen_transfer_forecasts":forecasts,"new_labels_fitted":False,"hardware_alias":"alveo_u250_firesim_gemmini_rocket_stock","hardware_identity_required":{k:engine[k] for k in ("hw_config","hwdb_sha256","bitstream_sha256")},"allowed_runs":1,"scope":"Fixed typed8-lane source packets and same emitted instruction/branch/stack counts, with different pure independent clamp/conversion ordering. Explicit nontrapping/unobserved FP flags, unchanged source default. Calibrate causal source dependence instead of inferring it from changed loop packet counts. Record all original frozen transfer failures; future new-battery fit trains only0/1/4/5, never held2/3.","unknowns":["instruction order changes linked footprint and GPR distance distributions","logical stack requests do not identify cache traffic","no uncertainty or whole composition price"],"automatic_export":False,"pins":{str(p):pin(p)["sha256"] for p in (args.training_packet,args.training_score,args.new_packet,strict_path,Path(__file__),Path(__file__).with_name("source_host_quant_spacing_battery_probe.py"))}}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"compiled_equal_pairs":3,"distinct_spacing":True,"pins":result["packet_pins_reclosed"],"new_labels_fitted":False,"conditional_fractions":{f["name"]:[p["conditional_fraction_group8_lower"] for p in f["pair_arithmetic"]] for f in forecasts}}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("training-packet","training-score","new-packet","census-directory","encoding-header","decode-header","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    freeze(parser.parse_args())
