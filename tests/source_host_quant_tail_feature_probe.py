"""Diagnose declared group features without touching forecasts or fitting labels."""

import argparse
import json
import math
from pathlib import Path

from current_host_quant_fixture_probe import pin


def run(args):
    if args.output.exists():
        raise ValueError("diagnostic requires a fresh output")
    rows=[]
    for packet_path in args.packet:
        packet=json.loads(packet_path.read_text())
        for ref in packet["flat_artifact_pins"]:
            if pin(ref["path"])!=ref:
                raise ValueError("source/feature input changed")
        for case in packet["manifest"]["cases"]:
            route=case["implementation"]["routes"][0]
            shape=route["shape"]
            axis, width=route["packet_axis"],route["lanes"]
            if any(type(n) is not int or n<=0 for n in shape) or type(axis) is not int or not 0<=axis<len(shape) or type(width) is not int or width<=0:
                raise ValueError("static schedule domain refused")
            invocation_count=packet["source_features"][str(case["id"])]["aggregate_invocations"]
            size=math.prod(shape)
            prefix=size//shape[axis]*invocation_count
            full=prefix*(shape[axis]//width)
            tail=prefix*bool(shape[axis]%width)
            logical=full+tail
            features=packet["source_features"][str(case["id"])]["features"]
            if size!=case["size"] or route["tail"]!=shape[axis]%width or features["source_elements"]!=size*invocation_count or features["source_lane_groups"]!=(size//width)*invocation_count:
                raise ValueError("original static work/feature definition differs")
            distance=features["source_fp_inverse_distance"]
            if distance!=3*logical:
                raise ValueError("reported ordered FP/group relation differs")
            rows.append({"packet":pin(packet_path),"case_id":case["id"],"extent":case["extent"],"lanes":width,"shape":shape,"axis":axis,"invocations":invocation_count,"full_logical_packets":full,"tail_logical_packets":tail,"all_logical_packets":logical,"tail_lane_operations":tail*route["tail"],"original_coarse_element_division_groups":features["source_lane_groups"],"original_fp_inverse_distance":distance,"coarse_group_difference":logical-features["source_lane_groups"],"FP_distance_equals_three_times_true_logical_groups":True})
    result={"schema":"source_host_quant_logical_tail_group_diagnostic_v1","status":"PASS","timing_labels_read":False,"coefficients_fitted":False,"original_features_forecasts_modified":False,"scope":"Postlabel diagnostic of existing source-bound schedules, not a new prospective predictor. Logical static packets do not claim dynamic semantic calls after LLVM inlining. Original coarse feature stays frozen.","identifiability":"Five-lane tails break distance=3*floor(totalElements/lanes), but ordered FP distance remains exactly3*actual logical schedule packets on all12 cases. This does not distinguish FP dependency cost from packet/branch overhead; separate causal prices are still unidentified.","next_experiment":"Hold source elements/full-tail packet geometry/opcode counts fixed while varying emitted intra-packet consumer order/spacing under source numeric/effect constraints; freeze hypothesis and source/target proof before timing.","rows":rows,"driver":pin(__file__)}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"PASS","cases":len(rows),"tails":[r["case_id"] for r in rows if r["tail_logical_packets"]],"causes_still_confounded":True,"labels_fitted":False}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
