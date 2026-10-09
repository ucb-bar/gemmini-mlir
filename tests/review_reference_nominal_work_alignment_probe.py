"""Reclose source-function request accounting against owned reference evidence."""

import argparse
import json
import math
from pathlib import Path

from current_host_quant_fixture_probe import pin


def review(args):
    receipt = json.loads(args.receipt.read_text())
    if args.output.exists():
        raise ValueError("independent review requires a fresh output")
    inputs = {}
    for ref in receipt["pins"]:
        if pin(ref["path"]) != ref:
            raise ValueError("bound accounting input changed")
        inputs[Path(ref["path"]).name] = json.loads(Path(ref["path"]).read_text())
    current = inputs["request_functions.json"]
    alignment = inputs["q1013_current2039_stock2046_profile_alignment.json"]
    reference = inputs["q1013_reference_heldout_operand_features.json"]
    catalog = inputs["device_catalog.json"]
    if alignment["control_job"] != 2039 or alignment["reference_job"] != 1876 or receipt["source_numeric_equivalence"]:
        raise ValueError("role identities or numeric-contract limit changed")
    for ref in current["pins"].values():
        observed = pin(ref["path"])
        if not {"path", "sha256"} <= set(ref) <= {"path", "sha256", "bytes"} or any(observed[k] != value for k, value in ref.items()):
            raise ValueError("current census execution/proof changed")
    if not current["whole_original_output_and_pc_histogram_byte_exact"] or not current["every_original_request_geometry_counter_exact"]:
        raise ValueError("original current execution not conserved")
    if receipt["current_final_elf"]["sha256"] != current["pins"]["elf"]["sha256"]:
        raise ValueError("actual current final ELF differs")
    by_symbol = {}
    for group in current["groups"]:
        for symbol in group["symbols"]:
            if symbol in by_symbol:
                raise ValueError("ambiguous current source-function alias")
            by_symbol[symbol] = group
    by_index = {layer["reference_index"]:layer for layer in reference["layers"]}
    declared = {row["reference_index"]:row for row in receipt["paired_geometry_rows"]}
    rebuilt = []
    for row in alignment["paired_layers"]:
        r = by_index[row["reference_index"]]
        c = by_symbol[row["current_symbol"]]["requests"]
        if r["geometry"] != row["geometry"]:
            raise ValueError("source-bound paired geometry changed")
        observed = {"reference_index":row["reference_index"],"reference_function":r["physical_function"],"current_function":row["current_symbol"],"role":row["category"],"geometry":row["geometry"],"current_nominal_rows":c["padded_compute_rows"],"reference_nominal_rows":r["features"]["array_padded_compute_rows"],"difference_nominal_rows":c["padded_compute_rows"]-r["features"]["array_padded_compute_rows"],"current_requested_load_bytes":c["requested_load_bytes"],"reference_requested_load_bytes":r["features"]["requested_dma_load_bytes"],"current_requested_store_bytes":c["requested_store_bytes"],"reference_requested_store_bytes":r["features"]["requested_dma_store_bytes"]}
        if observed != declared[row["reference_index"]]:
            raise ValueError("independent geometry/request join differs")
        rebuilt.append(observed)
    if len(rebuilt) != 54 or set(by_index) != set(declared):
        raise ValueError("paired scope coverage differs")
    residuals = []
    for old, typed in zip(receipt["current_residual_arithmetic"], catalog["residual_additions"], strict=True):
        c = by_symbol[typed["kernel"]]["requests"]
        coefficients = typed["proof"]["coefficients"]
        chunks = math.ceil(coefficients["p"]/127) + math.ceil(coefficients["q"]/127)
        rows = math.ceil(typed["m"]/16)*math.ceil(typed["n"]/16)*16*chunks
        if old["symbol"] != typed["kernel"] or old["coefficients"] != coefficients or old["chunks"] != chunks or old["nominal_rows"] != rows or c["padded_compute_rows"] != rows or old["coefficient_proof_sha256"] != typed["proof_sha256"]:
            raise ValueError("typed source residual/chunk/request proof differs")
        residuals.append({"symbol":typed["kernel"],"chunks":chunks,"nominal_rows":rows})
    current_total = sum(g["requests"]["padded_compute_rows"] for g in current["groups"])
    residual_current = sum(r["nominal_rows"] for r in residuals)
    residual_reference = reference["residual_aggregate_operand_features"]["features"]["array_padded_compute_rows"]
    mean_symbol = catalog["guarded_mean_additions"][0]["kernel"]
    mean_current = by_symbol[mean_symbol]["requests"]["padded_compute_rows"]
    paired_current = sum(r["current_nominal_rows"] for r in rebuilt)
    paired_reference = sum(r["reference_nominal_rows"] for r in rebuilt)
    if current_total != paired_current+residual_current+mean_current or mean_current != 8192:
        raise ValueError("complete current role totals not conserved")
    # Reference complete operand census was independently bound in root's prior
    # projection; use its actual total to check the unmatched mean remainder.
    frozen = json.loads(args.root_projection.read_text())
    reference_total = frozen["reference_features"]["nominal_padded_compute_rows"]
    if reference_total-paired_reference-residual_reference != 8192:
        raise ValueError("complete reference mean remainder differs")
    delta = current_total-reference_total
    residual_delta = residual_current-residual_reference
    stem = sum(r["difference_nominal_rows"] for r in rebuilt if r["role"] == "pooled_stem")
    spatial = sum(r["difference_nominal_rows"] for r in rebuilt if r["role"] == "direct")
    remaining = sum(r["difference_nominal_rows"] for r in rebuilt if r["role"] not in ("direct","pooled_stem"))
    if (delta,residual_current,residual_reference,residual_delta,stem,spatial,remaining) != (4937856,5193216,355328,4837888,81536,18432,0):
        raise ValueError("reported role decomposition differs")
    result={"schema":"root_current2039_reference_nominal_work_alignment_review_v1","status":"PASS","paired_source_geometries_independently_rebuilt":54,"residual_source_schedules_independently_bound":16,"active_current_functions":len(current["groups"]),"nominal_rows":{"current":current_total,"reference":reference_total,"difference":delta,"residual_difference":residual_delta,"residual_fraction":residual_delta/delta,"stem_difference":stem,"spatial_difference":spatial,"other_paired_difference":remaining,"mean_each":8192},"original_execution_conserved":True,"source_numeric_equivalence":False,"hardware_cycle_savings":None,"limits":"Nominal padded rows and logical requests, not usefulMACs/DDR/busy cycles. Different residual source numeric contracts. No reference timing fit or proportional whole-saving inference.","pins":{str(p):pin(p)["sha256"] for p in (args.receipt,args.root_projection,Path(__file__))},"input_pins_reclosed":receipt["pins"]}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"PASS","pairs":54,"residual_fraction":residual_delta/delta,"cycle_savings":None}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("receipt","root-projection","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    review(parser.parse_args())
