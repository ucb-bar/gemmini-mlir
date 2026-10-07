"""Admit a source-domain residual composition after rederiving its actual SSA facts."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from merlin.llvmlower import quantized_affine_domain as domain
from mlir_oot.domain_residual_catalog import inspect_choices
from mlir_oot.executed_features import parse_pc_histogram
from mlir_oot.no_fsm_audit import audit_elf
from review_rectifier_four_panel_stock_packet_probe import check, code, sha


def review(args):
    check(not args.output.exists(), "source-domain release must be fresh")
    q = json.loads(args.packet.read_text())
    bindings = {}
    for pin in q["flat_file_pins"]:
        path = Path(pin["path"])
        check(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"],
              "changed qualification artifact: " + str(path))
        check(str(path) not in bindings or bindings[str(path)] == pin["sha256"],
              "contradictory qualification binding")
        bindings[str(path)] = pin["sha256"]
    check(q["status"] == "PASS" and q["control_job"] == 2068
          and q["control_stock_cycles"] == 29891965 and q["control_byte_identical"],
          "actual current whole control differs")
    control, candidate = (Path(q[key]["path"]) for key in ("control_elf", "candidate_elf"))
    check(sha(control) == "2247aa1691bf1962951a879d44bced5a2e515d88e980bb9865807e830cdd0633"
          and sha(candidate) == q["candidate_elf"]["sha256"], "controlled final ELF changed")
    audit = audit_elf(candidate.read_bytes())
    check(audit["status"] == "pass" and not audit["unknown"] and not audit["forbidden"],
          "all executable sections must contain zero forbidden/unknown custom instructions")
    for gates in (q["normal_whole_gate"], q["controlled_whole_gate"]):
        for gate in (gates["native"], gates["spike"]):
            check(gate["elements"] == 1000 and gate["exact_equal"] and gate["quality_pass"]
                  and gate["atol"] == gate["rtol"] == 0, "original exact whole gate failed")
    strict = q["controlled_whole_gate"]["spike"]
    original = Path(q["normal_whole_gate"]["native"]["original_golden"])
    native = Path(strict["reference_path"])
    expected, actual = (np.load(path, allow_pickle=False) for path in (original, native))
    check(expected.dtype == actual.dtype == np.float32 and actual.size == 1000
          and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
          "all original native output bytes differ")
    raw = hashlib.sha256(actual.astype("<f4", copy=False).tobytes()).hexdigest()
    check(raw == strict["spike_output_sha256"] ==
          "0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787",
          "original complete target output digest differs")
    console_path = Path(strict["spike_console_path"])
    lines = console_path.read_text().splitlines()
    check(lines.count("DONE") == lines.count("METRIC memref_rank_mismatch 0") == 1
          and lines.count(f"OUT_SHA256 f32le 1000 4000 {raw}") == 1,
          "actual strict whole completion differs")
    selection_path = Path(q["normal_input_domain_binding"]["path"])
    selection = json.loads(selection_path.read_text())
    input_path = selection_path.parent.parent / "model.prepared.mlir"
    check(bindings.get(str(input_path)) == selection["input_sha256"],
          "actual normal callback source missing")
    changes = q["selected_source_ranges"]
    proposals = [{"source": row["source"], "predictor": row["choice"]["predictor"]} for row in changes]
    _, _, _, choices = inspect_choices(args.base, proposals, source=input_path)
    derived = [dict(row, new_products=sum((row["choice"]["predictor"][key] + 126) // 127
                                        for key in ("p", "q")))
               for row in choices if row["choice"] is not None]
    check(derived == changes and len(choices) == 16,
          "actual typed producer facts or selected source paths differ")
    check([(row["old_products"], row["new_products"]) for row in changes] == [(9, 3), (17, 16), (8, 7)],
          "admitted target representation differs")
    for row in changes:
        certificate = domain.validate(row["choice"]["certificate"])
        check(certificate["admitted_pairs"] == 32768
              and certificate["operand_intervals"] == [[-128, 127], [0, 127]]
              and not certificate["exact_for_complete_type_domain"],
              "excluded full-type pairs must remain inadmissible")
    link = q["controlled_link"]
    for path, digest in link["retained_input_pins"].items():
        check(sha(path) == digest, "retained link input differs")
    identity = link["executable_symbol_rebinding_identity"]
    check(code(identity["before"]) == code(identity["after"]),
          "symbol rebinding changed executable bytes")
    check(q["unchanged_13_other_residuals_and_first14_batch4"]
          and q["host_runtime_weights_other_target_objects_unchanged"]
          and link["all_other_thirteen_residual_routes_byte_identical"]
          and link["first_source_rectifier_four_panel_unchanged"]
          and link["all_host_runtime_weights_other_target_objects_unchanged"], "controlled link scope differs")
    work = q["actual_selected_execution_and_emitted_work"]
    hist_path = Path(work["qualified_histogram"])
    check(sha(hist_path) == work["histogram_sha256"], "actual final ELF histogram differs")
    histogram = parse_pc_histogram(hist_path.read_text())
    counts = {row["symbol"]: histogram.get(row["start"], 0) for row in work["symbol_ranges"]}
    check(counts == work["actual_entry_counts"]
          and all(count == (0 if symbol.startswith("domain_retained") else 1)
                  for symbol, count in counts.items()), "actual selected/retained entry coverage differs")
    removed = 0
    for row in work["selected_geometry_work"]:
        old, new = row["arms"]["control"], row["arms"]["candidate"]
        check(old["staticCFG_complete"] and new["staticCFG_complete"]
              and old["requested_i8_load_bytes"] == new["requested_i8_load_bytes"]
              and old["requested_i8_store_bytes"] == new["requested_i8_store_bytes"]
              and old["commands"]["fence"] == new["commands"]["fence"] == 2,
              "original-geometry traffic/fence work differs")
        removed += old["nominal_compute_rows"] - new["nominal_compute_rows"]
    check(removed == work["nominal_compute_rows_removed"] == q["target_nominal_compute_rows_removed"] == 338688
          and q["requested_traffic_delta"] == work["extra_requested_traffic"] == 0
          and not q["new_correction_or_private_storage"], "complete target demand differs")
    capsules = q["complete_admitted_target_capsules"]
    check(capsules["status"] == "PASS" and len(capsules["cases"]) == 3
          and all(row["complete_admitted_pairs"] == 32768 for row in capsules["cases"]),
          "complete admitted target domain coverage differs")
    for key in ("strict_stdout", "gsim_stdout"):
        raw_capsule = Path(capsules[key]["path"]).read_text()
        check(raw_capsule.splitlines().count("ADMITTED_DOMAIN_ALL PASS") == 1
              and len(re.findall(r"^ADMITTED_CASE [012] \d+ 0 0$", raw_capsule, re.M)) == 3,
              "actual admitted target pairs, guards or FP state failed")
    check(audit_elf(Path(capsules["elf"]["path"]).read_bytes())["status"] == "pass",
          "complete admitted target capsule noFSM failed")
    check(not q["default_enabled"] and q["whole_candidate_cycles"] == "UNKNOWN", "unmeasured/default boundary differs")
    result = {
        "schema": "root_source_domain_residual_whole_stock_release_v1",
        "status": "RELEASED_ONE_CONTROLLED_WHOLE_STOCK_OBSERVATION",
        "qualification_pins_reclosed": len(q["flat_file_pins"]),
        "actual_normal_SSA_domain_paths_and_all16_choices_rederived": True,
        "control_job": 2068, "control_cycles": 29891965, "control_elf_sha256": sha(control),
        "candidate": str(candidate), "candidate_elf_sha256": sha(candidate),
        "original_words_exact": 1000, "raw_original_sha256": raw,
        "numeric_gate": {"atol": 0, "rtol": 0}, "normal_and_controlled_whole_routes_qualified": True,
        "source_products": [(row["old_products"], row["new_products"]) for row in changes],
        "target_nominal_compute_rows_removed": removed, "requested_traffic_delta": 0,
        "complete_admitted_target_pairs": 98304, "active_entry_counts_rederived": counts,
        "host_runtime_weights_other_target_objects_unchanged": True,
        "final_noFSM": audit, "strict_retirement_only_proxy": q["whole_strict_forward_proxy"],
        "whole_cycle_prediction": "UNKNOWN", "hardware_alias": "alveo_u250_firesim_gemmini_rocket_stock",
        "hardware_config": "FireSimGemminiRocketConfig", "allowed_runs": 1,
        "scope": "Source-proven nonnegative RHS domains propagated through actual verified typed views, consumed by complete admitted affine certificates. Only three target routes change; first14batch4 and all host/runtime/weights/other routes retained. No workload/sample-range strategy selection.",
        "default_promotion": False,
        "requirements": ["Recovery submits once after root commit with actual staged ELF/stock configuration/bitstream receipts.",
                         "Original complete1000word0/0 gate, raw digest/DONE/rank0 and standard adapter.",
                         "Compare whole forward with2068; nominal rows/capsule/strict instructions are not a whole forecast."],
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.packet, candidate, control, original, native, console_path, hist_path,
                  selection_path, input_path, Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "qualification_pins_reclosed", "candidate_elf_sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "base", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
