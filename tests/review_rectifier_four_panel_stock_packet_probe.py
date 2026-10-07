"""Independently admit one controlled whole-model four-panel observation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from merlin.targetgen.elf_lanes import executable_sections
from mlir_oot.executed_features import parse_pc_histogram
from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def code(path):
    blob = Path(path).read_bytes()
    return [(name, address, blob[offset:offset + size])
            for name, offset, size, address in executable_sections(blob)]


def review(args):
    check(not args.output.exists(), "root release must be fresh")
    q = json.loads(args.packet.read_text())
    bindings = {}
    for pin in q["flat_file_pins"]:
        path = Path(pin["path"])
        check(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"],
              "changed qualification artifact: " + str(path))
        check(str(path) not in bindings or bindings[str(path)] == pin["sha256"],
              "conflicting evidence binding")
        bindings[str(path)] = pin["sha256"]
    check(q["status"] == "PASS" and q["control_job"] == 2066 and
          q["control_stock_cycles"] == 30169093 and q["control_byte_identical"],
          "current whole control differs")
    control = Path(q["control_elf"]["path"])
    candidate = Path(q["candidate_elf"]["path"])
    check(sha(control) == "16e8f69ad5b54a15e5620729d555bc5eea90fa6da7c23badb8a6124783c91ef9"
          and sha(candidate) == "2247aa1691bf1962951a879d44bced5a2e515d88e980bb9865807e830cdd0633",
          "controlled final ELF changed")
    audit = audit_elf(candidate.read_bytes())
    check(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
          "all executable sections must have zero forbidden/unknown custom instructions")
    for gate in (q["normal_whole_native"], q["normal_whole_spike"],
                 q["controlled_whole_gate"]["native"], q["controlled_whole_gate"]["spike"]):
        check(gate["elements"] == 1000 and gate["exact_equal"] and gate["quality_pass"]
              and gate["atol"] == gate["rtol"] == 0, "original exact whole gate failed")
    strict = q["controlled_whole_gate"]["spike"]
    original = Path(q["normal_whole_native"]["original_golden"])
    native = Path(strict["reference_path"])
    words, expected = (np.load(path, allow_pickle=False) for path in (native, original))
    check(words.dtype == expected.dtype == np.float32 and words.size == 1000 and
          words.shape == expected.shape and words.tobytes() == expected.tobytes(),
          "original native output words differ")
    raw = hashlib.sha256(words.astype("<f4", copy=False).tobytes()).hexdigest()
    check(raw == strict["spike_output_sha256"] ==
          "0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787",
          "original full output digest differs")
    console_path = Path(strict["spike_console_path"])
    check(sha(console_path) == strict["spike_console_sha256"], "target console changed")
    lines = console_path.read_text().splitlines()
    check(lines.count("DONE") == lines.count("METRIC memref_rank_mismatch 0") == 1 and
          lines.count(f"OUT_SHA256 f32le 1000 4000 {raw}") == 1,
          "complete target output terminal differs")
    link = q["controlled_link"]
    for path, digest in link["retained_input_pins"].items():
        check(sha(path) == digest, "retained current object differs")
    for identity in link["executable_symbol_rebinding_identity"]:
        check(code(identity["before"]) == code(identity["after"]),
              "symbol rebinding changed executable bytes")
    check(link["all_host_runtime_weights_other_target_objects_unchanged"] and
          q["runtime_host_weights_other_target_objects_unchanged"], "controlled scope differs")
    active = q["actual_selected_execution"]
    hist_path = next(Path(path) for path, digest in bindings.items()
                     if digest == active["histogram_sha256"])
    histogram = parse_pc_histogram(hist_path.read_text())
    counts = {entry["symbol"]: histogram.get(entry["start"], 0)
              for entry in active["symbol_ranges"]}
    check(counts == active["observed_entry_counts"] and list(counts.values()) == [1, 1, 0, 0, 0, 0]
          and active["returncode"] == 0, "selected or retained producer execution differs")
    route = q["one_changed_active_target_route"]
    check(route["panel_batch"] == 4 and route["compute_passes"] == 14,
          "target resource/product schedule differs")
    cost = q["ranked_cost"]
    check(cost["before_coalesced14"] == 1603489 and cost["after_batch4"] == 1322936 and
          q["no_additive_cycle_forecast"] and not q["default_enabled"],
          "matched cost/default boundary differs")
    result = {
        "schema": "root_rectifier_four_panel_whole_stock_release_v1",
        "status": "RELEASED_ONE_CONTROLLED_WHOLE_STOCK_OBSERVATION",
        "qualification_pins_reclosed": len(q["flat_file_pins"]),
        "control_job": 2066, "control_cycles": 30169093,
        "control_elf_sha256": sha(control), "candidate": str(candidate),
        "candidate_elf_sha256": sha(candidate), "original_words_exact": 1000,
        "raw_original_sha256": raw, "numeric_gate": {"atol": 0, "rtol": 0},
        "normal_and_controlled_whole_routes_qualified": True,
        "active_entry_counts_rederived": counts,
        "host_runtime_weights_other_target_objects_unchanged": True,
        "one_changed_active_target_route": route, "final_noFSM": audit,
        "complete_ranked_GSIM_cost": cost, "whole_cycle_prediction": "UNKNOWN",
        "hardware_alias": "alveo_u250_firesim_gemmini_rocket_stock",
        "hardware_config": "FireSimGemminiRocketConfig", "allowed_runs": 1,
        "requirements": [
            "Recovery submits only after root commit; exact released final ELF and stock bitstream identity.",
            "All1000 original words/full digest/DONE/rank0/standard adapter, original0/0 gate retained.",
            "One whole forward observation versus2066; section improvements are not a whole forecast.",
        ],
        "scope": "Four disjoint private panels share immutable coefficient operands and preserve store/reload/publication fences. Same14 products, exact finite-source certificate and typed three-ranked-argument ABI. No source/model ID selects compiler strategy.",
        "default_promotion": False,
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.packet, candidate, control, original, native, console_path, hist_path, Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "qualification_pins_reclosed", "candidate_elf_sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
