"""Reclose the single released stock encoder-composition group observation."""

import argparse
import json
import re
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "review must be fresh")
    terminal = json.loads(args.terminal.read_text())
    release = json.loads(args.release.read_text())
    for evidence in (terminal, release):
        for path, digest in evidence["pins"].items():
            require(sha(path) == digest, "frozen dependency differs: " + path)
    record = terminal["record"]
    require(record["job_id"] == 2072 and record["state"] == record["phase"] == "DONE" and
            record["exit_code"] == 0, "stock job did not complete")
    require(record["elf"] == release["candidate"] and
            record["elf_sha256"] == release["candidate_elf_sha256"] == sha(record["elf"]),
            "released/staged ELF differs")
    for key, value in release["hardware_identity_required"].items():
        require(record[key] == value, "stock hardware differs: " + key)
    job_root = Path(record["uart"]).parents[2]
    spec = json.loads((job_root / "runworkload-full.json").read_text())
    require(spec["user"] == "agustin" and spec["workload"] == "merlin-golden-nofsm-probe" and
            spec["stage_from"] == release["candidate"] and
            spec["hw_config"] == release["hardware_alias"] and
            spec["hwdb_config_artifact_sha256"] == record["hwdb_sha256"],
            "actual queue specification differs")
    staged_paths = []
    for suffix, key in (("_staged_elf_check.json", "elf_sha256"),
                        ("_staged_bitstream_check.json", "bitstream_sha256")):
        paths = [Path(path) for path in terminal["pins"] if path.endswith(suffix)]
        require(len(paths) == 1, "missing actual staging receipt")
        staged = json.loads(paths[0].read_text())
        require(staged["job_id"] == record["job_id"] and staged[key] == record[key],
                "actual staging differs")
        staged_paths += paths
    uarts = [Path(path) for path, digest in terminal["pins"].items()
             if path.endswith("_uart.txt") and digest == record["uart_sha256"]]
    require(len(uarts) == 1, "missing raw UART archive")
    stricts = [Path(path) for path in release["pins"] if path.endswith("/spike.stdout")]
    require(len(stricts) == 1, "missing strict control scope")
    console = uarts[0].read_text().replace("\r", "")
    strict = stricts[0].read_text().replace("\r", "")
    expected = release["uart_required"]
    for text in (console, strict):
        require(re.findall(r"^WORKSPACE_STAT (\d+) (\d+)$", text, re.M) ==
                [(str(i), str(v)) for i, v in enumerate(expected["stats"])], "stats differ")
        require(re.findall(r"^UNOBSERVED_CARRIER_DIFFERENCES (\d+)$", text, re.M) ==
                [str(expected["carrier_differences"])], "carrier observation differs")
        require(expected["pass_marker"] in text and "CAPSULE UNEXPECTED SOURCE REFUSAL" not in text,
                "original consumer/input/guards failed")
    require("*** PASSED ***" in console and 'COMMAND_EXIT_CODE="0"' in console and
            record["raw_metric_label"] == "WORKSPACE_GROUP_INSTRUCTIONS" and
            re.findall(r"^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$", console, re.M) ==
            [str(record["kernel_cycles"])] and
            re.findall(r"^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$", strict, re.M) ==
            [str(release["strict_group"]["candidate_instructions"])], "ROI/completion differs")
    audit = audit_elf(Path(record["elf"]).read_bytes())
    require(audit == release["final_noFSM"] and audit["status"] == "pass", "all-exec audit differs")
    require(terminal["control"]["job"] == release["control_job"] == 2069 and
            terminal["control"]["cycles"] == release["control_cycles"], "control differs")
    actual = record["kernel_cycles"]
    fraction = (release["control_cycles"] - actual) / release["control_cycles"]
    require(abs(fraction - terminal["cycle_reduction_fraction"]) < 1e-15, "comparison differs")
    frozen = release["conditional_pretiming_screen"]
    ratio = release["strict_group"]["candidate_instructions"] / release["strict_group"]["control_instructions"]
    require(frozen["ratio"] == ratio and frozen["candidate_group_cycles"] ==
            release["control_cycles"] * ratio and not frozen["new_labels_fitted"] and
            not frozen["resolved_prediction"], "frozen count screen differs")
    result = {
        "schema": "root_fused_encoder_stock2072_terminal_review_v1",
        "status": "VERIFIED_COMPLETE_SOURCE_GROUP_STOCK_WIN", "job_id": 2072,
        "stock_cycles": actual, "control_job": 2069, "control_cycles": release["control_cycles"],
        "fraction_lower": fraction, "terminal_pins_reclosed": len(terminal["pins"]),
        "release_pins_reclosed": len(release["pins"]),
        "original_i8_words": release["strict_group"]["original_i8"],
        "original_bf16_scales": release["strict_group"]["original_scales"],
        "raw_UART_consumer_stats_carrier_guards_reparsed": True,
        "actual_staged_ELF_bitstream_reclosed": True, "nofsm": audit,
        "hardware_identity": release["hardware_identity_required"],
        "frozen_conditional_screen_score": {"predicted_cycles": frozen["candidate_group_cycles"],
            "actual_cycles": actual, "relative_error": abs(frozen["candidate_group_cycles"] - actual) / actual,
            "qualified_cost_model": False, "new_label_fitted": False, "unpriced": frozen["unpriced"]},
        "whole_hardware_cycles": "UNKNOWN;1906 champion258621872969 unchanged",
        "default_promotion": False,
        "scope": "Allocation-aware original12head group, unchanged explicit experimental RMS4 policy; no whole projection or production approximate binder seal.",
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.terminal, args.release, uarts[0], Path(__file__), *staged_paths)},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "stock_cycles", "fraction_lower", "frozen_conditional_screen_score")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("terminal", "release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
