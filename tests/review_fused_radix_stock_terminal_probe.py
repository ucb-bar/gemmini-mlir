"""Reclose the released complete-group stock observation and its frozen screen."""

import argparse
import json
import re
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "terminal review output must be fresh")
    terminal = json.loads(args.terminal.read_text())
    release = json.loads(args.release.read_text())
    for receipt in (terminal, release):
        for path, expected in receipt["pins"].items():
            require(sha(path) == expected, "frozen dependency changed: " + path)
    record = terminal["record"]
    require(record["job_id"] == 2069 and record["state"] == record["phase"] == "DONE"
            and record["exit_code"] == 0, "stock job did not complete")
    require(record["elf"] == release["candidate"]
            and record["elf_sha256"] == release["candidate_elf_sha256"]
            and sha(record["elf"]) == record["elf_sha256"], "released ELF differs")
    for key, expected in release["hardware_identity_required"].items():
        require(record[key] == expected, "stock hardware differs: " + key)
    job_root = Path(record["uart"]).parents[2]
    spec = json.loads((job_root / "runworkload-full.json").read_text())
    require(spec["user"] == "agustin" and spec["workload"] == "merlin-golden-nofsm-probe"
            and spec["stage_from"] == release["candidate"]
            and spec["hw_config"] == release["hardware_alias"]
            and spec["hwdb_config_artifact_sha256"] == record["hwdb_sha256"],
            "actual queue workload differs")
    staged_paths = []
    for suffix, digest_key in (("_staged_elf_check.json", "elf_sha256"),
                               ("_staged_bitstream_check.json", "bitstream_sha256")):
        paths = [Path(path) for path in terminal["pins"] if path.endswith(suffix)]
        require(len(paths) == 1, "missing or ambiguous actual staging receipt")
        staged = json.loads(paths[0].read_text())
        require(staged["job_id"] == record["job_id"]
                and staged[digest_key] == record[digest_key], "actual staging differs")
        staged_paths.extend(paths)
    uart_paths = [Path(path) for path, digest in terminal["pins"].items()
                  if path.endswith("_uart.txt") and digest == record["uart_sha256"]]
    require(len(uart_paths) == 1, "missing immutable raw UART archive")
    uart_path = uart_paths[0]
    console = uart_path.read_text().replace("\r", "")
    strict_paths = [Path(path) for path in terminal["pins"] if path.endswith("/spike.stdout")]
    require(len(strict_paths) == 1, "missing matched strict work log")
    strict_console = strict_paths[0].read_text().replace("\r", "")
    for raw in (console, strict_console):
        require(re.findall(r"^WORKSPACE_STAT (\d+) (\d+)$", raw, re.M)
                == [(str(i), str(value)) for i, value in enumerate(release["uart_required"]["stats"])],
                "original complete consumer statistics differ")
        require(re.findall(r"^UNOBSERVED_CARRIER_DIFFERENCES (\d+)$", raw, re.M)
                == [str(release["uart_required"]["carrier_differences"])],
                "carrier diagnostic differs")
        require(re.findall(r"^" + re.escape(release["uart_required"]["pass_marker"]) + r"$", raw, re.M)
                == [release["uart_required"]["pass_marker"]]
                and "CAPSULE UNEXPECTED SOURCE REFUSAL" not in raw,
                "original consumer, input, descriptor or guard check failed")
    require("*** PASSED ***" in console and 'COMMAND_EXIT_CODE="0"' in console,
            "simulator completion failed")
    require(re.findall(r"^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$", console, re.M)
            == [str(record["kernel_cycles"])], "actual hardware ROI counter differs")
    require(re.findall(r"^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$", strict_console, re.M)
            == [str(release["strict_group"]["candidate_roi_instructions"])],
            "matched strict ROI retirement differs")
    require(record["raw_metric_label"] == "WORKSPACE_GROUP_INSTRUCTIONS",
            "original metric spelling differs")
    audit = audit_elf(Path(record["elf"]).read_bytes())
    require(audit == release["final_noFSM"] and audit["status"] == "pass",
            "all executable sections noFSM audit differs")
    control = terminal["control"]
    require(control["job"] == release["control_job"]
            and control["cycles"] == release["control_cycles"]
            and control["elf_sha256"] == release["control_elf_sha256"], "control differs")
    actual = record["kernel_cycles"]
    fraction_lower = (control["cycles"] - actual) / control["cycles"]
    require(abs(fraction_lower - terminal["cycle_reduction_fraction"]) < 1e-15,
            "cycle comparison differs")
    screen = release["conditional_pretiming_screen"]
    ratio = release["strict_group"]["candidate_roi_instructions"] / release["strict_group"]["control_roi_instructions"]
    require(screen["ratio"] == ratio
            and screen["candidate_group_cycles"] == control["cycles"] * ratio
            and screen["new_labels_fitted"] is False and screen["resolved_prediction"] is False,
            "frozen conditional screen changed")
    paths = [args.terminal, args.release, uart_path, Path(__file__), *staged_paths]
    result = {
        "schema": "root_fused_radix_stock2069_terminal_review_v1",
        "status": "VERIFIED_COMPLETE_SOURCE_GROUP_STOCK_WIN", "job_id": 2069,
        "stock_cycles": actual, "control_job": control["job"],
        "control_cycles": control["cycles"], "fraction_lower": fraction_lower,
        "terminal_pins_reclosed": len(terminal["pins"]),
        "release_pins_reclosed": len(release["pins"]),
        "raw_UART_original_consumer_stats_carrier_guards_reparsed": True,
        "original_i8_words": release["strict_group"]["complete_original_i8"],
        "original_bf16_scales": release["strict_group"]["complete_original_bf16_scales"],
        "output_checksum": "NOT_EMITTED; original complete consumer byte comparisons pass",
        "actual_staged_ELF_bitstream_receipts_reclosed": True,
        "hardware_identity": {key: record[key] for key in release["hardware_identity_required"]},
        "nofsm": audit,
        "frozen_conditional_screen_score": {
            "predicted_cycles": screen["candidate_group_cycles"], "actual_cycles": actual,
            "relative_error": abs(screen["candidate_group_cycles"] - actual) / actual,
            "qualified_cost_model": False, "unpriced": screen["unpriced"],
            "new_label_fitted": False,
        },
        "whole_hardware_cycles": "UNKNOWN;1906 champion258621872969 unchanged",
        "default_promotion": False,
        "scope": "One allocation-aware 12head group with explicit experimental RMS4 policy; no whole cycle projection or production approximate binder seal.",
        "pins": {str(path.resolve()): sha(path) for path in paths},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "stock_cycles", "fraction_lower", "frozen_conditional_screen_score")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("terminal", "release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
