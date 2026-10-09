"""Reparse the actual stock complete attention ABBA experiment."""

import importlib.util
import json
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review():
    root = Path(__file__).resolve().parents[1]
    packet = Path("/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/docs/perf_records/stock2073_complete_attention_terminal.json")
    terminal = json.loads(packet.read_text())
    record = terminal["record"]
    output = root / "docs/perf_records/root_tiny_complete_attention_stock2073_terminal_review_20261007.json"
    require(not output.exists(), "terminal review must be fresh")
    for path, digest in terminal["pins"].items():
        require(sha(path) == digest, "changed complete attention hardware evidence: " + path)
    release_path = Path(record["root_release"])
    require(sha(release_path) == record["root_release_sha256"], "root release changed")
    release = json.loads(release_path.read_text())
    for path, digest in release["pins"].items():
        require(sha(path) == digest, "changed released source/target evidence: " + path)
    require(record["job_id"] == 2073 and record["state"] == record["phase"] == "DONE" and
            record["exit_code"] == 0 and record["elf_sha256"] == release["candidate_elf_sha256"],
            "stock terminal or executable identity differs")
    for key, value in release["hardware_identity_required"].items():
        require(record[key] == value, "actual stock hardware identity differs")
    for key, field in (("staged_elf_receipt", "elf_sha256"), ("staged_bitstream_receipt", "bitstream_sha256")):
        observation = json.loads(Path(record[key]).read_text())
        require(observation["job_id"] == 2073 and observation[field] == record[field], "staging observation differs")
    elf_audit = audit_elf(Path(release["candidate"]).read_bytes())
    require(elf_audit["status"] == "pass" and elf_audit["elf_sha256"] == record["elf_sha256"],
            "all-executable instruction audit failed")
    parser_path = Path(record["parser_path"])
    spec = importlib.util.spec_from_file_location("root_complete_attention_uart_parser", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    uart_path = next(Path(p) for p, digest in terminal["pins"].items()
                     if p.endswith("_uart.txt") and digest == record["uart_sha256"])
    console = uart_path.read_text()
    report = parser.parse_report(console, release)
    require(report == record["report"] and "*** PASSED ***" in console and 'COMMAND_EXIT_CODE="0"' in console,
            "actual raw UART did not pass complete source/guard/counter checks")
    control = sum(r["cycles"] for r in report["rows"] if r["id"] == 0) / 2
    current = sum(r["cycles"] for r in report["rows"] if r["id"] == 1) / 2
    reduction = (control - current) / control
    require(control == terminal["means"]["source_control"] and current == terminal["means"]["current_masked"] and
            reduction == terminal["means"]["cycle_reduction_fraction"], "actual timing aggregation differs")
    result = {
        "schema": "root_complete_attention_stock2073_terminal_review_v1",
        "status": "VERIFIED_COMPLETE_CURRENT_ATTENTION_STOCK_PAIR",
        "job_id": 2073, "terminal_pins_reclosed": len(terminal["pins"]),
        "release_pins_reclosed": len(release["pins"]), "report": report,
        "mean_control_cycles": control, "mean_current_cycles": current, "fraction_lower": reduction,
        "actual_staged_ELF_bitstream_reclosed": True, "raw_UART_ABBA_source16384i8_modes_guards_reparsed": True,
        "final_noFSM": elf_audit,
        "hardware_identity": {k: record[k] for k in release["hardware_identity_required"]},
        "scope": release["complete_cost_scope"]["ROI"],
        "candidate_policy": release["candidate_policy"],
        "host_vs_accelerator": "This capsule executes the current scalar attention path; its final executable contains no Gemmini custom instructions. It prices host attention on the requested stock Gemmini system.",
        "whole_Tiny_champion": {"job_id": 2070, "cycles": 394765577, "unchanged": True},
        "all22_projection_or_whole_forecast": "UNKNOWN; one original context and its runtime/operands",
        "new_label_fitted": False, "new_numeric_policy_selected": False,
        "pins": {str(p.resolve()): sha(p) for p in (packet, release_path, uart_path, parser_path, Path(__file__))},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "job_id", "mean_control_cycles", "mean_current_cycles", "fraction_lower")}))


if __name__ == "__main__":
    review()
