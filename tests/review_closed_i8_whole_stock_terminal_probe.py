"""Reclose the released whole-model stock run against immutable original outputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "whole terminal review must be fresh")
    terminal, release = (json.loads(path.read_text()) for path in (args.terminal, args.release))
    for receipt in (terminal, release):
        for path, expected in receipt["pins"].items():
            require(sha(path) == expected, "immutable dependency changed: " + path)
    record, control = terminal["record"], terminal["control"]
    require(record["job_id"] == 2070 and record["state"] == record["phase"] == "DONE"
            and record["exit_code"] == 0 and record["marker_kind"] == "WHOLE_MODEL",
            "complete stock forward did not finish")
    require(record["elf"] == release["candidate"]
            and record["elf_sha256"] == release["candidate_elf_sha256"]
            and sha(record["elf"]) == record["elf_sha256"], "released final executable differs")
    for key, expected in release["hardware_identity_required"].items():
        require(record[key] == control[key] == expected, "stock identity differs: " + key)
    staged_paths = []
    for suffix, key in (("_staged_elf_check.json", "elf_sha256"),
                         ("_staged_bitstream_check.json", "bitstream_sha256")):
        paths = [Path(path) for path in terminal["pins"] if path.endswith(suffix)]
        require(len(paths) == 1, "missing actual staging observation")
        receipt = json.loads(paths[0].read_text())
        require(receipt["job_id"] == record["job_id"] and receipt[key] == record[key],
                "actual staged artifact differs")
        staged_paths.extend(paths)
    uart_paths = [Path(path) for path, digest in terminal["pins"].items()
                  if path.endswith("_uart.txt") and digest == record["uart_sha256"]]
    require(len(uart_paths) == 1, "immutable UART archive missing")
    uart_path = uart_paths[0]
    console = uart_path.read_text().replace("\r", "")
    actual = record["kernel_cycles"]
    require(re.findall(r"^METRIC cycles (\d+)$", console, re.M) == [str(actual)]
            and console.splitlines().count("DONE") == 1
            and console.splitlines().count("METRIC memref_rank_mismatch 0") == 1
            and "*** PASSED ***" in console and 'COMMAND_EXIT_CODE="0"' in console,
            "actual original full-forward metric or terminal differs")
    evidence = record["output_digest_evidence"]
    require(sha(evidence["reference"]) == evidence["reference_sha256"]
            and sha(evidence["validation"]) == evidence["validation_sha256"]
            and evidence["validation"] == release["standard_adapter"],
            "original standard reference adapter differs")
    adapter = json.loads(Path(evidence["validation"]).read_text())
    reference = np.load(evidence["reference"], allow_pickle=False)
    gold_path = Path(adapter["torch_golden_path"])
    require(sha(gold_path) == adapter["torch_golden_sha256"], "original Torch output changed")
    gold = np.load(gold_path, allow_pickle=False)
    raw = hashlib.sha256(reference.astype("<f4", copy=False).tobytes()).hexdigest()
    require(reference.dtype == np.float32 and reference.shape == gold.shape == (1, 8, 32000)
            and raw == evidence["sha256"] == release["raw_original_sha256"]
            and evidence["elements"] == 256000 and evidence["bytes"] == 1024000,
            "all original output bytes differ")
    require(np.allclose(reference, gold, atol=.03125, rtol=.02)
            and release["original_Torch_gate"] == {"atol": .03125, "rtol": .02, "passed": True},
            "original Torch elementwise gate failed")
    require(re.findall(r"^OUT_SHA256 f32le (\d+) (\d+) ([a-f0-9]{64})$", console, re.M)
            == [("256000", "1024000", raw)], "hardware full-output digest differs")
    require(control["job_id"] == release["control_job"] == 2062
            and control["kernel_cycles"] == release["control_cycles"]
            and control["elf_sha256"] == release["control_elf_sha256"]
            and control["state"] == control["phase"] == "DONE" and control["exit_code"] == 0,
            "released actual whole control differs")
    fraction_lower = (control["kernel_cycles"] - actual) / control["kernel_cycles"]
    require(abs(fraction_lower - terminal["cycle_reduction_fraction"]) < 1e-15,
            "measured whole comparison differs")
    audit = audit_elf(Path(record["elf"]).read_bytes())
    require(audit == release["final_noFSM"] and audit["status"] == "pass",
            "all executable sections instruction audit differs")
    paths = [args.terminal, args.release, uart_path, Path(evidence["reference"]),
             Path(evidence["validation"]), gold_path, Path(__file__), *staged_paths]
    result = {
        "schema": "root_closed_i8_stock2070_whole_terminal_review_v1",
        "status": "champion_whole_model_stock_observation", "job_id": 2070,
        "stock_cycles": actual, "control_job": 2062,
        "control_cycles": control["kernel_cycles"], "fraction_lower": fraction_lower,
        "difference_cycles": actual - control["kernel_cycles"],
        "terminal_pins_reclosed": len(terminal["pins"]),
        "release_pins_reclosed": len(release["pins"]),
        "original256000_words_digest_independently_reparsed": True,
        "raw_original_sha256": raw,
        "original_Torch_gate": release["original_Torch_gate"],
        "actual_staged_ELF_bitstream_receipts_reclosed": True,
        "hardware_identity": {key: record[key] for key in release["hardware_identity_required"]},
        "nofsm_all_executable_sections": audit,
        "remaining_300M_target_gap_cycles": actual - 300000000,
        "whole_cycle_prediction": "UNKNOWN", "default_promotion": False,
        "scope": "One controlled whole observation on frozen2062 source with22layers/eighttokens. Only selected model.o changes;155 device boundaries and11 other objects remain unchanged. Helper14.0703% becomes whole3.7502%; no fresh upstream recapture or default promotion.",
        "pins": {str(path.resolve()): sha(path) for path in paths},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "stock_cycles", "fraction_lower", "remaining_300M_target_gap_cycles")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("terminal", "release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
