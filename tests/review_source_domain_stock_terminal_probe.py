"""Close the exact source-domain whole-model stock observation."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review():
    records = Path(__file__).resolve().parents[1] / "docs/perf_records"
    archive = Path("/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/docs/perf_records")
    packet_path = archive / "stock2071_source_domain_whole_terminal.json"
    release_path = records / "root_resnet_source_domain_stock_release_20261007.json"
    output = records / "root_resnet_stock2071_source_domain_terminal_review_20261007.json"
    require(not output.exists(), "terminal review must be fresh")
    stock, release = (json.loads(path.read_text()) for path in (packet_path, release_path))
    for receipt in (stock, release):
        for path, digest in receipt["pins"].items():
            require(sha(path) == digest, "immutable terminal/release dependency changed: " + path)
    record, control = stock["record"], stock["control"]
    require(record["job_id"] == 2071 and record["state"] == record["phase"] == "DONE"
            and record["exit_code"] == 0 and record["elf_sha256"] == release["candidate_elf_sha256"]
            and sha(record["elf"]) == record["elf_sha256"], "released stock executable or completion differs")
    identity = {"hw_config": release["hardware_alias"],
                "hwdb_sha256": "5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126",
                "bitstream_sha256": "6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229"}
    require(all(record[key] == control[key] == value for key, value in identity.items()),
            "actual stock identity differs")
    for suffix, key in (("_staged_elf_check.json", "elf_sha256"),
                         ("_staged_bitstream_check.json", "bitstream_sha256")):
        staged_path = next(Path(path) for path in stock["pins"] if path.endswith(suffix))
        staged = json.loads(staged_path.read_text())
        require(staged["job_id"] == record["job_id"] and staged[key] == record[key],
                "actual staged artifact differs")
    uart_path = archive / "stock2071_source_domain_whole_uart.txt"
    require(sha(uart_path) == record["uart_sha256"], "immutable raw UART changed")
    console = uart_path.read_text().replace("\r", "")
    cycles = record["kernel_cycles"]
    require(re.findall(r"^METRIC cycles (\d+)$", console, re.M) == [str(cycles)]
            and console.splitlines().count("DONE") == 1
            and console.splitlines().count("METRIC memref_rank_mismatch 0") == 1
            and "*** PASSED ***" in console and 'COMMAND_EXIT_CODE="0"' in console,
            "actual complete forward counter or terminal differs")
    evidence = record["output_digest_evidence"]
    require(sha(evidence["reference"]) == evidence["reference_sha256"]
            and sha(evidence["validation"]) == evidence["validation_sha256"], "standard reference adapter changed")
    words = np.load(evidence["reference"], allow_pickle=False)
    raw = hashlib.sha256(words.astype("<f4", copy=False).tobytes()).hexdigest()
    require(words.dtype == np.float32 and words.size == evidence["elements"] == 1000
            and evidence["bytes"] == 4000 and raw == evidence["sha256"] == release["raw_original_sha256"]
            and console.splitlines().count(f"OUT_SHA256 f32le 1000 4000 {raw}") == 1
            and release["numeric_gate"] == {"atol": 0, "rtol": 0}, "original complete exact output differs")
    require(control["job_id"] == release["control_job"] == 2068
            and control["kernel_cycles"] == release["control_cycles"]
            and control["elf_sha256"] == release["control_elf_sha256"], "actual whole control differs")
    fraction = (control["kernel_cycles"] - cycles) / control["kernel_cycles"]
    require(abs(fraction - stock["cycle_reduction_fraction"]) < 1e-15, "measured comparison differs")
    audit = audit_elf(Path(record["elf"]).read_bytes())
    require(audit == release["final_noFSM"] and audit["status"] == "pass", "all executable section audit differs")
    result = {"schema": "root_resnet_stock2071_source_domain_terminal_review_v1",
              "status": "champion_whole_model_stock_observation", "job_id": 2071,
              "stock_cycles": cycles, "control_job": 2068, "control_cycles": control["kernel_cycles"],
              "fraction_lower": fraction, "difference_cycles": cycles - control["kernel_cycles"],
              "terminal_pins_reclosed": len(stock["pins"]), "release_pins_reclosed": len(release["pins"]),
              "original1000_words_digest_independently_reparsed": True,
              "original_numeric_gate": release["numeric_gate"], "actual_staged_ELF_bitstream_receipts_reclosed": True,
              "hardware_identity": identity, "nofsm_all_executable_sections": audit,
              "owned_reference_cycles": 22387449, "remaining_owned_reference_gap_cycles": cycles - 22387449,
              "whole_cycle_prediction": "UNKNOWN", "default_promotion": False,
              "scope": "One whole observation per arm; three source-proven domain routes only. Original host/runtime/weights and first14batch4 unchanged.338688 nominal rows is not a cycle forecast; owned ZIP/1876 reference only.",
              "pins": {str(path): sha(path) for path in
                       (packet_path, release_path, uart_path, Path(evidence["reference"]),
                        Path(evidence["validation"]), Path(__file__))}}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "stock_cycles", "fraction_lower", "remaining_owned_reference_gap_cycles")}))


if __name__ == "__main__":
    review()
