"""Close the released four-panel whole-model hardware observation."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review():
    root = Path(__file__).resolve().parents[1]
    records = root / "docs/perf_records"
    archive = Path("/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/docs/perf_records")
    packet_path = archive / "stock2068_four_panel_whole_terminal.json"
    uart_path = archive / "stock2068_four_panel_whole_uart.txt"
    release_path = records / "root_resnet_four_panel_stock_release_20261007.json"
    output = records / "root_resnet_stock2068_four_panel_terminal_review_20261007.json"
    require(not output.exists(), "root terminal review must be fresh")
    stock, release = (json.loads(path.read_text()) for path in (packet_path, release_path))
    for path, digest in stock["pins"].items():
        require(sha(path) == digest, "changed hardware terminal dependency: " + path)
    record = stock["record"]
    require(record["job_id"] == 2068 and record["state"] == record["phase"] == "DONE"
            and record["exit_code"] == 0 and record["hw_config"] == release["hardware_alias"],
            "hardware completion or stock configuration differs")
    require(sha(record["elf"]) == record["elf_sha256"] == release["candidate_elf_sha256"]
            and sha(uart_path) == record["uart_sha256"], "actual released ELF/raw UART changed")
    require(record["hwdb_sha256"] == "5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126"
            and record["bitstream_sha256"] == "6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229",
            "stock hardware identity differs")
    uart = uart_path.read_text()
    cycles = record["kernel_cycles"]
    require(re.findall(r"^METRIC cycles (\d+)\s*$", uart, re.M) == [str(cycles)]
            and uart.splitlines().count("DONE") == 1
            and uart.splitlines().count("METRIC memref_rank_mismatch 0") == 1
            and "*** PASSED ***" in uart, "complete forward counter or terminal differs")
    raw = release["raw_original_sha256"]
    require(uart.splitlines().count(f"OUT_SHA256 f32le 1000 4000 {raw}") == 1,
            "original complete output digest differs")
    evidence = record["output_digest_evidence"]
    require(sha(evidence["reference"]) == evidence["reference_sha256"]
            and sha(evidence["validation"]) == evidence["validation_sha256"],
            "standard reference adapter changed")
    words = np.load(evidence["reference"], allow_pickle=False)
    require(words.dtype == np.float32 and words.size == 1000
            and hashlib.sha256(words.astype("<f4", copy=False).tobytes()).hexdigest() == raw == evidence["sha256"],
            "all original float words differ")
    require(release["numeric_gate"] == {"atol": 0, "rtol": 0}
            and release["control_job"] == stock["control"]["job"] == 2066
            and release["control_cycles"] == stock["control"]["cycles"] == 30169093,
            "original exact gate or actual control differs")
    require(1 - cycles / release["control_cycles"] == stock["cycle_reduction_fraction"],
            "reported measured whole improvement differs")
    audit = audit_elf(Path(record["elf"]).read_bytes())
    require(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
            "all executable section instruction audit failed")
    staged = Path(record["actual_simulated_elf"])
    if staged.is_file():
        require(sha(staged) == record["elf_sha256"], "actual staged executable differs")
    reference_cycles = 22387449
    result = {
        "schema": "root_resnet_stock2068_four_panel_terminal_review_v1",
        "status": "champion_whole_model_stock_observation", "job_id": 2068,
        "stock_cycles": cycles, "control_job": 2066, "control_cycles": release["control_cycles"],
        "difference_cycles": cycles - release["control_cycles"],
        "reduction_percent": 100 * stock["cycle_reduction_fraction"],
        "permitted_reference_job": 1876, "permitted_reference_cycles": reference_cycles,
        "remaining_reference_gap_cycles": cycles - reference_cycles,
        "remaining_reference_gap_fraction": cycles / reference_cycles - 1,
        "terminal_pins_reclosed": len(stock["pins"]),
        "original1000_words_digest_independently_reparsed": True,
        "original_numeric_gate": release["numeric_gate"],
        "actual_staged_ELF_available_for_root_rehash": staged.is_file(),
        "nofsm_all_executable_sections": audit,
        "hardware_identity": {key: record[key] for key in ("hw_config", "hwdb_sha256", "bitstream_sha256")},
        "scope": "One whole observation per arm. Four private panels preserve14 exact products and required store/reload/publication fences; host/runtime/weights/other target objects unchanged. Section17.496% is not a whole forecast. Reference only from owned ZIP/1876 evidence.",
        "default_promotion": False,
        "pins": {str(path): sha(path) for path in
                 (packet_path, uart_path, release_path, Path(record["elf"]),
                  Path(evidence["reference"]), Path(evidence["validation"]), Path(__file__))},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "stock_cycles", "remaining_reference_gap_cycles")}))


if __name__ == "__main__":
    review()
