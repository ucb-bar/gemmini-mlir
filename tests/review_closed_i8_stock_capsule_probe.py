"""Review one immutable complete-helper stock cost capsule before submission."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            state.update(chunk)
    return state.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(args):
    require(not args.output.exists(), "release output must be fresh")
    packet = json.loads(args.packet.read_text())
    for path, expected in packet["pins"].items():
        require(sha(path) == expected, "changed capsule artifact: " + path)
    adapter_path = Path(packet["expected_adapter_path"])
    require(sha(adapter_path) == packet["expected_adapter_sha256"], "adapter changed")
    adapter = json.loads(adapter_path.read_text())
    pairs = (
        ("elf", "elf_sha256"), ("parser_module_path", "parser_sha256"),
        ("declaration_path", "declaration_sha256"), ("reference_path", "reference_sha256"),
        ("source_reference_npy_path", "source_reference_npy_sha256"),
        ("strict_console_path", "strict_console_sha256"),
        ("qualification_path", "qualification_sha256"),
        ("full35_target_source_environment_path", "full35_target_source_environment_sha256"),
        ("numerical_preparation_packet_path", "numerical_preparation_packet_sha256"),
    )
    for path_key, sha_key in pairs:
        require(sha(adapter[path_key]) == adapter[sha_key], "changed adapter dependency: " + path_key)
    require(adapter["elf_sha256"] == packet["timing_elf_sha256"] ==
            "b0635db509ff33c99e396b60dc94741e1d1350fc7f5db7f41a80db477fc7b6ad",
            "immutable complete timing ELF differs")
    expected = Path(adapter["reference_path"]).read_bytes()
    source = np.load(adapter["source_reference_npy_path"])
    require(source.dtype == np.int8 and source.size == 45056 and source.tobytes() == expected,
            "original complete source integer words differ")
    declaration = json.loads(Path(adapter["declaration_path"]).read_text())
    require(declaration["original_i8_words"] == 45056 and
            declaration["elf_sha256"] == adapter["elf_sha256"] and
            declaration["no_whole_speedup_inference"], "declaration scope differs")
    spec = importlib.util.spec_from_file_location("root_reviewed_i8_capsule", adapter["parser_module_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strict_report = module.parse_report(Path(adapter["strict_console_path"]).read_text(),
                                        declaration, expected)
    require(strict_report["original_digest"] == 0xb702909ca80dfe2a,
            "original fixture digest differs")
    # Reclose exact timing and all-environment qualification objects independently.
    for key in ("qualification_path", "full35_target_source_environment_path"):
        qualification = json.loads(Path(adapter[key]).read_text())
        require(qualification["status"] == "pass", "complete target qualification failed")
        for path, expected_sha in qualification.get("pins", {}).items():
            require(sha(path) == expected_sha, "changed target qualification dependency: " + path)
        for path, expected_sha in qualification.get("objects", {}).items():
            require(sha(path) == expected_sha, "changed actual linked object: " + path)
    audit = audit_elf(Path(adapter["elf"]).read_bytes())
    require(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
            "all executable section instruction audit failed")
    preparation = json.loads(args.preparation_review.read_text())
    require(preparation["independent_tests"] == 50 and preparation["actual_contexts"] == 22
            and preparation["original_whole_native_words"] == 256000,
            "source/native preparation review differs")
    require("10 passed" in args.parser_tests.read_text(), "independent protocol tests failed")
    result = {
        "schema": "root_closed_i8_complete_M8_stock_cost_release_v1",
        "status": "RELEASED_ONE_STOCK_COST_CAPSULE",
        "candidate": {"path": adapter["elf"], "sha256": adapter["elf_sha256"]},
        "custom_expected_adapter": str(adapter_path),
        "declaration": adapter["declaration_path"],
        "packet_pins_reclosed": len(packet["pins"]),
        "original_complete_integer_words": 45056, "parser_tests": 10,
        "original_native_whole_words": 256000,
        "all5frm7sticky_companion": adapter["full35_target_source_environment_path"],
        "all_executable_zero_FSM": audit,
        "strict_retirement_only_observation": strict_report,
        "hardware_config": "alveo_u250_firesim_gemmini_rocket_stock",
        "hardware_identity_required": {
            "hw_config": "alveo_u250_firesim_gemmini_rocket_stock",
            "hwdb_sha256": "5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126",
            "bitstream_sha256": "6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229",
        },
        "requirements": [
            "Recovery alone submits after this release is committed; retain queue/staging/workload completion receipts.",
            "Check actual staged ELF, hardware config, bitstream and job-owned UART/DONE/rank0.",
            "Parse exactly four complete ABBA windows and original45056-byte digest/guards/inputs with frozen adapter.",
            "Keep live GSIM unchanged; this stock observation is an independent engine label.",
        ],
        "scope": "Complete helper dequant/table/certificate/original source fallback/finish/store/frame. Validation outside each ROI. No whole target ELF admitted.",
        "prospective_cycle_model": "UNKNOWN; interval, frame, memory and cross-call costs unpriced",
        "whole_cycle_forecast": "UNKNOWN", "whole_champion_changes": False,
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.packet, adapter_path, args.preparation_review, args.parser_tests, Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "packet_pins_reclosed", "original_complete_integer_words")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "preparation-review", "parser-tests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
