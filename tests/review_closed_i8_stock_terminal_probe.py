"""Read back the exact released complete-helper stock experiment."""

import argparse
import importlib.util
import json
from pathlib import Path

from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "terminal review output must be fresh")
    terminal = json.loads(args.terminal.read_text())
    for path, expected in terminal["pins"].items():
        require(sha(path) == expected, "changed terminal dependency: " + path)
    record = terminal["record"]
    release_path = Path(record["root_release"])
    require(sha(release_path) == record["root_release_sha256"], "root release changed")
    release = json.loads(release_path.read_text())
    require(record["job_id"] == 2067 and record["state"] == record["phase"] == "DONE"
            and record["exit_code"] == 0 and record["elf_sha256"] == release["candidate"]["sha256"],
            "released stock job identity or terminal differs")
    for key, value in release["hardware_identity_required"].items():
        require(record[key] == value, "actual stock hardware differs: " + key)
    for key, digest_key in (("staged_elf_receipt", "elf_sha256"),
                            ("staged_bitstream_receipt", "bitstream_sha256")):
        staged = json.loads(Path(record[key]).read_text())
        require(staged["job_id"] == record["job_id"] and staged[digest_key] == record[digest_key],
                "actual staged hardware/executable observation differs")
    adapter = json.loads(Path(release["custom_expected_adapter"]).read_text())
    declaration = json.loads(Path(adapter["declaration_path"]).read_text())
    expected = Path(adapter["reference_path"]).read_bytes()
    spec = importlib.util.spec_from_file_location("root_stock_i8_parser", adapter["parser_module_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    uart_path = next(Path(path) for path, digest in terminal["pins"].items()
                     if path.endswith("_uart.txt") and digest == record["uart_sha256"])
    console = uart_path.read_text()
    report = module.parse_report(console, declaration, expected)
    require(report == record["report"], "raw UART original byte/counter contract differs")
    require("*** PASSED ***" in console and 'COMMAND_EXIT_CODE="0"' in console,
            "actual simulator completion failed")
    strict = release["strict_retirement_only_observation"]
    require([row["instructions"] for row in report["rows"]] ==
            [row["instructions"] for row in strict["rows"]], "matched compiled work differs")
    result = {
        "schema": "root_closed_i8_stock2067_terminal_review_v1",
        "status": "VERIFIED_COMPLETE_M8_STOCK_WIN", "job_id": 2067,
        "terminal_pins_reclosed": len(terminal["pins"]),
        "raw_UART_original_words_and_ABBA_reparsed": True,
        "report": report, "fraction_lower": -report["fraction_change"],
        "actual_staged_ELF_bitstream_receipts_reclosed": True,
        "staging_scope": "Root rehashes immutable source ELF, raw UART and collector observations taken before teardown; actual staged paths are not required to survive.",
        "hardware_identity": {key: record[key] for key in release["hardware_identity_required"]},
        "whole_hardware_cycles": "UNKNOWN; current2062 champion unchanged",
        "whole_cycle_prediction": "UNKNOWN", "new_label_fitted": False,
        "scope": "One frozen four-window complete helper ABBA measurement; all original45056 byte checks/guards outsideROI. Enables successor whole build qualification, not whole performance/default promotion.",
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.terminal, release_path, uart_path, Path(release["custom_expected_adapter"]), Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "job_id", "fraction_lower")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("terminal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
