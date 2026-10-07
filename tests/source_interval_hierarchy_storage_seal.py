"""Close a no-interwindow-table-scan successor, retaining the first packet."""

from __future__ import annotations

import json
from pathlib import Path

import source_interval_hierarchy_census as census
import source_interval_hierarchy_storage_v2 as successor
from merlin.perf.layer_bench import build_program

from mlir_oot.no_fsm_audit import audit_elf

FIRST = (
    census.HERE / "docs/perf_records/source_interval_hierarchy_M8_20261007/receipt.json"
)
OUT = census.HERE / "docs/perf_records/source_interval_hierarchy_M8_storage_v2_20261007"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    base = json.loads(FIRST.read_text())
    assert all(census.sha(p) == digest for p, digest in base["pins"].items())
    original_q = json.loads((successor.OUT / "qualification.json").read_text())
    parsed = successor.parse_report((successor.OUT / "spike.stdout").read_text())
    symbols = (successor.OUT / "symbols.stdout").read_text().splitlines()
    actual_symbols = []
    for record in original_q["symbols"]:
        fields = next(
            line.split() for line in symbols if line.split()[-1] == record["name"]
        )
        actual_symbols.append(
            {
                "name": fields[-1],
                "address": fields[0],
                "bytes": int(fields[1], 16),
                "kind": fields[2],
            }
        )
    final_q = {
        **original_q,
        "symbols": actual_symbols,
        "symbol_metadata_reclosure": "First successor qualification reused previous ELF symbol list while all actual runtime pointers were independently matched to new nm output. This adjunct binds the new actual symbol addresses; original observation remains unchanged.",
    }
    census.save(OUT / "qualification.json", final_q)
    helper_before = {
        p: h for p, h in original_q["objects"].items() if not p.endswith("/main.o")
    }
    assert helper_before == {
        p: h
        for p, h in json.loads(
            (successor.first.OUT / "qualification.json").read_text()
        )["objects"].items()
        if not p.endswith("/main.o")
    }
    assert all(census.sha(p) == h for p, h in original_q["objects"].items())
    rebuilt = build_program(
        [Path(p) for p in original_q["objects"]],
        OUT / "relink",
        target="gemmini",
        max_loaded_bytes=None,
    )
    assert census.sha(rebuilt.elf) == original_q["elf_sha256"]
    assert audit_elf(rebuilt.elf.read_bytes())["status"] == "pass"
    mutated = (
        (successor.OUT / "spike.stdout")
        .read_text()
        .replace("INTEGER_RESULT_TABLE_HASH_PASS", "INCOMPLETE")
    )
    try:
        successor.parse_report(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError("missing post-ROI table closure accepted")
    pins = dict(base["pins"])
    for directory in [successor.OUT, OUT]:
        for p in directory.rglob("*"):
            if p.is_file():
                pins[str(p)] = census.sha(p)
    for p in [
        FIRST,
        census.HERE / "tests/source_interval_hierarchy_storage_v2.py",
        Path(__file__),
    ]:
        pins[str(p)] = census.sha(p)
    receipt = {
        **{
            k: v
            for k, v in base.items()
            if k
            not in {
                "pins",
                "actual_elf",
                "actual_elf_sha256",
                "parser_path",
                "parser_function",
                "expected_storage",
                "named_storage_and_tables",
                "before_after",
                "cold_warm_scope",
            }
        },
        "schema": "source_interval_hierarchy_storage_v2_release_v1",
        "original_300_pin_packet": str(FIRST),
        "original_packet_sha256": census.sha(FIRST),
        "actual_elf": original_q["elf_path"],
        "actual_elf_sha256": original_q["elf_sha256"],
        "parser_path": str(
            census.HERE / "tests/source_interval_hierarchy_storage_v2.py"
        ),
        "parser_function": "parse_report",
        "expected_storage": parsed["named_storage"][0],
        "named_storage_and_tables": actual_symbols,
        "before_after": {
            "compiled_retired_control_mean": 1726434,
            "compiled_retired_candidate_mean": 1764719,
            "retired_delta_percent": (1764719 / 1726434 - 1) * 100,
            "hardware_cycles": "UNKNOWN",
            "whole_cycles": "UNKNOWN",
        },
        "cold_warm_scope": original_q["cold_warm_scope"],
        "measurement_harness_change": "No full-table hash before/between ABBA windows; source-derived expected byte hashes materialized as integer constants and actual table bytes checked after all four windows. Named address records remain outside each ROI. Initial candidate source gate retained.",
        "helper_control_candidate_and_table_objects_unchanged": True,
        "default_or_source_numeric_policy_change": False,
        "first_packet_execution": "Root reports2096 startup cancelled after first harness residency concern; no validated hardware timing or profitability from that attempt.",
        "actual_symbols_adjunct": str(OUT / "qualification.json"),
        "fresh_relink_ELF_byteexact": True,
        "pins": pins,
    }
    census.save(OUT / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "receipt": str(OUT / "receipt.json"),
                "sha256": census.sha(OUT / "receipt.json"),
                "pins": len(pins),
                "elf": original_q["elf_path"],
                "elf_sha256": original_q["elf_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
