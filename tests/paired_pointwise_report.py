"""Strict three-arm complete sibling/observer capsule protocol, never a model.

The expected artifact, immutable pointer tuple and original output digest are
sealed before hardware. Reported cycles cover the complete compound ROI;
Spike's counters are functional retirement diagnostics, not hardware labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fields(line):
    return dict(piece.split("=", 1) for piece in line.split()[1:])


def parse_report(console, expected):
    lines = console.splitlines()
    if any("PAIRED_FAIL" in line for line in lines):
        raise ValueError("compound source/guard refusal")
    for marker in (
        "PAIRED_SOURCE_GATE original45056i8 all5frm sticky7 guards sourcecontinuation",
        "PAIRED_PASS original90112i32 original45056i8 allguards immutableinputs rank0 DONE",
    ):
        if lines.count(marker) != 1:
            raise ValueError("missing or repeated complete source gate")
    pointers = [_fields(line) for line in lines if line.startswith("PAIRED_POINTERS ")]
    if pointers != [expected["pointers"]]:
        raise ValueError("all named common storage addresses must match sealed ELF")
    initial = [_fields(line) for line in lines if line.startswith("PAIRED_INITIAL ")]
    if initial != expected["initial"]:
        raise ValueError("initial original outputs or protocol differ")
    rows = [_fields(line) for line in lines if line.startswith("PAIRED_ROW ")]
    if len(rows) != 6:
        raise ValueError("six complete rows required")
    for row, reference in zip(rows, expected["strict_rows"], strict=True):
        if set(row) != set(reference):
            raise ValueError("row schema changed")
        if any(row[key] != reference[key] for key in row if key != "cycles"):
            raise ValueError("row order/instructions/source digest/protocol changed")
        if int(row["cycles"]) <= 0:
            raise ValueError("invalid complete counter window")
    means = {
        name: sum(int(r["cycles"]) for r in rows if int(r["id"]) == index) / 2
        for index, name in enumerate(("original_full", "serial_tiles", "overlap_tiles"))
    }
    original = means["original_full"]
    return {
        "schema": "complete_sibling_pointwise_three_arm_report_v1",
        "status": "pass",
        "scope": expected["scope"],
        "rows": rows,
        "mean_cycles": means,
        "relative_to_original_percent": {
            name: (value / original - 1) * 100 for name, value in means.items()
        },
        "all_named_common_storage": pointers[0],
        "whole_cycles": "UNKNOWN",
        "pure_array_or_physical_memory_attribution": "UNKNOWN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("console", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse_report(
        args.console.read_text(), json.loads(args.expected.read_text())
    )
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
