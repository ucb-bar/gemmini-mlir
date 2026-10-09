"""Strict manifest-bound UART protocol for the original ranked residual pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse(stdout, *, arm, elements):
    if (
        type(arm) is not int
        or arm not in (0, 1)
        or type(elements) is not int
        or elements <= 0
    ):
        raise ValueError("explicit arm0/1 and positive original element count required")
    lines = stdout.replace("\r", "").splitlines()
    if any("FAIL" in line or "FAILED" in line for line in lines):
        raise ValueError("device/source/guard runtime failure")
    counters = [line for line in lines if line.startswith("RECTIFIED_RANKED_COUNTER ")]
    markers = [line for line in lines if line.startswith("RECTIFIED_RANKED_PASS ")]
    if len(counters) != 1 or len(markers) != 1:
        raise ValueError("exactly one counter and complete ranked marker required")
    fields = counters[0].split()[1:]
    if len(fields) != 5 or any(field.count("=") != 1 for field in fields):
        raise ValueError("malformed counter protocol")
    pairs = [field.split("=") for field in fields]
    if len({key for key, value in pairs}) != 5:
        raise ValueError("duplicate counter fields")
    row = {key: int(value) for key, value in pairs}
    if set(row) != {"arm", "cycles", "instructions", "before", "after"}:
        raise ValueError("unexpected counter fields")
    if row["arm"] != arm or min(row["cycles"], row["instructions"]) <= 0:
        raise ValueError("wrong arm or invalid timed window")
    if row["before"] != 0 or row["after"] != 0:
        raise ValueError("original declared floating flag context changed")
    expected = f"RECTIFIED_RANKED_PASS arm{arm} all{elements} guards4096 inputs{2 * elements} descriptors flags"
    if markers[0] != expected:
        raise ValueError(
            "original complete output/input/descriptor/guard marker changed"
        )
    return {
        "passed": True,
        "counters": row,
        "elements": elements,
        "marker": expected,
        "scope": "full producer and ranked adapter; all source outputs/inputs/descriptors/output guards/flags checked outside ROI",
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stdout", type=Path, required=True)
    cli.add_argument("--arm", type=int, required=True)
    cli.add_argument("--elements", type=int, required=True)
    args = cli.parse_args()
    print(
        json.dumps(
            parse(args.stdout.read_text(), arm=args.arm, elements=args.elements),
            indent=2,
        )
    )
