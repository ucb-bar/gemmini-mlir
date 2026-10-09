"""Aggregate observed primitive operands into Merlin's feature JSON pointers.

The isolated observer and its numeric/histogram qualification are the execution
producer. This reader checks transport/geometry/count invariants, not that a
file originated from a claimed engine. Caller-owned receipts bind that claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .tables import rtl_facts as F


def _integer(value, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive) or value >= 1 << 64:
        raise ValueError(f"invalid telemetry {name}")
    return value


def read_telemetry(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "gemmini_primitive_operand_telemetry_v1":
        raise ValueError("unknown primitive operand telemetry schema")
    if (data.get("dim"), data.get("elem_bytes"), data.get("acc_bytes")) != (
        F.DIM,
        1,
        4,
    ):
        raise ValueError("telemetry engine geometry differs from recorded target facts")
    mode = data.get("pc_aggregation")
    if mode not in ("exact_pc_geometry", "scope_geometry"):
        raise ValueError("telemetry PC aggregation mode must be explicit")
    total = _integer(data.get("total_commands"), "total commands", positive=True)
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("telemetry contains no complete primitive rows")
    commands = 0
    for row in rows:
        funct = _integer(row.get("funct"), "funct")
        name = F.FUNCT_NAMES.get(funct)
        if name is None or funct not in F.LEGAL_FUNCTS or name.startswith("LOOP_"):
            raise ValueError("telemetry includes an unsupported or FSM command")
        count = _integer(row.get("commands"), "commands", positive=True)
        commands += count
        first = _integer(row.get("first_event"), "first event", positive=True)
        last = _integer(row.get("last_event"), "last event", positive=True)
        if first > last or last > total or count > last - first + 1:
            raise ValueError("telemetry row command extent is incomplete")
        for key in (
            "pc",
            "scope_id",
            "invocation",
            "requested_load_bytes",
            "requested_store_bytes",
            "unknown_dma_commands",
            "padded_mac_slots",
            "padded_compute_rows",
        ):
            _integer(row.get(key), key)
        for key in ("a_rows", "a_cols", "b_rows", "b_cols"):
            if not 0 <= _integer(row.get(key), key) <= 65535:
                raise ValueError("telemetry command geometry exceeds encoded extent")
        if row["pc"] % 2 or (mode == "scope_geometry" and row["pc"] != 0):
            raise ValueError("telemetry PC violates its declared aggregation")
        if row["unknown_dma_commands"] > count:
            raise ValueError("unknown DMA commands exceed command count")
        compute_count = count if funct in (4, 5) else 0
        if (
            row["padded_mac_slots"] != compute_count * F.DIM**3
            or row["padded_compute_rows"] != compute_count * F.DIM
        ):
            raise ValueError("telemetry padded work disagrees with executed primitives")
    if commands != total:
        raise ValueError("telemetry command rows do not conserve total count")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise TypeError("telemetry scope entries are absent")
    previous = 0
    seen = Counter()
    for entry in entries:
        event = _integer(entry.get("event"), "entry event", positive=True)
        scope = _integer(entry.get("scope_id"), "entry scope", positive=True)
        invocation = _integer(
            entry.get("invocation"), "entry invocation", positive=True
        )
        pc = _integer(entry.get("pc"), "entry PC")
        seen[scope] += 1
        if event <= previous or event > total or invocation != seen[scope] or pc % 2:
            raise ValueError("telemetry scope entries are unordered or incomplete")
        previous = event
    for row in rows:
        if row["scope_id"] and row["invocation"] > seen[row["scope_id"]]:
            raise ValueError("telemetry rows refer to an unobserved invocation")
    return data


def summarize(
    data: dict, *, scope_id: int | None = None, invocation: int | None = None
) -> dict:
    """Select actual observer scopes; never split shared calls by assumed counts."""
    if invocation is not None and scope_id is None:
        raise ValueError("an invocation requires an explicit scope")
    if scope_id is not None:
        _integer(scope_id, "selected scope", positive=True)
    if invocation is not None:
        _integer(invocation, "selected invocation", positive=True)
    rows = [
        row
        for row in data["rows"]
        if (scope_id is None or row["scope_id"] == scope_id)
        and (invocation is None or row["invocation"] == invocation)
    ]
    if not rows:
        raise ValueError("selected telemetry scope has no commands")
    if scope_id is not None and any(row["invocation"] == 0 for row in rows):
        raise ValueError("selected commands precede their scope entry marker")
    commands = Counter(
        {name: 0 for funct, name in F.FUNCT_NAMES.items() if funct in F.LEGAL_FUNCTS}
    )
    geometry = Counter()
    for row in rows:
        commands[F.FUNCT_NAMES[row["funct"]]] += row["commands"]
        if row["funct"] in (4, 5):
            geometry[row["a_rows"]] += row["commands"]
    unknown = sum(row["unknown_dma_commands"] for row in rows)
    reads = sum(row["requested_load_bytes"] for row in rows)
    writes = sum(row["requested_store_bytes"] for row in rows)
    observed = {
        "status": "observed_functional_operand_execution",
        "producer_qualification": "Required external copied-engine numeric/histogram receipt",
    }
    return {
        "schema": "gemmini_observed_operand_features_v1",
        "features": {
            "primitive_commands": dict(sorted(commands.items())),
            "array_work": sum(row["padded_mac_slots"] for row in rows),
            "array_padded_compute_rows": sum(
                row["padded_compute_rows"] for row in rows
            ),
            "requested_dma_bytes": None if unknown else reads + writes,
            "requested_dma_load_bytes": None if unknown else reads,
            "requested_dma_store_bytes": None if unknown else writes,
        },
        "feature_status": {
            "/features/array_work": dict(
                observed,
                unit="nominal_padded_mac_slot",
                caveat="DIM^3 per executed compute; not nonzero MACs, physical activity or cycles",
            ),
            "/features/array_padded_compute_rows": dict(
                observed, unit="nominal_padded_compute_row"
            ),
            "/features/primitive_commands": dict(observed, unit="command"),
            "/features/requested_dma_bytes": dict(
                observed
                if not unknown
                else {
                    "status": "UNKNOWN",
                    "reason": "Unconfigured/normalization DMA requests present",
                },
                unit="requested_payload_byte",
                caveat="Not physical DRAM traffic",
            ),
            **{
                "/features/" + name: dict(
                    observed if not unknown else {"status": "UNKNOWN"},
                    unit="requested_payload_byte",
                    caveat="Not physical DRAM traffic",
                )
                for name in ("requested_dma_load_bytes", "requested_dma_store_bytes")
            },
            **{
                "/features/primitive_commands/" + name: dict(observed, unit="command")
                for name in commands
            },
        },
        "scope": {
            "scope_id": scope_id,
            "invocation": invocation,
            "first_event": min(row["first_event"] for row in rows),
            "last_event": max(row["last_event"] for row in rows),
            "source_or_hardware_interval_closure": "Requires source-bound external scope receipt",
        },
        "details": {
            "unknown_dma_commands": unknown,
            "compute_a_row_histogram": {str(k): v for k, v in sorted(geometry.items())},
        },
        "unknown_features": {
            "physical_dram_bytes": "UNKNOWN",
            "bank_hazards": "UNKNOWN",
            "accelerator_dispatch_overlap": "UNKNOWN",
            "hardware_cycles": "UNKNOWN",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--scope-id", type=int)
    parser.add_argument("--invocation", type=int)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        read_telemetry(args.telemetry),
        scope_id=args.scope_id,
        invocation=args.invocation,
    )
    result["artifacts"] = {
        "telemetry": {
            "path": str(args.telemetry.resolve()),
            "sha256": hashlib.sha256(args.telemetry.read_bytes()).hexdigest(),
        },
        "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
