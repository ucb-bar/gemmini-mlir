"""Record a comparable, numerically verified GSIM kernel optimization.

This small machine-readable ledger keeps the later automatic tuner from
learning from different shapes, engines, failed runs, or unaudited ELFs.
FireSim whole-model and class results need a separate measurement window key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _checked(path: Path) -> dict:
    result = json.loads(path.read_text())
    audit_path = path.parent / "nofsm_audit.json"
    audit = json.loads(audit_path.read_text())
    if result.get("status") != "pass" or result.get("kernel_cycles") is None:
        raise ValueError(f"probe did not pass: {path}")
    if audit.get("status") != "pass":
        raise ValueError(f"linked ELF did not pass the no-FSM audit: {audit_path}")
    if not result.get("gsim_engine_sha256") or not result.get("elf_sha256"):
        raise ValueError(f"probe is missing pinned engine or ELF identity: {path}")
    if audit.get("elf_sha256") != result["elf_sha256"]:
        raise ValueError(f"no-FSM audit belongs to a different ELF: {audit_path}")
    return {"result_path": str(path.resolve()),
            "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "elf_sha256": result["elf_sha256"],
            "object_sha256": result.get("object_sha256"),
            "cycles": result["kernel_cycles"],
            "status": result["status"]}


def compare(baseline_path: Path, candidate_path: Path, change: str) -> dict:
    base_raw = json.loads(baseline_path.read_text())
    cand_raw = json.loads(candidate_path.read_text())
    for key in ("shape", "gsim_engine_sha256"):
        if base_raw.get(key) != cand_raw.get(key):
            raise ValueError(f"cannot compare probes with different {key}")
    baseline = _checked(baseline_path)
    candidate = _checked(candidate_path)
    return {"schema": "golden_gsim_optimization_record_v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "change": change,
            "measurement_window": "rdcycle_around_kernel_call",
            "shape": base_raw["shape"],
            "engine_sha256": base_raw["gsim_engine_sha256"],
            "baseline": baseline, "candidate": candidate,
            "cycle_delta": candidate["cycles"] - baseline["cycles"],
            "speedup": baseline["cycles"] / candidate["cycles"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--change", required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    record = compare(args.baseline, args.candidate, args.change)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"cycle_delta": record["cycle_delta"],
                      "speedup": record["speedup"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
