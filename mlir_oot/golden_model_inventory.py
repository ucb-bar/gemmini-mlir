"""Census exact rewritten integer GEMMs and their peak-mesh arithmetic floors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .contraction_patterns import match_integer_gemm
from .frontend.parse import parse_module
from .golden_gemm import _ceil_div
from .tables import rtl_facts as F


def inventory(source: Path) -> dict:
    raw = source.read_bytes()
    module = parse_module(raw.decode())
    families = Counter()
    macs = padded_array_cycles = computes = 0
    rows = []
    for op in module.walk():
        shape = match_integer_gemm(op)
        if shape is None:
            continue
        rank = len(op.operands[0].type.get_shape())
        family = "batched_3d" if rank == 3 else "matrix_2d"
        families[family] += 1
        work = shape.batch * shape.m * shape.n * shape.k
        ncompute = (shape.batch * _ceil_div(shape.m, F.DIM) *
                    _ceil_div(shape.n, F.DIM) * _ceil_div(shape.k, F.DIM))
        macs += work
        computes += ncompute
        padded_array_cycles += ncompute * F.DIM
        rows.append({"region": getattr(op.attributes.get("prov.region_id"), "data", ""),
                     "batch": shape.batch, "m": shape.m, "n": shape.n, "k": shape.k,
                     "macs": work, "compute_commands": ncompute})
    return {
        "schema": "golden_integer_model_inventory_v1",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "exact_contractions": sum(families.values()),
        "families": dict(families),
        "integer_macs": macs,
        "peak_array_math_floor_cycles": _ceil_div(macs, F.DIM * F.DIM),
        "compute_commands": computes,
        "padded_array_issue_floor_cycles": padded_array_cycles,
        "largest_contractions": sorted(rows, key=lambda x: x["macs"], reverse=True)[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()
    reports = {p.stem: inventory(p) for p in args.input}
    result = {"models": reports}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
