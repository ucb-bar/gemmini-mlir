"""Prepare a quantized upstream model for the primitive Gemmini target.

The existing Merlin integer rewrite owns QDQ recognition and arithmetic.  This
adapter invokes it rather than reimplementing that pass, writes its exact IR,
then verifies and inventories the result with this OOT backend.  The optional
pre-gather rewrite changes quantization granularity and needs model accuracy
validation before a golden claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .frontend.linalg_reader import read
from .frontend.parse import parse_module
from .contraction_patterns import match_integer_gemm
from .lowering.model_lane import mesh_eligible


def prepare(source: Path, output: Path, *, prequant_gather: bool = False) -> dict:
    try:
        from merlin.frontends.quant_ext import parse_quant_mlir
        from merlin.llvmlower.quant_passes import apply_quant
    except ImportError as exc:
        raise RuntimeError("Merlin Python package is required for upstream QDQ rewriting") from exc
    module = parse_quant_mlir(source)
    details: dict = {}
    counts = apply_quant(module, passes=["contraction_int8"],
                         named_contraction=True, prequant_gather=prequant_gather,
                         report_out=details)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(module) + "\n")
    checked = parse_module(output.read_text())
    wl = read(checked)
    lane_of = {r.region_id: r.lane for r in wl.regions}
    eligible = sum(mesh_eligible(op, lane_of) for op in checked.walk()
                   if op.name in ("linalg.matmul", "linalg.generic"))
    batched = sum(1 for op in checked.walk()
                  if (match := match_integer_gemm(op)) is not None and
                  len(op.operands[0].type.get_shape()) == 3)
    receipt = {
        "schema": "gemmini_upstream_quant_prepare_v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "passes": {"contraction_int8": True, "named_contraction": True,
                   "prequant_gather": prequant_gather},
        "prequant_changes_integer_arithmetic": prequant_gather,
        "model_accuracy_check_required": True,
        "rewrite_counts": counts,
        "rewrite_report": details,
        "mesh_regions": len(wl.mesh_regions),
        "mesh_eligible_matmuls": eligible,
        "exact_batched_gemms": batched,
    }
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--prequant-gather", action="store_true")
    args = ap.parse_args()
    print(json.dumps(prepare(args.input, args.output,
                             prequant_gather=args.prequant_gather), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
