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


def prepare(source: Path, output: Path, *, prequant_gather: bool = False,
            integer_nonlinears: bool = False,
            alias_identity_im2col: bool = True) -> dict:
    try:
        from merlin.frontends.quant_ext import parse_quant_mlir
        from merlin.llvmlower.quant_passes import apply_quant
        from merlin.llvmlower.im2col_identity_view import rewrite_module as rewrite_identity_im2col
    except ImportError as exc:
        raise RuntimeError("Merlin Python package is required for upstream QDQ rewriting") from exc
    module = parse_quant_mlir(source)
    details: dict = {}
    selected_passes = ["contraction_int8"]
    if integer_nonlinears:
        selected_passes += ["softmax_int", "gelu_int", "silu_int", "rsqrt_int"]
    counts = apply_quant(module, passes=selected_passes,
                         named_contraction=True, prequant_gather=prequant_gather,
                         report_out=details)
    # Use Merlin's structural view pass after named contraction rewriting;
    # it now recognizes both generic and named matmul im2col chains. The OOT
    # parser below verifies the mixed i8*i8->i32 named form after serialization.
    alias_report = rewrite_identity_im2col(module).to_dict() if alias_identity_im2col else None
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
                   "prequant_gather": prequant_gather,
                   "integer_nonlinears": integer_nonlinears,
                   "alias_identity_im2col": alias_identity_im2col},
        "prequant_changes_integer_arithmetic": prequant_gather,
        "nonlinear_approximations_change_arithmetic": integer_nonlinears,
        "model_accuracy_check_required": True,
        "rewrite_counts": counts,
        "rewrite_report": details,
        "mesh_regions": len(wl.mesh_regions),
        "mesh_eligible_matmuls": eligible,
        "exact_batched_gemms": batched,
        "identity_im2col_alias_report": alias_report,
    }
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--prequant-gather", action="store_true")
    ap.add_argument("--integer-nonlinears", action="store_true",
                    help="try scoped integer exp/GELU/SiLU/rsqrt rewrites; requires accuracy check")
    ap.add_argument("--alias-identity-im2col", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="prove and remove exact 1x1 stride-one im2col copies (default: on)")
    args = ap.parse_args()
    print(json.dumps(prepare(args.input, args.output,
                             prequant_gather=args.prequant_gather,
                             integer_nonlinears=args.integer_nonlinears,
                             alias_identity_im2col=args.alias_identity_im2col), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
