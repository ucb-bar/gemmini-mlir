"""Compile one exact upstream integer GEMM, including batched attention GEMM.

The receipt identifies the source operation and its dense pointer ABI.  This
does not bind the operation to a whole-model allocation or lower surrounding
host operations; that must be done by the model compiler before execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from .contraction_patterns import IntegerGemm, match_integer_gemm
from .frontend.parse import parse_module
from .golden_batched_gemm import build as build_batched
from .golden_device_compile import compile_module
from .golden_gemm import GoldenGemm, Shape, _ceil_div
from .golden_tuning import tune
from .tables import rtl_facts as F


def choose_shape(dims: IntegerGemm) -> Shape:
    """One shape-based schedule rule shared by selected and model-wide compilation."""
    shape, _ = tune(Shape(dims.m, dims.n, dims.k, output_dtype="i32"))
    shape = replace(shape, reuse_b=shape.bm > 1)
    if _ceil_div(dims.m, F.DIM) > shape.bm:
        try:
            candidate = replace(shape, cache_b=True)
            candidate.validate()
            shape = candidate
        except ValueError:
            pass
    for flag, eligible in (("wide_a", dims.k in (32, 48, 64)),
                           ("wide_b", dims.n in (32, 48, 64))):
        if eligible:
            try:
                candidate = replace(shape, **{flag: True})
                candidate.validate()
                shape = candidate
            except ValueError:
                pass
    shape.validate()
    return shape


def select(source: str, region_id: str) -> tuple[IntegerGemm, Shape, dict]:
    module = parse_module(source)
    selected = []
    for ordinal, op in enumerate(module.walk()):
        rid = getattr(op.attributes.get("prov.region_id"), "data", None)
        if rid != region_id or op.name not in ("linalg.matmul", "linalg.generic"):
            continue
        matched = match_integer_gemm(op)
        if matched is None:
            continue
        selected.append((ordinal, op, matched))
    if len(selected) != 1:
        raise ValueError(f"expected one integer GEMM in region {region_id!r}, found {len(selected)}")
    ordinal, op, dims = selected[0]
    shape = choose_shape(dims)
    binding = {
        "region_id": region_id,
        "source_operation_ordinal": ordinal,
        "source_operation": op.name,
        "abi": "gemmini_golden_batched_gemm" if len(op.operands[0].type.get_shape()) == 3
               else "gemmini_golden_gemm",
        "pointer_order": ["lhs", "rhs", "output"],
        "tensor_types": [str(v.type) for v in op.operands[:2]] + [str(op.results[0].type)],
        "dense_row_major_required": True,
        "zero_initialized_output_required": False,
    }
    return dims, shape, binding


def compile_selected(source_path: Path, region_id: str, llvm_bin: Path,
                     workdir: Path) -> dict:
    source = source_path.read_text()
    dims, shape, binding = select(source, region_id)
    module = (build_batched(dims.batch, shape) if binding["abi"] == "gemmini_golden_batched_gemm"
              else GoldenGemm(shape).build())
    receipt = compile_module(module, llvm_bin, workdir)
    result = {
        "schema": "golden_upstream_integer_gemm_v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "dimensions": asdict(dims),
        "schedule": asdict(shape),
        "binding": binding,
        "compilation": receipt,
        "whole_model_memory_bound": False,
        "whole_model_correctness_verified": False,
    }
    (workdir / "upstream_contraction_binding.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--region", required=True)
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    result = compile_selected(args.input, args.region, args.llvm_bin, args.workdir)
    print(json.dumps({"dimensions": result["dimensions"], "binding": result["binding"],
                      "object_sha256": result["compilation"]["object_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
