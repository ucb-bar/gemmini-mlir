"""Export typed source-view/resource proofs without changing model policies."""

import argparse
import hashlib
import json
from pathlib import Path

from merlin.common.mlir_query import parse
from merlin.llvmlower.segmented_matrix_view import prove_segmented_matrix
from xdsl.dialects.func import CallOp

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import GoldenGemm, Shape


def pin(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--physical-llvm", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = parse(args.prepared.read_text())
    module.verify()
    catalog = json.loads(args.catalog.read_text())
    bindings = {entry["symbol"]: entry for entry in catalog["fused_requantizations"]}
    matches = []
    refusals = []
    for op in module.walk():
        if not isinstance(op, CallOp) or len(op.operands) != 3 or len(op.results) != 1:
            continue
        try:
            proof = prove_segmented_matrix(op.operands[0])
        except ValueError:
            continue
        right = op.operands[1].type
        if not hasattr(right, "get_shape"):
            continue
        k, n = right.get_shape()
        if k != proof.address.cols:
            continue
        m = proof.address.rows
        binding = bindings[op.callee.root_reference.data]
        shape = Shape(**binding["schedule"])
        assert (shape.m, shape.n, shape.k) == (m, n, k)
        control = GoldenGemm(shape)
        if (
            proof.address.row_stride == proof.address.cols
            and proof.address.segment_stride
            == proof.address.segment_rows * proof.address.cols
        ):
            refusals.append(
                {
                    "source_callee": op.callee.root_reference.data,
                    "source_view": proof.to_dict(),
                    "refusal": "Contiguous logical matrix needs no segmented-input loader",
                }
            )
            continue
        try:
            candidate = GoldenGemm(
                control.shape,
                prefetch_b_rows=control.prefetch_b_rows,
                resident_a_load_tiles=control.resident_a_load_tiles,
                input_view=proof.address,
            )
        except ValueError as failure:
            refusals.append(
                {
                    "source_callee": op.callee.root_reference.data,
                    "source_view": proof.to_dict(),
                    "refusal": str(failure),
                }
            )
            continue
        emitted = candidate.build()
        emitted.verify()
        lower(emitted.clone()).verify()
        matches.append(
            {
                "source_callee": op.callee.root_reference.data,
                "owner_callee": proof.owner.owner.callee.root_reference.data
                if isinstance(proof.owner.owner, CallOp)
                else None,
                "source_view": proof.to_dict(),
                "shape": candidate.shape.__dict__,
                "selected_family": binding["schedule_kind"],
                "source_numeric_proof": binding["proof"],
                "resource_and_primitive_lowering": "PASS",
                "normal_model_binding": "NOT ENABLED: consumer declaration/owner allocation/lifetime/native oracle closure still required",
            }
        )
    result = {
        "schema": "typed_segmented_input_source_probe_v1",
        "pins": {
            "prepared": pin(args.prepared),
            "physical_llvm": pin(args.physical_llvm),
            "catalog": pin(args.catalog),
            "source_view_provider": pin(
                Path(prove_segmented_matrix.__code__.co_filename)
            ),
            "device_generator": pin(Path(GoldenGemm.__init__.__code__.co_filename)),
            "driver": pin(Path(__file__)),
        },
        "matches": matches,
        "refusals": refusals,
        "policy": "Matches derive from typed slice/reshape chains and resource facts; symbol names are audit identity only",
        "physical_evidence": "Recovery independently observed matching dense i8 residual allocation→projection i8 copy in the pinned physical LLVM; this driver does not structurally validate LLVM producer ownership",
        "numeric_contract": "No source arithmetic, source model, input, weights, accuracy gate or default policy changed",
        "target_cycles": "UNKNOWN: no original-model matched capsule or hardware measurement for this view option",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"matched_views": len(matches), "output": str(args.output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
