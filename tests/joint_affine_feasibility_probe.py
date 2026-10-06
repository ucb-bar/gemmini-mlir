"""Source-derived bounded joint predictor search; no runtime policy selection.

Only complete certificates grant semantic feasibility. Ranking by diagonal
chunk count is target work bookkeeping, not an elapsed-cycle estimate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from merlin.common.digest import sha256_file
from merlin.llvmlower.quantized_affine_joint import derive_joint, nearby_candidates
from merlin.llvmlower.quantized_affine_pair import predictor_table, source_table

from mlir_oot.golden_wide_resadd import ResidualPrefetchPlan


def chunks(contract):
    # Constructor closes signed-i32 accumulation and scratchpad/ACC resources.
    plan = ResidualPrefetchPlan(32, contract["p"], contract["q"], True)
    return (plan.p + 126) // 127 + (plan.q + 126) // 127


def search(source, *, max_denominator):
    expected = source_table(**source)
    first_candidates = []
    for contract in nearby_candidates(**source, max_denominator=127):
        if max(contract["p"], contract["q"]) <= 127:
            errors = int(np.sum(predictor_table(**contract) != expected))
            first_candidates.append((errors, contract))
    if not first_candidates:
        raise ValueError("source ratio has no candidate in explicit two-chunk search")
    errors, first = min(
        first_candidates,
        key=lambda row: (row[0], row[1]["p"] + row[1]["q"], -row[1]["scale"]),
    )
    rows, accepted = [], []
    for second in nearby_candidates(**source, max_denominator=max_denominator):
        proof = derive_joint(**source, predictors=[first, second])
        row = {
            "predictor": second,
            "chunks": chunks(second),
            "conflicting_tuples": proof["conflicting_tuples"],
            "different_prediction_pairs": proof["different_prediction_pairs"],
        }
        rows.append(row)
        if proof["decoder_exact_for_all_pairs"]:
            accepted.append(proof)
    chosen = min(
        accepted,
        key=lambda proof: (
            chunks(proof["predictors"][1]),
            len(bytes.fromhex(proof["decoder_hex"])),
            proof["different_prediction_pairs"],
            sum(proof["predictors"][1][key] for key in ("p", "q")),
        ),
        default=None,
    )
    two_chunk = [row for row in rows if row["chunks"] == 2]
    return {
        "source": source,
        "first_predictor": first,
        "first_predictor_mismatched_pairs": errors,
        "first_candidates": len(first_candidates),
        "second_denominator_bound": max_denominator,
        "second_candidates": len(rows),
        "accepted_candidates": len(accepted),
        "two_chunk_candidates": len(two_chunk),
        "two_chunk_minimum_conflicting_tuples": min(
            (row["conflicting_tuples"] for row in two_chunk), default=None
        ),
        "selected_certificate": chosen,
        "candidates": rows,
        "ranking_scope": "Bounded source-derived search only; smallest existing diagonal chunk count then decoder size/disagreement count. No global optimality or cycle prediction",
        "backend_predictor_implementation": "UNKNOWN until independent full-domain target validation",
        "production_policy": "Unchanged; feasibility experiment does not enable an alternative",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--max-denominator", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_certificate.read_text())["source"]
    result = search(source, max_denominator=args.max_denominator)
    result.update(
        schema="gemmini_joint_affine_bounded_feasibility_v1",
        source_certificate={
            "path": str(args.source_certificate.resolve()),
            "sha256": sha256_file(args.source_certificate),
        },
        driver={"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        token_usage_available=False,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "JOINT_FEASIBILITY",
        result["accepted_candidates"],
        result["two_chunk_minimum_conflicting_tuples"],
        flush=True,
    )


if __name__ == "__main__":
    main()
