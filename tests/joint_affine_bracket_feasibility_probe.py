"""Screen source-derived shared affine brackets; no target selection or timing.

The explicit source manifest supplies each original ordered affine relation.
Every accepted readout family is independently proved over all signed-byte
pairs. Integer primitive work ranks guesses; it is not a cycle cost model.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from gsim_joint_affine_pair_probe import pin
from merlin.llvmlower.quantized_affine_bracket import derive_bracket
from merlin.llvmlower.quantized_affine_joint import nearby_candidates
from merlin.llvmlower.quantized_affine_pair import derive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--known-certificate", type=Path, action="append", default=[])
    parser.add_argument("--max-denominator", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_denominator <= 32767:
        parser.error("positive bounded denominator required")
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    known = []
    for path in args.known_certificate:
        proof = json.loads(path.read_text())
        assert proof == derive_bracket(**proof["source"], p=proof["p"], q=proof["q"])
        known.append(proof)
    rows, start = [], time.monotonic()
    payload = json.loads(args.source_manifest.read_text())
    for route in payload["routes"]:
        source, old = route["proof"]["source"], route["proof"]["coefficients"]
        control = derive(**source, **old)
        if control["mismatched_pairs"]:
            raise ValueError(
                "original control coefficients are not exact for the supplied source relation"
            )
        old_chunks = sum((old[key] + 126) // 127 for key in ("p", "q"))
        tried, accepted = 0, None
        for proof in known:
            if (
                source == proof["source"]
                and (proof["two_readouts_exact"] or proof["single_readout_exact"])
                and sum((proof[key] + 126) // 127 for key in ("p", "q")) < old_chunks
            ):
                accepted = proof
                break
        if accepted is None:
            pairs = {
                (row["p"], row["q"])
                for row in nearby_candidates(
                    **source, max_denominator=args.max_denominator
                )
            }
            ordered = sorted(
                pairs,
                key=lambda pair: (sum((v + 126) // 127 for v in pair), sum(pair), pair),
            )
            for p, q in ordered:
                chunks = (p + 126) // 127 + (q + 126) // 127
                if chunks >= old_chunks:
                    break
                tried += 1
                proof = derive_bracket(**source, p=p, q=q)
                if proof["two_readouts_exact"] or proof["single_readout_exact"]:
                    accepted = proof
                    break
        row = {
            "source_symbol_for_traceability_only": route["symbol"],
            "source": source,
            "source_extent": math.prod(route["shape"]),
            "old_exact_chunks": old_chunks,
            "bounded_denominator": args.max_denominator,
            "source_derived_coefficient_pairs_examined": tried,
            "candidate": None,
            "elapsed_s": time.monotonic() - start,
        }
        if accepted:
            # Derived order determines filenames; source identifiers do not
            # affect coefficient generation, ranking or proof eligibility.
            proof_path = work / f"proof{len(rows)}.json"
            proof_path.write_text(json.dumps(accepted, indent=2) + "\n")
            row["candidate"] = {
                "p": accepted["p"],
                "q": accepted["q"],
                "chunks": (accepted["p"] + 126) // 127 + (accepted["q"] + 126) // 127,
                "predictors": accepted["predictors"],
                "single_readout_exact": accepted["single_readout_exact"],
                "two_readouts_exact": accepted["two_readouts_exact"],
                "proof": pin(proof_path),
            }
            if accepted["joint_certificate"]:
                row["candidate"]["individual_mismatched_pairs"] = accepted[
                    "joint_certificate"
                ]["individual_mismatched_pairs"]
        else:
            row["status"] = (
                "No lower-chunk accepted corresponding-threshold candidate in bounded source-nearby search; no global impossibility claim."
            )
        rows.append(row)
        print(
            "SOURCE_BRACKET",
            route["symbol"],
            old_chunks,
            row["candidate"]["chunks"] if accepted else None,
            tried,
            flush=True,
        )
        record = {
            "schema": "source_scale_shared_bracket_feasibility_v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest": pin(args.source_manifest),
            "driver": pin(__file__),
            "known_certificates": [pin(path) for path in args.known_certificate],
            "bounded_coefficient_family": "For each bounded positive q, positive floor/ceil(source_lhs/source_rhs*q); integer primitive chunks rank guesses only; every candidate independently complete-domain certified.",
            "cost_scope": "Integer chunks only; two stores, decoder, private storage, target arithmetic, alias/lifetime and performance UNKNOWN until independently qualified.",
            "normal_selection_enabled": False,
            "rows": rows,
            "token_usage_available": False,
        }
        (work / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
