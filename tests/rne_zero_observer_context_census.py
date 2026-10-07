"""Original all-context numerical work census, without cycle extrapolation."""

from __future__ import annotations

import numpy as np
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    find_closed_scalar_i8_observers,
)
from rne_zero_observer_capsule import OLD, OUT, TYPED, save, sha


def f32(word):
    return np.array([word], np.uint32).view(np.float32)[0]


def main():
    import json

    capture = OLD / "out/artifacts/probes/source-continuation-contexts-v3-20261007"
    features = json.loads((capture / "features_v2/contexts.json").read_text())
    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refused = find_closed_scalar_i8_observers(
        parse_mlir_text(TYPED.read_text()), effects=effects
    )
    assert len(proofs) == 22 and not refused
    table = build_source_interval_table(
        proofs[0].expression,
        effects=effects,
        leading_bits=16,
        max_table_bytes=512 * 1024,
    )
    intervals = np.frombuffer(table.data, "<f4").reshape(-1, 2)
    zero = prepare_bounded_rne_word_cells(proofs[0], effects=effects)
    half = zero.ranges[128][1] ^ 0x80000000
    rows, pins = [], {str(TYPED): sha(TYPED)}
    for row in features["rows"]:
        index = row["context"]
        directory = capture / f"context_{index:02d}"
        paths = [
            directory / (name + ".npy")
            for name in ("a", "scale_a", "b", "scale_b", "expected")
        ]
        a, sa, b, sb, expected = [np.load(p) for p in paths]
        pins.update({str(p): sha(p) for p in paths})
        constant, scale = (
            f32(row["preparation_factor_bits"]),
            f32(row["quant_factor_bits"]),
        )
        assert any(p.quant_factor_bits == row["quant_factor_bits"] for p in proofs)
        # Every original source cast and rounded FMUL is explicit. No gold
        # output affects decisions; expected is only an independent check.
        with np.errstate(all="ignore"):
            x = np.float32(np.float32(np.float32(a) * constant) * sa)
            up = np.float32(np.float32(np.float32(b) * constant) * sb)
            cells = intervals[x.view(np.uint32) >> np.uint32(16)]
            lo, hi = cells[..., 0], cells[..., 1]
            low_scaled = np.float32(np.float32(lo * up) * scale)
            high_scaled = np.float32(np.float32(hi * up) * scale)
        valid = lo <= hi
        magnitude = (
            low_scaled.view(np.uint32) | high_scaled.view(np.uint32)
        ) & np.uint32(0x7FFFFFFF)
        certified_zero = valid & (magnitude <= half)
        assert np.all(expected[certified_zero] == 0)
        finite = np.isfinite(low_scaled) & np.isfinite(high_scaled)
        low_q = np.rint(np.clip(low_scaled, -128, 127))
        high_q = np.rint(np.clip(high_scaled, -128, 127))
        numeric_same = valid & finite & (low_q == high_q)
        rows.append(
            {
                "context": index,
                "source_output_words": int(expected.size),
                "zero_predicate_admitted": int(certified_zero.sum()),
                "source_output_zeros_check_only": int((expected == 0).sum()),
                "refused_table_cells": int((~valid).sum()),
                "finite_endpoint_quantizer_disagreement": int(
                    (valid & finite & (low_q != high_q)).sum()
                ),
                "unknown_nonfinite_endpoints": int((valid & ~finite).sum()),
                "finite_same_quantization": int(numeric_same.sum()),
                "quantizer_calls_removed": int(certified_zero.sum()) * 2,
                "prices": "UNKNOWN",
            }
        )
    assert len(rows) == 22
    save(
        OUT / "all22_zero_context_census.json",
        {
            "schema": "exact_source_zero_observer_allcontext_work_v1",
            "rows": rows,
            "source_output_words": sum(r["source_output_words"] for r in rows),
            "zero_predicate_admitted": sum(r["zero_predicate_admitted"] for r in rows),
            "decision_source": "Only retained typed observer bin and current immutable source-derived bounds/rounded FMUL inputs. Original source expected words validate observations, never choose policy.",
            "physical_table_traffic_cache_and_whole_price": "UNKNOWN",
            "scope": "Cheap original operand execution-work census; no all22 compiled successor or whole candidate claim",
            "pins": pins,
            "token_usage_available": False,
        },
    )
    print(
        "ALL22_ZERO_NUMERIC_WORK",
        sum(r["zero_predicate_admitted"] for r in rows),
        flush=True,
    )


if __name__ == "__main__":
    main()
