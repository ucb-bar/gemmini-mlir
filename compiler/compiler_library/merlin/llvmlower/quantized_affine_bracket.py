"""Complete source-derived f32 scale brackets for one integer affine producer.

This is a corresponding-output-threshold family, not a global optimizer over
arbitrary predictor tuples. Both stores must satisfy their complete independently
qualified contracts; shared hardware arithmetic/lifetimes remain provider facts.
"""

import hashlib
import json

import numpy as np

from .quantized_affine_joint import derive_joint
from .quantized_affine_pair import _scales, predictor_table, source_table


def _readout(z, bits, relu):
    scale = np.array(bits, np.uint32).view(np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        value = np.rint(np.float32(np.float32(z) * scale))
    return int(np.clip(value, 0 if relu else -128, 127))


def _allowed_bits(z, threshold, *, greater_equal, relu):
    """Inclusive finite positive-f32 interval for one monotone readout predicate."""
    first, last = 1, 0x7F7FFFFF

    def predicate(bits):
        output = _readout(z, bits, relu)
        return output >= threshold if greater_equal else output <= threshold

    begin, end = predicate(first), predicate(last)
    if begin == end:
        return (first, last) if begin else None
    lo, hi = first, last
    while lo < hi:
        middle = (lo + hi) // 2
        if predicate(middle) == begin:
            lo = middle + 1
        else:
            hi = middle
    return (lo, last) if end else (first, lo - 1)


def derive_bracket(lhs_scale, rhs_scale, output_scale, *, p, q, relu=False):
    """Derive scales from every reachable source transition, refusing raw collisions.

    Greedy interval stabbing proves the minimum number of scales within the
    corresponding-threshold family. A separate complete joint certificate,
    rather than interval coverage alone, grants a two-readout decoder.
    """
    source, reciprocal = _scales(lhs_scale, rhs_scale, output_scale, relu)
    with np.errstate(over="ignore", under="ignore"):
        extent = np.float32(np.float32(128 * source["lhs_scale"]) + np.float32(128 * source["rhs_scale"]))
        extent = np.float32(extent * np.float32(reciprocal))
    if not np.isfinite(extent):
        raise ValueError("finite ordered binary32 source intermediates required over the complete domain")
    predictor_table(p, q, 1.0, relu=relu)  # Existing positive/i32 coefficient contract.
    lhs = np.repeat(np.arange(-128, 128, dtype=np.int64), 256)
    rhs = np.tile(np.arange(-128, 128, dtype=np.int64), 256)
    z = p * lhs + q * rhs
    expected = source_table(**source).ravel()
    keys, inverse = np.unique(z, return_inverse=True)
    low, high = np.full(len(keys), 128, np.int16), np.full(len(keys), -129, np.int16)
    np.minimum.at(low, inverse, expected)
    np.maximum.at(high, inverse, expected)
    conflicts = np.flatnonzero(low < high)
    witnesses = []
    for key in conflicts[:8]:
        positions = np.flatnonzero(inverse == key)
        pair = [int(positions[np.flatnonzero(expected[positions] == y)[0]]) for y in (low[key], high[key])]
        witnesses.append(
            {
                "sum": int(keys[key]),
                "pairs": [
                    {"lhs": int(lhs[index]), "rhs": int(rhs[index]), "output": int(expected[index])} for index in pair
                ],
            }
        )
    monotone = not len(conflicts) and bool(np.all(low[1:] >= low[:-1]))
    intervals, chosen, disjoint, minimum = [], [], [], None
    refusal = "raw affine sum requires different source outputs" if len(conflicts) else None
    if not len(conflicts) and not monotone:
        refusal = "source output is not monotone in the selected affine sum"
    if monotone:
        for output in range(int(low.min()), int(low.max())):
            last = int(keys[low <= output].max())
            first = int(keys[low >= output + 1].min())
            lower = _allowed_bits(first, output + 1, greater_equal=True, relu=relu)
            upper = _allowed_bits(last, output, greater_equal=False, relu=relu)
            if lower is None or upper is None or max(lower[0], upper[0]) > min(lower[1], upper[1]):
                refusal = "no f32 readout can represent a corresponding source transition"
                break
            intervals.append(
                {
                    "output": output,
                    "last_sum": last,
                    "first_sum": first,
                    "scale_bits_min": max(lower[0], upper[0]),
                    "scale_bits_max": min(lower[1], upper[1]),
                }
            )
        if refusal is None:
            remaining = intervals[:]
            while remaining:
                row = min(remaining, key=lambda item: item["scale_bits_max"])
                bits = row["scale_bits_max"]
                chosen.append(bits)
                disjoint.append(row)
                remaining = [item for item in remaining if not item["scale_bits_min"] <= bits <= item["scale_bits_max"]]
            minimum = len(chosen)
            if not chosen:  # Zero is the only constant affine-source output (A=B=0).
                chosen = [1]
            if len(chosen) > 2:
                refusal = "more than two corresponding-threshold f32 scales required"
    predictors = [
        {"p": p, "q": q, "scale": float(np.array(bits, np.uint32).view(np.float32)), "relu": relu} for bits in chosen
    ]
    joint = derive_joint(**source, predictors=predictors) if refusal is None and len(predictors) == 2 else None
    single_exact = bool(
        refusal is None and len(predictors) == 1 and np.array_equal(predictor_table(**predictors[0]).ravel(), expected)
    )
    if joint is not None and not joint["decoder_exact_for_all_pairs"]:
        refusal = "complete joint tuple reconstruction refuses despite threshold coverage"
    if refusal is None and len(predictors) == 1 and not single_exact:
        refusal = "complete single readout reconstruction refuses despite threshold coverage"
    interval_bytes = json.dumps(intervals, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "quantized_affine_corresponding_threshold_bracket_v1",
        "source": source,
        "reciprocal": reciprocal,
        "p": p,
        "q": q,
        "pairs": 65536,
        "source_table_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "raw_sum_table_sha256": hashlib.sha256(z.astype("<i4").tobytes()).hexdigest(),
        "reachable_sums": len(keys),
        "raw_sum_min": int(z.min()),
        "raw_sum_max": int(z.max()),
        "raw_conflicting_sums": len(conflicts),
        "raw_collision_witnesses": witnesses,
        "source_output_monotone": monotone,
        "intervals": intervals,
        "interval_sha256": hashlib.sha256(interval_bytes).hexdigest(),
        "minimum_scales_in_corresponding_threshold_family": minimum,
        "disjoint_interval_witnesses": disjoint,
        "predictors": predictors,
        "single_readout_exact": single_exact,
        "two_readouts_exact": bool(joint and joint["decoder_exact_for_all_pairs"]),
        "joint_certificate": joint,
        "refusal": refusal,
        "scope": (
            "Complete signed-byte source domain; minimal interval cover only for corresponding "
            "hardware output thresholds, not all possible predictor families"
        ),
        "backend_implementation": (
            "UNKNOWN; both target readouts and common-producer lifetimes require independent closure"
        ),
    }
