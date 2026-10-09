"""Experiment: all-pair scale intervals for one f32 affine predictor.

The production source arithmetic is unchanged. A found candidate is independently
checked against every source pair; this experiment does not qualify hardware.
The interval argument requires exactly representable i32-to-f32 accumulators.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from merlin.llvmlower.quantized_affine_pair import source_table


def solve(source, max_q, seeds):
    expected = source_table(**source).reshape(-1)
    a = np.repeat(np.arange(-128, 128, dtype=np.int64), 256)
    b = np.tile(np.arange(-128, 128, dtype=np.int64), 256)
    low = np.full(128, -np.inf, dtype=np.float64)
    high = np.full(128, np.inf, dtype=np.float64)
    for output in range(128):
        if output:
            half = np.float32(output - 0.5)
            adjacent = np.nextafter(half, np.float32(np.inf if output & 1 else -np.inf))
            low[output] = (float(half) + float(adjacent)) * 0.5
        if output < 127:
            half = np.float32(output + 0.5)
            adjacent = np.nextafter(half, np.float32(-np.inf if output & 1 else np.inf))
            high[output] = (float(half) + float(adjacent)) * 0.5
    lower, upper = low[expected], high[expected]
    positive_expected = expected > 0
    ratio = source['lhs_scale'] / source['rhs_scale']
    found, tried, best_gap = [], 0, None
    # An independent exact existing predictor closes interval semantics first.
    for q in range(1, max_q + 1):
        midpoint = round(q * ratio)
        for p in range(max(1, midpoint - 1), midpoint + 2):
            for seed in seeds:
                tried += 1
                acc = p * a + q * b + seed
                if np.max(np.abs(acc)) >= 1 << 24:
                    continue
                if np.any(positive_expected & (acc <= 0)):
                    continue
                valid = acc > 0
                quot_low = np.where(positive_expected, lower, 0.) / np.where(valid, acc, 1)
                quot_high = np.where(valid, upper, np.inf) / np.where(valid, acc, 1)
                lo_index, hi_index = int(np.argmax(quot_low)), int(np.argmin(quot_high))
                lo, hi = float(quot_low[lo_index]), float(quot_high[hi_index])
                gap = (lo - hi) / max(abs(lo), abs(hi), np.finfo(float).tiny)
                if best_gap is None or gap < best_gap['relative_gap']:
                    best_gap = dict(p=p, q=q, seed=seed, relative_gap=gap, lo=lo, hi=hi)
                # One f32 ulp margin admits boundary candidates without claiming
                # float64 extrema themselves establish mathematical refusal.
                start, stop = np.float32(lo), np.float32(hi)
                if not np.isfinite(start) or not np.isfinite(stop):
                    continue
                start = np.nextafter(start, np.float32(-np.inf))
                stop = np.nextafter(stop, np.float32(np.inf))
                if start > stop:
                    continue
                candidates, value = [], start
                while value <= stop and len(candidates) < 32:
                    candidates.append(float(value))
                    value = np.nextafter(value, np.float32(np.inf))
                for scale in candidates:
                    if scale <= 0:
                        continue
                    predicted = np.clip(np.rint(acc.astype(np.float32) * np.float32(scale)), 0, 127).astype(np.int8)
                    if np.array_equal(predicted, expected):
                        row = dict(p=p, q=q, seed=seed, scale=scale, scale_hex=scale.hex(),
                            all_65536_source_outputs_exact=True,
                            predictor_table_sha256=hashlib.sha256(predicted.tobytes()).hexdigest(),
                            full_i32_accumulator_range=[int(acc.min()), int(acc.max())],
                            chunks=math.ceil(p / 127) + math.ceil(q / 127),
                            lower_active_pair=[int(a[lo_index]), int(b[lo_index])],
                            upper_active_pair=[int(a[hi_index]), int(b[hi_index])])
                        found.append(row)
                        break
                if found and found[-1]['p'] == p and found[-1]['q'] == q and found[-1]['seed'] == seed:
                    print(json.dumps(found[-1]), flush=True)
        # Source search objective is minimal emitted signed-i8 weight chunks,
        # not smallest coefficients or a workload-specific preferred pair.
        if found and q > 127 and min(x['chunks'] for x in found) <= math.ceil(q / 127):
            break
    return dict(source=source, tried=tried, max_q=max_q, seeds=seeds, found=found,
        best_gap=best_gap, source_table_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        scope='General affine family numerical experiment; every reported winner full-domain checked, hardware not yet validated',
        no_solution_claim='Search failure is not a proof against all possible algorithms or all coefficient ratios')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--certificate', type=Path, required=True)
    parser.add_argument('--max-q', type=int, default=512)
    parser.add_argument('--seed-radius', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    certificate = json.loads(args.certificate.read_text())
    started = time.monotonic()
    result = solve(certificate['source'], args.max_q, list(range(-args.seed_radius, args.seed_radius + 1)))
    result.update(elapsed_seconds=time.monotonic()-started,
        original_certificate_sha256=hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'found'}), flush=True)


if __name__ == '__main__':
    main()
