"""Analytical block search for the primitive-only golden GEMM.

The numbers are geometric bounds, not fitted FireSim predictions. MAC/256 is
the peak-array floor; unique tensor bytes/16 is an optimistic DRAM floor.
Requested panel bytes/16 describes a pessimistic no-cache transfer time and
must not be called a lower bound.  All must be compared with a complete-model
FireSim measurement before a cycle claim is made. The search minimizes
repeated A/B panel loads and CPU tile-loop dispatch within RTL capacities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from .golden_gemm import Shape, _ceil_div
from .tables import rtl_facts as F


def estimate(shape: Shape) -> dict:
    shape.validate()
    mt, nt, kt = (_ceil_div(d, F.DIM) for d in (shape.m, shape.n, shape.k))
    a_loads = b_loads = a_commands = b_commands = output_blocks = mvout_commands = 0
    for m0 in range(0, mt, shape.bm):
        am = min(shape.bm, mt - m0)
        for n0 in range(0, nt, shape.bn):
            bn = min(shape.bn, nt - n0)
            output_blocks += 1
            a_loads += am * kt
            a_commands += am if shape.wide_a else am * kt
            if not shape.cache_b:
                b_loads += bn * kt
                b_commands += kt if shape.wide_b else bn * kt
            if shape.wide_store and shape.output_dtype == "i8":
                full_cols = min(bn, max(0, shape.n // F.DIM - n0))
                mvout_commands += am * (_ceil_div(full_cols, 4) + bn - full_cols)
            else:
                mvout_commands += am * bn
    if shape.cache_b:
        b_loads = nt * kt
        b_commands = kt if shape.wide_b else nt * kt
    c_tiles = mt * nt
    computes = c_tiles * kt
    macs = shape.m * shape.n * shape.k
    # A load carries at most DIM^2 bytes, B likewise.  Exact edge bytes may be
    # lower, so this is a conservative DMA request-volume estimate.
    requested_bytes = (a_loads + b_loads) * F.DIM * F.DIM
    output_bytes = 4 if shape.output_dtype == "i32" else 1
    requested_bytes += shape.m * shape.n * output_bytes
    if shape.bias:
        requested_bytes += c_tiles * F.DIM * F.DIM * 4
    unique_bytes = (shape.m * shape.k + shape.k * shape.n +
                    shape.m * shape.n * output_bytes +
                    (shape.n * 4 if shape.bias else 0))
    commands = (a_commands + b_commands + 2 * computes + mvout_commands +
                (c_tiles if shape.bias else 0))
    return {
        "m": shape.m, "n": shape.n, "k": shape.k,
        "bm": shape.bm, "bn": shape.bn,
        "macs": macs,
        "array_floor_cycles": _ceil_div(macs, F.DIM * F.DIM),
        "mesh_compute_commands": computes,
        "padded_array_floor_cycles": computes * F.DIM,
        "dma_request_bytes_upper": requested_bytes,
        "dma_request_transfer_cycles_at_16B_upper": _ceil_div(requested_bytes, 16),
        "unique_tensor_bytes": unique_bytes,
        "unique_dram_floor_cycles_at_16B": _ceil_div(unique_bytes, 16),
        "primitive_command_count": commands,
        "output_blocks": output_blocks,
        "mvout_commands": mvout_commands,
        "a_panel_loads": a_loads,
        "b_panel_loads": b_loads,
        "a_mvin_commands": a_commands,
        "b_mvin_commands": b_commands,
    }


def tune(shape: Shape) -> tuple[Shape, dict]:
    mt, nt = _ceil_div(shape.m, F.DIM), _ceil_div(shape.n, F.DIM)
    acc_tiles = F.ACC_ROWS // F.DIM
    best = None
    for bm in range(1, min(mt, acc_tiles) + 1):
        for bn in range(1, min(nt, acc_tiles // bm) + 1):
            candidate = replace(shape, bm=bm, bn=bn)
            try:
                row = estimate(candidate)
            except ValueError:
                continue
            # Command issue and memory traffic both matter, but neither is a
            # calibrated cycle model.  Rank lexicographically to avoid a fake
            # scalar conversion from bytes to hardware time.
            key = (row["primitive_command_count"], row["dma_request_bytes_upper"],
                   row["output_blocks"], -bm * bn, abs(bm - bn))
            if best is None or key < best[0]:
                best = key, candidate, row
    if best is None:
        raise ValueError("no legal block shape")
    return best[1], best[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", action="append", required=True,
                    help="M x N x K, for example 3136x64x64")
    ap.add_argument("--output-dtype", choices=("i8", "i32"), default="i8")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--wide-store", action="store_true")
    ap.add_argument("--cache-b", action="store_true")
    ap.add_argument("--reuse-b", action="store_true")
    ap.add_argument("--wide-a", action="store_true")
    ap.add_argument("--wide-b", action="store_true")
    args = ap.parse_args()
    results = []
    for spelling in args.shape:
        m, n, k = (int(x) for x in spelling.lower().split("x"))
        base = Shape(m, n, k, args.output_dtype, bias=args.bias,
                     wide_store=args.wide_store, cache_b=args.cache_b,
                     reuse_b=args.reuse_b, wide_a=args.wide_a,
                     wide_b=args.wide_b)
        chosen, row = tune(base)
        results.append({"shape": spelling, "schedule": asdict(chosen),
                        "estimate": row, "default_4x4": estimate(base)})
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
