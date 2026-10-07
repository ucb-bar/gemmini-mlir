"""One predeclared source-wide storage alternative; no timing or promotion."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    find_closed_scalar_i8_observers,
)
from merlin.perf.address_locality import address_locality
from rne_zero_observer_capsule import HERE, OLD, TYPED, save, sha

OUT = HERE / "out/artifacts/probes/source-interval-storage-census-20261007"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "driver_snapshot.py").write_bytes(Path(__file__).read_bytes())
    save(
        OUT / "declaration.json",
        {
            "source_control_partition_bits": 16,
            "candidate_partition_bits": 13,
            "capacity_budget_bytes": [512 * 1024, 64 * 1024],
            "source_effects_and_i8_observer": "Unchanged source expression/all-use closure and original rounded finishing operations; original source fallback for uncertifiable endpoints.",
            "hypothesis": "A smaller immutable source-wide bound table trades logical requested region extent/reuse against more original source replay. Physical traffic/cache behavior and cycle prices are UNKNOWN.",
            "selection": "One independently chosen storage budget, not a precision sweep or golden-derived eligibility",
            "no_new_elf_or_hardware": True,
        },
    )
    capture = OLD / "out/artifacts/probes/source-continuation-contexts-v3-20261007"
    metadata = json.loads((capture / "features_v2/contexts.json").read_text())
    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refusals = find_closed_scalar_i8_observers(
        parse_mlir_text(TYPED.read_text()), effects=effects
    )
    assert len(proofs) == 22 and not refusals
    rows, pins = (
        [],
        {
            str(TYPED): sha(TYPED),
            str(capture / "features_v2/contexts.json"): sha(
                capture / "features_v2/contexts.json"
            ),
        },
    )
    for leading in (16, 13):
        table = build_source_interval_table(
            proofs[0].expression,
            effects=effects,
            leading_bits=leading,
            max_table_bytes=(1 << leading) * 8,
        )
        binary = OUT / f"table_b{leading}.bin"
        binary.write_bytes(table.data)
        pins[str(binary)] = sha(binary)
        bounds = np.frombuffer(table.data, "<f4").reshape(-1, 2)
        for meta in metadata["rows"]:
            directory = capture / f"context_{meta['context']:02d}"
            paths = [
                directory / (name + ".npy")
                for name in ("a", "scale_a", "b", "scale_b", "expected")
            ]
            a, sa, b, sb, expected = [np.load(p) for p in paths]
            pins.update({str(p): sha(p) for p in paths})
            factor = np.array([meta["preparation_factor_bits"]], np.uint32).view(
                np.float32
            )[0]
            quant = np.array([meta["quant_factor_bits"]], np.uint32).view(np.float32)[0]
            with np.errstate(all="ignore"):
                x = np.float32(np.float32(np.float32(a) * factor) * sa)
                up = np.float32(np.float32(np.float32(b) * factor) * sb)
                indexes = x.view(np.uint32) >> np.uint32(32 - leading)
                cells = bounds[indexes]
                lo, hi = cells[..., 0], cells[..., 1]
                low = np.float32(np.float32(lo * up) * quant)
                high = np.float32(np.float32(hi * up) * quant)
                lower_q, upper_q = (
                    np.rint(np.clip(low, -128, 127)),
                    np.rint(np.clip(high, -128, 127)),
                )
            # The source emitter takes the original continuation on NaN bins.
            admitted = (lo <= hi) & (lower_q == upper_q)
            assert np.array_equal(lower_q[admitted].astype(np.int8), expected[admitted])
            # Original lowered helper performs two adjacent f32 table requests
            # per point in row-major lane0/lane1 order. This is logical table-only
            # chronology, not an actual memory trace of stack/source/other loads.
            requests = np.empty(indexes.size * 2, np.uint64)
            requests[::2] = indexes.reshape(-1).astype(np.uint64) * 8
            requests[1::2] = requests[::2] + 4
            locality = address_locality(
                (int(a) for a in requests),
                granule=64,
                max_requests=int(requests.size),
                capacities=(0, 64, 256, 1024),
            )
            rows.append(
                {
                    "partition_bits": leading,
                    "context": meta["context"],
                    "source_points": int(x.size),
                    "integer_observations_certified": int(admitted.sum()),
                    "source_replays": int((~admitted).sum()),
                    "table_bytes": len(table.data),
                    "source_bound_table_sha256": table.sha256,
                    "table_only_locality": asdict(locality),
                    "requests_base": "Logical zero-aligned immutable owner. Candidate actual ELF VMA is UNKNOWN; no ELF emitted.",
                    "source_fallback_work_unpriced": {
                        "original_point_calls": int((~admitted).sum()),
                        "includes": "Original activation source expression+rounded finishing FMULs+original quant observer/frame/dispatch",
                    },
                    "cycle_price": "UNKNOWN",
                }
            )
    assert len(rows) == 44
    save(
        OUT / "receipt.json",
        {
            "schema": "source_wide_storage_tradeoff_allcontext_census_v1",
            "rows": rows,
            "source_original_words_each_partition": sum(
                r["source_points"] for r in rows if r["partition_bits"] == 16
            ),
            "source_replay_counts_by_partition": {
                str(b): sum(
                    r["source_replays"] for r in rows if r["partition_bits"] == b
                )
                for b in (16, 13)
            },
            "all_certified_observations_match_original": True,
            "scope": "Cheap actual original22-operand source-exact table/certificate/locality screen; no newly compiled native/target/helper/whole gate or ranking",
            "unknowns": [
                "Actual candidate VMA/cache/physical traffic",
                "Complete request chronology with other data and stack",
                "Replay and source/finish cycle prices",
                "Whole cycle prediction",
            ],
            "pins": {
                **pins,
                str(Path(__file__)): sha(__file__),
                str(OUT / "driver_snapshot.py"): sha(OUT / "driver_snapshot.py"),
                str(OUT / "declaration.json"): sha(OUT / "declaration.json"),
            },
            "token_usage_available": False,
        },
    )
    print(
        "SOURCE_INTERVAL_STORAGE_CENSUS",
        {
            b: sum(r["source_replays"] for r in rows if r["partition_bits"] == b)
            for b in (16, 13)
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
