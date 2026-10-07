"""One fixed source-derived hierarchy, all original contexts, no cycle price."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    find_closed_scalar_i8_observers,
)
from merlin.llvmlower.source_interval_hierarchy import (
    build_source_interval_hierarchy,
    validate_source_interval_hierarchy,
)

HERE = Path(__file__).resolve().parents[1]
CORE = Path("/scratch/agustin/tmp/merlin-source-table-hierarchy-main-20261007")
RNE = Path("/scratch/agustin/tmp/gemmini-rne-observer-cells-20261007")
OLD = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
TYPED = Path(
    "/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006/out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic/typed_prepacket.generic.mlir"
)
CAPTURE = OLD / "out/artifacts/probes/source-continuation-contexts-v3-20261007"
OUT = HERE / "out/artifacts/probes/source-interval-hierarchy-census-v2-20261007"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def f32(word):
    return np.asarray([word], "u4").view("f4")[0]


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    effects = IntervalEffectContract(True, True, True, True, True)
    source = TYPED.read_text()
    proofs, refused = find_closed_scalar_i8_observers(
        parse_mlir_text(source), effects=effects
    )
    assert len(proofs) == 22 and not refused
    hierarchy = build_source_interval_hierarchy(
        proofs[0].expression,
        effects=effects,
        partition_bits=(13, 16),
        max_table_bytes=576 * 1024,
    )
    validate_source_interval_hierarchy(hierarchy)
    cells = prepare_bounded_rne_word_cells(proofs[0], effects=effects)
    half = cells.ranges[128][1] ^ 0x80000000
    metadata = json.loads((CAPTURE / "features_v2/contexts.json").read_text())
    pins = {
        str(path): sha(path)
        for path in [
            Path(__file__),
            TYPED,
            CAPTURE / "features_v2/contexts.json",
            CORE / "src/merlin/llvmlower/source_interval_hierarchy.py",
            CORE / "src/merlin/llvmlower/source_expression_interval.py",
            CORE / "src/merlin/llvmlower/bounded_rne_word_cells.py",
        ]
    }
    save(
        OUT / "declaration.json",
        {
            "schema": "source_interval_hierarchy_predeclared_work_v1",
            "partitions": [13, 16],
            "total_immutable_capacity_bytes": 576 * 1024,
            "policy": "Current source-wide exact i8 observer; coarse first, unchanged fine certificate on ambiguity, unchanged source replay after final ambiguity. Original rounded finishing MULs and explicit RNE/gradual/nontrapping/unobserved flags remain.",
            "hypothesis": "Avoid most large-table requests using a small source-derived level, at cost of additional exact certificates; no memory/cache/cycle price inferred.",
            "baseline": "current2085 helper uses one16bit/512KiB table and exact zero-bin observer",
            "selection": "One explicit storage/partition plan chosen independently of expected output bytes, not a sweep",
            "cycles": "UNKNOWN",
            "pins": pins,
        },
    )
    arrays = {}
    for table in hierarchy.tables:
        path = OUT / f"table_b{table.leading_bits}.bin"
        path.write_bytes(table.data)
        pins[str(path)] = sha(path)
        arrays[table.leading_bits] = np.frombuffer(table.data, "<f4").reshape(-1, 2)
    old_fine = (
        RNE
        / "out/artifacts/probes/source-interval-storage-census-20261007/table_b16.bin"
    )
    old_coarse = (
        RNE
        / "out/artifacts/probes/source-interval-storage-census-20261007/table_b13.bin"
    )
    assert (OUT / "table_b16.bin").read_bytes() == old_fine.read_bytes()
    assert (OUT / "table_b13.bin").read_bytes() == old_coarse.read_bytes()
    pins.update({str(path): sha(path) for path in (old_fine, old_coarse)})
    rows = []
    for item in metadata["rows"]:
        directory = CAPTURE / f"context_{item['context']:02d}"
        paths = [
            directory / (name + ".npy")
            for name in ("a", "scale_a", "b", "scale_b", "expected")
        ]
        a, sa, b, sb, expected = [np.load(path) for path in paths]
        pins.update({str(path): sha(path) for path in paths})
        assert any(p.quant_factor_bits == item["quant_factor_bits"] for p in proofs)
        with np.errstate(all="ignore"):
            x = np.float32(
                np.float32(np.float32(a) * f32(item["preparation_factor_bits"])) * sa
            )
            up = np.float32(
                np.float32(np.float32(b) * f32(item["preparation_factor_bits"])) * sb
            )
        decisions = {}
        for bits in (13, 16):
            idx = x.view("u4") >> np.uint32(32 - bits)
            lo, hi = arrays[bits][idx][..., 0], arrays[bits][idx][..., 1]
            valid = lo <= hi
            with np.errstate(all="ignore"):
                low = np.float32(np.float32(lo * up) * f32(item["quant_factor_bits"]))
                high = np.float32(np.float32(hi * up) * f32(item["quant_factor_bits"]))
                zero = valid & (
                    ((low.view("u4") | high.view("u4")) & np.uint32(0x7FFFFFFF)) <= half
                )
                lq, hq = (
                    np.rint(np.clip(low, -128, 127)),
                    np.rint(np.clip(high, -128, 127)),
                )
                same = valid & (lq == hq)
            admitted = zero | same
            assert np.array_equal(lq[admitted].astype("i1"), expected[admitted])
            decisions[bits] = {
                "index": idx,
                "valid": valid,
                "zero": zero,
                "admitted": admitted,
            }
        refine = ~decisions[13]["admitted"]
        replay = refine & ~decisions[16]["admitted"]
        source_fine_replay = ~decisions[16]["admitted"]
        current_quantizer_calls = 2 * int(
            (decisions[16]["valid"] & ~decisions[16]["zero"]).sum()
        )
        hierarchical_quantizer_calls = 2 * int(
            (decisions[13]["valid"] & ~decisions[13]["zero"]).sum()
        ) + 2 * int((refine & decisions[16]["valid"] & ~decisions[16]["zero"]).sum())
        rows.append(
            {
                "context_for_binding_only": item["context"],
                "source_points": int(x.size),
                "coarse_certified": int((~refine).sum()),
                "fine_requests": int(refine.sum()),
                "source_replays": int(replay.sum()),
                "control_source_replays": int(source_fine_replay.sum()),
                "baseline_table_cell_requests": int(x.size),
                "candidate_table_cell_requests": int(x.size + refine.sum()),
                "baseline_endpoint_finish_fmul": 4 * int(decisions[16]["valid"].sum()),
                "candidate_endpoint_finish_fmul": 4 * int(decisions[13]["valid"].sum())
                + 4 * int((refine & decisions[16]["valid"]).sum()),
                "baseline_endpoint_quantizer_calls": current_quantizer_calls,
                "candidate_endpoint_quantizer_calls": hierarchical_quantizer_calls,
                "source_expected_is_validation_only": True,
                "requested_coarse_logical64B_regions": int(
                    np.unique(decisions[13]["index"] // 8).size
                ),
                "requested_fine_logical64B_regions": int(
                    np.unique(decisions[16]["index"][refine] // 8).size
                ),
                "baseline_fine_logical64B_regions": int(
                    np.unique(decisions[16]["index"] // 8).size
                ),
                "extra_source_scanner_or_preparation": 0,
                "source_input_cast_FMUL_order": "unchanged",
                "prices": "UNKNOWN: logical requests/regions and source operations are not cycles or physical cache/DDR events",
            }
        )
    summed = {
        key: sum(row[key] for row in rows)
        for key in rows[0]
        if type(rows[0][key]) is int and key != "context_for_binding_only"
    }
    save(
        OUT / "qualification.json",
        {
            "schema": "source_interval_hierarchy_all22_work_v1",
            "status": "pass",
            "rows": rows,
            "sums": summed,
            "original_i8_outputs": summed["source_points"],
            "typed_source_observers": len(proofs),
            "universal_cells_rederived": True,
            "original_coarse_and_fine_bytes_exact": True,
            "total_table_bytes": hierarchy.table_bytes,
            "expression_sha256": hierarchy.expression.canonical_sha256,
            "effects": asdict(effects),
            "native_target_compiled_successor": "NOT YET RUN",
            "cycles": "UNKNOWN",
            "physical_memory_and_whole_price": "UNKNOWN",
            "token_usage_available": False,
            "pins": {
                **pins,
                str(OUT / "declaration.json"): sha(OUT / "declaration.json"),
            },
        },
    )
    print("SOURCE_HIERARCHY_ALL22", json.dumps(summed), flush=True)


if __name__ == "__main__":
    main()
