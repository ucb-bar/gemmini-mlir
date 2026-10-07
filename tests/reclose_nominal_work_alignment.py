"""Reclose a pinned geometry/role request reconciliation without cycle prices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def reclose(path):
    receipt = json.loads(path.read_text())
    inputs = []
    for row in receipt["pins"]:
        data = Path(row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("pinned request/source input changed")
        inputs.append(json.loads(data))
    current, alignment, reference, catalog = inputs
    assert current["whole_original_output_and_pc_histogram_byte_exact"]
    assert current["every_original_request_geometry_counter_exact"]
    assert alignment["control_job"] == receipt["current_job"] == 2039
    assert alignment["profile_job"] == 2046
    current_by_symbol = {
        symbol: group["requests"]
        for group in current["groups"]
        for symbol in group["symbols"]
    }
    reference_by_index = {row["reference_index"]: row for row in reference["layers"]}
    observed = []
    for pair in alignment["paired_layers"]:
        old = reference_by_index[pair["reference_index"]]
        assert old["geometry"] == pair["geometry"]
        new = current_by_symbol[pair["current_symbol"]]
        values = old["features"]
        observed.append(
            {
                "reference_index": pair["reference_index"],
                "reference_function": old["physical_function"],
                "current_function": pair["current_symbol"],
                "role": pair["category"],
                "geometry": pair["geometry"],
                "current_nominal_rows": new["padded_compute_rows"],
                "reference_nominal_rows": values["array_padded_compute_rows"],
                "difference_nominal_rows": new["padded_compute_rows"]
                - values["array_padded_compute_rows"],
                "current_requested_load_bytes": new["requested_load_bytes"],
                "reference_requested_load_bytes": values["requested_dma_load_bytes"],
                "current_requested_store_bytes": new["requested_store_bytes"],
                "reference_requested_store_bytes": values["requested_dma_store_bytes"],
            }
        )
    assert len(observed) == 54 and observed == receipt["paired_geometry_rows"]
    residual = catalog["residual_additions"]
    assert len(residual) == len(receipt["current_residual_arithmetic"]) == 16
    for route, expected in zip(residual, receipt["current_residual_arithmetic"]):
        p, q = [route["proof"]["coefficients"][key] for key in ("p", "q")]
        chunks = (p + 126) // 127 + (q + 126) // 127
        assert expected == {
            "symbol": route["kernel"],
            "shape": route["shape"],
            "physical_matrix": [route["m"], route["n"]],
            "source": route["proof"]["source"],
            "coefficients": route["proof"]["coefficients"],
            "chunks": chunks,
            "nominal_rows": current_by_symbol[route["kernel"]]["padded_compute_rows"],
            "complete_domain_pairs": route["proof"]["pairs"],
            "coefficient_proof_sha256": route["proof_sha256"],
            "kernel_object_sha256": route["compilation"]["object_sha256"],
        }
    new_residual = sum(
        row["nominal_rows"] for row in receipt["current_residual_arithmetic"]
    )
    old_residual = reference["residual_aggregate_operand_features"]["features"][
        "array_padded_compute_rows"
    ]
    assert new_residual == receipt["residual_current_nominal_rows"] == 5193216
    assert old_residual == receipt["residual_reference_nominal_rows"] == 355328
    new_total = sum(row["requests"]["padded_compute_rows"] for row in current["groups"])
    old_total = reference["whole_program_target_features"]["features"][
        "array_padded_compute_rows"
    ]
    new_mean = (
        new_total - sum(row["current_nominal_rows"] for row in observed) - new_residual
    )
    old_mean = (
        old_total
        - sum(row["reference_nominal_rows"] for row in observed)
        - old_residual
    )
    assert new_mean == old_mean == receipt["integer_mean_nominal_rows_each"] == 8192
    assert new_total - old_total == receipt["whole_delta_nominal_rows"] == 4937856
    assert (
        new_residual - old_residual == receipt["residual_delta_nominal_rows"] == 4837888
    )
    assert (
        sum(row["difference_nominal_rows"] for row in observed)
        + new_residual
        - old_residual
        == new_total - old_total
    )
    return {
        "status": "PASS",
        "paired_geometries": len(observed),
        "source_residual_routes": len(residual),
        "current_nominal_rows": new_total,
        "reference_nominal_rows": old_total,
        "integer_mean_nominal_rows_each": new_mean,
        "delta_nominal_rows": new_total - old_total,
        "residual_delta_nominal_rows": new_residual - old_residual,
        "scope": "Observed requests only; source numerics differ, no cycle attribution or hardware capability grant.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(reclose(parser.parse_args().receipt), indent=2))
