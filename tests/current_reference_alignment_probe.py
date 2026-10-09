"""Reclose current stock wrapper intervals against owned reference geometry.

This receipt compares different numerical contracts and timer scopes. It does not
select compiler policies or claim that the reference is an equivalent model.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mlir_oot.golden_device_profile import parse_profile


def pin(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    return {"path": str(path), "sha256": digest}


def reconcile(
    archive_path,
    qualification_path,
    catalog_path,
    pairing_path,
    reference_path,
    reference_staged_elf_path,
    reference_staged_bitstream_path,
):
    pins = {}
    inputs = {
        "stock_archive": archive_path,
        "profile_qualification": qualification_path,
        "source_catalog": catalog_path,
        "historical_geometry_pairing": pairing_path,
        "reference_profile": reference_path,
        "reference_staged_elf_observation": reference_staged_elf_path,
        "reference_staged_bitstream_observation": reference_staged_bitstream_path,
        "driver": __file__,
    }
    for name, path in inputs.items():
        pins[name] = pin(path)
    archive, qualification, catalog, pairing, reference = [
        json.loads(Path(path).read_text())
        for path in (
            archive_path,
            qualification_path,
            catalog_path,
            pairing_path,
            reference_path,
        )
    ]
    current = archive["result"]
    build = qualification["build"]
    assert current["job_id"] == 1919 and reference["job_id"] == 1876
    assert current["control_job"] == build["control_job"] == 1903
    assert build["control_reproduced_byteidentical"]
    assert build["source_semantics_equal1874"]
    pins["stock_terminal_receipt"] = pin(
        archive["receipt_path"], archive["receipt_sha256"]
    )
    for name, receipt in (("current", current), ("reference", reference)):
        assert receipt["state"] == receipt["phase"] == "DONE"
        assert receipt["exit_code"] == 0 and receipt["numeric_pass"]
        for field in ("elf", "uart"):
            pins[name + "_" + field] = pin(receipt[field], receipt[field + "_sha256"])
        staged = Path(receipt["actual_simulated_elf"])
        if staged.exists():
            pins[name + "_staged_elf"] = pin(staged, receipt["elf_sha256"])
    # Simulation teardown removes staged bytes; retained observations are explicit.
    for path, evidence in archive["staged_evidence"].items():
        pins["staged_observation:" + path] = pin(path, evidence["sha256"])
        observation = json.loads(Path(path).read_text())
        assert observation == evidence["record"] and observation["job_id"] == 1919
        for field in ("elf_sha256", "bitstream_sha256"):
            if field in observation:
                assert observation[field] == current[field]
    for path, field in (
        (reference_staged_elf_path, "elf_sha256"),
        (reference_staged_bitstream_path, "bitstream_sha256"),
    ):
        observation = json.loads(Path(path).read_text())
        assert observation["job_id"] == 1876 and observation[field] == reference[field]
    hardware_fields = (
        "hw_config",
        "hwdb_sha256",
        "bitstream_archive_sha256",
        "bitstream_sha256",
    )
    for field in hardware_fields:
        assert current[field] == reference[field], field
    pins["unprofiled_control_elf"] = pin(build["base_elf"], build["base_elf_sha256"])
    assert build["elf_sha256"] == current["elf_sha256"]
    # Close the diagnostic wrapper, validation, audit and all actual linked leaves.
    for path, digest in qualification["pins"].items():
        pins["qualification:" + path] = pin(path, digest)
    for path, digest in build["objects"].items():
        pins["object:" + path] = pin(path, digest)
    for component in build["leaf_component_reproduction"]:
        pins["component:" + component["component"]] = pin(
            component["component"], component["sha256"]
        )
        for path, digest in component["leaves"].items():
            pins["leaf:" + path] = pin(path, digest)
    pins["catalog_source"] = pin(catalog["source_snapshot"], catalog["source_sha256"])
    # Geometry correspondence is retained from the earlier source audit, never its timings.
    pin(reference_path, pairing["pins"]["reference_profile"]["sha256"])
    reference_geometry = json.loads(
        Path(pairing["pins"]["reference_shapes"]["path"]).read_text()
    )
    pins["reference_geometry"] = pin(
        path=pairing["pins"]["reference_shapes"]["path"],
        expected=pairing["pins"]["reference_shapes"]["sha256"],
    )
    assert len(reference_geometry) == 53
    profile = parse_profile(Path(current["uart"]).read_text(), build)
    assert profile == current["boundary_profile"]
    boundaries = {entry["id"]: entry for entry in build["boundaries"]}
    assert set(boundaries) == set(range(70))
    geometry_pairs = {row["current_region"]: row for row in pairing["paired_layers"]}
    classifier_binding = catalog["bindings"][0]
    assert len(catalog["bindings"]) == len(catalog["kernels"]) == 1
    classifier = catalog["kernels"][0]
    assert classifier["symbol"] == classifier_binding["symbol"]
    source_bound, by_region, classes = [], {}, Counter()
    for ordinal, symbol_id, host_gap, cycles in profile["events"]:
        boundary = boundaries[symbol_id]
        shape = boundary["shape"]
        region = boundary["source_region"]
        if boundary["category"] == "residual":
            category = "residual"
        elif boundary["category"] == "pooled_stem":
            category = "pooled_stem"
            assert shape == catalog["pooled_stem"]["shape"]
        elif boundary["category"] == "unary":
            category = "direct" if "cin" in shape else "pointwise"
        else:
            assert boundary["symbol"] == classifier["symbol"]
            assert boundary["category"] == "dense" and shape is None and region is None
            category, region, shape = (
                "classifier",
                classifier_binding["region"],
                classifier["schedule"],
            )
        row = {
            "call_index": ordinal,
            "symbol_id": symbol_id,
            "symbol": boundary["symbol"],
            "region": region,
            "category": category,
            "actual_device_cycles": cycles,
            "preceding_host_gap_cycles": host_gap,
            "shape": shape,
            "cpu_integer_readout_after": boundary.get("cpu_integer_readout_after"),
            "timing_scope": "Primitive wrapper; ranked adapter and CPU epilogue remain outside",
        }
        assert region not in by_region
        source_bound.append(row)
        by_region[region] = row
        classes[category] += cycles
    assert len(source_bound) == 70 and classes.total() == profile["device_counter"]
    reference_timers = {record["index"]: record for record in reference["layer_cycles"]}
    assert set(reference_timers) == set(range(1, 55))
    paired, reference_classes = [], Counter()
    for region, old in geometry_pairs.items():
        entry = by_region[region]
        index, geometry, shape = old["reference_index"], old["geometry"], entry["shape"]
        assert entry["call_index"] == old["current_call_index"]
        if entry["category"] in ("pointwise", "classifier"):
            assert geometry == {axis: shape[axis] for axis in ("m", "k", "n")}
        elif entry["category"] == "direct":
            assert geometry == {
                "m": ((shape["h"] - 1) // shape["stride"] + 1)
                * ((shape["w"] - 1) // shape["stride"] + 1),
                "k": 9 * shape["cin"],
                "n": shape["cout"],
            }
        else:
            assert entry["category"] == "pooled_stem" and geometry == {
                "m": 12544,
                "k": 147,
                "n": 64,
            }
        if index <= 53:
            assert tuple(geometry[axis] for axis in ("m", "k", "n")) == tuple(
                reference_geometry[index - 1][2:5]
            )
        ref_cycles = reference_timers[index]["cycles"]
        reference_classes[entry["category"]] += ref_cycles
        paired.append(
            {
                "reference_index": index,
                "reference_layer": old["reference_layer"],
                "reference_function": old.get("reference_function"),
                "category": entry["category"],
                "current_region": region,
                "geometry": geometry,
                "reference1876_cycles": ref_cycles,
                "current1919_device_cycles": entry["actual_device_cycles"],
                "difference_cycles": entry["actual_device_cycles"] - ref_cycles,
                "current1919_call_index": entry["call_index"],
                "current1919_preceding_host_gap": entry["preceding_host_gap_cycles"],
                "current1903_shape": shape,
                "cpu_integer_readout_after": entry["cpu_integer_readout_after"],
                "source_numeric_equivalence": False,
            }
        )
    aggregate = {}
    for line in reference["aggregate_lines"]:
        label, value = line.split(": ", 1)
        aggregate[label] = int(value.split()[0])
    reference_classes["residual"] = aggregate["Res add cycles"]
    reference_classes["host_intervals"] = (
        aggregate["Other cycles"] + aggregate["FM uncounted delta"]
    )
    classes["host_intervals"] = profile["host_gap_counter"]
    assert reference_classes.total() == reference["kernel_cycles"]
    assert classes.total() == profile["forward_counter"]
    comparison = [
        {
            "category": kind,
            "reference1876_cycles": reference_classes[kind],
            "current1919_cycles": classes[kind],
            "difference_cycles": classes[kind] - reference_classes[kind],
        }
        for kind in (
            "pointwise",
            "direct",
            "residual",
            "pooled_stem",
            "classifier",
            "host_intervals",
        )
    ]
    assert (
        sum(row["difference_cycles"] for row in comparison)
        == profile["forward_counter"] - reference["kernel_cycles"]
    )
    return {
        "schema": "q1013_1903_current_profile_alignment_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "owner": "reference_parity",
        "pins": pins,
        "hardware": {field: current[field] for field in hardware_fields},
        "control_job": 1903,
        "control_cycles": current["control_cycles"],
        "profile_job": 1919,
        "profile_metric_cycles": current["kernel_cycles"],
        "forward_cycles": profile["forward_counter"],
        "device_cycles": profile["device_counter"],
        "host_gap_cycles": profile["host_gap_counter"],
        "tail_cycles": profile["tail_counter"],
        "instrumentation_delta_cycles": current["profiling_cycle_delta"],
        "class_comparison": comparison,
        "paired_layers": sorted(paired, key=lambda row: row["reference_index"]),
        "all70_source_bound_events": source_bound,
        "largest_host_intervals": sorted(
            source_bound, key=lambda row: row["preceding_host_gap_cycles"], reverse=True
        )[:8],
        "largest_device_interval_differences": sorted(
            paired, key=lambda row: row["difference_cycles"], reverse=True
        )[:10],
        "source_numeric_equivalence": False,
        "causal_savings_claim": False,
        "residual_scope": "Current16 calls versus reference aggregate; no per-reference residual timers",
        "scope": [
            "Reference1876 modified diagnostic retains reference own input/weights/scales and narrowing",
            "1919 is the instrumented exact1903 object set, not a newer schedule composition",
            "Reference layer scopes may include CPU epilogues; ours time primitive calls only",
            "Current f32 NCHW input requires host quantization/packing; reference int8 input differs",
            "All preceding host operations and diagnostic overhead belong to host gaps",
            "Two SAT8 host adapter replacements are included in pinned current component leaves",
            "Earlier1850/1899 receipts remain historical; geometry correspondence only is reused",
        ],
        "token_usage_available": False,
        "staged_bytes_rehashed_now": {
            name: Path(receipt["actual_simulated_elf"]).exists()
            for name, receipt in (("current", current), ("reference", reference))
        },
        "staged_evidence_scope": "Archived observations are rehashed; staged bytes were removed at simulation teardown",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "archive",
        "qualification",
        "catalog",
        "pairing",
        "reference",
        "reference-staged-elf",
        "reference-staged-bitstream",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    receipt = reconcile(
        args.archive,
        args.qualification,
        args.catalog,
        args.pairing,
        args.reference,
        args.reference_staged_elf,
        args.reference_staged_bitstream,
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["class_comparison"], indent=2))
