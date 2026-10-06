"""Evaluate paired engine corrections with complete geometries held out.

This experiment binds existing execution evidence and delegates fitting and
validation to Merlin. It cannot license a production cost provider, physical
service rate, or whole-program prediction from one profiled executable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast
from mlir_oot.golden_device_profile import parse_profile
from mlir_oot.no_fsm_audit import audit_elf


def geometry(boundary):
    shape = boundary["shape"]
    if isinstance(shape, dict):
        shape = {
            key: value for key, value in shape.items()
            if key in ("h", "w", "cin", "cout", "m", "n", "k", "stride", "output_dtype", "bias")
        }
    elif isinstance(shape, list):
        # Residual descriptor: rows, columns, then numerical coefficients.
        shape = shape[:2]
    elif shape is not None:
        raise ValueError("unknown geometry descriptor")
    return canonical_sha256([boundary["category"], shape])


def verify(path, digest, pins):
    path = Path(path)
    if sha256_file(path) != digest:
        raise ValueError(f"evidence changed: {path}")
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError("one path has conflicting identities")
    pins[str(path)] = digest
    return path


def evaluate(rows, pointers, include_fixed):
    result = fast.cross_validate(
        rows,
        lambda training: fast.fit_linear_screen(
            training, pointers=pointers, include_fixed=include_fixed,
            maximum_condition=1000,
        ),
        maximum_relative_error=0.2,
        minimum_predictions=len(rows),
        minimum_rank_rate=0.8,
        minimum_decided=6,
        minimum_slice_decided=2,
        minimum_slices=2,
    )
    try:
        fit = fast.fit_linear_screen(
            rows, pointers=pointers, include_fixed=include_fixed,
            maximum_condition=1000,
        )
        result["all_rows_diagnostic_fit"] = {
            "coefficients": fit.coefficients,
            "fixed_cycles": fit.fixed_cycles,
            "feature_domains": fit.domains,
            "fit_sha256": fit.provenance_sha256,
            "role": "diagnostic only; training error never licenses a provider",
        }
    except ValueError as error:
        result["all_rows_diagnostic_fit"] = {"refused": str(error)}
    return result


def run(gsim_path, stock_path, operand_path):
    pins = {}
    for path in (gsim_path, stock_path, operand_path):
        verify(path, sha256_file(path), pins)
    gsim = json.loads(gsim_path.read_text())
    stock_record = json.loads(stock_path.read_text())
    stock = stock_record["result"]
    for path, digest in gsim["artifacts"].items():
        verify(path, digest, pins)
    verify(stock_record["receipt_path"], stock_record["receipt_sha256"], pins)
    for path, row in stock_record["staged_evidence"].items():
        verify(path, row["sha256"], pins)
    for path, digest in (
        (stock["elf"], stock["elf_sha256"]),
        (stock["uart"], stock["uart_sha256"]),
        (stock["output_digest_evidence"]["reference"], stock["output_digest_evidence"]["reference_sha256"]),
        (stock["output_digest_evidence"]["validation"], stock["output_digest_evidence"]["validation_sha256"]),
    ):
        verify(path, digest, pins)
    if stock["state"] != "DONE" or stock["exit_code"] != 0:
        raise ValueError("stock job did not complete")
    if gsim["elf_sha256"] != stock["elf_sha256"]:
        raise ValueError("engine observations do not bind the identical executable")
    if audit_elf(Path(stock["elf"]).read_bytes())["status"] != "pass":
        raise ValueError("final executable instruction policy failed")
    manifest_path = next(Path(p) for p in gsim["artifacts"] if p.endswith("profile_build.json"))
    manifest = json.loads(manifest_path.read_text())
    gsim_uart = next(Path(p) for p in gsim["artifacts"] if p.endswith("uart.log"))
    parsed_gsim = parse_profile(gsim_uart.read_text(), manifest)
    parsed_stock = parse_profile(Path(stock["uart"]).read_text(), manifest)
    if parsed_stock != stock["boundary_profile"]:
        raise ValueError("retained stock profile differs from actual console")
    for key, field in (("forward_counter", "forward_cycles"), ("device_counter", "device_cycles"), ("host_gap_counter", "host_gap_cycles")):
        if parsed_gsim[key] != gsim[field]:
            raise ValueError("GSIM interval conservation differs from actual console")
    expected_digest = stock["output_digest_evidence"]["sha256"]
    marker = f"OUT_SHA256 f32le 1000 4000 {expected_digest}"
    if marker not in gsim_uart.read_text() or marker not in Path(stock["uart"]).read_text():
        raise ValueError("engine numerical observations differ")
    domain = canonical_sha256({
        "sim_engine": gsim["engine_sha256"],
        "hardware": {key: stock[key] for key in ("hwdb_sha256", "bitstream_sha256", "bitstream_archive_sha256")},
        "scope": "same instrumented executable; separate callback and preceding host intervals",
    })
    paired = {}
    actual = {event[0]: event for event in parsed_stock["events"]}
    for kind, column in (("callback", 3), ("preceding_host", 2)):
        rows = []
        for event in gsim["events"]:
            ordinal, boundary = event["ordinal"], event["boundary"]
            if actual[ordinal][1] != boundary["id"]:
                raise ValueError("source callback binding differs across engines")
            feature = event["device_cycles"] if column == 3 else event["preceding_host_gap_cycles"]
            sim_event = parsed_gsim["events"][ordinal]
            if sim_event[:2] != [ordinal, boundary["id"]] or sim_event[column] != feature:
                raise ValueError("GSIM source callback binding differs")
            rows.append(fast.Observation(
                stock["elf_sha256"], canonical_sha256([ordinal, kind]), geometry(boundary), domain,
                {"/features/simulator_cycles": feature}, actual[ordinal][column], tuple(pins.values()),
            ))
        paired[kind] = {
            "geometries": len({row.group for row in rows}),
            "raw_cycles": {"simulator": sum(row.features["/features/simulator_cycles"] for row in rows), "stock": sum(row.cycles for row in rows)},
            "scale_only": evaluate(rows, ("/features/simulator_cycles",), False),
            "scale_and_fixed": evaluate(rows, ("/features/simulator_cycles",), True),
        }
    operand = json.loads(operand_path.read_text())
    for ref in operand["pins"].values():
        if isinstance(ref, dict) and "path" in ref and "sha256" in ref:
            verify(ref["path"], ref["sha256"], pins)
    proxies = {}
    for names in (("issue_compute_max_proxy",), ("issue_compute_max_proxy", "instruction_footprint"), ("issue_compute_max_proxy", "requested_store_bytes", "instruction_footprint")):
        rows = []
        for row in operand["observations"]:
            features = row["features"]
            values = {
                "issue_compute_max_proxy": max(features["executed_instructions"], features["array_padded_compute_rows"]),
                "instruction_footprint": features["touched_instruction_bytes"],
                "requested_store_bytes": features["requested_dma_store_bytes"],
            }
            source = row["source_intervals"][0]
            rows.append(fast.Observation(
                operand["pins"]["elf"]["sha256"], canonical_sha256(row["scope_id"]), geometry(source),
                canonical_sha256([operand["schema"], "separate historical stock callback domain"]),
                {"/features/" + name: values[name] for name in names}, row["hardware_cycles"], tuple(pins.values()),
            ))
        proxies["+".join(names)] = evaluate(rows, tuple("/features/" + name for name in names), False)
    return {
        "schema": "paired_engine_geometry_screen_diagnostic_v1", "pins": pins,
        "driver_sha256": sha256_file(Path(__file__)), "shared_validator_sha256": sha256_file(Path(fast.__file__)),
        "paired_executable": stock["elf_sha256"], "paired_stock_job": stock["job_id"],
        "paired_engine_corrections": paired, "separate_spike_operand_proxies": proxies,
        "reference_cycle_labels_used": False, "production_enabled": False,
        "licence": "Geometry holdout within one executable; independent program prediction and matched-variant ranking remain UNKNOWN",
        "whole_prediction": "UNKNOWN: host/tail errors and independent executable generalization remain unqualified",
        "limitations": ["GSIM simulation time is not a fast Spike estimate", "Requested bytes do not establish physical traffic", "Instruction footprint does not establish cache misses", "The max proxy does not calibrate overlap or physical resource services", "Callbacks include CPU issue, DMA and fences"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsim", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--operand-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.gsim, args.stock, args.operand_features)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("PAIRED_ENGINE_SCREEN_RECHECK_PASS", len(result["pins"]), "pins; production disabled")
