"""Close one immutable service battery across strict Spike and pinned GSIM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mlir_oot.executed_features import _symbol_ranges, census
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.service_battery import (
    feature_rows,
    import_stock_report,
    parse_service_report,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(work, gsim_directory=None, output=None, spike_only=False, stock_record=None):
    manifest_path = work / "generated_v3/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    built = json.loads((work / "built.json").read_text())
    elf = Path(built["elf"])
    assert sha(elf) == built["elf_sha256"]
    audit = audit_elf(elf.read_bytes())
    assert audit["status"] == "pass"
    symbols = [
        "empty",
        "div_dep",
        "div_ind",
        "fma_dep",
        "fma_ind",
        "memory",
        "footprint_1024",
        "footprint_4096",
        "footprint_16384",
    ]
    ranges = _symbol_ranges(elf, symbols, "readelf")
    sizes = {row["symbol"]: row["end"] - row["start"] for row in ranges}
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    spike_engine = {
        "kind": "spike_functional",
        "path": str(spike),
        "sha256": sha(spike),
        "isa": "rv64gc",
        "extension": "gemmini",
        "mcycle_interpretation": "retired instruction proxy, not hardware cycles",
    }
    observations = {}
    for name, filename, engine in [("spike", "spike_hist.stdout", spike_engine)]:
        stdout = work / filename
        report = parse_service_report(stdout.read_text(), manifest)
        provenance = {
            "elf_sha256": sha(elf),
            "stdout_sha256": sha(stdout),
            "manifest_sha256": sha(manifest_path),
            "scope": "same fenced common kernel-call windows",
        }
        observations[name] = {
            "report": report,
            "features": feature_rows(
                report,
                manifest,
                engine=engine,
                provenance=provenance,
                function_sizes=sizes,
            ),
        }
    direct = parse_service_report((work / "spike.stdout").read_text(), manifest)
    assert direct == observations["spike"]["report"]
    hist = work / "spike_hist.stderr"
    counts = {}
    for symbol in symbols:
        scope = "all_invocations_of_" + symbol
        counts[symbol] = census(
            elf,
            hist,
            symbols=[symbol],
            scope_id=scope,
            execution_binding={
                "elf_sha256": sha(elf),
                "histogram_sha256": sha(hist),
                "scope_id": scope,
                "engine": spike_engine,
            },
        )
        assert counts[symbol]["status"] == "pass"
    for symbol, kind in [
        ("div_dep", "fp_div"),
        ("div_ind", "fp_div"),
        ("fma_dep", "fp_fma"),
        ("fma_ind", "fp_fma"),
    ]:
        assert counts[symbol]["features"]["cpu_opcode_classes"][kind] == sum(
            case["size"] * manifest["repetitions"]
            for case in manifest["cases"]
            if case["kernel"] == symbol
        )
    assert counts["memory"]["features"]["cpu_opcode_classes"]["store_integer"] == sum(
        case["size"] * manifest["repetitions"]
        for case in manifest["cases"]
        if case["family"] == "memory"
    )
    # Shared functions vary counts across sizes; retain aggregate CPU census. Do
    # not divide its aliases or invocation counts to manufacture per-case data.
    gsim_directory = gsim_directory or work
    if not spike_only and (gsim_directory / "gsim.json").exists():
        run = json.loads((gsim_directory / "gsim.json").read_text())
        assert run["returncode"] == 0 and run["finish"]["done"]
        stdout = gsim_directory / "gsim.stdout"
        assert sha(stdout) == run["stdout_sha256"]
        report = parse_service_report(stdout.read_text(), manifest)
        for row, other in zip(
            report["rows"], observations["spike"]["report"]["rows"], strict=True
        ):
            assert row["instructions"] == other["instructions"], (row, other)
        engine = {
            "kind": "gsim_rtl",
            "provenance": run["engine"],
            "load_path": run["load_path"],
            "memory_regime": "elaborated GSIM harness; distinct from stock FireSim",
        }
        provenance = {
            "elf_sha256": sha(elf),
            "stdout_sha256": sha(stdout),
            "manifest_sha256": sha(manifest_path),
            "scope": "same fenced common kernel-call windows",
        }
        observations["gsim"] = {
            "report": report,
            "features": feature_rows(
                report,
                manifest,
                engine=engine,
                provenance=provenance,
                function_sizes=sizes,
            ),
        }
    if stock_record is not None:
        parser_path = (
            Path(__file__).resolve().parents[1] / "mlir_oot/service_battery.py"
        )
        report, record = import_stock_report(
            stock_record, manifest, elf=elf, parser_path=parser_path
        )
        for row, other in zip(
            report["rows"], observations["spike"]["report"]["rows"], strict=True
        ):
            assert row["instructions"] == other["instructions"], (row, other)
        engine = {
            "kind": "firesim_stock",
            "job_id": record["job_id"],
            "hw_config": record["hw_config"],
            "hwdb_sha256": record["hwdb_sha256"],
            "bitstream_sha256": record["bitstream_sha256"],
            "memory_regime": "actual pinned stock FireSim; no GSIM slopes transplanted",
        }
        provenance = {
            "elf_sha256": sha(elf),
            "stdout_sha256": record["uart_sha256"],
            "manifest_sha256": sha(manifest_path),
            "stock_record": str(stock_record),
            "stock_record_sha256": sha(stock_record),
            "scope": "same fenced common kernel-call windows",
            "collector_actual_stage": {
                name: record[name]
                for name in (
                    "expected_elf_receipt",
                    "expected_elf_receipt_sha256",
                    "staged_elf_receipt",
                    "staged_elf_receipt_sha256",
                    "staged_bitstream_receipt",
                    "staged_bitstream_receipt_sha256",
                    "runworkload_receipt",
                    "runworkload_receipt_sha256",
                )
            },
        }
        observations["firesim"] = {
            "report": report,
            "features": feature_rows(
                report,
                manifest,
                engine=engine,
                provenance=provenance,
                function_sizes=sizes,
            ),
        }
    out = {
        "schema": "rv64gc_service_battery_qualified_v1",
        "built": built,
        "manifest": manifest,
        "observations": observations,
        "executed_function_census": counts,
        "function_sizes": sizes,
        "function_ranges": ranges,
        "nofsm_audit": audit,
        "scope_limitations": [
            "Empty-window overhead retained separately; no automatic subtraction",
            "Per-case FP/add/requestedCPU byte counts bound by generated source and actual function histogram totals, not unobserved operand telemetry",
            "Function aggregate PC histograms include all varied-size invocations; not per-case attribution",
            "No physical memory traffic/cache misses or accelerator overlap inferred",
            "Middle sizes and both lane/stride arms withheld together; never fitted here",
            "No Jack/model timing labels fitted",
        ],
        "token_usage_available": False,
    }
    (output or work / "qualified.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "elf_sha256": built["elf_sha256"],
                "engines": list(observations),
                "cases": len(manifest["cases"]),
                "function_sizes": sizes,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--gsim-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spike-only", action="store_true")
    parser.add_argument("--stock-record", type=Path)
    args = parser.parse_args()
    export(
        args.workdir.resolve(),
        args.gsim_directory.resolve() if args.gsim_directory else None,
        args.output.resolve() if args.output else None,
        args.spike_only,
        args.stock_record.resolve() if args.stock_record else None,
    )
