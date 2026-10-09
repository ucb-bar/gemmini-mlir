"""Validate exact RV64GC service-battery reports and expose measured features.

This target-side reader does not fit cycle costs. Counter windows include the
common call and fences; empty windows remain separate observations. Requested
CPU bytes and function footprint never assert physical traffic or cache misses.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_ROW = re.compile(
    r"SERVICE_ROW id=(-?\d+) repeat=(\d+) cycles=(\d+) instructions=(\d+) frm=(\d+) flags_before=(\d+) flags_after=(\d+)"
)
_FP = re.compile(
    r"SERVICE_FP id=(\d+) repeat=(\d+) words=([0-9a-f]{8}(?:,[0-9a-f]{8})*)"
)
_PASS = re.compile(r"SERVICE_PASS cases=(\d+) repeats=(\d+) checksum=([0-9a-f]{16})")


def parse_service_report(text: str, manifest: dict) -> dict:
    """Refuse incomplete, duplicate, misordered or numerically wrong reports."""
    cases = manifest["cases"]
    if [case["id"] for case in cases] != list(range(len(cases))):
        raise ValueError("manifest case IDs must be contiguous")
    repeats, empties = manifest["repetitions"], manifest["empty_windows"]
    if (
        type(repeats) is not int
        or repeats <= 0
        or type(empties) is not int
        or empties <= 0
    ):
        raise ValueError("manifest repetition counts are invalid")
    rows, fp, passes = [], {}, []
    for line in text.splitlines():
        if line.startswith("SERVICE_ROW"):
            match = _ROW.fullmatch(line)
            if not match:
                raise ValueError("malformed counter row")
            row = dict(
                zip(
                    (
                        "id",
                        "repeat",
                        "cycles",
                        "instructions",
                        "frm",
                        "flags_before",
                        "flags_after",
                    ),
                    map(int, match.groups()),
                    strict=True,
                )
            )
            if (
                not 0 < row["cycles"] <= 0xFFFFFFFF
                or not 0 < row["instructions"] <= 0xFFFFFFFF
            ):
                raise ValueError("counter range is invalid")
            if row["frm"] != 0 or row["flags_before"] != 0:
                raise ValueError("floating-point state is not the declared RNE window")
            rows.append(row)
        elif line.startswith("SERVICE_FP"):
            match = _FP.fullmatch(line)
            if not match:
                raise ValueError("malformed FP row or numeric failure")
            key = (int(match[1]), int(match[2]))
            if key in fp:
                raise ValueError("duplicate FP row")
            fp[key] = [int(value, 16) for value in match[3].split(",")]
        elif line.startswith("SERVICE_PASS"):
            match = _PASS.fullmatch(line)
            if not match:
                raise ValueError("malformed terminal marker")
            passes.append((int(match[1]), int(match[2]), match[3]))
        elif line.startswith("SERVICE_"):
            raise ValueError("unexpected service failure or report marker")
    expected_order = [(-1, repeat) for repeat in range(empties)] + [
        (case["id"], repeat) for case in cases for repeat in range(repeats)
    ]
    if [(row["id"], row["repeat"]) for row in rows] != expected_order:
        raise ValueError("counter coverage or order is incomplete")
    expected_fp = {
        (case["id"], repeat): case["expected_words"]
        for case in cases
        if case["family"] in ("div", "fma")
        for repeat in range(repeats)
    }
    if fp != expected_fp:
        raise ValueError("FP exact-word coverage disagrees with the manifest")
    if passes != [(len(cases), repeats, manifest["expected_checksum"])]:
        raise ValueError("terminal checksum or count disagrees with the manifest")
    for row in rows:
        expected_flags = (
            0 if row["id"] == -1 else cases[row["id"]].get("expected_fflags", 0)
        )
        if row["flags_after"] != expected_flags:
            raise ValueError("observed exception flags disagree with the source proof")
    return {
        "rows": rows,
        "fp_words": [
            {"id": key[0], "repeat": key[1], "words": values}
            for key, values in fp.items()
        ],
        "checksum": passes[0][2],
        "complete": True,
    }


def feature_rows(
    report: dict,
    manifest: dict,
    *,
    engine: dict,
    provenance: dict,
    function_sizes: dict,
) -> list[dict]:
    """Export arbitrary numeric feature pointers for Merlin's shared fitter."""
    result = []
    for row in report["rows"]:
        case = None if row["id"] == -1 else manifest["cases"][row["id"]]
        family = "empty_window" if case is None else case["family"]
        features = {
            "retired_instructions": row["instructions"],
            "fp_div_ops": case["size"] if family == "div" else 0,
            "fp_fma_ops": case["size"] if family == "fma" else 0,
            "independent_lanes": case.get("independent_lanes", 0) if case else 0,
            "source_dependent_fp_chain_ops": case["size"] // case["independent_lanes"]
            if family in ("div", "fma")
            else 0,
            "requested_cpu_load_bytes": case.get("requested_cpu_load_bytes", 0)
            if case
            else 0,
            "requested_cpu_store_bytes": case.get("requested_cpu_store_bytes", 0)
            if case
            else 0,
            "memory_stride_bytes": case["stride"] * 8 if family == "memory" else 0,
            "memory_extent_bytes": case.get("source_and_destination_extent_bytes", 0)
            if case
            else 0,
            "memory_working_set_bytes": case.get("working_set_bytes", 0) if case else 0,
            "gather_index_count": case["size"] if family == "gather" else 0,
            "gather_unique_requested_words": case.get("unique_requested_words", 0)
            if case
            else 0,
            "gather_unique_requested_64B_regions": case.get(
                "unique_requested_64B_regions", 0
            )
            if case
            else 0,
            "integer_add_ops": case.get("total_add_instructions", 0) if case else 0,
            "requested_instruction_body_bytes": case["size"]
            if family == "footprint"
            else 0,
            "kernel_function_bytes": function_sizes[
                case["kernel"] if case else "empty"
            ],
        }
        result.append(
            {
                "id": row["id"],
                "repeat": row["repeat"],
                "family": family,
                "partition": case["partition"] if case else "overhead_only",
                "features": features,
                "metric": {
                    "mcycle_delta": row["cycles"],
                    "minstret_delta": row["instructions"],
                    "mcycle_status": "retired_instruction_proxy"
                    if engine["kind"] == "spike_functional"
                    else "observed_hardware_counter",
                    "timer_scope": "fenced common kernel call only; input setup and all validation outside",
                },
                "feature_status": {
                    "/features/" + name: {
                        "status": "observed_counter"
                        if name == "retired_instructions"
                        else "source_and_emitted_code_bound",
                        "provenance": provenance,
                    }
                    for name in features
                },
                "engine": engine,
                "provenance": provenance,
                "unknown_features": {
                    "physical_memory_traffic": "UNKNOWN",
                    "cache_misses": "UNKNOWN",
                    "instruction_misses": "UNKNOWN",
                    "CPU_accelerator_overlap": "No accelerator work in this battery",
                },
            }
        )
    return result


def import_stock_report(
    record_path: Path, manifest: dict, *, elf: Path, parser_path: Path
) -> tuple[dict, dict]:
    """Reclose collector UART and named actual-stage receipts before import."""

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    record_path = Path(record_path)
    record = json.loads(record_path.read_text())
    if (
        record.get("schema") != "stock_cpu_service_battery_v1"
        or record.get("state") != "DONE"
        or record.get("exit_code") != 0
    ):
        raise ValueError("stock observation must be an exact terminal service receipt")
    job = record["job_id"]
    if type(job) is not int or job <= 0:
        raise ValueError("stock job identity is invalid")
    if (
        digest(elf) != record["elf_sha256"]
        or digest(record["elf"]) != record["elf_sha256"]
    ):
        raise ValueError("stock observation uses another ELF")
    if (
        digest(record["manifest_path"]) != record["manifest_sha256"]
        or json.loads(Path(record["manifest_path"]).read_text()) != manifest
    ):
        raise ValueError("stock manifest binding changed")
    if (
        digest(record["parser_path"]) != record["parser_sha256"]
        or digest(parser_path) != record["parser_sha256"]
    ):
        raise ValueError("stock parser binding changed")
    uart = Path(record["uart"])
    if digest(uart) != record["uart_sha256"]:
        raise ValueError("stock UART binding changed")
    report = parse_service_report(uart.read_text().replace("\r", ""), manifest)
    if report != record["report"]:
        raise ValueError("stock decoded rows disagree with the pinned UART")
    receipts = {}
    for name in (
        "expected_elf_receipt",
        "staged_elf_receipt",
        "staged_bitstream_receipt",
        "runworkload_receipt",
    ):
        path = Path(record[name])
        if digest(path) != record[name + "_sha256"]:
            raise ValueError("stock actual-stage receipt changed")
        receipts[name] = json.loads(path.read_text())
    for name in (
        "expected_elf_receipt",
        "staged_elf_receipt",
        "staged_bitstream_receipt",
    ):
        if receipts[name]["job_id"] != job:
            raise ValueError("stock actual-stage job binding disagrees")
    if any(
        receipts[name]["elf_sha256"] != record["elf_sha256"]
        for name in ("expected_elf_receipt", "staged_elf_receipt")
    ):
        raise ValueError("stock actual staged ELF differs")
    if (
        receipts["staged_bitstream_receipt"]["bitstream_sha256"]
        != record["bitstream_sha256"]
    ):
        raise ValueError("stock actual staged bitstream differs")
    workload = receipts["runworkload_receipt"]
    if (
        workload["hw_config"] != record["hw_config"]
        or workload["hwdb_config_artifact_sha256"] != record["hwdb_sha256"]
    ):
        raise ValueError("stock hardware configuration differs")
    if digest(workload["stage_from"]) != record["elf_sha256"]:
        raise ValueError("stock staged source ELF differs")
    expected = receipts["expected_elf_receipt"]
    if (
        expected["manifest_sha256"] != record["manifest_sha256"]
        or expected["parser_sha256"] != record["parser_sha256"]
    ):
        raise ValueError("stock intended manifest/parser differs")
    return report, record
