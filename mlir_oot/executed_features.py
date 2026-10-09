"""Executed Gemmini feature census for Merlin's existing calibration pointers.

A Spike PC histogram proves instruction and primitive class multiplicities.
It does not contain operand values, chronological order or hardware cycles.
Geometry, physical traffic and overlap therefore remain unknown in this path.
Function filters describe an explicit measurement scope; aliases and shared
invocations are never followed or divided by guessed geometric multiplicities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from merlin.targetgen.elf_lanes import executable_sections, instruction_words

from .cpu_opcode_census import CLASSES, classify
from .no_fsm_audit import AuditError, _instruction_bytes, audit_elf
from .tables import rtl_facts as F


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pc_histogram(text: str) -> dict[int, int]:
    """Read Spike's complete final histogram, refusing duplicates/truncation."""
    lines = text.splitlines()
    heads = [i for i, line in enumerate(lines) if line.startswith("PC Histogram size:")]
    if len(heads) != 1:
        raise ValueError("execution evidence needs one complete PC histogram")
    index = heads[0]
    size = int(lines[index].split(":", 1)[1])
    rows = [line.split() for line in lines[index + 1 :] if line.strip()]
    if size <= 0 or len(rows) != size:
        raise ValueError("PC histogram extent is empty or truncated")
    result = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError("PC histogram row must contain address and count")
        pc, count = int(row[0], 16), int(row[1], 10)
        if not 0 <= pc < 1 << 64 or pc % 2 or count <= 0:
            raise ValueError("PC histogram address or count is invalid")
        if pc in result:
            raise ValueError("PC histogram repeats an executed address")
        result[pc] = count
    return result


def _symbol_ranges(elf: Path, symbols: Sequence[str], readelf: str) -> list[dict]:
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("measurement symbols must be nonempty and unique")
    text = subprocess.run(
        [readelf, "-sW", str(elf)], check=True, capture_output=True, text=True
    ).stdout
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[3] != "FUNC" or fields[-1] not in symbols:
            continue
        if fields[-1] in rows:
            raise ValueError("ELF measurement symbol is ambiguous")
        start = int(fields[1], 16)
        size = int(fields[2], 16 if fields[2].startswith("0x") else 10)
        if fields[6] in ("UND", "ABS") or size <= 0:
            raise ValueError("measurement symbol has no executable function body")
        rows[fields[-1]] = {"symbol": fields[-1], "start": start, "end": start + size}
    if set(rows) != set(symbols):
        raise ValueError("ELF does not define every measurement symbol")
    return [rows[symbol] for symbol in symbols]


def census(
    elf: Path,
    histogram: Path,
    *,
    symbols: Sequence[str] | None = None,
    scope_id: str = "whole_program_including_harness",
    execution_binding: Mapping | None = None,
    readelf: str = "readelf",
) -> dict:
    """Export numeric feature pointers, status and exact source provenance.

    The caller's execution binding must pin ELF/histogram bytes, scope and
    engine independently. It is a provenance declaration, not proof that the
    histogram was generated from those bytes. Its authority remains explicit.
    """
    elf, histogram = Path(elf), Path(histogram)
    if not scope_id:
        raise ValueError("feature census requires an explicit measurement scope")
    data = elf.read_bytes()
    audit = audit_elf(data)
    if audit["status"] != "pass":
        raise AuditError("executed feature ELF does not pass the final no-FSM audit")
    elf_sha, hist_sha = hashlib.sha256(data).hexdigest(), _sha(histogram)
    binding = dict(execution_binding or {})
    if binding and (
        binding.get("elf_sha256") != elf_sha
        or binding.get("histogram_sha256") != hist_sha
        or binding.get("scope_id") != scope_id
    ):
        raise ValueError("execution binding disagrees with ELF, histogram or scope")
    hist = parse_pc_histogram(histogram.read_text())
    sections = executable_sections(data)
    words = {}
    encodings = {}
    starts = set()
    for _, offset, size, address in sections:
        pos = 0
        while pos < size:
            pc = address + pos
            if pc in starts:
                raise ValueError("executable ELF instruction ranges overlap")
            starts.add(pc)
            width = _instruction_bytes(struct.unpack_from("<H", data, offset + pos)[0])
            encodings[pc] = (
                int.from_bytes(data[offset + pos : offset + pos + width], "little"),
                width,
            )
            pos += width
        for pc, word in instruction_words(data[offset : offset + size], address):
            if pc in words:
                raise ValueError("executable ELF instruction ranges overlap")
            words[pc] = word
    for pc in hist:
        if (
            any(address <= pc < address + size for _, _, size, address in sections)
            and pc not in starts
        ):
            raise ValueError("PC histogram address is inside an instruction")
    ranges = _symbol_ranges(elf, symbols, readelf) if symbols is not None else []
    if ranges:
        for row in ranges:
            if not any(
                address <= row["start"] < row["end"] <= address + size
                for _, _, size, address in sections
            ):
                raise ValueError("measurement function is outside executable bytes")
            if row["start"] not in starts or (
                row["end"] not in starts
                and not any(
                    row["end"] == address + size for _, _, size, address in sections
                )
            ):
                raise ValueError("measurement function bounds split an instruction")
        selected = {
            pc: count
            for pc, count in hist.items()
            if any(row["start"] <= pc < row["end"] for row in ranges)
        }
    else:
        selected = hist
    if not selected:
        raise ValueError("measurement scope contains no executed instructions")
    primitive = Counter(
        {name: 0 for funct, name in F.FUNCT_NAMES.items() if funct in F.LEGAL_FUNCTS}
    )
    cpu_classes = Counter({name: 0 for name in CLASSES})
    unknown = Counter()
    mapped = 0
    for pc, count in selected.items():
        encoding = encodings.get(pc)
        cpu_classes[classify(*encoding) if encoding else "outside_elf_unknown"] += count
        if any(address <= pc < address + size for _, _, size, address in sections):
            mapped += count
        word = words.get(pc)
        if word is None or word & 0x7F != F.CUSTOM_OPCODE:
            continue
        funct = word >> 25
        name = F.FUNCT_NAMES.get(funct)
        if name is None or funct not in F.LEGAL_FUNCTS or (word >> 12) & 7 != F.FUNCT3:
            unknown[str(funct)] += count
        else:
            primitive[name] += count
    provenance = [
        "elf_sha256:" + elf_sha,
        "pc_histogram_sha256:" + hist_sha,
        "scope_id:" + scope_id,
    ]
    observed = {
        "status": "observed_functional_execution",
        "unit": "instruction",
        "provenance": provenance,
        "cycle_interpretation": "UNKNOWN: these are retired instructions, not hardware cycles",
        "execution_producer_binding": "declared" if binding else "UNKNOWN",
    }
    unavailable = {
        "status": "UNKNOWN",
        "value_source": None,
        "reason": "PC histogram contains no operands or chronological command order",
        "provenance": provenance,
    }
    command_status = dict(
        observed,
        unit="command",
        attribution_scope="ELF mapped executed PCs only",
    )
    if mapped != sum(selected.values()):
        command_status.update(
            status="UNKNOWN",
            reason="Some scoped executed PCs are outside this ELF and cannot be decoded",
        )
    return {
        "schema": "gemmini_executed_feature_census_v1",
        "features": {
            "executed_instructions": sum(selected.values()),
            "primitive_commands": dict(sorted(primitive.items())),
            "cpu_opcode_classes": dict(sorted(cpu_classes.items())),
            "unique_executed_pcs": len(selected),
            "touched_instruction_bytes": (
                sum(encodings[pc][1] for pc in selected)
                if all(pc in encodings for pc in selected)
                else None
            ),
            "array_work": None,
            "requested_dma_bytes": None,
        },
        "feature_status": {
            "/features/executed_instructions": observed,
            "/features/primitive_commands": command_status,
            "/features/cpu_opcode_classes": dict(observed, unit="instruction"),
            "/features/unique_executed_pcs": dict(observed, unit="unique_pc"),
            "/features/touched_instruction_bytes": (
                dict(observed, unit="instruction_byte")
                if all(pc in encodings for pc in selected)
                else dict(
                    unavailable,
                    reason="Scoped PCs outside ELF have unknown encoding widths",
                )
            ),
            **{
                "/features/cpu_opcode_classes/" + name: dict(
                    observed, unit="instruction"
                )
                for name in cpu_classes
            },
            **{
                "/features/primitive_commands/" + name: dict(command_status)
                for name in primitive
            },
            "/features/array_work": dict(unavailable),
            "/features/requested_dma_bytes": dict(unavailable),
        },
        "unknown_features": {
            "physical_dram_bytes": "UNKNOWN",
            "dma_dependencies": "UNKNOWN",
            "bank_hazards": "UNKNOWN",
            "accelerator_dispatch_overlap": "UNKNOWN",
            "hardware_cycles": "UNKNOWN",
            "instruction_cache_line_footprint": "UNKNOWN: cache geometry not pinned",
            "instruction_cache_misses": "UNKNOWN: unordered histogram has no fetch chronology",
        },
        "scope": {
            "id": scope_id,
            "selection": "explicit_function_union" if ranges else "whole_pc_histogram",
            "symbol_ranges": ranges,
            "unique_pcs": len(selected),
            "source_histogram_unique_pcs": len(hist),
            "elf_mapped_instructions": mapped,
            "outside_elf_instructions": sum(selected.values()) - mapped,
            "alias_or_shared_invocation_attribution": "UNKNOWN; no implicit alias following or per-call division",
            "excludes_harness": "UNKNOWN: function selection alone does not prove a measurement window",
            "caveat": "A function union is its exact PC set; call targets outside it and the hardware timing window require separate closure",
        },
        "artifacts": {
            "elf": {"path": str(elf.resolve()), "sha256": elf_sha},
            "histogram": {"path": str(histogram.resolve()), "sha256": hist_sha},
            "provider_sha256": _sha(Path(__file__)),
        },
        "execution_binding": binding
        or {"status": "UNKNOWN: execution producer/engine binding absent"},
        "engine": binding.get("engine", {"status": "UNKNOWN: engine not pinned"}),
        "nofsm_audit": audit,
        "unknown_primitive_funct_counts": dict(unknown),
        "status": "pass"
        if not unknown and mapped == sum(selected.values())
        else "UNKNOWN",
        "calibration_state": "No cycles fitted here; features bind Merlin's existing arbitrary JSON-pointer providers",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("histogram", type=Path)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--execution-binding", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    binding = (
        json.loads(args.execution_binding.read_text())
        if args.execution_binding
        else None
    )
    result = census(
        args.elf,
        args.histogram,
        symbols=args.symbol,
        scope_id=args.scope_id,
        execution_binding=binding,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
