"""Seal a named-storage successor without changing either measured helper."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import source_interval_hierarchy_census as census
from merlin.perf.layer_bench import build_program

from mlir_oot.no_fsm_audit import audit_elf

FIRST = (
    census.HERE / "out/artifacts/probes/source-interval-hierarchy-complete-M8-20261007"
)
OUT = (
    census.HERE
    / "out/artifacts/probes/source-interval-hierarchy-storage-successor-20261007"
)
LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
GCC = Path(
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
)


def parse_report(log):
    """Refuse incomplete ABBA, output, named-storage or immutable-table evidence."""
    if (
        "INTEGER_RESULT_FAIL" in log
        or "INTEGER_RESULT_SOURCE_GATE initialRNE words=45056 original guards PASS"
        not in log
    ):
        raise ValueError("missing original complete source gate")
    if "INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0" not in log:
        raise ValueError("missing full output, guard or input closure")
    rows, storage = [], []
    for line in log.splitlines():
        if line.startswith("INTEGER_RESULT_ROW "):
            rows.append(dict(p.split("=", 1) for p in line.split()[1:]))
        if line.startswith("INTEGER_RESULT_STORAGE "):
            storage.append(dict(p.split("=", 1) for p in line.split()[1:]))
    if len(rows) != 4 or len(storage) != 4:
        raise ValueError("exactly four complete paired windows required")
    named = {
        "a",
        "sa",
        "b",
        "sb",
        "expected",
        "guarded",
        "out",
        "fine",
        "coarse",
        "fine_hash",
        "coarse_hash",
    }
    for sample, (row, owner) in enumerate(zip(rows, storage)):
        if set(row) != {"id", "sample", "cycles", "instructions", "digest"} or set(
            owner
        ) != named | {"id", "sample"}:
            raise ValueError("complete known window and storage schema required")
        if (
            int(row["sample"]) != sample
            or int(owner["sample"]) != sample
            or int(row["id"]) != (0, 1, 1, 0)[sample]
            or row["id"] != owner["id"]
        ):
            raise ValueError("ABBA order/storage association changed")
        if (
            row["digest"] != "b702909ca80dfe2a"
            or int(row["cycles"]) <= 0
            or int(row["instructions"]) <= 0
        ):
            raise ValueError("original output or complete counters changed")
        if {k: owner[k] for k in named} != {k: storage[0][k] for k in named}:
            raise ValueError("physical owners or immutable table words changed")
        if int(owner["out"], 16) != int(owner["guarded"], 16) + 64:
            raise ValueError("public output does not retain its guard arena")
    return {
        "rows": rows,
        "named_storage": storage,
        "actual_hardware_cycles": "UNKNOWN until stock execution",
        "scope": "complete first M8 helper; prevalidation warms both table/helper paths; no independent cold-cache observation",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    prior = json.loads((FIRST / "timing/qualification.json").read_text())
    assert all(census.sha(p) == h for p, h in prior["objects"].items())
    commands = []

    def run(argv, stem):
        argv = [str(a) for a in argv]
        commands.append(argv)
        r = subprocess.run(argv, check=False, capture_output=True, text=True)
        (OUT / (stem + ".stdout")).write_text(r.stdout)
        (OUT / (stem + ".stderr")).write_text(r.stderr)
        r.check_returncode()
        return r

    text = (FIRST / "timing/main.c").read_text()
    text = text.replace(
        "static int8_t guarded[45184]",
        "extern const uint32_t table_b16[][2],hierarchy_coarse_table[][2];\nstatic int8_t guarded[45184]",
    )
    anchor = " F pair[2]={cells_M8,hierarchy_M8};"
    assert text.count(anchor) == 1
    text = text.replace(
        anchor,
        " uint64_t finepin=hash(table_b16,524288),coarsepin=hash(hierarchy_coarse_table,65536);\n"
        + anchor,
    )
    anchor = "  if(valid())return 3;"
    assert text.count(anchor) == 1
    text = text.replace(
        anchor,
        anchor
        + '\n  if(hash(table_b16,524288)!=finepin||hash(hierarchy_coarse_table,65536)!=coarsepin)return 5;\n  printf("INTEGER_RESULT_STORAGE id=%u sample=%u a=%lx sa=%lx b=%lx sb=%lx expected=%lx guarded=%lx out=%lx fine=%lx coarse=%lx fine_hash=%lx coarse_hash=%lx\\n",id,sample,(unsigned long)a,(unsigned long)scale_a,(unsigned long)b,(unsigned long)scale_b,(unsigned long)expected,(unsigned long)guarded,(unsigned long)(guarded+64),(unsigned long)table_b16,(unsigned long)hierarchy_coarse_table,(unsigned long)finepin,(unsigned long)coarsepin);',
    )
    (OUT / "main.c").write_text(text)
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O3",
        "-ffreestanding",
        "-fno-builtin",
        "-ffp-contract=off",
    ]
    run(
        [LLVM / "clang", *flags, "-c", OUT / "main.c", "-o", OUT / "main.o"],
        "main_compile",
    )
    objects = [Path(p) for p in prior["objects"] if not p.endswith("/main.o")] + [
        OUT / "main.o"
    ]
    before = {str(p): census.sha(p) for p in objects}
    built = build_program(
        objects, OUT / "build", target="gemmini", max_loaded_bytes=None
    )
    assert before == {str(p): census.sha(p) for p in objects}
    audit = audit_elf(built.elf.read_bytes())
    assert audit["status"] == "pass"
    census.save(OUT / "nofsm.json", audit)
    log = run(
        [GCC.with_name("spike"), "--isa=rv64gc", "--extension=gemmini", built.elf],
        "spike",
    ).stdout
    parsed = parse_report(log)
    symbols = run(
        [LLVM / "llvm-nm", "--print-size", "--defined-only", built.elf], "symbols"
    ).stdout
    run([LLVM / "llvm-objdump", "-dr", built.elf], "disassembly")
    selected_symbols = []
    names = {
        "a",
        "b",
        "scale_a",
        "scale_b",
        "expected",
        "guarded",
        "table_b16",
        "hierarchy_coarse_table",
        "cells_M8_rne",
        "hierarchy_M8_rne",
        "cells_M8",
        "hierarchy_M8",
    }
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[-1] in names:
            selected_symbols.append(
                {
                    "name": fields[-1],
                    "address": fields[0],
                    "bytes": int(fields[1], 16),
                    "kind": fields[2],
                }
            )
    assert {s["name"] for s in selected_symbols} == names
    for key, symbol in {
        "a": "a",
        "sa": "scale_a",
        "b": "b",
        "sb": "scale_b",
        "expected": "expected",
        "guarded": "guarded",
        "fine": "table_b16",
        "coarse": "hierarchy_coarse_table",
    }.items():
        address = next(s["address"] for s in selected_symbols if s["name"] == symbol)
        assert int(parsed["named_storage"][0][key], 16) == int(address, 16)
    census.save(
        OUT / "qualification.json",
        {
            "schema": "source_hierarchy_named_storage_successor_v1",
            "status": "pass",
            **parsed,
            "elf_path": str(built.elf),
            "elf_sha256": census.sha(built.elf),
            "objects": before,
            "original_timing_manifest": str(FIRST / "timing/qualification.json"),
            "original_all5FRM_7sticky_manifest": str(
                FIRST / "all_modes/qualification.json"
            ),
            "helper_objects_unchanged": True,
            "private_arena": "No dynamic/caller-owned scratch arena; complete mutable destination is guarded[45184]. Per-function stack frames differ and stay unpriced.",
            "source_tables": "fine524288B and coarse65536B readonly bytes hashed outside ROI every window; global addresses equal for all arms",
            "unused_readonly_owners": "Legacy table18/20 data remain in both arms from immutable original object list; no claim whole ELF has only589824readonly bytes",
            "cold_warm_scope": parsed["scope"],
            "symbols": selected_symbols,
            "commands": commands,
            "token_usage_available": False,
        },
    )
    print(
        json.dumps(
            {
                "elf": str(built.elf),
                "sha256": census.sha(built.elf),
                "rows": parsed["rows"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
