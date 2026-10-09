"""Named owners with table validation after timing, not between windows."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import source_interval_hierarchy_census as census
import source_interval_hierarchy_packet as first
from merlin.perf.layer_bench import build_program

from mlir_oot.no_fsm_audit import audit_elf

OUT = census.HERE / "out/artifacts/probes/source-interval-hierarchy-storage-v2-20261007"


def fnv(data):
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(value, "x")


def parse_report(log):
    result = first.parse_report(log)
    record = [
        line
        for line in log.splitlines()
        if line.startswith("INTEGER_RESULT_TABLE_HASH_PASS ")
    ]
    if len(record) != 1:
        raise ValueError("one post-ROI full immutable-table check required")
    fields = dict(p.split("=", 1) for p in record[0].split()[1:])
    owner = result["named_storage"][0]
    if fields != {
        "fine_hash": owner["fine_hash"],
        "coarse_hash": owner["coarse_hash"],
        "after_windows": "4",
    }:
        raise ValueError("post-ROI immutable-table evidence differs")
    result["table_validation_scope"] = (
        "No runtime full-table hash before or between timing windows. Final hash matches source-derived original bytes after all4 windows. Initial candidate source gate remains outside ROI."
    )
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    prior = json.loads((first.OUT / "qualification.json").read_text())
    objects = [Path(p) for p in prior["objects"] if not p.endswith("/main.o")]
    assert all(census.sha(p) == h for p, h in prior["objects"].items())
    fine = fnv((census.OUT / "table_b16.bin").read_bytes())
    coarse = fnv((census.OUT / "table_b13.bin").read_bytes())
    text = (first.OUT / "main.c").read_text()
    before = "uint64_t finepin=hash(table_b16,524288),coarsepin=hash(hierarchy_coarse_table,65536);"
    perrow = "if(hash(table_b16,524288)!=finepin||hash(hierarchy_coarse_table,65536)!=coarsepin)return 5;"
    assert text.count(before) == text.count(perrow) == 1
    text = text.replace(before, f"uint64_t finepin=0x{fine}ul,coarsepin=0x{coarse}ul;")
    text = text.replace(
        perrow, "/* Full immutable-byte scan is after all ROI windows. */"
    )
    end = (
        ' printf("INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0\\n");'
    )
    assert text.count(end) == 1
    text = text.replace(
        end,
        ' if(hash(table_b16,524288)!=finepin||hash(hierarchy_coarse_table,65536)!=coarsepin)return 5;\n printf("INTEGER_RESULT_TABLE_HASH_PASS fine_hash=%lx coarse_hash=%lx after_windows=4\\n",(unsigned long)finepin,(unsigned long)coarsepin);\n'
        + end,
    )
    (OUT / "main.c").write_text(text)
    commands = []

    def run(argv, name):
        argv = list(map(str, argv))
        commands.append(argv)
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        (OUT / (name + ".stdout")).write_text(result.stdout)
        (OUT / (name + ".stderr")).write_text(result.stderr)
        result.check_returncode()
        return result

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
        [first.LLVM / "clang", *flags, "-c", OUT / "main.c", "-o", OUT / "main.o"],
        "main_compile",
    )
    objects.append(OUT / "main.o")
    pins = {str(p): census.sha(p) for p in objects}
    built = build_program(
        objects, OUT / "build", target="gemmini", max_loaded_bytes=None
    )
    assert pins == {str(p): census.sha(p) for p in objects}
    audit = audit_elf(built.elf.read_bytes())
    assert audit["status"] == "pass"
    census.save(OUT / "nofsm.json", audit)
    log = run(
        [
            first.GCC.with_name("spike"),
            "--isa=rv64gc",
            "--extension=gemmini",
            built.elf,
        ],
        "spike",
    ).stdout
    parsed = parse_report(log)
    symbols = run(
        [first.LLVM / "llvm-nm", "--print-size", "--defined-only", built.elf], "symbols"
    ).stdout
    actual_symbols = []
    for record in prior["symbols"]:
        fields = next(
            line.split()
            for line in symbols.splitlines()
            if line.split()[-1] == record["name"]
        )
        assert int(fields[1], 16) == record["bytes"]
        actual_symbols.append(
            {
                "name": fields[-1],
                "address": fields[0],
                "bytes": int(fields[1], 16),
                "kind": fields[2],
            }
        )
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
        line = next(
            line.split() for line in symbols.splitlines() if line.split()[-1] == symbol
        )
        assert int(parsed["named_storage"][0][key], 16) == int(line[0], 16)
    q = {
        **prior,
        **parsed,
        "schema": "source_hierarchy_named_storage_v2_v1",
        "elf_path": str(built.elf),
        "elf_sha256": census.sha(built.elf),
        "objects": pins,
        "symbols": actual_symbols,
        "commands": commands,
        "prior_successor": str(first.OUT / "qualification.json"),
        "cold_warm_scope": "Initial actual candidate source gate warms original firstM8 requests only; no full-table hash before/between windows. No independent cold-cache measurement; all post-gate validation outside ROI.",
    }
    census.save(OUT / "qualification.json", q)
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
