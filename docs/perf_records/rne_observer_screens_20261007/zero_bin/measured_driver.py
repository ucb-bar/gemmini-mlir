"""Matched complete current-source M8 exact zero-bin observer alternative.

Only the generic Merlin observer representation changes. The target mode guard,
original source/table/scanners and complete common-address ROI remain pinned.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.late_quant_rne import _tokens
from merlin.llvmlower.scaled_integer_finite_llvm import bind_finite_scale_helpers
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    emit_source_interval_i8_lookup,
    find_closed_scalar_i8_observers,
)
from merlin.perf.layer_bench import build_program

from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from mlir_oot.no_fsm_audit import audit_elf

HERE = Path(__file__).resolve().parents[1]
CORE = Path("/scratch/agustin/tmp/merlin-rne-observer-cells-main-20261007")
FINITE = Path("/scratch/agustin/tmp/gemmini-tiny-finite-domain-20261007")
CONTROL = FINITE / "out/artifacts/probes/finite-scale-M8-v2-20261007"
OLD = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
SOURCE = OLD / "out/artifacts/probes/closed-i8-interval-result-20261007"
TYPED = Path(
    "/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006/out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic/typed_prepacket.generic.mlir"
)
LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
GCC = Path(
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
)
OUT = HERE / "out/artifacts/probes/rne-zero-observer-M8-20261007"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def rename(source, mapping):
    changes = [
        (t.start, t.end, mapping[t.text]) for t in _tokens(source) if t.text in mapping
    ]
    for first, last, value in reversed(changes):
        source = source[:first] + value + source[last:]
    return source


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "measured_driver.py").write_bytes(Path(__file__).read_bytes())
    commands = []

    def run(argv):
        argv = list(map(str, argv))
        commands.append(argv)
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refusals = find_closed_scalar_i8_observers(
        parse_mlir_text(TYPED.read_text()), effects=effects
    )
    assert len(proofs) == 22 and not refusals
    routes = json.loads((SOURCE / "source_binding.json").read_text())["routes"]
    helpers = bind_finite_scale_helpers(
        (SOURCE / "source.ll").read_text(),
        routes=routes,
        observers=proofs,
        effects=effects,
        immutable_inputs=True,
        fresh_disjoint_output=True,
    )
    assert len(helpers) == 1
    route_keys = {
        (r["source_expression_sha256"], r["quant_factor_bits"]) for r in routes
    }
    selected_proofs = tuple(
        p
        for p in proofs
        if (p.expression.canonical_sha256, p.quant_factor_bits) in route_keys
    )
    cells = tuple(
        prepare_bounded_rne_word_cells(p, effects=effects) for p in selected_proofs
    )
    table = build_source_interval_table(
        proofs[0].expression,
        effects=effects,
        leading_bits=16,
        max_table_bytes=512 * 1024,
    )
    ordinary = emit_source_interval_i8_lookup(
        table_name="table_b16",
        activation_name="finite_source_activation",
        quantizer_name="finite_quantize",
        lookup_name="finite_lookup_activation",
        leading_bits=16,
        finite_inputs=helpers,
        finite_table=table,
    )
    assert ordinary == (CONTROL / "lookup.c").read_text()
    lookup = emit_source_interval_i8_lookup(
        table_name="table_b16",
        activation_name="cells_source_activation",
        quantizer_name="cells_quantize",
        lookup_name="cells_lookup_activation",
        leading_bits=16,
        finite_inputs=helpers,
        finite_table=table,
        zero_observer_cells=cells,
    )
    (OUT / "lookup.c").write_text(lookup)
    mapping = {
        "@finite_M8_rne": "@cells_M8_rne",
        "@finite_lookup_activation": "@cells_lookup_activation",
    }
    (OUT / "selected.ll").write_text(
        rename((CONTROL / "selected.ll").read_text(), mapping)
    )
    (OUT / "activation.ll").write_text(
        rename(
            (CONTROL / "activation.ll").read_text(),
            {"@finite_source_activation": "@cells_source_activation"},
        )
    )
    (OUT / "quantize.ll").write_text(
        rename(
            (CONTROL / "quantize.ll").read_text(),
            {"@finite_quantize": "@cells_quantize"},
        )
    )
    scanner = (
        (CONTROL / "scale_guard.c")
        .read_text()
        .replace("prepared_M8", "cells_prepared_M8")
        .replace("finite_M8_rne", "cells_M8_rne")
    )
    (OUT / "scale_guard.c").write_text(scanner)
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
    for stem in ("lookup", "scale_guard"):
        run(
            [
                LLVM / "clang",
                *flags,
                "-S",
                "-emit-llvm",
                OUT / (stem + ".c"),
                "-o",
                OUT / (stem + ".ll"),
            ]
        )
    run(
        [
            LLVM / "llvm-link",
            "-S",
            *[
                OUT / (stem + ".ll")
                for stem in (
                    "selected",
                    "lookup",
                    "activation",
                    "quantize",
                    "scale_guard",
                )
            ],
            "-o",
            OUT / "linked.ll",
        ]
    )
    run(
        [
            LLVM / "opt",
            "-S",
            "-passes=always-inline",
            OUT / "linked.ll",
            "-o",
            OUT / "inlined.ll",
        ]
    )
    merlin_host_llvm_transform(LLVM, combine_clamp=True)(
        OUT / "inlined.ll", OUT / "late_rne"
    )
    run(
        [
            LLVM / "clang",
            *flags,
            "-c",
            OUT / "late_rne/model.ll",
            "-o",
            OUT / "candidate.o",
        ]
    )
    assert sha(OUT / "candidate.o") != sha(CONTROL / "candidate.o")
    (OUT / "candidate.dump").write_text(
        run([LLVM / "llvm-objdump", "-dr", OUT / "candidate.o"]).stdout
    )
    # Target FRM binding delegates unsupported rounding to unchanged source.
    mode_guard = (
        (CONTROL / "target_guard.c")
        .read_text()
        .replace("prepared_M8", "cells_prepared_M8")
        .replace("finite_M8(", "cells_M8(")
    )
    (OUT / "target_guard.c").write_text(mode_guard)
    run(
        [
            LLVM / "clang",
            *flags,
            "-c",
            OUT / "target_guard.c",
            "-o",
            OUT / "target_guard.o",
        ]
    )
    old = json.loads((CONTROL / "timing/qualification.json").read_text())
    for path, digest in old["objects"].items():
        assert sha(path) == digest
    existing = [Path(p) for p in old["objects"] if not p.endswith("/main.o")]
    events = []
    for name in ("all_modes", "timing"):
        text = (CONTROL / name / "main.c").read_text()
        text = text.replace(
            "F pair[2]={integer_M8_b16,finite_M8}", "F pair[2]={finite_M8,cells_M8}"
        )
        text = text.replace(
            "extern void finite_M8(int32_t*,float*,int32_t*,float*,int8_t*);",
            "extern void finite_M8(int32_t*,float*,int32_t*,float*,int8_t*);\nextern void cells_M8(int32_t*,float*,int32_t*,float*,int8_t*);",
        )
        text = text.replace(
            "finite_M8(a,scale_a,b,scale_b,guarded+64)",
            "cells_M8(a,scale_a,b,scale_b,guarded+64)",
        )
        # All-mode source and control comparisons remain unchanged, candidate
        # has exactly the existing output guard placement and immutable inputs.
        case = OUT / name
        case.mkdir()
        (case / "main.c").write_text(text)
        run([LLVM / "clang", *flags, "-c", case / "main.c", "-o", case / "main.o"])
        objects = [
            *existing,
            OUT / "candidate.o",
            OUT / "target_guard.o",
            case / "main.o",
        ]
        before = {str(p): sha(p) for p in objects}
        build = build_program(
            objects, case / "build", target="gemmini", max_loaded_bytes=None
        )
        assert before == {str(p): sha(p) for p in objects}
        audit = audit_elf(build.elf.read_bytes())
        assert audit["status"] == "pass"
        save(case / "nofsm_audit.json", audit)
        result = run(
            [GCC.with_name("spike"), "--isa=rv64gc", "--extension=gemmini", build.elf]
        )
        (case / "spike.stdout").write_text(result.stdout)
        (case / "spike.stderr").write_text(result.stderr)
        log = result.stdout + result.stderr
        assert (
            "INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0" in log
            and "INTEGER_RESULT_FAIL" not in log
        )
        rows = [
            dict(part.split("=", 1) for part in line.split()[1:])
            for line in log.splitlines()
            if line.startswith("INTEGER_RESULT_ROW ")
        ]
        assert len(rows) == 4 and [int(r["id"]) for r in rows] == [0, 1, 1, 0]
        qualification = {
            "schema": "bounded_rne_source_word_cells_complete_M8_v1",
            "status": "pass",
            "mode_scope": "all5modes/7sticky"
            if name == "all_modes"
            else "initialRNE plus ABBA",
            "original_i8_words": 45056,
            "rows": rows,
            "objects": before,
            "elf_path": str(build.elf),
            "elf_sha256": sha(build.elf),
            "observer_cells_bytes": 0,
            "activation_table_bytes": len(table.data),
            "complete_scope": "Both immutable scale scans, original source table, all finishing rounded multiplies, original cold source fallback, frame, dispatch and complete output stores are inside each ROI.",
            "source_policy": "Existing exact i8 source-observer closure; unsupported rounding original source; explicit unobserved flags/nontrapping permission for omitted second observer.",
            "stock_cycles": "UNKNOWN",
            "commands": commands,
            "token_usage_available": False,
        }
        save(case / "qualification.json", qualification)
        events.append(qualification)
        print("OBSERVER_CELLS_STRICT_PASS", name, rows, flush=True)
    save(
        OUT / "qualification.json",
        {
            "status": "pass",
            "events": events,
            "table_sha256": table.sha256,
            "typed_source_sha256": sha(TYPED),
            "ordinary_control_emission_byteexact": True,
            "pins": {
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    TYPED,
                    *OUT.rglob("*"),
                    *existing,
                    CORE / "src/merlin/llvmlower/source_expression_interval.py",
                    CORE / "src/merlin/llvmlower/bounded_rne_word_cells.py",
                ]
                if p.is_file()
            },
            "token_usage_available": False,
        },
    )


if __name__ == "__main__":
    main()
