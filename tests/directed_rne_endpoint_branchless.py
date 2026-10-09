"""Complete current-source M8 branchless directed far-edge observation pair."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


old = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
census = SimpleNamespace(
    sha=sha,
    save=save,
    OLD=old,
    RNE=Path("/scratch/agustin/tmp/gemmini-rne-observer-cells-20261007"),
    TYPED=Path(
        "/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006/out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic/typed_prepacket.generic.mlir"
    ),
    CAPTURE=old / "out/artifacts/probes/source-continuation-contexts-v3-20261007",
    OUT=Path(
        "/scratch/agustin/tmp/gemmini-source-table-hierarchy-20261007/out/artifacts/probes/source-interval-hierarchy-census-v2-20261007"
    ),
)
CORE = Path("/scratch/agustin/tmp/merlin-directed-rne-endpoint-branchless-20261007")
OUT = HERE / "out/artifacts/probes/directed-rne-endpoint-branchless-M8-20261007"
CONTROL = census.RNE / "out/artifacts/probes/rne-zero-observer-M8-20261007"
SOURCE = census.OLD / "out/artifacts/probes/closed-i8-interval-result-20261007"
LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
GCC = Path(
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
)


def rename(source, mapping):
    changes = [
        (t.start, t.end, mapping[t.text]) for t in _tokens(source) if t.text in mapping
    ]
    for begin, end, value in reversed(changes):
        source = source[:begin] + value + source[end:]
    return source


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    census.save(
        OUT / "declaration.json",
        {
            "schema": "source_directed_complete_M8_predeclaration_v1",
            "control": "Current2085 exact zero observer +16bit512KiB table, original source scans/continuation/finishing",
            "candidate": "Same general exact far RNE-bin proof; source multiplier sign selects one prederived edge, inverted ordered key replaces direction branch. Zero-bin/table/products/source continuation unchanged. First direction-branch arm preserved separately.",
            "source_DAG_shape_and_input": "Same actual source8x5632 helper, original captured45056 outputs, identical physical input/output/table objects in paired ELF",
            "cost_scope": "Both scale scans, all source input casts/roundedproducts/tabletraffic/observer certificates/refinement/original cold source continuation/frame/stores within each ROI",
            "prelabel_forecast": "UNKNOWN; one redundant saturated RNE removed versus sign/key/edge load/branch work and new2KiB owner. No calibrated model prices these dependencies or physical memory.",
            "immutable_storage_added_bytes": 2048,
            "original_source_table_bytes": 524288,
            "no_whole_model_forecast": True,
            "stock_cycles": "UNKNOWN",
            "original_capsule_gate": "Every45056 original i8 byte/guards/sourcehash/all5FRM+7sticky outside timing prerequisite",
            "prior_source_capture": str(census.CAPTURE / "features_v2/contexts.json"),
        },
    )
    commands = []

    def run(argv, label=None):
        argv = list(map(str, argv))
        commands.append(argv)
        result = subprocess.run(argv, capture_output=True, text=True)
        if label:
            (OUT / (label + ".stdout")).write_text(result.stdout)
            (OUT / (label + ".stderr")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refused = find_closed_scalar_i8_observers(
        parse_mlir_text(census.TYPED.read_text()), effects=effects
    )
    assert len(proofs) == 22 and not refused
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
    keys = {(r["source_expression_sha256"], r["quant_factor_bits"]) for r in routes}
    selected = tuple(
        p
        for p in proofs
        if (p.expression.canonical_sha256, p.quant_factor_bits) in keys
    )
    zero = tuple(prepare_bounded_rne_word_cells(p, effects=effects) for p in selected)
    table = build_source_interval_table(
        selected[0].expression,
        effects=effects,
        leading_bits=16,
        max_table_bytes=512 * 1024,
    )
    fine = emit_source_interval_i8_lookup(
        table_name="table_b16",
        activation_name="cells_source_activation",
        quantizer_name="cells_quantize",
        lookup_name="cells_lookup_activation",
        leading_bits=16,
        finite_inputs=helpers,
        finite_table=table,
        zero_observer_cells=zero,
    )
    assert fine == (CONTROL / "lookup.c").read_text()
    assert table.data == (census.OUT / "table_b16.bin").read_bytes()
    c = emit_source_interval_i8_lookup(
        table_name="table_b16",
        activation_name="cells_source_activation",
        quantizer_name="cells_quantize",
        lookup_name="directed_lookup_activation",
        leading_bits=16,
        finite_inputs=helpers,
        finite_table=table,
        zero_observer_cells=zero,
        ordered_observer_word_cells=zero,
    )
    (OUT / "lookup.c").write_text(c)
    mapping = {
        "@cells_M8_rne": "@directed_M8_rne",
        "@cells_lookup_activation": "@directed_lookup_activation",
    }
    (OUT / "selected.ll").write_text(
        rename((CONTROL / "selected.ll").read_text(), mapping)
    )
    scanner = (
        (CONTROL / "scale_guard.c")
        .read_text()
        .replace("cells_prepared_M8", "directed_prepared_M8")
        .replace("cells_M8_rne", "directed_M8_rne")
    )
    (OUT / "scale_guard.c").write_text(scanner)
    guard = (
        (CONTROL / "target_guard.c")
        .read_text()
        .replace("cells_prepared_M8", "directed_prepared_M8")
        .replace("cells_M8(", "directed_M8(")
    )
    (OUT / "target_guard.c").write_text(guard)
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
    # The original callbacks are immutable executed control definitions. Linking
    # their LLVM lets the existing target RNE legalizer see precisely that source
    # observer. Strong unused duplicate exports are renamed for final linkage.
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
            ],
            stem + "_compile",
        )
    for stem in ("activation", "quantize"):
        (OUT / (stem + ".ll")).write_bytes((CONTROL / (stem + ".ll")).read_bytes())
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
        ],
        "link_llvm",
    )
    run(
        [
            LLVM / "opt",
            "-S",
            "-passes=always-inline",
            OUT / "linked.ll",
            "-o",
            OUT / "inlined.ll",
        ],
        "inline",
    )
    merlin_host_llvm_transform(LLVM, combine_clamp=True)(
        OUT / "inlined.ll", OUT / "late_rne"
    )
    actual = rename(
        (OUT / "late_rne/model.ll").read_text(),
        {
            "@cells_source_activation": "@directed_source_activation",
            "@cells_quantize": "@directed_quantize",
        },
    )
    (OUT / "compiled.ll").write_text(actual)
    run(
        [LLVM / "clang", *flags, "-c", OUT / "compiled.ll", "-o", OUT / "candidate.o"],
        "candidate_compile",
    )
    run(
        [
            LLVM / "clang",
            *flags,
            "-c",
            OUT / "target_guard.c",
            "-o",
            OUT / "target_guard.o",
        ],
        "guard_compile",
    )
    (OUT / "candidate.dump").write_text(
        run([LLVM / "llvm-objdump", "-dr", OUT / "candidate.o"]).stdout
    )
    prior = json.loads((CONTROL / "timing/qualification.json").read_text())
    for path, digest in prior["objects"].items():
        assert census.sha(path) == digest
    existing = [Path(path) for path in prior["objects"] if not path.endswith("/main.o")]
    cases = []
    for name in ("all_modes", "timing"):
        case = OUT / name
        case.mkdir()
        text = (CONTROL / name / "main.c").read_text()
        text = text.replace(
            "F pair[2]={finite_M8,cells_M8}", "F pair[2]={cells_M8,directed_M8}"
        )
        text = text.replace(
            "extern void cells_M8(int32_t*,float*,int32_t*,float*,int8_t*);",
            "extern void cells_M8(int32_t*,float*,int32_t*,float*,int8_t*);\nextern void directed_M8(int32_t*,float*,int32_t*,float*,int8_t*);",
        )
        text = text.replace(
            "cells_M8(a,scale_a,b,scale_b,guarded+64)",
            "directed_M8(a,scale_a,b,scale_b,guarded+64)",
        )
        (case / "main.c").write_text(text)
        run(
            [LLVM / "clang", *flags, "-c", case / "main.c", "-o", case / "main.o"],
            name + "_main_compile",
        )
        objects = [
            *existing,
            OUT / "candidate.o",
            OUT / "target_guard.o",
            case / "main.o",
        ]
        before = {str(p): census.sha(p) for p in objects}
        built = build_program(
            objects, case / "build", target="gemmini", max_loaded_bytes=None
        )
        assert before == {str(p): census.sha(p) for p in objects}
        audit = audit_elf(built.elf.read_bytes())
        assert audit["status"] == "pass"
        census.save(case / "nofsm.json", audit)
        result = run(
            [GCC.with_name("spike"), "--isa=rv64gc", "--extension=gemmini", built.elf],
            name + "_spike",
        )
        assert (
            "INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0"
            in result.stdout
        )
        assert "INTEGER_RESULT_FAIL" not in result.stdout
        rows = [
            dict(part.split("=", 1) for part in line.split()[1:])
            for line in result.stdout.splitlines()
            if line.startswith("INTEGER_RESULT_ROW ")
        ]
        assert len(rows) == 4 and [int(r["id"]) for r in rows] == [0, 1, 1, 0]
        q = {
            "schema": "source_directed_complete_M8_strict_v1",
            "status": "pass",
            "mode_scope": "all5FRM/7sticky_source_word_comparisons"
            if name == "all_modes"
            else "initialRNE/original45056 gate plus ABBA",
            "original_i8_outputs": 45056,
            "rows": rows,
            "elf_path": str(built.elf),
            "elf_sha256": census.sha(built.elf),
            "objects": before,
            "cycles": "UNKNOWN: functional Spike counters are retired work",
            "cost_scope": "Both source scans, casts/roundedinputproducts, unchanged table traffic/exact bin certificates/original sourcecontinuation/frames/outputstores insideROI. Guards/inputhash outsideROI. Actual same data/table objects linked for both arms; no hierarchy/refinement.",
            "new_storage_bytes": 2048,
            "total_used_table_bytes": len(table.data),
            "token_usage_available": False,
        }
        census.save(case / "qualification.json", q)
        cases.append(q)
        print("DIRECTED_COMPLETE_STRICT", name, rows, flush=True)
    pins = {
        str(p): census.sha(p)
        for p in [
            Path(__file__),
            census.TYPED,
            census.OUT / "qualification.json",
            CORE / "src/merlin/llvmlower/source_expression_interval.py",
            *existing,
            *OUT.rglob("*"),
        ]
        if p.is_file()
    }
    census.save(
        OUT / "qualification.json",
        {
            "schema": "source_directed_complete_M8_all_qualification_v1",
            "status": "pass",
            "cases": cases,
            "source_control_lookup_C_byteexact": True,
            "original_fine_table_bytes_exact": True,
            "generic_source_observers": len(proofs),
            "selected_helper_observers": len(selected),
            "target_ISA_ABI": "OOT existing qualified RNE/clamp andFRMguard; noISA inMerlin",
            "all_guard_and_source_effects_retained": True,
            "direction_proof": "Source-proven strictly positive finite factor and universally rederived finite table endpoints; current actual finite input producer/scans. Rounded products monotone in source endpoint under actual up sign, first observation in its preimage, only far edge needed. Signed zeros share zero-bin; unsupported mode/factor original source.",
            "physical_memory_cost_and_whole_price": "UNKNOWN",
            "commands": commands,
            "pins": pins,
            "token_usage_available": False,
        },
    )


if __name__ == "__main__":
    main()
