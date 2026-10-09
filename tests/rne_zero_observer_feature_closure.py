"""Same-ELF source-bound observer cost adjunct; all cycle prices unknown."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.scaled_integer_finite_llvm import bind_finite_scale_helpers
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    emit_source_interval_i8_lookup,
    find_closed_scalar_i8_observers,
)
from rne_zero_observer_capsule import CONTROL, CORE, OUT, SOURCE, TYPED, save, sha

from mlir_oot.cpu_fp_distances import census as distances
from mlir_oot.executed_features import _symbol_ranges, census, parse_pc_histogram


def main():
    elf = OUT / "timing/build/layer.elf"
    hist = OUT / "timing/histogram_g.stderr"
    stdout = OUT / "timing/histogram_g.stdout"
    symbols = [
        "finite_M8_rne",
        "cells_M8_rne",
        "finite_source_activation",
        "cells_source_activation",
    ]
    binding = {
        "elf_sha256": sha(elf),
        "histogram_sha256": sha(hist),
        "scope_id": "source_observer_same_elf_function_union",
        "engine": {
            "path": "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike",
            "sha256": sha(
                "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike"
            ),
            "argv": ["--isa=rv64gc", "--extension=gemmini", "-g", str(elf)],
            "stdout": str(stdout),
            "stdout_sha256": sha(stdout),
        },
    }
    assert (
        "INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0"
        in stdout.read_text()
    )
    # Exact histogram and actual symbol entry multiplicities, not guessed work.
    counts = parse_pc_histogram(hist.read_text())
    entries = {
        r["symbol"]: counts.get(r["start"], 0)
        for r in _symbol_ranges(elf, symbols, "readelf")
    }
    assert entries["finite_M8_rne"] == 2 and entries["cells_M8_rne"] == 3
    features = {}
    for symbol in symbols:
        scoped_binding = {**binding, "scope_id": symbol}
        features[symbol] = census(
            elf,
            hist,
            symbols=[symbol],
            scope_id=symbol,
            execution_binding=scoped_binding,
        )
    save(
        OUT / "execution_features.json",
        {
            "features": features,
            "actual_entry_counts": entries,
            "scope": "2 complete finite-control calls and3 zero-successor calls including one initial RNE gate. Symbol-only CPU scopes exclude mode/scan wrappers, source call children counted separately; ROI costs remain complete in primary timing.",
            "cycles": "UNKNOWN",
        },
    )
    inc = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/include/riscv")
    fp = distances(elf, hist, symbols, inc / "encoding.h", inc / "decode.h")
    save(OUT / "fp_dependencies.json", fp)
    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refused = find_closed_scalar_i8_observers(
        parse_mlir_text(TYPED.read_text()), effects=effects
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
    keys = {(r["source_expression_sha256"], r["quant_factor_bits"]) for r in routes}
    cells = tuple(
        prepare_bounded_rne_word_cells(p, effects=effects)
        for p in proofs
        if (p.expression.canonical_sha256, p.quant_factor_bits) in keys
    )
    table = build_source_interval_table(
        proofs[0].expression,
        effects=effects,
        leading_bits=16,
        max_table_bytes=512 * 1024,
    )
    kwargs = dict(
        table_name="table_b16",
        activation_name="cells_source_activation",
        quantizer_name="cells_quantize",
        lookup_name="cells_lookup_activation",
        leading_bits=16,
        finite_inputs=helpers,
        finite_table=table,
    )
    zero = emit_source_interval_i8_lookup(**kwargs, zero_observer_cells=cells)
    full = emit_source_interval_i8_lookup(**kwargs, observer_word_cells=cells)
    assert zero == (OUT / "lookup.c").read_text()
    word_out = OUT.parent / "rne-observer-cells-M8-v2-20261007"
    assert full == (word_out / "lookup.c").read_text()
    ordinary = emit_source_interval_i8_lookup(
        **{
            **kwargs,
            "activation_name": "finite_source_activation",
            "quantizer_name": "finite_quantize",
            "lookup_name": "finite_lookup_activation",
        }
    )
    assert ordinary == (CONTROL / "lookup.c").read_text()
    pins = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            *[
                CORE / "src/merlin/llvmlower" / name
                for name in (
                    "bounded_rne_word_cells.py",
                    "source_expression_interval.py",
                )
            ],
            elf,
            hist,
            stdout,
            TYPED,
            inc / "encoding.h",
            inc / "decode.h",
            OUT / "lookup.c",
            word_out / "lookup.c",
            CONTROL / "lookup.c",
            OUT / "execution_features.json",
            OUT / "fp_dependencies.json",
        ]
    }
    save(
        OUT / "normalized_source_reemission.json",
        {
            "schema": "bounded_observer_normalized_source_reemission_v1",
            "status": "pass",
            "ordinary_word_and_zero_lookup_C_byteidentical_to_executed": True,
            "table_sha256": table.sha256,
            "original_primary_receipts_unchanged": True,
            "measured_code_snapshots": str(OUT / "measured_core_identity.json"),
            "pins": pins,
        },
    )
    save(
        OUT / "prospective_cost_declaration.json",
        {
            "schema": "source_observer_complete_cost_prospective_v1",
            "control": "Complete current finite-scanned original source M8 helper",
            "candidate": "Only exact sufficient zero-bin test before unchanged original i8 observers",
            "hypothesis": "Proven zero bins remove two FMIN/FMAX/FCVT chains. Extra integer magnitude operations/branch can lose on nonzero data; complete M8 stock label is needed.",
            "prediction": {
                "winner": "UNKNOWN",
                "cycles": "UNKNOWN",
                "reason": "Existing stream/calibrated helpers do not price this mixed branch/FP/continuation dependency path. No fit or instruction-to-cycle conversion.",
            },
            "physical_table_memory_and_cross_call_order": "UNKNOWN",
            "all22_context_transfer": "UNKNOWN; first context has a different output distribution",
            "new_storage_bytes": 0,
            "original_source_table_bytes": 512 * 1024,
            "outside_ROI": "Existing data/table/initial guard qualification; immutable compiler-generated table is shipped in ELF, no lazy generation excluded",
            "no_whole_forecast": True,
            "pins": pins,
            "token_usage_available": False,
        },
    )
    print("RNE_ZERO_FEATURE_REEMISSION_PASS", entries, flush=True)


if __name__ == "__main__":
    main()
