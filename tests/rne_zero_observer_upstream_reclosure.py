"""Re-emit measured normal helpers with the published generic Merlin topic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import merlin.llvmlower.bounded_rne_word_cells as cells_module
import merlin.llvmlower.source_expression_interval as interval_module
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.scaled_integer_finite_llvm import (
    bind_finite_scale_helpers,
    emit_finite_scale_helper,
)
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    emit_source_interval_i8_lookup,
    find_closed_scalar_i8_observers,
)
from rne_zero_observer_whole_prepare import CONTROL, CORE, OLD, OUT, SOURCE, save, sha

UPSTREAM = Path("/scratch/agustin/tmp/merlin-rounded-polynomial-main-20261007")
HEAD = "c0f40f8f8d100b841c26fe8bbd6c09e79b6de17c"


def main():
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
        ).strip()
        == HEAD
    )
    for module in (cells_module, interval_module):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(UPSTREAM / "src")
        assert sha(path) == sha(CORE / path.relative_to(UPSTREAM))
    typed = Path(
        "/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006/out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic/typed_prepacket.generic.mlir"
    )
    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refused = find_closed_scalar_i8_observers(
        parse_mlir_text(typed.read_text()), effects=effects
    )
    assert len(proofs) == 22 and not refused
    binding = json.loads(
        (CONTROL / "selected/target/host_llvm/source_binding.json").read_text()
    )
    helpers = bind_finite_scale_helpers(
        SOURCE.read_text(),
        routes=binding["source_bindings"]["routes"],
        observers=proofs,
        effects=effects,
        immutable_inputs=True,
        fresh_disjoint_output=True,
    )
    table = build_source_interval_table(
        proofs[0].expression,
        effects=effects,
        leading_bits=16,
        max_table_bytes=512 * 1024,
    )
    assert table.sha256 == binding["source_table_sha256"]
    for arm in ("control", "selected"):
        source = emit_source_interval_i8_lookup(
            table_name="source_interval_table",
            activation_name="source_activation",
            quantizer_name="source_quantize",
            lookup_name="source_lookup_activation",
            leading_bits=16,
            finite_inputs=helpers,
            finite_table=table,
            zero_observer_cells=tuple(
                prepare_bounded_rne_word_cells(p, effects=effects) for p in proofs
            )
            if arm == "selected"
            else (),
        )
        assert source == (OUT / arm / "lookup.c").read_text()
        for case in ("native", "target"):
            glue = (OUT / arm / case / "host_llvm/mode_guard.c").read_text()
            for i, helper in enumerate(helpers):
                scan = emit_finite_scale_helper(
                    helper,
                    wrapper_symbol=f"prepared_{i}",
                    finite_symbol=f"rne_{i}",
                    fallback_symbol=f"source_{i}",
                )
                assert glue.count(scan) == 1
    upstream_qualification = (
        UPSTREAM / "out/rne_observer_qualification/qualification.json"
    )
    paths = [
        Path(__file__),
        typed,
        SOURCE,
        upstream_qualification,
        Path(cells_module.__file__),
        Path(interval_module.__file__),
        UPSTREAM / "src/merlin/llvmlower/scaled_integer_finite.py",
        UPSTREAM / "src/merlin/llvmlower/scaled_integer_finite_llvm.py",
        UPSTREAM / "src/merlin/llvmlower/source_expression_interval_llvm.py",
        *[OUT / a / "lookup.c" for a in ("control", "selected")],
        *[
            OUT / a / c / "host_llvm/mode_guard.c"
            for a in ("control", "selected")
            for c in ("native", "target")
        ],
    ]
    record = {
        "schema": "published_generic_observer_normal_reemission_v1",
        "status": "pass",
        "upstream_head": HEAD,
        "original_measured_topic": "bcd6d3fc452e9de064a39db56b5b37fb58f73782",
        "implementation_modules_byteidentical": True,
        "control_and_selected_lookup_C_byteidentical": True,
        "all22finite_scan_wrappers_in_native_and_target_byteidentical": True,
        "all22typed_source_observers_rederived": True,
        "source_table_sha256": table.sha256,
        "source_contexts": 22,
        "source_producer_chains": 44,
        "scope": "Published Merlin main independently derives the same typed observer/table/producer scanner contracts and exactly reproduces every compiled lookup/scanner C body. Whole upstream recapture is outside this controlled normal-host-hook scope.",
        "pins": {str(p): sha(p) for p in paths},
        "token_usage_available": False,
    }
    save(OUT / "upstream_reemission.json", record)
    print("UPSTREAM_NORMAL_ZERO_REEMISSION_PASS", len(record["pins"]), flush=True)


if __name__ == "__main__":
    main()
