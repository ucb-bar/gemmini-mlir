"""Seal one rejected numerical policy with independent original observations."""

from __future__ import annotations

import json
from pathlib import Path

import source_aware_pv_native_screen as screen


def main():
    out = screen.HERE / "docs/perf_records/source_aware_projected_pv_negative_20261007"
    out.mkdir(parents=True, exist_ok=False)
    native = json.loads((screen.OUT / "native_validation.json").read_text())
    declaration = json.loads((screen.OUT / "declaration.json").read_text())
    observer = screen.OUT.parent / "source-aware-projected-pv-observers-20261007"
    diagnostics = json.loads((observer / "diagnostics.json").read_text())
    assert (
        native["status"] == "fail"
        and native["compiled_gate_failures"] == 189885
        and native["torch_gate_failures"] == 189884
    )
    assert (
        native["calls"] == 22
        and native["refusals"] == 0
        and native["source_onehot_rows_retained"] == 704
    )
    assert (
        diagnostics["status"] == "pass"
        and diagnostics["whole_token0_all32000_logits_bits_exact"]
    )
    pins = {}
    for receipt in (native, declaration, diagnostics):
        for path, digest in receipt["pins"].items():
            assert screen.sha(path) == digest, (path, digest)
            pins[path] = digest
    roots = [
        screen.OUT,
        observer,
        screen.HERE / "out/artifacts/probes/source-aware-pv-logs",
        screen.CORE / "out/artifacts/probes/projected-integer-pv",
    ]
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                pins[str(path)] = screen.sha(path)
    paths = [
        Path(__file__),
        screen.HERE / "tests/source_aware_pv_native_screen.py",
        screen.HERE / "tests/source_aware_pv_observer_diagnostics.py",
        screen.CORE / "src/merlin/llvmlower/projected_integer_pv.py",
        screen.CORE / "merlin/tests/ir/test_projected_integer_pv.py",
    ]
    for path in paths:
        pins[str(path)] = screen.sha(path)
    for name, value in [
        ("declaration.json", declaration),
        ("native_validation.json", native),
        ("observer_diagnostics.json", diagnostics),
    ]:
        screen.save(out / name, value)
    record = {
        "schema": "source_aware_projected_pv_one_policy_negative_archive_v1",
        "status": "sealed",
        "hypothesis": "Preserving original i32 V and scales, exact integer products and source onehot rows may avoid lossy V column quantization while one fixed probability grid improves array admission.",
        "ownership": {
            "generic_numerical_policy": "Merlin isolated prototype",
            "native_experiment_bindings": "OOT",
            "target_layout_ISA_ABI": "not implemented",
        },
        "actual_change": "ONLY PV endpoints after unchanged source QK/mask/P, fixedP14; original projected integer/scales/GQA and original source results on704 onehot rows.",
        "policy_default_off": True,
        "production_routing_enabled": False,
        "source_exact_f32_certificate": False,
        "independent_native_tests": 18,
        "original_whole_source_off_control": "all256000 bits exact, originalTorchgate passes",
        "whole_unchanged_gate": {
            "atol": 0.03125,
            "rtol": 0.02,
            "compiled_failures": 189885,
            "torch_failures": 189884,
            "maxabs": native["max_abs_vs_torch"],
            "calls": 22,
            "source_onehot_rows_retained": 704,
            "runtime_refusals": 0,
            "token0_all32000_logits_bits_exact": True,
        },
        "observer_witness": {
            "typed_source_bound_consumers": 22,
            "i8_words_per_consumer": 16384,
            "floating_escapes": 0,
            "paired_local_changed_i8_total": sum(
                r["changed_observed_i8"] for r in diagnostics["contexts"]
            ),
            "first_context_changed_i8": 4,
            "first_crossing": diagnostics["contexts"][0]["first_changed_observation"],
            "later_context_scope": "source/candidate local paired endpoints on propagated candidate inputs; not frozen original trajectory",
        },
        "decision": "REJECTED; no target/wholehardware/precision ladder. Small local floating changes crossing original i8 rounding boundaries amplify downstream.",
        "compiler_legality_scope": "Actual typed projection/GQA/all-use consumer/source-context witnesses and original compiled consumers reclosed. Numerical approximation is explicit, not an exact source theorem. Runtime source fallback remains.",
        "limitations": [
            "Native stand-in executes original PV first for diagnostics/fallback, so timings would not price an offload.",
            "No target signed-digit provider implemented. Universal source V bounds may require four signed-byte digits; any six-plane estimate was conditional, not a complete legal plan.",
            "No universal FENV/flag equivalence claimed: explicit RNE/nontrapping/flags-unobserved approximation domain.",
            "No performance or <=300M prediction from source operation counts.",
        ],
        "hardware_cycles": "UNKNOWN; no run",
        "token_usage_available": False,
        "native_primary_path": str(screen.OUT / "native_validation.json"),
        "native_primary_sha256": screen.sha(screen.OUT / "native_validation.json"),
        "diagnostic_path": str(observer / "diagnostics.json"),
        "diagnostic_sha256": screen.sha(observer / "diagnostics.json"),
        "pins": pins,
    }
    screen.save(out / "receipt.json", record)
    print(out / "receipt.json", screen.sha(out / "receipt.json"), len(pins), flush=True)


if __name__ == "__main__":
    main()
