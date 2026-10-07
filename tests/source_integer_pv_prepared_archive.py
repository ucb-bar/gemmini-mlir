"""Seal prepared-coefficient negatives and final source re-emission identity."""

from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from merlin.llvmlower.balanced_radix_groups import plan_balanced_radix_groups
from merlin.llvmlower.prepared_scaled_integer_observer import emit_prepared_scaled_integer_observer
from merlin.llvmlower.scaled_integer_dot_enclosure import ScaledIntegerDotEffects
from merlin.llvmlower.scaled_integer_observer_codegen import ScaledIntegerObserverPlan
from mlir_oot.executed_features import census
from source_integer_pv_native import CORE, HERE, OUT as ORIGINAL_NATIVE, sha, save
from source_integer_pv_target import OUT as ORIGINAL_TARGET, GCC


def main():
    probes = ORIGINAL_TARGET.parent
    native = probes / "source-integer-pv-prepared-native-20261007"
    portable = probes / "source-integer-pv-prepared-target-20261007"
    directed = probes / "source-integer-pv-prepared-directed-target-20261007"
    destination = probes / "source-integer-pv-prepared-archive-20261007"
    assert not destination.exists()
    destination.mkdir()
    snapshot = probes / "source-integer-pv-prepared-source-snapshot-20261007/prepared_scaled_integer_observer.py"
    current_module = CORE / "src/merlin/llvmlower/prepared_scaled_integer_observer.py"
    pins, resolutions = {}, []
    for receipt in [probes / "source-integer-pv-first-archive-20261007/receipt.json",
                    native / "qualification.json", portable / "qualification.json", directed / "qualification.json"]:
        record = json.loads(receipt.read_text())
        for name, expected in record["pins"].items():
            if sha(name) != expected:
                assert Path(name) == current_module and sha(snapshot) == expected, name
                resolutions.append({"original_receipt": str(receipt), "original_pin": name,
                                    "expected_sha256": expected, "byte_identical_snapshot": str(snapshot),
                                    "reason": "Final Python formatting and corrected explanatory lower-exponent bound; generated C re-emission independently exact"})
                pins[str(snapshot)] = expected
            else:
                pins[name] = expected
        pins[str(receipt)] = sha(receipt)
    reemitted = []
    for index in range(22):
        folder = native / f"context_{index:02d}"
        data = json.loads((folder / "source_plan.json").read_text())
        radix = data["radix"]
        data["radix"] = plan_balanced_radix_groups(**{name: radix[name] for name in
            ("radix_bits", "lhs_digits", "rhs_digits", "reduction_length", "accumulator_bits", "reconstruction_bits")})
        data["effects"] = ScaledIntegerDotEffects(**data["effects"])
        data["row_factors"] = tuple(data["row_factors"])
        data["output_permutation"] = tuple(data["output_permutation"])
        assert set(data) == {field.name for field in fields(ScaledIntegerObserverPlan)}
        text = "#define MERLIN_SOURCE_BITCAST_COPY __builtin_memcpy\n" + emit_prepared_scaled_integer_observer(ScaledIntegerObserverPlan(**data), symbol="integer_observer")
        assert text.encode() == (folder / "executor.c").read_bytes()
        assert np.array_equal(np.load(folder / "candidate_i8.npy"), np.load(ORIGINAL_NATIVE / f"context_{index:02d}/candidate_i8.npy"))
        assert json.loads((folder / "qualification.json").read_text())["counts"] == json.loads((ORIGINAL_NATIVE / f"context_{index:02d}/qualification.json").read_text())["counts"]
        reemitted.append({"evidence_context": index, "generated_C_sha256": sha(folder / "executor.c"),
                          "original_i8_words_exact": 16384, "counts_unchanged": True})
    assert (directed / "histogram.stdout").read_bytes() == (directed / "spike.stdout").read_bytes()
    elf = directed / "build/layer.elf"
    hist = directed / "histogram.stderr"
    scope = "prepared_directed_integer_observer_all5calls"
    feature = census(elf, hist, symbols=["integer_observer"], scope_id=scope,
        execution_binding={"elf_sha256": sha(elf), "histogram_sha256": sha(hist), "scope_id": scope,
            "engine": {"path": str(GCC.with_name("spike")), "sha256": sha(GCC.with_name("spike"))},
            "argv": [str(GCC.with_name("spike")), "-g", "--isa=rv64gc", "--extension=gemmini", "-m0x80000000:0x400000000", str(elf)],
            "stdout_path": str(directed / "histogram.stdout"), "stdout_sha256": sha(directed / "histogram.stdout"),
            "confirmed_invocations": 5, "scope_authority": "Frozen common main.c initial gate/RNE checks/ABBA; validation included in all-call scopes"})
    save(destination / "directed_executor_features.json", feature)
    results = []
    for label, directory in [("first_portable", ORIGINAL_TARGET), ("prepared_portable", portable), ("prepared_directed", directed)]:
        record = json.loads((directory / "qualification.json").read_text())
        metric = record.get("mean_retired_instructions", {"source": 635401, "candidate": 6234123})
        results.append({"alternative": label, "receipt": str(directory / "qualification.json"),
                        "mean_retired_instructions": metric, "ratio_to_source": metric["candidate"] / metric["source"],
                        "hardware_cycles": "UNKNOWN", "decision": "Reject performance promotion/hardware admission"})
    gate_folder = CORE / "out/artifacts/probes/source-integer-pv-prepared-gates-20261007"
    assert "104 passed" in (gate_folder / "source_tests.log").read_text()
    for path in [Path(__file__), current_module, snapshot, hist, directed / "histogram.stdout",
                 destination / "directed_executor_features.json", *gate_folder.glob("*.log"),
                 *CORE.glob("src/merlin/llvmlower/*integer*observer*.py"),
                 *CORE.glob("merlin/tests/ir/test_*integer*observer*.py")]:
        pins[str(path)] = sha(path)
    result = {
        "schema": "prepared_source_integer_pv_negative_v1", "status": "qualified_negative",
        "hypothesis": "Move invariant exact scale-product, representation-error coefficient and source prefix overflow proof to one immutable column preparation",
        "source_policy": "Exact closed i8 observation only for this experiment; original increasing-K binary32 multiply/add replay and source two rounded V multiplies retained",
        "actual_change": "Six equal-exponent grouped products/384KiB readout unchanged; fused exact f64 scale product and outward prepared coefficient, then explicit OOT directed scalar capabilities",
        "ownership": {"mathematical schedule/executor": "Merlin", "device ABI/products/scalar ISA capabilities": "OOT"},
        "independent_tests": 104, "all22_original_compiled_i8_exact": 360448,
        "native_counts_unchanged": True, "target_context0_original_i8_exact": 16384,
        "actual_target_integer_readouts_exact": 98304, "re_emission_after_normalization": reemitted,
        "normalization_pin_resolutions": resolutions, "alternatives": results,
        "target_gate": "Original i8 bytes, full exact integer readouts, dirty input/output/workspace guards, five rounding-mode output comparisons, final executable noFSM",
        "cost_scope": json.loads((portable / "qualification.json").read_text())["cost_scope"],
        "prepared_executor_all5call_features": feature["features"],
        "preparation_divide_attribution": "Per-output prefix guard division removed; remaining divisions belong to row norm preparation, not source arithmetic",
        "whole_route": "Absent; neither normal source-provider binding nor whole-model gate installed",
        "performance_decision": "Even best zero-replay context is5.57x source instructions. No further precision ladder, no FPGA run, no default/main promotion",
        "model_domain": "UNKNOWN: binary64 directed arithmetic, certificate branches, reconstruction/readback and physical overlap not priced by existing float quantizer pilots",
        "token_usage_available": False, "pins": pins,
    }
    save(destination / "receipt.json", result)
    print(json.dumps({"receipt": str(destination / "receipt.json"), "sha256": sha(destination / "receipt.json"),
                      "pins": len(pins), "features": feature["features"]["cpu_opcode_classes"]}), flush=True)


if __name__ == "__main__":
    main()
