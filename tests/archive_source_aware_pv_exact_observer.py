"""Seal exact-observer correctness and complete target instruction cost.

Hardware cycles remain unknown. This archive does not promote the local generic
prototype, alter the earlier approximation rejection, or install a model route.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import source_aware_pv_native_screen as screen
from merlin.llvmlower.balanced_radix_groups import plan_balanced_radix_groups
from merlin.llvmlower.projected_integer_pv import ProjectedIntegerPVPolicy
from merlin.llvmlower.projected_integer_pv_codegen import (
    ProjectedIntegerPVObserverPlan,
    emit_projected_integer_pv_observer,
)


def main():
    out = (
        screen.HERE
        / "docs/perf_records/source_aware_pv_exact_observer_negative_20261007"
    )
    out.mkdir(parents=True, exist_ok=False)
    roots = {
        "original_operands": screen.OUT.parent
        / "source-aware-pv-prepared-norm-screen-20261007",
        "whole_native": screen.OUT.parent
        / "source-aware-pv-exact-observer-whole-native-20261007",
        "complete_target": screen.OUT.parent
        / "source-aware-pv-prepared-norm-complete-target-20261007",
    }
    records = {
        "original_operands": json.loads(
            (roots["original_operands"] / "qualification.json").read_text()
        ),
        "whole_native": json.loads(
            (roots["whole_native"] / "native_validation.json").read_text()
        ),
        "complete_target": json.loads(
            (roots["complete_target"] / "qualification.json").read_text()
        ),
    }
    pins = {}
    for record in records.values():
        assert record["status"] == "pass"
        for path, digest in record["pins"].items():
            assert screen.sha(path) == digest, (path, digest)
            pins[path] = digest
    whole = records["whole_native"]
    assert whole["calls"] == 22 and whole["refusals"] == 0
    assert whole["original_compiled_i8_observations_exact"] == 360448
    assert whole["certified_nononehot_outputs"] == 310604
    assert whole["ambiguous_point_fallbacks"] == 4788
    assert whole["source_onehot_rows_retained"] == 704
    output = np.load(roots["whole_native"] / "output.npy")
    assert hashlib.sha256(output.astype("<f4").tobytes()).hexdigest() == screen.RAW
    assert np.array_equal(
        output.view("u4"), np.load(screen.BASE / "output.npy").view("u4")
    )
    assert np.allclose(
        output, np.load(screen.BUNDLE / "golden.npy"), atol=0.03125, rtol=0.02
    )
    target = records["complete_target"]
    assert target["exact_device_readout_words"] == 65536
    assert target["original_i8_exact"] == 16384 and target["zeroFSM"]
    means = target["mean_retired_instructions"]
    assert means == {"0": 635401.0, "1": 2414031.0}
    plan_fields = json.loads((roots["complete_target"] / "plan.json").read_text())
    radix = plan_fields.pop("radix")
    policy = ProjectedIntegerPVPolicy(**plan_fields.pop("policy"))
    plan = ProjectedIntegerPVObserverPlan(
        **plan_fields,
        policy=policy,
        radix=plan_balanced_radix_groups(
            **{
                key: radix[key]
                for key in (
                    "radix_bits",
                    "lhs_digits",
                    "rhs_digits",
                    "reduction_length",
                    "accumulator_bits",
                    "reconstruction_bits",
                )
            }
        ),
    )
    regenerated = emit_projected_integer_pv_observer(
        plan, symbol="integer_observer"
    ).encode()
    assert regenerated == (roots["complete_target"] / "executor.c").read_bytes()
    # The formatter changed no source bytes; retain snapshots and the fresh
    # exact-emission check rather than silently invalidating original pins.
    snapshots = screen.OUT.parent / "source-aware-pv-final-source-style-20261007"
    for item in json.loads((snapshots / "before.json").read_text()):
        assert screen.sha(item["original_path"]) == item["sha256"]
        assert screen.sha(item["snapshot_path"]) == item["sha256"]
    tests = (
        screen.CORE
        / "out/artifacts/probes/projected-integer-pv/exact_observer_tests.log"
    )
    assert "50 passed" in tests.read_text()
    extra = [Path(__file__), tests]
    for root in [*roots.values(), snapshots]:
        extra.extend(path for path in root.rglob("*") if path.is_file())
    extra.extend(
        path
        for path in (
            screen.HERE / "out/artifacts/probes/source-aware-pv-logs"
        ).iterdir()
        if path.is_file()
    )
    for name in [
        "projected_integer_pv.py",
        "projected_integer_pv_codegen.py",
        "projected_integer_pv_enclosure.py",
        "balanced_radix_groups.py",
        "AGENT.md",
    ]:
        extra.append(screen.CORE / "src/merlin/llvmlower" / name)
    extra.extend(
        screen.CORE / "merlin/tests/ir" / name
        for name in [
            "test_projected_integer_pv.py",
            "test_projected_integer_pv_codegen.py",
            "test_projected_integer_pv_enclosure.py",
            "test_balanced_radix_groups.py",
        ]
    )
    extra.append(screen.CORE / "docs/design/agent_compiler_performance.md")
    for path in extra:
        pins[str(path)] = screen.sha(path)
    for name, value in records.items():
        screen.save(out / (name + ".json"), value)
    record = {
        "schema": "source_aware_pv_exact_observer_complete_negative_archive_v1",
        "status": "sealed",
        "hypothesis": "Prepared source-valid probability row and original V/code column norms can preserve exact downstream integer observations while using fixed-grid integer products and selective original point replay.",
        "ownership": {
            "norms_numerics_grouping_reconstruction_portable_executor": "Merlin isolated prototype",
            "experimental_source_binding_target_callback_layout_ABI_ISA": "OOT",
        },
        "actual_emission": "Same P14 center, original i32 V and source scales/GQA; six logical plane pairs in four grouped callbacks/16 physical KV matrix calls. Prepared bounds certify complete original14opquant bins; ambiguous points and onehot rows execute original increasingK separateF32MUL/ADD.",
        "generic_core_head": "6de060cd9",
        "first_approximation_rejection_preserved": str(
            screen.HERE
            / "docs/perf_records/source_aware_projected_pv_negative_20261007/receipt.json"
        ),
        "original_input_screen": {
            "contexts": 22,
            "i8_observations_exact": 360448,
            "certified_nononehot": 310604,
            "ambiguous_point_fallbacks": 4788,
        },
        "whole_native_gate": {
            "tokens": 8,
            "layers": 22,
            "elements": 256000,
            "atol": 0.03125,
            "rtol": 0.02,
            "compiled_reference_bits_exact": True,
            "torch_gate_pass": True,
            "refusals": 0,
            "source_onehot_rows": 704,
            "all_original_live_inputs_equal": True,
            "unobserved_float_carrier_changes": 310527,
            "native_integer_product_standin": True,
        },
        "complete_first_target_gate": {
            "original_i8_outputs": 16384,
            "real_readout_words_exact": 65536,
            "readout_bytes": 262144,
            "grouped_callbacks": 4,
            "physical_KV_matrix_calls": 16,
            "private_workspace_bytes": 246016,
            "dirty_input_output_guards": True,
            "zeroFSM": True,
            "five_FRM_two_sticky_input_contexts": True,
            "nonRNE_refuses_before_output_then_actual_source": True,
            "fflags_observation_permission": "explicit nontrapping and unobserved; no source/candidate flag parity claim",
            "elf_sha256": target["elf_sha256"],
        },
        "first_target_runtime_counts": {
            "certified": 14306,
            "replayed": 2078,
            "onehot_rows": 32,
            "onehot_outputs": 2048,
            "ambiguous_outputs": 30,
            "source_muladd_pairs": 16624,
        },
        "complete_cost": {
            "scope": target["cost_scope"],
            "source_retired_instructions": 635401,
            "candidate_retired_instructions": 2414031,
            "ratio": means["1"] / means["0"],
            "instruction_change_percent": (means["1"] / means["0"] - 1) * 100,
            "actual_hardware_cycles": "UNKNOWN; no RTL/FireSim run",
            "whole_prediction": "UNKNOWN; no operational model prices certificate/packing/readout/latency",
        },
        "independent_tests": {
            "native_numeric_group_plan_compiled_executor": 50,
            "structure_check": "PASS",
            "format_import_sort": "PASS unchanged bytes",
            "regenerated_complete_executor_C": "byte-identical",
        },
        "test_fixture_diagnostic": "Initial parameterized compiledC test reused one .so pathname, causing process loader reuse. Unique per-shape library filenames fixed isolation; production C/exact gates unchanged. Initial tool failure transcript was not saved in a local log.",
        "decision": "Correctness-qualified local prototype; performance-unpromoted. Complete target instruction cost is unfavorable and no credible latency model ranks a win. No normal route, whole target ELF, hardware run or main publication.",
        "scope_limits": [
            "Caller supplies typed source casts, scales, zero seed/order, complete all-use i8 closure, immutable representation epoch and exact device product proof.",
            "Native whole callback retains original source PV first for functional fallback, so native duration is not candidate cost.",
            "First universal code range admits three value digits; other source instances may need four. First target costs do not price every context.",
            "Allocation/poison are outside both paired windows with an explicit reusable private workspace; normal allocation/lifetime integration remains unimplemented.",
            "No source/model/golden identifier selects generic numeric behavior.",
        ],
        "token_usage_available": False,
        "primary_receipts": {
            name: str(
                root
                / (
                    "native_validation.json"
                    if name == "whole_native"
                    else "qualification.json"
                )
            )
            for name, root in roots.items()
        },
        "pins": pins,
    }
    screen.save(out / "receipt.json", record)
    print(out / "receipt.json", screen.sha(out / "receipt.json"), len(pins), flush=True)


if __name__ == "__main__":
    main()
