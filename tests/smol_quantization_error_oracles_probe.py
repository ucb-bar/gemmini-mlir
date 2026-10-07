"""Compare frozen diagnostic arms to distinct prequant and quantized oracles."""

import json
from pathlib import Path

import numpy as np

from review_closed_i8_stock_capsule_probe import require, sha
from review_smol_torchao_recipe_probe import metrics


def review():
    root = Path(__file__).resolve().parents[1]
    audit = root / "docs/perf_records/root_smol_torchao_recipe_quality_review_20261007.json"
    q = json.loads(audit.read_text())
    for path, expected in q["pins"].items():
        require(sha(path) == expected, "changed quantization audit source")
    directory = root / "out/artifacts/probes/smol-torchao-recipe-audit-20261007/forward"
    oracles = {"unquantized_mixed_BF16_F32_checkpoint": np.load(directory / "unquantized_checkpoint_output.npy"),
               "original_quantized_golden": np.load(directory / "torchao_quantized_output.npy")}
    stems = {"original_ordered_control": "selected_graph_ordered_source_control",
             "rejected_single7": "rms4_single_digit_selected_graph",
             "rejected_two_digit": "rms4_two_digit_selected_graph",
             "rejected_wide64_unchanged_operands": "selected_graph_f64_reduction_control"}
    candidates = Path("/scratch/agustin/tmp/gemmini-rms-selective-replay-20261007/out")
    pins = {str(audit): sha(audit), str(Path(__file__).resolve()): sha(__file__)}
    for path in directory.glob("*.npy"):
        pins[str(path)] = sha(path)
    rows = []
    for label, stem in stems.items():
        path = candidates / stem / "native/output.npy"
        values = np.load(path, allow_pickle=False)
        require(values.dtype == np.float32 and values.shape == (1, 50, 32), "frozen output differs")
        pins[str(path)] = sha(path)
        row = {"arm": label, "oracles": {name: metrics(reference, values) for name, reference in oracles.items()}}
        rows.append(row)
    require(rows[0]["oracles"]["original_quantized_golden"]["changed_f32_words"] == 0,
            "original compiler control does not reproduce quantized golden")
    require([r["oracles"]["original_quantized_golden"]["failures_at_existing_elementwise_gate"] for r in rows] ==
            [0, 131, 106, 144], "frozen rejected outcomes differ")
    result = {
        "schema": "root_smol_distinct_quantization_oracle_diagnostics_v1",
        "status": "DIAGNOSTIC_ONLY_EXISTING_REJECTIONS_PRESERVED",
        "rows": rows,
        "comparison_scope": "Same retained1600 outputs, original quantized compiler control and already-rejected frozen native arms; no new parameter/representation search or hardware run.",
        "interpretation": "The prequant checkpoint oracle and compiler fidelity oracle answer different questions. A rejected arm's smaller fixture error to the prequant checkpoint does not prove robot quality, semantic equivalence or acceptance. Quantization and compiler/approximation errors are not additive scalar budgets.",
        "existing_acceptance": {"oracle": "original quantized golden", "atol": .03125, "rtol": .02, "unchanged": True},
        "task_quality": "UNKNOWN", "policy_selected": False, "golden_replaced": False,
        "pins": pins,
    }
    output = root / "docs/perf_records/root_smol_quantization_error_oracles_20261007.json"
    require(not output.exists(), "fresh diagnostic required")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rows": rows}))


if __name__ == "__main__":
    review()
