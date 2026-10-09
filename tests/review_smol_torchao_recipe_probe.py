"""Close the retained quantization audit without rerunning or replacing its golden."""

import base64
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from review_closed_i8_stock_capsule_probe import require, sha


def metrics(reference, candidate):
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "elements": reference.size,
        "changed_f32_words": int(np.count_nonzero(reference.view(np.uint32) != candidate.view(np.uint32))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / np.linalg.norm(reference.astype(np.float64).ravel())),
        "failures_at_existing_elementwise_gate": int(np.count_nonzero(~np.isclose(candidate, reference, atol=.03125, rtol=.02))),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def review():
    root = Path(__file__).resolve().parents[1]
    directory = root / "out/artifacts/probes/smol-torchao-recipe-audit-20261007"
    packet = directory / "forward/validation.json"
    output = root / "docs/perf_records/root_smol_torchao_recipe_quality_review_20261007.json"
    require(not output.exists(), "review output must be fresh")
    q = json.loads(packet.read_text())
    require(q["status"] == "REPRODUCED_ORIGINAL_QUANTIZED_GOLDEN" and len(q["pins"]) == 27,
            "retained recipe audit did not close")
    for path, digest in q["pins"].items():
        require(sha(path) == digest, "changed original source or audit evidence: " + path)
    baseline = np.load(directory / "forward/unquantized_checkpoint_output.npy", allow_pickle=False)
    quantized = np.load(directory / "forward/torchao_quantized_output.npy", allow_pickle=False)
    bundle = Path(next(path for path in q["pins"] if path.endswith("/capture_receipt.json"))).parent
    golden = np.load(bundle / "golden.npy", allow_pickle=False)
    require(baseline.shape == quantized.shape == golden.shape == (1, 50, 32) and
            all(a.dtype == np.float32 for a in (baseline, quantized, golden)), "output shape/dtype differs")
    require(quantized.tobytes() == golden.tobytes(), "original quantized golden is not reproduced")
    comparisons = {"quantized_vs_unquantized_checkpoint": metrics(baseline, quantized),
                   "reproduced_quantized_vs_original_golden": metrics(golden, quantized)}
    for name, measured in comparisons.items():
        for field, value in measured.items():
            require(np.isclose(value, q[name][field], rtol=1e-12, atol=1e-12),
                    "independent baseline metric differs: " + name + "/" + field)
    before, after = q["linear_module_census_before"], q["linear_module_census_after"]
    require(len(before) == len(after) == 303 and
            [(x["name"], x["shape"]) for x in before] == [(x["name"], x["shape"]) for x in after],
            "registered Linear census differs")
    for x in after:
        require(x["weight_class"].endswith(".LinearActivationQuantizedTensor") and
                x["stored_weight"]["dtype"] == "torch.int8" and
                x["stored_weight"]["shape"] == x["shape"] and
                x["stored_weight"]["scale_shape"] == [x["shape"][0]],
                "actual stored i8/per-output-channel weights differ")
    trace = json.loads((bundle / "frontend-trace.json").read_text())
    graphs = {stage: x["by_target"] for stage, x in trace["graphs"].items()}
    require(graphs["original"]["aten.linear.default"] == 302 and
            graphs["quantized"]["aten._int_mm.default"] == 302 and
            graphs["quantized"]["torchao.choose_qparams_affine.default"] == 302 and
            graphs["quantized"]["torchao.quantize_affine.default"] == 302 and
            graphs["quantized"]["aten.matmul.default"] == 64 and
            graphs["quantized"]["aten.scaled_dot_product_attention.default"] == 12 and
            graphs["prepared"]["aten.bmm.default"] == 88, "capture quantization scope differs")
    require(q["unquantized_parameter_dtype_counts"] == {"torch.bfloat16": 474, "torch.float32": 26}
            and q["recipe"]["static_calibration_performed"] is False and q["inputs_unchanged"] is True,
            "original recipe or source assumptions differ")
    # An import-only repair was necessary in the owned overlay. Removing exactly
    # its first logger declaration must recover the installed wheel's RECORD hash.
    repaired = directory / "deps/diffusers/quantizers/torchao/torchao_quantizer.py"
    record = directory / "deps/diffusers-0.35.2.dist-info/RECORD"
    wheel_row = next(row for row in csv.reader(record.open()) if row[0] == "diffusers/quantizers/torchao/torchao_quantizer.py")
    added = b"logger = logging.get_logger(__name__)\n\n\n"
    original_import_source = repaired.read_bytes().replace(added, b"", 1)
    encoded = base64.urlsafe_b64encode(hashlib.sha256(original_import_source).digest()).decode().rstrip("=")
    require(wheel_row[1] == "sha256=" + encoded and int(wheel_row[2]) == len(original_import_source),
            "overlay repair contains an unrecorded change")
    extra = [packet, Path(__file__), directory / "forward.log", repaired, record,
             Path("/scratch/agustin/tmp/model2mlir-silu-opmath-20261005/uv.lock")]
    extra += sorted(directory.glob("forward_failed_import_*.log"))
    result = {
        "schema": "root_smol_torchao_original_recipe_quality_review_v1",
        "status": "VERIFIED_ORIGINAL_QUANTIZED_GOLDEN_AND_SEPARATE_QUANTIZATION_LOSS",
        "original_audit_pins_reclosed": len(q["pins"]),
        "checkpoint": q["checkpoint"], "versions": q["versions"], "recipe": q["recipe"],
        "original_quantized_golden_all1600words_exact": True,
        "metrics": comparisons,
        "registered_linear_modules": 303, "registered_linear_modules_stored_i8": 303,
        "captured_linear_call_sites_integerized": 302,
        "call_site_counting_unit": "static captured call sites, not registered module count or dynamic execution count",
        "remaining_attention": {"quantized_graph_floating_matmul_call_sites": 64,
                                "quantized_graph_floating_SDPA_call_sites": 12,
                                "prepared_graph_floating_bmm_call_sites": 88},
        "unquantized_parameter_dtype_counts": q["unquantized_parameter_dtype_counts"],
        "weight_scale_dtype_counts": dict(Counter(x["stored_weight"]["scale_dtype"] for x in after)),
        "historical_full_dependency_environment_proven": False,
        "overlay_import_repair": {"path": str(repaired), "change": "Initialize Diffusers logger before its torch-safe-global compatibility handler; original trailing initializer retained.",
                                  "original_wheel_record_sha256_recovered": hashlib.sha256(original_import_source).hexdigest(),
                                  "only_40_byte_import_initialization_added": True, "inference_or_arithmetic_change": False},
        "interpretation": "TorchAO applies the recorded Linear recipe and reproduces the retained golden on this fixture. Prequantization quality loss is distinct from additional compiler/attention approximation error against that quantized golden. The current compiler acceptance gate is unchanged.",
        "not_established": ["robot task success or held-out action quality", "uniform FP32 checkpoint baseline", "historical full dependency closure", "quality or recipe audit for other models", "compiler error measured by this forward audit", "complete quantization of attention"],
        "input_dataset_attribution": "UNKNOWN", "inputs_unchanged": True,
        "new_capture_or_golden": False, "new_numeric_policy_selected": False,
        "compiler_gate": {"atol": .03125, "rtol": .02, "oracle": "unchanged original quantized golden"},
        "pins": {str(p.resolve()): sha(p) for p in extra},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "metrics": comparisons,
                      "registered_linear_modules": 303, "captured_integer_linear_call_sites": 302}))


if __name__ == "__main__":
    review()
