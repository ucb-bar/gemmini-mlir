"""Measure original recipe quantization separately from compiler error on owned inputs."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchao
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/artifacts/probes/smol-torchao-recipe-audit-20261007/forward"
BUNDLE = Path("/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle")
SOURCE = Path("/scratch/agustin/tmp/model2mlir-silu-opmath-20261005")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(reference, candidate):
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    norm = np.linalg.norm(reference.astype(np.float64).ravel())
    return {
        "elements": reference.size,
        "changed_f32_words": int(np.count_nonzero(reference.view(np.uint32) != candidate.view(np.uint32))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / norm) if norm else None,
        "failures_at_existing_elementwise_gate": int(np.count_nonzero(
            ~np.isclose(candidate, reference, atol=.03125, rtol=.02))),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def census(model):
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        record = {"name": name, "weight_class": type(weight).__module__ + "." + type(weight).__name__,
                  "logical_dtype": str(weight.dtype), "shape": list(weight.shape)}
        original = getattr(weight, "original_weight_tensor", weight)
        implementation = getattr(original, "tensor_impl", None)
        if implementation is not None and hasattr(implementation, "get_plain"):
            integer, scale, zero = implementation.get_plain()
            record["stored_weight"] = {"dtype": str(integer.dtype), "shape": list(integer.shape),
                                       "scale_dtype": str(scale.dtype), "scale_shape": list(scale.shape),
                                       "zero_dtype": str(zero.dtype) if zero is not None else None,
                                       "zero_nonzero": int(torch.count_nonzero(zero)) if zero is not None else None}
        rows.append(record)
    return rows


def main():
    assert not OUT.exists(), "fresh output required"
    OUT.mkdir(parents=True)
    receipt = json.loads((BUNDLE / "capture_receipt.json").read_text())
    pins = {str(BUNDLE / "capture_receipt.json"): sha(BUNDLE / "capture_receipt.json"),
            str(Path(__file__).resolve()): sha(__file__)}
    for relative, digest in receipt["tool"]["source_sha256"].items():
        path = SOURCE / relative
        assert sha(path) == digest, "original capture source changed: " + relative
        pins[str(path)] = digest
    loader_path = Path(receipt["source"]["path"])
    assert sha(loader_path) == receipt["source"]["sha256"]
    pins[str(loader_path)] = sha(loader_path)
    for name in ("inputs.npz", "golden.npy", "meta.json", "frontend-trace.json"):
        assert sha(BUNDLE / name) == receipt["artifacts"][name]["sha256"]
        pins[str(BUNDLE / name)] = sha(BUNDLE / name)
    assert torch.__version__ == receipt["framework"]["torch"]
    assert torchao.__version__ == receipt["framework"]["torchao"]
    os.environ["M2M_SMOLVLA_PRETRAINED"] = "1"
    os.environ["M2M_SEQ"] = "8"
    for variable in ("M2M_SMOLVLA_SESSION", "M2M_LLAMA_SESSION", "M2M_SMOLVLA_TAP",
                     "M2M_SMOLVLA_VLM_LAYERS", "M2M_SMOLVLA_EXPERT_LAYERS"):
        os.environ.pop(variable, None)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    spec = importlib.util.spec_from_file_location("owned_smol_recipe_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    start = time.monotonic()
    model, _ = loader.get_model_and_inputs()
    print("ORIGINAL_CHECKPOINT_LOADED", time.monotonic() - start, flush=True)
    z = np.load(BUNDLE / "inputs.npz", allow_pickle=False)
    inputs = tuple(torch.from_numpy(z["in" + str(i)]) for i in range(6))
    input_bytes = [x.numpy().tobytes() for x in inputs]
    before = census(model)
    parameter_dtypes = dict(Counter(str(p.dtype) for p in model.parameters()))
    with torch.no_grad():
        start = time.monotonic()
        baseline = model(*inputs).detach().float().cpu().numpy().copy()
        baseline_seconds = time.monotonic() - start
    np.save(OUT / "unquantized_checkpoint_output.npy", baseline)
    print("UNQUANTIZED_FORWARD_DONE", baseline_seconds, baseline.shape, flush=True)
    config = QuantizationConfig(scheme="int8_dyn_act_int8_weight", calibration_samples=1)
    model = apply_quantization(model, config, example_inputs=inputs)
    after = census(model)
    with torch.no_grad():
        start = time.monotonic()
        candidate = model(*inputs).detach().float().cpu().numpy().copy()
        candidate_seconds = time.monotonic() - start
    np.save(OUT / "torchao_quantized_output.npy", candidate)
    assert input_bytes == [x.numpy().tobytes() for x in inputs], "forward mutated source inputs"
    original = np.load(BUNDLE / "golden.npy", allow_pickle=False)
    assert baseline.shape == candidate.shape == original.shape == (1, 50, 32)
    matched = candidate.tobytes() == original.tobytes()
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchao", "lerobot", "transformers", "numpy", "draccus", "safetensors")}
    record = {
        "schema": "owned_smol_torchao_recipe_forward_audit_v1",
        "status": "REPRODUCED_ORIGINAL_QUANTIZED_GOLDEN" if matched else "DIAGNOSTIC_ENVIRONMENT_OR_FORWARD_MISMATCH",
        "source_loader": str(loader_path), "versions": versions, "interpreter": sys.executable,
        "checkpoint": "lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "recipe": {"scheme": config.scheme, "calibration_samples_argument": 1,
                   "static_calibration_performed": False, "scope": "TorchAO default Linear filter; attention matmuls not selected"},
        "unquantized_parameter_dtype_counts": parameter_dtypes,
        "unquantized_is_uniform_fp32": False,
        "quantized_vs_unquantized_checkpoint": metrics(baseline, candidate),
        "reproduced_quantized_vs_original_golden": metrics(original, candidate),
        "original_quantized_golden_all1600words_exact": matched,
        "linear_module_census_before": before, "linear_module_census_after": after,
        "native_forward_seconds": {"unquantized": baseline_seconds, "quantized": candidate_seconds},
        "inputs_unchanged": True,
        "model_quality_claim": "None: one retained input fixture with unknown dataset attribution; no robot success/held-out action quality assessment.",
        "acceptance_gate_unchanged": {"atol": .03125, "rtol": .02},
        "compiler_error_measured_here": False,
        "historical_full_dependency_environment_proven": False,
        "no_new_capture_or_golden_no_numeric_policy_selection": True,
        "pins": {**pins, **{str(path): sha(path) for path in OUT.glob("*.npy")}},
    }
    (OUT / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "versions", "quantized_vs_unquantized_checkpoint",
                                                 "reproduced_quantized_vs_original_golden")}), flush=True)


if __name__ == "__main__":
    main()
