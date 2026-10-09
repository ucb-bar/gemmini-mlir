"""Reclose attention stages and replay their actual retained quantized consumer."""

import ctypes as C
import json
from pathlib import Path

import numpy as np

from review_closed_i8_stock_capsule_probe import require, sha


def descriptor(a):
    rank = a.ndim

    class Descriptor(C.Structure):
        _fields_ = [("allocated", C.c_void_p), ("aligned", C.c_void_p),
                    ("offset", C.c_int64), ("sizes", C.c_int64 * rank),
                    ("strides", C.c_int64 * rank)]

    return Descriptor(a.ctypes.data, a.ctypes.data, 0,
                      (C.c_int64 * rank)(*a.shape),
                      (C.c_int64 * rank)(*(x // a.itemsize for x in a.strides)))


def metrics(reference, candidate):
    d = candidate.astype(np.float64) - reference.astype(np.float64)
    return {"elements": reference.size,
            "bits_differ": int(np.count_nonzero(reference.view(np.uint32) != candidate.view(np.uint32))),
            "maxabs": float(np.max(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d * d))),
            "relative_l2": float(np.linalg.norm(d.ravel()) / np.linalg.norm(reference.astype(np.float64).ravel())),
            "diagnostic_same_tolerance_failures": int(np.count_nonzero(~np.isclose(candidate, reference, atol=.03125, rtol=.02)))}


def review():
    root = Path(__file__).resolve().parents[1]
    audit = Path("/scratch/agustin/tmp/gemmini-attention-boundary-audit-20261007")
    stage_path = audit / "docs/perf_records/smol_attention_boundary_numerical_audit.json"
    escape_path = audit / "docs/perf_records/smol_wide64_original_state_escape_audit.json"
    stage, escape = (json.loads(p.read_text()) for p in (stage_path, escape_path))
    output = root / "docs/perf_records/root_smol_attention_boundary_review_20261007.json"
    require(not output.exists(), "review output must be fresh")
    require(len(stage["pins"]) == 234 and len(escape["pins"]) == 80, "retained audit extent differs")
    for packet in (stage, escape):
        for path, digest in packet["pins"].items():
            require(sha(path) == digest, "changed attention boundary source/evidence: " + path)
    directory = audit / "out/boundaries"
    library = C.CDLL(stage["compiled_consumer"]["library"])
    consume = library._mlir_ciface_source_quant_frontier
    consume.argtypes = [C.c_void_p] * 3
    consume.restype = None

    def quantize(carrier):
        require(carrier.shape == (1, 12, 1024, 64) and carrier.dtype == np.uint16,
                "actual consumer input shape differs")
        snapshot = carrier.tobytes()
        q = np.full((1, 1024, 768), 17, dtype=np.int8)
        scale = np.full((1, 1024), 17, dtype=np.uint16)
        arguments = [descriptor(x) for x in (carrier, q, scale)]
        consume(*(C.byref(d) for d in arguments))
        require(carrier.tobytes() == snapshot, "consumer mutates input")
        return q, scale

    stage_names = ["qk_f32", "shifted_scaled_score_f32", "polynomial_y_f32",
                   "probability_bf16_widened", "denominator_f32", "alpha_f32",
                   "pv_partials_f32", "attention_output_bf16_widened"]
    observations = {}
    for arm in ("ordered", "single7", "two_digit", "wide64"):
        carrier = np.load(directory / arm / "output.npy", allow_pickle=False)
        require(carrier.tobytes() == np.load(directory / (arm + "_uninstrumented") / "output.npy").tobytes(),
                "instrumentation changes actual endpoint")
        qi, scale = quantize(np.concatenate([carrier] * 4, axis=2))
        qi, scale = qi[:, :256], scale[:, :256]
        require(qi.tobytes() == np.load(directory / arm / "consumer_i8.npy").tobytes() and
                scale.tobytes() == np.load(directory / arm / "consumer_scale.npy").tobytes(),
                "actual compiled consumer replay differs")
        if arm == "ordered":
            original_q, original_scale = qi, scale
        measured = {"i8_changed": int(np.count_nonzero(qi != original_q)),
                    "bf16_scale_changed": int(np.count_nonzero(scale != original_scale))}
        expected = stage["compiled_consumer"]["results"][arm]
        require(all(value == expected[key] for key, value in measured.items()), "consumer counts differ")
        observations[arm] = {"consumer": measured, "stages": {}}
        if arm != "ordered":
            for index, name in enumerate(stage_names):
                reference = np.load(directory / "ordered" / f"stage{index}.npy", allow_pickle=False)
                candidate = np.load(directory / arm / f"stage{index}.npy", allow_pickle=False)
                values = metrics(reference, candidate)
                saved = stage["stage_ledger"]["stages"][arm][name]
                require(all(np.isclose(value, saved[key], rtol=1e-12, atol=1e-12) for key, value in values.items()),
                        "independent stage metrics differ: " + arm + "/" + name)
                observations[arm]["stages"][name] = values
    rows = escape["rows"]
    require(len(rows) == 48 and all(x["index"] == i and x["input_hashes_original"] for i, x in enumerate(rows)),
            "independent original state sequence differs")
    totals = {name: sum(row[name] for row in rows) for name in ("carrier_changes", "i8_changes", "scale_changes")}
    totals["groups_with_escaping_changes"] = sum(bool(x["i8_changes"] or x["scale_changes"]) for x in rows)
    require(all(totals[k] == escape["summary"][k] for k in totals) and
            totals == {"carrier_changes": 2258, "i8_changes": 81, "scale_changes": 0, "groups_with_escaping_changes": 35},
            "original-state aggregate counts differ")
    first = directory / "first_wide64_escape"
    first_arrays = {}
    for arm in ("original", "candidate"):
        a = np.load(first / (arm + "_carriers.npy"), allow_pickle=False)
        qi, scales = quantize(a)
        require(qi.tobytes() == np.load(first / (arm + "_i8.npy")).tobytes() and
                scales.tobytes() == np.load(first / (arm + "_scales.npy")).tobytes(),
                "first escaping actual consumer replay differs")
        first_arrays[arm] = qi, scales
    oq, os = first_arrays["original"]
    cq, cs = first_arrays["candidate"]
    coordinates = np.argwhere(oq != cq)
    first_coordinate = tuple(map(int, coordinates[0]))
    require(first_coordinate == (0, 135, 689) and int(oq[first_coordinate]) == 56 and
            int(cq[first_coordinate]) == 57 and os.tobytes() == cs.tobytes(), "first quantized escape differs")
    first_group = next(x["index"] for x in rows if x["i8_changes"] or x["scale_changes"])
    require(first_group == 4 and escape["whole_original_bits_preserved"] is True, "original-state diagnostic differs")
    result = {
        "schema": "root_smol_attention_boundary_and_original_state_review_v1",
        "status": "VERIFIED_STAGE_DIAGNOSIS_NO_POLICY_PROMOTION",
        "stage_source_pins_reclosed": 234, "original_state_source_pins_reclosed": 80,
        "independent_first_group_stage_metrics_and_compiled_consumers": observations,
        "independent_original_state_aggregate_counts": totals,
        "first_escaping_actual_compiled_consumer_replay": {"group": first_group, "row": 135, "channel": 689,
                                                         "original_i8": 56, "wide64_i8": 57, "escaping_scale_changes": 0},
        "scope": "First complete12-head stage arrays/consumer replay and all48 retained original-state ledger inputs/pins/aggregate counts; original outputs were fed downstream in this diagnostic. Later candidate numerical arrays are not all independently recomputed by root.",
        "interpretation": "Wider accumulation of unchanged floating operands can cross downstream integer bins. First-group absence of integer escapes does not prove all48 groups. Independent-state81 escaping integer differences do not attribute all sequential whole144 gate failures to these81 words.",
        "whole_model_gate": {"atol": .03125, "rtol": .02, "unchanged": True},
        "new_policy_selected": False, "new_hardware_run": False,
        "task_quality": "UNKNOWN", "compiler_bug_asserted": False,
        "pins": {str(p.resolve()): sha(p) for p in (stage_path, escape_path, Path(__file__))},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "totals": totals,
                      "first_escape": result["first_escaping_actual_compiled_consumer_replay"]}))


if __name__ == "__main__":
    review()
