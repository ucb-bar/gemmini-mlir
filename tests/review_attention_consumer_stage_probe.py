"""Independently close rejected attention observations at their actual quantizer."""

import argparse
import json
from pathlib import Path

import numpy as np
from review_closed_i8_stock_capsule_probe import require, sha


def ordered_dot(a, b, seed):
    result = seed.copy()
    for k in range(a.shape[-1]):
        product = np.multiply(a[..., :, k, None], b[..., k, None, :], dtype=np.float32)
        result = np.add(product, result, dtype=np.float32)
    return result


def consumer(carrier, factor):
    raw = np.multiply(carrier, factor, dtype=np.float32)
    codes = np.rint(np.clip(raw, np.float32(-128), np.float32(127))).astype(np.int8)
    return raw, codes.transpose(1, 0, 2).reshape(1, 8, 2048)


def review(args):
    require(not args.output.exists(), "review output must be fresh")
    q = json.loads(args.packet.read_text())
    require(q["status"] == "pass" and len(q["contexts"]) == 22,
            "complete source audit is required")
    for path, digest in q["pins"].items():
        require(sha(path) == digest, "changed source/stage evidence: " + path)
    root = args.packet.parent
    captures = root.parent / "dynamic-i8-attention-original-attribution-20261007"
    totals = {"qk_only_changed_consumer_i8": 0, "pv_only_changed_consumer_i8": 0,
              "query0_pv_changed_consumer_i8": 0, "pv_endpoint_gate_failures": 0,
              "qk_endpoint_gate_failures": 0, "consumer_words_per_arm": 0}
    rows = []
    for c in q["contexts"]:
        index = c["context"]
        require(index == len(rows), "context sequence changed")
        def load(name, family):
            return np.load(captures / f"context_{index:02d}_family{family}" / (name + ".npy"),
                           allow_pickle=False)
        query, key, seed, score = (load(name, 0) for name in ("a", "b", "seed", "original"))
        p, v, vseed, endpoint = (load(name, 1) for name in ("a", "b", "seed", "original"))
        require(ordered_dot(query, key, seed).tobytes() == score.tobytes() and
                ordered_dot(p, v, vseed).tobytes() == endpoint.tobytes(),
                "source separate F32 MUL/ADD order does not close")
        require(np.all(p[:, 0, 0] == 1) and np.all(p[:, 0, 1:] == 0) and
                endpoint[:, 0].tobytes() == v[:, 0].tobytes(), "one-hot isolation differs")
        saved = root / f"context_{index:02d}"
        factor = np.array([c["consumer"]["factor_f32_word"]], dtype=np.uint32).view(np.float32)[0]
        original_raw, original_codes = consumer(endpoint, factor)
        require(original_codes.tobytes() == np.load(saved / "original_consumer.npy").tobytes(),
                "actual source quantizer differs")
        record = {"context": index, "factor_f32": float(factor), "arms": {}}
        for arm, filename, countkey, gatekey in (
            ("qk_candidate", "qk_only_endpoint", "qk_only_changed_consumer_i8", "qk_endpoint_gate_failures"),
            ("pv_candidate", "approximate", "pv_only_changed_consumer_i8", "pv_endpoint_gate_failures"),
        ):
            candidate = load(filename, 1)
            raw, codes = consumer(candidate, factor)
            saved_codes = np.load(saved / (arm + "_consumer.npy"), allow_pickle=False)
            require(codes.tobytes() == saved_codes.tobytes(), "candidate actual consumer differs")
            diff = codes != original_codes
            by_query = [int(np.count_nonzero(diff[:, row])) for row in range(8)]
            require(by_query == c["consumer"][arm + "_changed_i8_by_query"],
                    "source-bound per-query counts differ")
            gate_failures = int(np.count_nonzero(~np.isclose(candidate, endpoint, atol=.03125, rtol=.02)))
            totals[countkey] += int(np.count_nonzero(diff))
            totals[gatekey] += gate_failures
            if arm == "pv_candidate":
                totals["query0_pv_changed_consumer_i8"] += by_query[0]
            code_delta = codes.astype(np.int16) - original_codes.astype(np.int16)
            result = {"changed_i8": int(np.count_nonzero(diff)), "by_query": by_query,
                      "endpoint_gate_failures": gate_failures,
                      "max_abs_i8_change": int(np.max(np.abs(code_delta))),
                      "max_abs_preclamp_change": float(np.max(np.abs(raw.astype(np.float64) - original_raw))) }
            coordinates = np.argwhere(diff)
            if coordinates.size:
                _, row, flattened = map(int, coordinates[0])
                head, channel = divmod(flattened, 64)
                x = float(original_raw[head, row, channel])
                y = float(raw[head, row, channel])
                result["first_changed_consumer"] = {
                    "head": head, "query": row, "channel": channel,
                    "original_f32_word": int(endpoint.view(np.uint32)[head, row, channel]),
                    "candidate_f32_word": int(candidate.view(np.uint32)[head, row, channel]),
                    "original_preclamp": x, "candidate_preclamp": y,
                    "original_code": int(original_codes[0, row, flattened]),
                    "candidate_code": int(codes[0, row, flattened]),
                    "distance_to_nearest_half_integer": abs(x - (np.floor(x) + .5)),
                }
            record["arms"][arm] = result
        totals["consumer_words_per_arm"] += original_codes.size
        rows.append(record)
    require(totals["qk_only_changed_consumer_i8"] == 42271 and
            totals["pv_only_changed_consumer_i8"] == 75595 and
            totals["query0_pv_changed_consumer_i8"] == 7936 and
            totals["pv_endpoint_gate_failures"] == 0 and totals["qk_endpoint_gate_failures"] == 4,
            "complete stage totals differ")
    result = {
        "schema": "root_rejected_tiny_attention_consumer_stage_review_v1",
        "status": "VERIFIED_STAGE_DIAGNOSIS_NO_POLICY_PROMOTION",
        "source_stage_pins_reclosed": len(q["pins"]), "contexts": 22,
        "totals": totals, "consumer_contract": "Original scalar immutable factor; separately rounded F32 multiply, clamp[-128,127], RNE, i8, actual head→flattened-channel mapping.",
        "rows": rows,
        "interpretation": "Local floating output tolerance does not establish same downstream quantization bins or whole-model tolerance. PV-only changes even a source-exact one-hot probability row. No mask/indexing/saturation explanation established for this failure.",
        "gate": {"atol": .03125, "rtol": .02, "whole_model_acceptance_unchanged": True},
        "not_established": ["original Torch intermediate oracle", "task-level quality impact", "complete target attention cycles", "qualified alternate representation"],
        "new_policy_selected": False, "new_hardware_run": False,
        "pins": {str(path.resolve()): sha(path) for path in (args.packet, Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "totals": totals, "first_PV_bin_crossing": rows[0]["arms"]["pv_candidate"]["first_changed_consumer"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    review(parser.parse_args())
