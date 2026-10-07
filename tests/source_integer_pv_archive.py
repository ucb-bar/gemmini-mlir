"""Archive exact-observer correctness and its complete negative instruction screen."""

from __future__ import annotations

import json
from pathlib import Path

from mlir_oot.executed_features import census
from source_integer_pv_native import CORE, OUT as NATIVE, sha, save
from source_integer_pv_target import OUT as TARGET, GCC


def main():
    destination = TARGET.parent / "source-integer-pv-first-archive-20261007"
    assert not destination.exists()
    destination.mkdir()
    pins = {}
    for receipt in (NATIVE / "qualification.json", TARGET / "qualification.json"):
        record = json.loads(receipt.read_text())
        assert record["status"] == "pass"
        for name, expected in record["pins"].items():
            assert sha(name) == expected, name
            pins[name] = expected
        pins[str(receipt)] = sha(receipt)
    histogram = TARGET / "complete_histogram.txt"
    assert histogram.read_bytes() == (TARGET / "histogram_g.stderr").read_bytes()
    assert (TARGET / "histogram_g.stdout").read_bytes() == (TARGET / "spike.stdout").read_bytes()
    elf = TARGET / "build/layer.elf"
    invocations = {
        "integer_observer": 5,
        "original_pv": 20,
        "original_quant": 20,
        "integer_product_8": 10,
        "integer_product_16": 10,
        "integer_product_24": 10,
        "product": 30,
    }
    scopes = {}
    for symbol, calls in invocations.items():
        scope = "complete_source_integer_pv_histogram_" + symbol
        binding = {
            "elf_sha256": sha(elf), "histogram_sha256": sha(histogram), "scope_id": scope,
            "producer": "strict functional Spike same-ELF complete execution",
            "argv": [str(GCC.with_name("spike")), "-g", "--isa=rv64gc", "--extension=gemmini",
                     "-m0x80000000:0x400000000", str(elf)],
            "engine": {"path": str(GCC.with_name("spike")), "sha256": sha(GCC.with_name("spike"))},
            "stdout_path": str(TARGET / "histogram_g.stdout"),
            "stdout_sha256": sha(TARGET / "histogram_g.stdout"),
            "invocation_authority": "Frozen main.c control/candidate mode checks and ABBA order; no inferred call count",
            "timing_scope": "Complete functional execution includes validation; exact function scopes, not a hardware ROI",
        }
        full = census(elf, histogram, symbols=[symbol], scope_id=scope, execution_binding=binding)
        save(destination / (symbol + "_features.json"), full)
        count = full["features"]["executed_instructions"]
        scopes[symbol] = {
            "actual_source_invocations": calls, "all_invocations_instructions": count,
            "per_invocation_instruction_mean": count / calls,
            "cpu_opcode_classes_all_invocations": full["features"]["cpu_opcode_classes"],
            "caveat": "Per-invocation mean includes the symbol's internal work only; indirect callback validation/callees require separate scope",
        }
    rows = []
    for line in (TARGET / "spike.stdout").read_text().splitlines():
        if line.startswith("INTEGER_PV_ROW "):
            fields = dict(field.split("=", 1) for field in line.split()[1:])
            rows.append({key: int(value, 16 if key == "digest" else 10) for key, value in fields.items()})
    assert [(row["id"], row["sample"]) for row in rows] == [(0, 0), (1, 1), (1, 2), (0, 3)]
    original = sum(row["instructions"] for row in rows if row["id"] == 0) / 2
    candidate = sum(row["instructions"] for row in rows if row["id"] == 1) / 2
    assert original == 635401 and candidate == 6234123
    for path in [Path(__file__), CORE / "src/merlin/llvmlower/exact_binary32_rows.py",
                 CORE / "src/merlin/llvmlower/scaled_integer_dot_enclosure.py",
                 histogram, TARGET / "histogram_g.stdout", TARGET / "histogram_g.stderr",
                 *destination.glob("*_features.json")]:
        pins[str(path)] = sha(path)
    result = {
        "schema": "source_integer_pv_complete_negative_v1", "status": "qualified_negative",
        "hypothesis": "Exact binary32 P lattice and original i32 V codes avoid lossy V re-encoding; equal-exponent radix groups reduce readout",
        "ownership": {"exact representation/range/enclosure/observer": "Merlin", "grouped products/CPU ISA/ABI": "OOT"},
        "native_original_compiled_i8_exact": 360448, "target_original_compiled_i8_exact": 16384,
        "target_actual_integer_readouts_exact": 98304, "radix_groups": 6,
        "target_calls": "6 batched calls, 24 KV products", "readout_bytes": 393216,
        "workspace_bytes": 255232, "timing_rows": rows,
        "mean_retired_instructions": {"source": original, "candidate": candidate},
        "instruction_ratio": candidate / original,
        "cost_scope": json.loads((TARGET / "qualification.json").read_text())["cost_scope"],
        "gate": "All original i8 words, dirty output/workspace/input guards, all five rounding-mode output comparisons, final executable zero FSM",
        "scope": "One complete PV and quantizer; QK/softmax/projection/GQA view preparation excluded from both arms",
        "current_policy": "Exact observed i8 experiment, stronger than user's unchanged whole tolerance; no default policy or normal routing",
        "decision": "Reject hardware admission/promotion: best zero-replay context costs 9.81x source retired instructions",
        "hardware_cycles": "UNKNOWN", "whole_accuracy": "NOT RUN: no whole route",
        "negative_attempts": [
            {"path": str(TARGET.parent / "source-integer-pv-target-20261007"), "result": "wide-A resource refusal for small K; no executable"},
            {"path": str(TARGET.parent / "source-integer-pv-target-v2-20261007"), "result": "diagnostic C compile error; no executable"},
            {"path": str(TARGET.parent / "source-integer-pv-target-v3-20261007"), "result": "bare-metal diagnostic library link refusal; no executable"},
            {"path": str(TARGET / "histogram.stderr"), "result": "unsupported Spike --histogram option, retained; subsequent -g is qualified"},
        ],
        "attribution": scopes, "token_usage_available": False, "pins": pins,
    }
    save(destination / "receipt.json", result)
    print(json.dumps({"receipt": str(destination / "receipt.json"), "sha256": sha(destination / "receipt.json"),
                      "source_instructions": original, "candidate_instructions": candidate, "pins": len(pins)}), flush=True)


if __name__ == "__main__":
    main()
