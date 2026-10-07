"""Original compiled consumer observations of the single rejected PV policy.

This is attribution after the frozen negative, never policy selection, tolerance
changes or replacement of an oracle. All paired arrays are native live inputs
from that same attempt; later contexts include propagated earlier differences.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import source_aware_pv_native_screen as screen
from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.llvmlower.quantized_consumer_frontier import (
    find_quantized_consumer_frontiers,
    quantized_consumer_semantic_sha256,
    validate_quantized_consumer_frontier,
)

ROOT = screen.OUT
OUT = ROOT.parent / "source-aware-projected-pv-observers-20261007"
OBS = (
    screen.OLD / "out/artifacts/probes/dynamic-i8-attention-quant-observations-20261007"
)
TYPED = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_host_abi/model.mlir"
)


def main():
    OUT.mkdir(exist_ok=False)
    record = json.loads((ROOT / "native_validation.json").read_text())
    assert (
        record["status"] == "fail" and record["calls"] == 22 and record["refusals"] == 0
    )
    assert all(screen.sha(p) == h for p, h in record["pins"].items())
    previous = json.loads((OBS / "observations.json").read_text())
    original_consumer = ctypes.CDLL(str(OBS / "original_quant_helpers.so"))
    llvm = (OBS / "original_quant_helpers.ll").read_text()
    current_llvm = screen.SOURCE.read_text()
    module = parse_mlir_file(TYPED)
    function = next(
        op
        for op in module.ops
        if op.name == "func.func" and op.sym_name.data == "forward"
    )
    operations = list(function.body.block.ops)
    declaration = json.loads((ROOT / "declaration.json").read_text())
    results = []
    pins = {
        str(p): screen.sha(p)
        for p in [
            Path(__file__),
            ROOT / "native_validation.json",
            ROOT / "declaration.json",
            TYPED,
            OBS / "observations.json",
            OBS / "original_quant_helpers.so",
            OBS / "original_quant_helpers.ll",
            screen.SOURCE,
        ]
    }
    for i, (binding, source) in enumerate(
        zip(
            previous["source_bindings"],
            declaration["typed_source_complete_attention"],
            strict=True,
        )
    ):
        assert source["pv_source_ordinal"] == binding["source_pv_ordinal"]
        pv = operations[binding["source_pv_ordinal"]]
        frontiers = find_quantized_consumer_frontiers(
            module, source_values=[pv.results[0]]
        )
        selected = [
            f for f in frontiers if f.integer_output.owner.name == "linalg.generic"
        ]
        assert len(selected) == 1
        frontier = selected[0]
        validate_quantized_consumer_frontier(frontier)
        assert (
            frontier.source_uses_closed
            and not frontier.floating_escapes
            and not frontier.unquantized_source_escapes
        )
        assert (
            quantized_consumer_semantic_sha256(frontier)
            == binding["consumer_semantic_sha256"]
        )
        assert [str(v.type) for v in frontier.observation_outputs] == [
            "tensor<1x8x2048xi8>"
        ]
        symbol = binding["compiled_actual_consumer_symbol"]
        start = current_llvm.index("define internal void @" + symbol + "(")
        end = current_llvm.index("\n}", start) + 2
        body = current_llvm[start:end]
        assert (
            hashlib.sha256(body.encode()).hexdigest() == binding["compiled_body_sha256"]
        )
        start = llvm.index(f"define void @attention_quant_{i}(")
        end = llvm.index("\n}", start) + 2
        words = re.findall(r"fmul float %\d+, f0x([0-9A-F]{8})", llvm[start:end])
        assert len(words) == 1
        factor = np.asarray([int(words[0], 16)], "u4").view("f4")[0]
        caller = getattr(original_consumer, f"attention_quant_{i}")
        caller.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        caller.restype = None
        here = OUT / f"context_{i:02d}"
        here.mkdir()
        carriers = {
            name: np.load(ROOT / f"context_{i:02d}" / (source_name + ".npy"))
            for name, source_name in [
                ("source", "source_PV"),
                ("candidate", "candidate_PV"),
            ]
        }
        observations = {}
        for name, carrier in carriers.items():
            pin = (
                ROOT
                / f"context_{i:02d}"
                / ({"source": "source_PV", "candidate": "candidate_PV"}[name] + ".npy")
            )
            pins[str(pin)] = screen.sha(pin)
            storage = np.full(16384 + 128, 73, "i1")
            target = storage[64:-64].reshape(1, 8, 2048)
            before = carrier.tobytes()
            caller(carrier.ctypes.data, target.ctypes.data)
            assert (
                np.all(storage[:64] == 73)
                and np.all(storage[-64:] == 73)
                and before == carrier.tobytes()
            )
            raw = np.asarray(carrier * factor, dtype="f4")
            independent = (
                np.rint(np.clip(raw, np.float32(-128), np.float32(127)))
                .astype("i1")
                .transpose(1, 0, 2)
                .reshape(target.shape)
            )
            assert np.array_equal(target, independent)
            observations[name] = target.copy()
            np.save(here / (name + "_i8.npy"), target)
        differing = observations["source"] != observations["candidate"]
        coords = np.argwhere(differing)
        first = None
        if coords.size:
            _, q, c = map(int, coords[0])
            h, ch = divmod(c, 64)
            s, candidate = [
                carriers[name][h, q, ch] for name in ("source", "candidate")
            ]
            sr, cr = np.float32(s * factor), np.float32(candidate * factor)
            bin_edge = np.float32(np.floor(np.float64(sr)) + 0.5)
            first = {
                "head": h,
                "query": q,
                "channel": ch,
                "source_word": int(s.view("u4")),
                "candidate_word": int(candidate.view("u4")),
                "source_value": float(s),
                "candidate_value": float(candidate),
                "source_i8": int(observations["source"][0, q, c]),
                "candidate_i8": int(observations["candidate"][0, q, c]),
                "source_preclamp": float(sr),
                "candidate_preclamp": float(cr),
                "nearest_half_boundary": float(bin_edge),
                "source_distance_to_half": float(abs(np.float64(sr) - bin_edge)),
                "candidate_distance_to_half": float(abs(np.float64(cr) - bin_edge)),
            }
        assert np.array_equal(
            observations["source"][:, 0], observations["candidate"][:, 0]
        )
        if i == 0:
            original = np.load(
                screen.OLD
                / "out/artifacts/probes/dynamic-i8-attention-original-attribution-20261007/context_00_family1/original.npy"
            )
            assert np.array_equal(original.view("u4"), carriers["source"].view("u4"))
            assert np.array_equal(
                observations["source"], np.load(OBS / "context_00/original.npy")
            )
        results.append(
            {
                "context_for_binding_only": i,
                "typed_consumer_semantic_sha256": binding["consumer_semantic_sha256"],
                "compiled_current_source_consumer_body_exact": True,
                "closed_i8_observations": 16384,
                "floating_escapes": 0,
                "source_quant_factor_word": int(factor.view("u4")),
                "changed_observed_i8": int(differing.sum()),
                "changed_per_query": [int(differing[0, row].sum()) for row in range(8)],
                "first_changed_observation": first,
                "original_compiled_consumer_input_guards": True,
                "query0_all2048_i8_exact": True,
                "scope": "paired local consumer outputs on actual live propagated inputs; contexts after0 are not original full-model trajectory",
            }
        )
        print(
            "PV_ORIGINAL_OBSERVER",
            i,
            results[-1]["changed_observed_i8"],
            first,
            flush=True,
        )
    full = np.load(ROOT / "output.npy")
    control = np.load(ROOT / "control.npy")
    assert np.array_equal(full[:, 0].view("u4"), control[:, 0].view("u4"))
    for path in OUT.rglob("*"):
        if path.is_file():
            pins[str(path)] = screen.sha(path)
    screen.save(
        OUT / "diagnostics.json",
        {
            "schema": "source_aware_projected_pv_rejected_observer_diagnostic_v1",
            "status": "pass",
            "candidate_status": "REJECTED original whole gate",
            "earliest_changed_context": next(
                r["context_for_binding_only"]
                for r in results
                if r["changed_observed_i8"]
            ),
            "contexts": results,
            "all22_same_complete_typed_i8_consumers": True,
            "no_unquantized_float_escape": True,
            "source_query0_all22_i8_exact": True,
            "whole_token0_all32000_logits_bits_exact": True,
            "new_policy_or_threshold": "NONE: unchanged frozen P14 policy; no precision ladder",
            "cycles": "UNKNOWN, target not run",
            "token_usage_available": False,
            "pins": pins,
        },
    )


if __name__ == "__main__":
    main()
