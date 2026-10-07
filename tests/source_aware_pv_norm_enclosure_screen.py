"""Prepared row/column exact-observer bound screen on original source inputs.

The original P14 approximation stays rejected. This is a distinct exact i8
observation experiment with source PV fallback for every ambiguous point. It
uses the same fixed grid, not a precision ladder or a golden-derived threshold.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import source_aware_pv_native_screen as screen
from merlin.llvmlower.projected_integer_pv_enclosure import enclose_projected_integer_pv

OUT = screen.OUT.parent / "source-aware-pv-prepared-norm-screen-20261007"
LOCAL = (
    screen.OLD
    / "out/artifacts/probes/dynamic-i8-attention-original-attribution-20261007"
)
CAP = (
    screen.OLD
    / "out/artifacts/probes/tiny-attention-original-v-projection-capture-v2-20261007"
)
OBS = (
    screen.OLD / "out/artifacts/probes/dynamic-i8-attention-quant-observations-20261007"
)


def main():
    OUT.mkdir(exist_ok=False)
    quant = ctypes.CDLL(str(OBS / "original_quant_helpers.so"))
    bindings = json.loads((CAP / "qualification.json").read_text())["source_contexts"]
    pins = {
        str(p): screen.sha(p)
        for p in [
            Path(__file__),
            CAP / "qualification.json",
            OBS / "original_quant_helpers.so",
            OBS / "observations.json",
            screen.CORE / "src/merlin/llvmlower/projected_integer_pv_enclosure.py",
            screen.CORE / "src/merlin/llvmlower/projected_integer_pv.py",
        ]
    }
    screen.save(
        OUT / "declaration.json",
        {
            "schema": "prepared_norm_exact_i8_observer_declaration_v1",
            "fixed_probability_grid_bits": 14,
            "policy": "exact source i8 observation; same P14 center only if original complete quantizer interval bins agree, actual original point PV fallback otherwise",
            "source_bound": "prepared P error/L1 row norms + i32/V column maxima + separate source scale roundoff + increasing-K separate MUL/ADD gamma and gradual underflow",
            "source_effects": "RNE/gradual/nontrapping/unobserved flags; original QK/P/mask and original typed all-use i8 closure retained",
            "source_predicate": "no measured original/candidate output selects certification",
            "cost": "not yet measured; no array kernel generated, all packing/dispatch/readout/refinement unpriced",
            "pins": pins,
        },
    )
    rows = []
    for i, binding in enumerate(bindings):
        here = OUT / f"context_{i:02d}"
        here.mkdir()
        paths = {
            "p": LOCAL / f"context_{i:02d}_family1/a.npy",
            "original": LOCAL / f"context_{i:02d}_family1/original.npy",
            "code": CAP / f"context_{i:02d}/code.npy",
            "beta": CAP / f"context_{i:02d}/channel_scale.npy",
            "source_v": CAP / f"context_{i:02d}/v.npy",
        }
        arrays = {key: np.load(path) for key, path in paths.items()}
        pins.update({str(path): screen.sha(path) for path in paths.values()})
        p = arrays["p"]
        code = np.ascontiguousarray(arrays["code"].reshape(8, 4, 64).transpose(1, 0, 2))
        beta = arrays["beta"].reshape(4, 64)
        v = np.ascontiguousarray(arrays["source_v"][::8])
        alpha = np.asarray([binding["activation_scale_word"]], "u4").view("f4")[0]
        q = np.rint(p.astype("f8") * 16384).astype("i8")
        dots = q.reshape(4, 64, 8) @ code.astype("i8")
        center, lo, hi, metadata = enclose_projected_integer_pv(
            p,
            code,
            np.asarray(alpha, "f4"),
            beta,
            v,
            dots,
            policy=screen.POLICY,
            epoch=f"original-source-context:{i}",
            rounding_mode="rne",
        )
        original = arrays["original"]
        assert np.all(lo.astype("f8") <= original.astype("f8")) and np.all(
            original.astype("f8") <= hi.astype("f8")
        )
        caller = getattr(quant, f"attention_quant_{i}")
        caller.argtypes = [ctypes.c_void_p] * 2
        caller.restype = None
        observations = {}
        for name, carrier in [
            ("original", original),
            ("center", center),
            ("lower", lo),
            ("upper", hi),
        ]:
            target = np.full(16384 + 128, 73, "i1")
            before = carrier.tobytes()
            caller(carrier.ctypes.data, target[64:-64].ctypes.data)
            assert (
                np.all(target[:64] == 73)
                and np.all(target[-64:] == 73)
                and before == carrier.tobytes()
            )
            observations[name] = target[64:-64].copy().reshape(1, 8, 2048)
        assert np.array_equal(
            observations["original"], np.load(OBS / f"context_{i:02d}/original.npy")
        )
        certified = observations["lower"] == observations["upper"]
        assert np.array_equal(
            observations["center"][certified], observations["original"][certified]
        )
        mask = certified.reshape(8, 32, 64).transpose(1, 0, 2)
        source_rows = (np.count_nonzero(p, axis=2) == 0) | (
            (np.count_nonzero(p, axis=2) == 1) & np.any(p == np.float32(1), axis=2)
        )
        mask[source_rows] = False
        carrier = np.where(mask, center, original)
        output = np.empty((1, 8, 2048), "i1")
        caller(carrier.ctypes.data, output.ctypes.data)
        assert np.array_equal(output, observations["original"])
        for name, array in [
            ("center", center),
            ("lower", lo),
            ("upper", hi),
            ("exact_observer_carrier", carrier),
            ("certified", mask),
            ("source_i8", output),
            ("integer_dots", dots),
        ]:
            np.save(here / (name + ".npy"), array)
        record = {
            **metadata,
            "context_for_binding_only": i,
            "source_outputs_enclosed": 16384,
            "exact_observed_i8": 16384,
            "certified_nononehot_outputs": int(mask.sum()),
            "source_retained_onehot_outputs": int(source_rows.sum()) * 64,
            "ambiguous_nononehot_fallback_outputs": int(
                (~mask).sum() - source_rows.sum() * 64
            ),
            "point_fallback_source_muladd_pairs": int(
                (~mask).sum() - source_rows.sum() * 64
            )
            * 8,
            "original_compiled_consumer_input_and_output_guards": True,
        }
        rows.append(record)
        screen.save(here / "qualification.json", record)
        print(
            "PREPARED_NORM_FRONTIER",
            i,
            record["certified_nononehot_outputs"],
            record["ambiguous_nononehot_fallback_outputs"],
            record["max_absolute_source_error_bound"],
            flush=True,
        )
    for path in OUT.rglob("*"):
        if path.is_file():
            pins[str(path)] = screen.sha(path)
    screen.save(
        OUT / "qualification.json",
        {
            "schema": "source_aware_pv_prepared_norm_all22_screen_v1",
            "status": "pass",
            "contexts": rows,
            "original_complete_i8_observations_exact": 360448,
            "certified_nononehot_outputs": sum(
                r["certified_nononehot_outputs"] for r in rows
            ),
            "ambiguous_nononehot_fallback_outputs": sum(
                r["ambiguous_nononehot_fallback_outputs"] for r in rows
            ),
            "all_enclosures_cover_actual_original_source": True,
            "whole_original_gate": "NOT RUN for this successor",
            "cost": "UNKNOWN: native integer stand-in, no target products or certificate timing",
            "promotion": "disabled pending complete costs and normal whole gates",
            "token_usage_available": False,
            "pins": pins,
        },
    )


if __name__ == "__main__":
    main()
