"""Functional whole closure for the prepared-norm exact-observer prototype.

The unchanged source PV runs first. Certified centers replace only its closed
floating carrier; ambiguous and one-hot points retain that actual source value.
This is a native integer-product stand-in, not a normal target implementation
or a performance measurement. The earlier approximate-policy failure remains
immutable and is not used to choose a different probability precision.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import source_aware_pv_native_screen as screen
from merlin.llvmlower import quant_hoist
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.projected_integer_pv_enclosure import enclose_projected_integer_pv
from merlin.runtime.dispatch_runtime import resolve_forward_args

OUT = screen.OUT.parent / "source-aware-pv-exact-observer-whole-native-20261007"
NORMS = screen.OUT.parent / "source-aware-pv-prepared-norm-screen-20261007"
OBS = (
    screen.OLD / "out/artifacts/probes/dynamic-i8-attention-quant-observations-20261007"
)
LOCAL = (
    screen.OLD
    / "out/artifacts/probes/dynamic-i8-attention-original-attribution-20261007"
)
CAP = (
    screen.OLD
    / "out/artifacts/probes/tiny-attention-original-v-projection-capture-v2-20261007"
)


def main():
    OUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    originals = [
        screen.OUT / "declaration.json",
        screen.OUT / "native_validation.json",
        NORMS / "qualification.json",
        screen.OUT / "model.so",
        screen.OUT / "model.native.ll",
        OBS / "original_quant_helpers.so",
        OBS / "observations.json",
        screen.BASE / "output.npy",
        screen.BUNDLE / "golden.npy",
        screen.BUILD / quant_hoist.PLAN_FILE,
        screen.BUILD / quant_hoist.VALUES_FILE,
        screen.CORE / "src/merlin/llvmlower/projected_integer_pv_enclosure.py",
        screen.CORE / "src/merlin/llvmlower/projected_integer_pv.py",
        Path(__file__),
    ]
    pins = {str(path): screen.sha(path) for path in originals}
    declaration = json.loads((screen.OUT / "declaration.json").read_text())
    bindings = declaration["typed_source_joins"]
    prior_norm = json.loads((NORMS / "qualification.json").read_text())
    assert prior_norm["status"] == "pass" and len(bindings) == 22
    # Reuse the immutable compiled bridge. Its original source and complete
    # all-use quantized consumer witnesses were already qualified; the callback
    # alone changes. No observed source output decides certification.
    screen.save(
        OUT / "declaration.json",
        {
            "schema": "source_aware_pv_exact_observer_whole_declaration_v1",
            "declared_utc": started,
            "policy": "same fixed P14 center; exact original i8 observation certificate; original point PV on ambiguity; original onehot rows",
            "source": "unchanged 2085 QK, mask, nonlinear, P, V i32 reconstruction/scales/GQA and original quantizer",
            "proof": "prepared source-valid row/column error enclosure; no floating escape; finite RNE/gradual/nontrapping/unobserved flags",
            "original_approximate_failure_unchanged": str(
                screen.OUT / "native_validation.json"
            ),
            "whole_gate": {
                "tokens": 8,
                "layers": 22,
                "elements": 256000,
                "atol": 0.03125,
                "rtol": 0.02,
            },
            "cost": "UNKNOWN; original source PV executes first and native exact integer dot is a functional stand-in",
            "pins": pins,
        },
    )
    library = ctypes.CDLL(str(screen.OUT / "model.so"))
    library.projected_environment_supported.restype = ctypes.c_int
    assert library.projected_environment_supported()
    model = HostModel.load(str(screen.OUT / "model.so"))
    quant = ctypes.CDLL(str(OBS / "original_quant_helpers.so"))
    args = resolve_forward_args(screen.BUNDLE)
    values = quant_hoist.read_values(screen.BUILD)
    args.extend(
        np.ascontiguousarray(values[entry.key])
        for entry in quant_hoist.read_plan(screen.BUILD)
    )
    assert len(args) == 668
    original = np.load(screen.BASE / "output.npy")
    golden = np.load(screen.BUNDLE / "golden.npy")
    control = np.empty_like(original)
    model([(a.ctypes.data, a.shape) for a in [*args, control]])
    assert hashlib.sha256(control.astype("<f4").tobytes()).hexdigest() == screen.RAW
    assert np.array_equal(control.view("u4"), original.view("u4"))
    assert np.allclose(control, golden, atol=0.03125, rtol=0.02)
    np.save(OUT / "control.npy", control)
    print("EXACT_OBSERVER_SAME_ABI_CONTROL_PASS", flush=True)
    vtype = ctypes.CFUNCTYPE(None, ctypes.c_int32, *[ctypes.c_void_p] * 3)
    ptype = ctypes.CFUNCTYPE(None, *[ctypes.c_void_p] * 3)
    state = {"next_v": 0, "next_pv": 0, "live": None}
    errors = []
    events = []

    @vtype
    def vtap(context, codeptr, scaleptr, vptr):
        try:
            assert context == state["next_v"] and state["live"] is None
            code = (
                np.frombuffer(ctypes.string_at(codeptr, 8 * 256 * 4), "i4")
                .copy()
                .reshape(8, 256)
            )
            beta = (
                np.frombuffer(ctypes.string_at(scaleptr, 256 * 4), "f4")
                .copy()
                .reshape(4, 64)
            )
            v = (
                np.frombuffer(ctypes.string_at(vptr, 32 * 8 * 64 * 4), "f4")
                .copy()
                .reshape(32, 8, 64)
            )
            alpha = np.asarray([bindings[context]["activation_scale_word"]], "u4").view(
                "f4"
            )[0]
            source_v = np.asarray(
                np.asarray(code.astype("f4") * alpha, "f4") * beta.reshape(1, 256), "f4"
            )
            expected = np.repeat(
                source_v.reshape(8, 4, 64).transpose(1, 0, 2), 8, axis=0
            )
            assert np.array_equal(expected.view("u4"), v.view("u4"))
            state["live"] = (context, code, beta, v, alpha)
            state["next_v"] += 1
        except Exception as error:
            errors.append(("V", repr(error)))

    @ptype
    def pv_tap(outptr, pptr, vptr):
        try:
            context, code, beta, v, alpha = state["live"]
            assert context == state["next_pv"]
            p = (
                np.frombuffer(ctypes.string_at(pptr, 32 * 8 * 8 * 4), "f4")
                .copy()
                .reshape(32, 8, 8)
            )
            actual_v = np.frombuffer(ctypes.string_at(vptr, v.nbytes), "f4").reshape(
                v.shape
            )
            assert np.array_equal(actual_v.view("u4"), v.view("u4"))
            source = (
                np.frombuffer(ctypes.string_at(outptr, 32 * 8 * 64 * 4), "f4")
                .copy()
                .reshape(32, 8, 64)
            )
            before = tuple(a.tobytes() for a in (p, code, beta, v))
            original_paths = {
                "probability": LOCAL / f"context_{context:02d}_family1/a.npy",
                "code": CAP / f"context_{context:02d}/code.npy",
                "channel_scale": CAP / f"context_{context:02d}/channel_scale.npy",
                "source_V": CAP / f"context_{context:02d}/v.npy",
                "source_PV": LOCAL / f"context_{context:02d}_family1/original.npy",
            }
            live = (p, code, beta.reshape(256), v, source)
            all_original_inputs = True
            for array, (name, path) in zip(live, original_paths.items(), strict=True):
                reference = np.load(path)
                assert np.array_equal(
                    array.view("u4"), reference.reshape(array.shape).view("u4")
                ), name
                pins[str(path)] = screen.sha(path)
            code_group = np.ascontiguousarray(code.reshape(8, 4, 64).transpose(1, 0, 2))
            caller = getattr(quant, f"attention_quant_{context}")
            caller.argtypes = [ctypes.c_void_p] * 2
            caller.restype = None

            def observe(carrier):
                carrier = np.ascontiguousarray(carrier, "f4")
                guarded = np.full(16384 + 128, 73, "i1")
                input_bytes = carrier.tobytes()
                caller(carrier.ctypes.data, guarded[64:-64].ctypes.data)
                assert (
                    input_bytes == carrier.tobytes()
                    and np.all(guarded[:64] == 73)
                    and np.all(guarded[-64:] == 73)
                )
                return guarded[64:-64].copy().reshape(8, 32, 64).transpose(1, 0, 2)

            metadata = {}
            refusal = None
            try:
                if not library.projected_environment_supported():
                    raise ValueError("unsupported FENV")
                q = np.rint(p.astype("f8") * 16384).astype("i8")
                dots = q.reshape(4, 64, 8) @ code_group.astype("i8")
                center, lo, hi, metadata = enclose_projected_integer_pv(
                    p,
                    code_group,
                    np.asarray(alpha, "f4"),
                    beta,
                    np.ascontiguousarray(v[::8]),
                    dots,
                    policy=screen.POLICY,
                    epoch=f"source-exact-observer:{context}",
                    rounding_mode="rne",
                )
                certified = observe(lo) == observe(hi)
                retained = (np.count_nonzero(p, axis=2) == 0) | (
                    (np.count_nonzero(p, axis=2) == 1)
                    & np.any(p == np.float32(1), axis=2)
                )
                certified[retained] = False
                carrier = np.where(certified, center, source)
                assert np.all(lo.astype("f8") <= source.astype("f8")) and np.all(
                    source.astype("f8") <= hi.astype("f8")
                )
            except ValueError as error:
                carrier = source.copy()
                certified = np.zeros(source.shape, bool)
                retained = np.ones((32, 8), bool)
                refusal = str(error)
            observed_source, observed_candidate = observe(source), observe(carrier)
            assert np.array_equal(observed_source, observed_candidate)
            ctypes.memmove(outptr, carrier.ctypes.data, carrier.nbytes)
            assert before == tuple(a.tobytes() for a in (p, code, beta, v))
            directory = OUT / f"context_{context:02d}"
            directory.mkdir()
            for name, array in [
                ("carrier", carrier),
                ("certified", certified),
                ("source_i8", observed_source),
                ("candidate_i8", observed_candidate),
            ]:
                np.save(directory / (name + ".npy"), array)
            event = {
                **metadata,
                "context_for_binding_only": context,
                "runtime_refusal": refusal,
                "certified_nononehot_outputs": int(certified.sum()),
                "ambiguous_outputs": int((~certified).sum() - retained.sum() * 64),
                "source_onehot_rows": int(retained.sum()),
                "changed_float_carrier_words": int(
                    np.count_nonzero(carrier.view("u4") != source.view("u4"))
                ),
                "all_original_live_inputs_bits_exact": all_original_inputs,
                "original_compiled_i8_observations_exact": 16384,
                "source_QK_P_mask_nonlinear_and_quant_consumer_unchanged": True,
                "actual_source_binding": bindings[context],
            }
            events.append(event)
            screen.save(directory / "qualification.json", event)
            screen.save(OUT / "calls_live.json", events)
            print(
                "EXACT_OBSERVER_PV",
                context,
                event["certified_nononehot_outputs"],
                event["ambiguous_outputs"],
                refusal,
                flush=True,
            )
            state["live"] = None
            state["next_pv"] += 1
        except Exception as error:
            errors.append(("PV", repr(error)))

    library.projected_install.argtypes = [vtype, ptype]
    library.projected_install(vtap, pv_tap)
    output = np.empty_like(original)
    model([(a.ctypes.data, a.shape) for a in [*args, output]])
    assert not errors and state == {"next_v": 22, "next_pv": 22, "live": None}, (
        errors,
        state,
    )
    assert all(screen.sha(path) == digest for path, digest in pins.items())
    np.save(OUT / "output.npy", output)
    raw = hashlib.sha256(output.astype("<f4").tobytes()).hexdigest()
    assert raw == screen.RAW and np.array_equal(output.view("u4"), original.view("u4"))
    assert np.allclose(output, golden, atol=0.03125, rtol=0.02)
    for path in OUT.rglob("*"):
        if path.is_file():
            pins[str(path)] = screen.sha(path)
    record = {
        "schema": "source_aware_pv_exact_observer_whole_native_v1",
        "status": "pass",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_off_same_ABI_control_all256000_bits_exact": True,
        "compiled_gate_pass": True,
        "torch_gate_pass": True,
        "original_atol": 0.03125,
        "original_rtol": 0.02,
        "whole_geometry": {"tokens": 8, "layers": 22, "elements": 256000},
        "output_raw_sha256": raw,
        "calls": len(events),
        "refusals": sum(e["runtime_refusal"] is not None for e in events),
        "all_original_live_inputs_bits_exact": True,
        "original_compiled_i8_observations_exact": 360448,
        "source_onehot_rows_retained": sum(e["source_onehot_rows"] for e in events),
        "certified_nononehot_outputs": sum(
            e["certified_nononehot_outputs"] for e in events
        ),
        "ambiguous_point_fallbacks": sum(e["ambiguous_outputs"] for e in events),
        "changed_float_carrier_words": sum(
            e["changed_float_carrier_words"] for e in events
        ),
        "native_integer_product_standin": True,
        "normal_target_route": "NOT ENABLED; complete first target separately instruction-negative",
        "cycles": "UNKNOWN",
        "token_usage_available": False,
        "pins": pins,
    }
    screen.save(OUT / "native_validation.json", record)
    print(
        "FULL_EXACT_OBSERVER_GATE",
        json.dumps({k: v for k, v in record.items() if k != "pins"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
