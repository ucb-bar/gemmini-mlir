"""One predeclared source-aware approximate PV policy, immutable whole gate.

Experimental symbols/indices join retained typed source instances. They select
no production strategy. The reusable numerical policy lives in Merlin. This
native bridge executes source PV first solely as a diagnostic/fallback oracle;
no target execution or performance is claimed.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from merlin.llvmlower import quant_hoist
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.integer_producer_range import ConstantIntegerSumProductsRange
from merlin.llvmlower.projected_integer_pv import (
    ProjectedIntegerPVPolicy,
    evaluate_projected_integer_pv,
)
from merlin.runtime.dispatch_runtime import resolve_forward_args

HERE = Path(__file__).resolve().parents[1]
CORE = Path("/scratch/agustin/tmp/merlin-source-aware-pv-main-20261007")
OLD = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
REF = Path("/scratch/agustin/tmp/gemmini-residual-domain-20261007")
CHAMP = Path(
    "/scratch/agustin/tmp/gemmini-rne-observer-cells-20261007/out/artifacts/probes/rne-zero-observer-normal-2076-20261007"
)
SOURCE = CHAMP / "selected/native/host_llvm/expanded.native.ll"
BASE = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host"
)
BUILD = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build"
)
BUNDLE = Path(
    "/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle"
)
RUNTIME = Path(
    "/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/abi/mlir_runtime.c"
)
CLANG = Path(
    "/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang"
)
OUT = HERE / "out/artifacts/probes/source-aware-projected-pv-native-20261007"
POLICY = ProjectedIntegerPVPolicy(14, 8, True, True, True)
RAW = "ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def extract(text, symbol):
    header = f"define internal void @{symbol}(ptr %0, ptr %1, ptr %2) #1 {{\n"
    assert text.count(header) == 1
    start = text.index(header)
    end = text.index("\n}", start) + 2
    return start, end, text[start:end], header


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    POLICY.validate()
    binding_path = (
        REF / "docs/perf_records/tiny_attention_projection_binding_audit.json"
    )
    typed_path = (
        REF / "docs/perf_records/attention_typed_representation_source_audit.json"
    )
    seed_path = REF / "docs/perf_records/tiny_attention_projection_seed_audit.json"
    strategy_path = (
        OLD
        / "out/artifacts/probes/tiny-attention-source-integer-strategy-20261007/result.json"
    )
    bindings = json.loads(binding_path.read_text())["contexts"]
    typed = json.loads(typed_path.read_text())
    prior_strategy = json.loads(strategy_path.read_text())
    assert (
        typed["status"] == "PASS" and not typed["tiny_gqa_or_quant_axis_defect_found"]
    )
    assert len(bindings) == len(typed["tiny"]) == len(prior_strategy["contexts"]) == 22
    original_text = SOURCE.read_text()
    # Pin source, complete original source witnesses, actual input data and policy
    # before running any candidate. These are bindings, never fitting labels.
    pins = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            SOURCE,
            binding_path,
            typed_path,
            seed_path,
            strategy_path,
            BUNDLE / "model.mlir",
            BUNDLE / "golden.npy",
            BUILD / quant_hoist.PLAN_FILE,
            BUILD / quant_hoist.VALUES_FILE,
            CORE / "src/merlin/llvmlower/projected_integer_pv.py",
            CORE / "src/merlin/llvmlower/integer_producer_range.py",
            CLANG,
        ]
    }
    args = resolve_forward_args(BUNDLE)
    base_count = len(args)
    plan = quant_hoist.read_plan(BUILD)
    values = quant_hoist.read_values(BUILD)
    assert base_count == 513 and len(plan) == 155
    args.extend(np.ascontiguousarray(values[item.key]) for item in plan)
    assert len(args) == 668
    joins = []
    for i, binding in enumerate(bindings):
        assert binding["context_for_audit"] == i
        slot = binding["original_weight_function_argument_index"] - base_count
        weight = args[binding["original_weight_function_argument_index"]]
        assert weight.shape == (2048, 256) and weight.dtype == np.int8
        actual_weight_sha = hashlib.sha256(weight.tobytes()).hexdigest()
        assert (
            actual_weight_sha
            == prior_strategy["contexts"][i]["constant_weight_array_sha256"]
        )
        # Universal proof uses all legal signed-byte activations, not captured
        # code magnitudes. Source zero initialization and selected peeled first-K
        # overwrite remain separately retained in the strategy witness.
        ranges = [
            ConstantIntegerSumProductsRange(
                tuple(map(int, weight[:, col])), -128, 127
            ).require_exact_binary_conversion(significand_bits=24)
            for col in range(256)
        ]
        assert max(max(abs(a), abs(b)) for a, b in ranges) <= 2**24
        start, end, body, header = extract(
            original_text, binding["captured_producer_symbol"]
        )
        assert (
            hashlib.sha256(body.encode()).hexdigest()
            == binding["captured_producer_body_sha256"]
        )
        assert typed["tiny"][i]["v_reconstruction_body"] == [
            "arith.sitofp",
            "arith.mulf",
            "arith.mulf",
            "linalg.yield",
        ]
        joins.append(
            {
                "context_for_binding_only": i,
                "source_producer_symbol": binding["captured_producer_symbol"],
                "source_producer_body_sha256": sha_text(body),
                "activation_scale_word": binding["activation_scale_word"],
                "weight_argument": binding["original_weight_function_argument_index"],
                "weight_hoist_slot": slot,
                "weight_hoist_key": plan[slot].key,
                "weight_bytes_sha256": actual_weight_sha,
                "universal_source_code_absbound": max(
                    max(abs(a), abs(b)) for a, b in ranges
                ),
                "source_channel_scale_argument": binding[
                    "weight_channel_scale_source_argument_index"
                ],
                "group_map": "group=head//8; code[key,group*64+channel]; scale[group*64+channel]",
                "onehot_behavior": "actual original PV evaluator result retained on exact onehot/zero probability rows",
            }
        )
    pv_symbol = "forward.extracted.112"
    _, _, pv_body, _ = extract(original_text, pv_symbol)
    declaration = {
        "schema": "one_source_aware_projected_pv_policy_v1",
        "declared_utc": started,
        "policy": asdict(POLICY),
        "only_policy": True,
        "default_off": True,
        "source_changes": "ONLY PV endpoint replacement. QK/mask/scale/max/source nonlinear/softmax/P/reduction/quant consumer unchanged.",
        "source_exact_float_contract": False,
        "explicit_approximation": "P fixed power-of-two ties-even grid; original projected integer V products; post-dot source activation/channel scale placement changes source rounding.",
        "typed_source_joins": joins,
        "typed_source_complete_attention": typed["tiny"],
        "original_gate": {
            "elements": 256000,
            "tokens": 8,
            "layers": 22,
            "atol": 0.03125,
            "rtol": 0.02,
            "compiled_reference": str(BASE / "output.npy"),
            "torch_reference": str(BUNDLE / "golden.npy"),
        },
        "runtime_refusal": "unsupported fenv/nonfinite/map/cast/range => actual original PV; onehot/zero rows actual original PV",
        "epoch": "one invocation-local V producer and current PV; no pointer-keyed cache or retained preparation across epochs",
        "scope": "functional native stand-in; original PV runs first for fallback/error evidence. No target offload or cycles.",
        "original_PV_body_sha256": sha_text(pv_body),
        "token_usage_available": False,
        "pins": pins,
    }
    save(OUT / "declaration.json", declaration)
    text = original_text
    for i, binding in enumerate(bindings):
        start, end, body, header = extract(text, binding["captured_producer_symbol"])
        assert body.count("  ret void") == 1
        changed = body.replace(
            "  ret void",
            f"  call void @projected_v_tap(i32 {i},ptr %0,ptr %1,ptr %2)\n  ret void",
            1,
        )
        text = text[:start] + changed + text[end:]
    start, end, body, header = extract(text, pv_symbol)
    assert (
        body.count("  ret void") == 1
        and text.count("call void @" + pv_symbol + "(") == 22
    )
    changed = body.replace(
        "  ret void",
        "  call void @projected_pv_tap(ptr %0,ptr %1,ptr %2)\n  ret void",
        1,
    )
    text = text[:start] + changed + text[end:]
    text += "\ndeclare void @projected_v_tap(i32,ptr,ptr,ptr)\ndeclare void @projected_pv_tap(ptr,ptr,ptr)\n"
    (OUT / "model.native.ll").write_text(text)
    bridge = OUT / "bridge.c"
    bridge.write_text("""#include <stdint.h>
#include <fenv.h>
#include <xmmintrin.h>
typedef void(*vtap_t)(int32_t,void*,void*,void*);
typedef void(*ptap_t)(void*,void*,void*);
static vtap_t vtap; static ptap_t ptap;
void projected_install(vtap_t v,ptap_t p){vtap=v;ptap=p;}
void projected_v_tap(int32_t i,void*a,void*b,void*c){if(vtap)vtap(i,a,b,c);}
void projected_pv_tap(void*a,void*b,void*c){if(ptap)ptap(a,b,c);}
int projected_environment_supported(void){return fegetround()==FE_TONEAREST && !(_mm_getcsr() & 0x8040);}
""")
    argv = [
        str(CLANG),
        "-O2",
        "-fPIC",
        "-c",
        str(OUT / "model.native.ll"),
        "-o",
        str(OUT / "model.o"),
    ]
    deps = [
        BASE / "reference.c",
        BASE / "device/device_catalog_shim.c",
        RUNTIME,
        bridge,
    ]
    link = [
        "cc",
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        str(OUT / "model.o"),
        *map(str, deps),
        "-lm",
        "-o",
        str(OUT / "model.so"),
    ]
    save(
        OUT / "build.json",
        {
            "compile_argv": argv,
            "link_argv": link,
            "pins": {str(p): sha(p) for p in [*deps, CLANG, OUT / "model.native.ll"]},
        },
    )
    for command, name in [(argv, "compile"), (link, "link")]:
        result = subprocess.run(command, capture_output=True, text=True)
        (OUT / (name + ".stdout")).write_text(result.stdout)
        (OUT / (name + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, result.stderr
    library = ctypes.CDLL(str(OUT / "model.so"))
    library.projected_environment_supported.restype = ctypes.c_int
    assert library.projected_environment_supported() == 1
    model = HostModel.load(str(OUT / "model.so"))
    golden = np.load(BUNDLE / "golden.npy")
    compiled = np.load(BASE / "output.npy")
    control = np.empty_like(golden)
    model([(a.ctypes.data, a.shape) for a in [*args, control]])
    assert hashlib.sha256(control.astype("<f4").tobytes()).hexdigest() == RAW
    assert np.allclose(control, golden, atol=0.03125, rtol=0.02)
    np.save(OUT / "control.npy", control)
    save(
        OUT / "control.json",
        {
            "same_registered_abi": True,
            "source256000_bits_exact": True,
            "torch_gate": True,
            "raw_sha256": RAW,
            "source_2085_host_unchanged_except_observation_callbacks": True,
        },
    )
    print("SAME_ABI_EXACT_CONTROL_PASS", flush=True)
    vtype = ctypes.CFUNCTYPE(None, ctypes.c_int32, *[ctypes.c_void_p] * 3)
    ptype = ctypes.CFUNCTYPE(None, *[ctypes.c_void_p] * 3)
    state = {"next_v": 0, "next_pv": 0, "live": None}
    events = []
    errors = []

    @vtype
    def vtap(context, codeptr, scaleptr, vptr):
        try:
            assert context == state["next_v"] and state["live"] is None
            code = (
                np.frombuffer(ctypes.string_at(codeptr, 8 * 256 * 4), "i4")
                .copy()
                .reshape(8, 256)
            )
            scales = np.frombuffer(ctypes.string_at(scaleptr, 256 * 4), "f4").copy()
            v = (
                np.frombuffer(ctypes.string_at(vptr, 32 * 8 * 64 * 4), "f4")
                .copy()
                .reshape(32, 8, 64)
            )
            alpha = np.asarray([bindings[context]["activation_scale_word"]], "u4").view(
                "f4"
            )[0]
            expected = np.asarray(
                np.asarray(code.astype("f4") * alpha, dtype="f4") * scales[None, :],
                dtype="f4",
            ).reshape(8, 4, 64)
            assert np.array_equal(
                np.repeat(expected.transpose(1, 0, 2), 8, axis=0).view("u4"),
                v.view("u4"),
            )
            state["live"] = (context, code, scales, v, alpha)
            state["next_v"] += 1
        except Exception as error:
            errors.append(("V", repr(error)))

    @ptype
    def pv_tap(outputptr, pptr, vptr):
        try:
            context, code, scales, v, alpha = state["live"]
            assert context == state["next_pv"]
            p = (
                np.frombuffer(ctypes.string_at(pptr, 32 * 8 * 8 * 4), "f4")
                .copy()
                .reshape(32, 8, 8)
            )
            actual_v = (
                np.frombuffer(ctypes.string_at(vptr, v.nbytes), "f4")
                .copy()
                .reshape(v.shape)
            )
            assert np.array_equal(actual_v.view("u4"), v.view("u4"))
            original = (
                np.frombuffer(ctypes.string_at(outputptr, 32 * 8 * 64 * 4), "f4")
                .copy()
                .reshape(32, 8, 64)
            )
            unchanged_inputs = tuple(a.tobytes() for a in (p, code, scales, v))
            refusal = None
            try:
                if not library.projected_environment_supported():
                    raise ValueError("unsupported FENV")
                candidate, retained, metadata = evaluate_projected_integer_pv(
                    p,
                    np.ascontiguousarray(code.reshape(8, 4, 64).transpose(1, 0, 2)),
                    np.asarray(alpha, "f4"),
                    scales.reshape(4, 64),
                    policy=POLICY,
                    epoch=f"source-PV-invocation:{context}",
                    rounding_mode="rne",
                )
                candidate[retained] = original[retained]
                assert np.array_equal(
                    candidate[retained].view("u4"), original[retained].view("u4")
                )
                ctypes.memmove(outputptr, candidate.ctypes.data, candidate.nbytes)
            except ValueError as error:
                candidate = original.copy()
                retained = np.ones((32, 8), bool)
                metadata = {}
                refusal = str(error)
            assert unchanged_inputs == tuple(a.tobytes() for a in (p, code, scales, v))
            directory = OUT / f"context_{context:02d}"
            directory.mkdir()
            for name, array in [
                ("probability", p),
                ("code", code),
                ("channel_scale", scales),
                ("source_V", v),
                ("source_PV", original),
                ("candidate_PV", candidate),
                ("source_retained_rows", retained),
            ]:
                np.save(directory / (name + ".npy"), array)
            event = {
                **metadata,
                "context_for_binding_only": context,
                "runtime_refusal": refusal,
                "changed_source_PV_words": int(
                    np.count_nonzero(candidate.view("u4") != original.view("u4"))
                ),
                "max_abs_source_PV_error": float(
                    np.max(np.abs(candidate.astype("f8") - original.astype("f8")))
                ),
                "source_onehot_rows_exact": True,
                "inputs_unchanged": True,
                "source_QK_P_mask_semantics_unchanged": True,
                "actual_source_binding": joins[context],
            }
            events.append(event)
            save(directory / "metadata.json", event)
            save(OUT / "calls_live.json", events)
            print(
                "SOURCE_AWARE_PV",
                context,
                event["changed_source_PV_words"],
                event["max_abs_source_PV_error"],
                refusal,
                flush=True,
            )
            state["live"] = None
            state["next_pv"] += 1
        except Exception as error:
            errors.append(("PV", repr(error)))

    library.projected_install.argtypes = [vtype, ptype]
    library.projected_install(vtap, pv_tap)
    output = np.empty_like(golden)
    model([(a.ctypes.data, a.shape) for a in [*args, output]])
    assert not errors and state == {"next_v": 22, "next_pv": 22, "live": None}, (
        errors,
        state,
    )
    assert all(sha(path) == digest for path, digest in pins.items())
    np.save(OUT / "output.npy", output)
    source_ok = np.isclose(output, compiled, atol=0.03125, rtol=0.02)
    torch_ok = np.isclose(output, golden, atol=0.03125, rtol=0.02)
    record = {
        "schema": "one_source_aware_projected_pv_whole_native_v1",
        "status": "pass" if source_ok.all() and torch_ok.all() else "fail",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "policy": asdict(POLICY),
        "compiled_gate_pass": bool(source_ok.all()),
        "compiled_gate_failures": int((~source_ok).sum()),
        "torch_gate_pass": bool(torch_ok.all()),
        "torch_gate_failures": int((~torch_ok).sum()),
        "max_abs_vs_compiled": float(np.max(np.abs(output - compiled))),
        "max_abs_vs_torch": float(np.max(np.abs(output - golden))),
        "changed_source_words": int(
            np.count_nonzero(output.view("u4") != compiled.view("u4"))
        ),
        "original_gate_unchanged": True,
        "whole_geometry": {"tokens": 8, "layers": 22, "elements": 256000},
        "calls": len(events),
        "refusals": sum(e["runtime_refusal"] is not None for e in events),
        "source_onehot_rows_retained": sum(
            e.get("source_retained_rows", 256) for e in events
        ),
        "source_exact_float_certificate": False,
        "original_QK_probability_mask_nonlinear_quant_consumers_unchanged": True,
        "native_integer_product_standin": True,
        "target_execution": "NOT RUN",
        "cycles": "UNKNOWN",
        "promotion": "disabled until whole gate and target/cost qualification; failure rejects the one fixed policy, no precision ladder",
        "output_raw_sha256": hashlib.sha256(output.astype("<f4").tobytes()).hexdigest(),
        "token_usage_available": False,
        "pins": {
            str(p): sha(p)
            for p in [
                OUT / "declaration.json",
                OUT / "control.json",
                OUT / "build.json",
                OUT / "calls_live.json",
                OUT / "model.native.ll",
                OUT / "model.o",
                OUT / "model.so",
                OUT / "control.npy",
                OUT / "output.npy",
                bridge,
            ]
        },
    }
    save(OUT / "native_validation.json", record)
    print(
        "FULL_ORIGINAL_GATE",
        json.dumps({k: v for k, v in record.items() if k != "pins"}),
        flush=True,
    )


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    main()
