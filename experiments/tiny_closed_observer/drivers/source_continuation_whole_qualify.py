"""Independent frozen whole output/Torch gate for normal lazy continuations.

Native device stand-ins remain disclosed. Strict target qualification verifies
every original output word from the actual final ELF; instruction counts are
functional evidence, never hardware cycles.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import source_continuation_whole_build as b
from merlin.llvmlower import quant_hoist
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args

BASE_NATIVE = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host"
)
BUNDLE = Path(
    "/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle"
)
FROZEN = Path("/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005")
CAPTURE = Path(
    "/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006/capture"
)
RAW = "ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3"
GOLD = "3a4d0d1105d4fad150b9ff1b80d4b7838d9fcdb3933a72b71000d6ecc0f5caee"


def now():
    return datetime.now(timezone.utc).isoformat()


def save_progress(path, value):
    # This is a live receipt owned by this one process; freeze at terminal state.
    path.write_text(json.dumps(value, indent=2) + "\n")


def reclose():
    build = json.loads((b.O / "build.json").read_text())
    for path, digest in build["pins"].items():
        assert b.sha(path) == digest, path
    assert build["baseline_byte_exact"] and build["all155devicebindings"]
    assert build["candidate_elf_sha256"] == b.sha(b.O / "target/model.elf")
    return build


def native(*, resume=False):
    build = reclose()
    host = b.O / "host"
    if resume:
        previous = json.loads((b.O / "initial_fixture_refusal.json").read_text())
        assert b.sha(previous["source_path"]) == previous["source_sha256"]
        assert b.sha(host / "model.o") == previous["native_object_sha256"]
        assert b.sha(host / "model.so") == previous["native_SO_sha256"]
    else:
        host.mkdir()
    source = b.O / "native/host_llvm/expanded.native.ll"
    reference, shim = (
        BASE_NATIVE / "reference.c",
        BASE_NATIVE / "device/device_catalog_shim.c",
    )
    runtime = FROZEN / "merlin/runtime/abi/mlir_runtime.c"
    old_gate = json.loads((BASE_NATIVE / "validation.json").read_text())
    assert old_gate["original_compiled_bits_exact"] and old_gate["allclose"]
    argv = [
        str(b.LLVM / "clang"),
        "-O2",
        "-fPIC",
        "-c",
        str(source),
        "-o",
        str(host / "model.o"),
    ]
    link = [
        "cc",
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        str(host / "model.o"),
        str(reference),
        str(shim),
        str(runtime),
        "-lm",
        "-o",
        str(host / "model.so"),
    ]
    record = {
        "schema": "tiny_source_continuation_full_native_v1",
        "status": "compiling",
        "started_utc": now(),
        "selected_llvm_path": str(source),
        "selected_llvm_sha256": b.sha(source),
        "compile_argv": argv,
        "link_argv": link,
        "native_boundary_sources": b.pin([reference, shim, runtime]),
        "scope": build["scope"]
        + " Native original int8/i32 device stand-ins and frozen original runtime; functional only.",
        "torch_golden_path": str(BUNDLE / "golden.npy"),
        "torch_golden_sha256": b.sha(BUNDLE / "golden.npy"),
        "torch_atol": 0.03125,
        "torch_rtol": 0.02,
        "token_usage_available": False,
    }
    receipt = host / ("validation_v2.json" if resume else "validation.json")
    save_progress(receipt, record)
    if not resume:
        subprocess.run(argv, check=True)
        subprocess.run(link, check=True)
    else:
        initial = json.loads((host / "validation.json").read_text())
        assert initial["selected_llvm_sha256"] == b.sha(source)
        assert initial["compile_argv"] == argv and initial["link_argv"] == link
        record["reused_same_compiled_native_image"] = previous
    print("FULL_NATIVE_COMPILED", flush=True)
    library = ctypes.CDLL(str(host / "model.so"))
    binding = json.loads((b.O / "native/host_llvm/source_binding.json").read_text())
    fixture = json.loads((b.TABLES / "generated.json").read_text())
    fixture_factor = fixture["source_bindings"][0]["source_binding"]["routes"][0][
        "quant_factor_bits"
    ]
    routes = binding["source_bindings"]["routes"]
    matching = {
        r["function_body_index"]
        for r in routes
        if r["quant_factor_bits"] == fixture_factor
    }
    assert len(matching) == 1
    indices = sorted({r["function_body_index"] for r in routes})
    guard = binding["whole_helper_guards"][indices.index(next(iter(matching)))]
    original, candidate = (
        getattr(library, guard["source_symbol"]),
        getattr(library, guard["original_symbol"]),
    )
    original.argtypes = candidate.argtypes = [ctypes.c_void_p] * 5
    original.restype = candidate.restype = None
    arrays = [
        np.ascontiguousarray(np.load(CAPTURE / (name + ".npy")))
        for name in ("a", "scale_a", "b", "scale_b")
    ]
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    expected = np.load(CAPTURE / "expected.npy").reshape(-1)
    env = ctypes.CDLL(None)
    old_mode = env.fegetround()
    events = []

    def call(function):
        storage = np.full(45056 + 128, 73, np.int8)
        function(*[a.ctypes.data for a in arrays], storage.ctypes.data + 64)
        assert np.all(storage[:64] == 73) and np.all(storage[-64:] == 73)
        return storage[64:-64].copy()

    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert env.fesetround(mode) == 0
            for preset in (0, 1, 4, 8, 16, 32, 61):
                env.feclearexcept(61)
                env.feraiseexcept(preset)
                a = call(original)
                af = env.fetestexcept(61)
                env.feclearexcept(61)
                env.feraiseexcept(preset)
                c = call(candidate)
                cf = env.fetestexcept(61)
                assert np.array_equal(a, c)
                if mode:
                    assert af == cf
                else:
                    assert np.array_equal(c, expected)
                events.append(
                    {
                        "mode": mode,
                        "preset": preset,
                        "source_flags": af,
                        "candidate_flags": cf,
                        "original45056words_exact": True,
                        "guards": True,
                        "flag_policy": "RNEflags unobserved by typed effect contract; nonRNE exact original continuation flags",
                    }
                )
    finally:
        assert env.fesetround(old_mode) == 0
    assert hashes == [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    b.save(
        host / "normal_helper_fenv.json",
        {
            "fixture_source_contract_path": str(b.TABLES / "generated.json"),
            "fixture_source_contract_sha256": b.sha(b.TABLES / "generated.json"),
            "fixture_quant_factor_bits": fixture_factor,
            "actual_guard_symbols": guard,
            "events": events,
            "inputs_unchanged": True,
            "source_callback_and_nonRNE_fallback_exact": True,
        },
    )
    args = resolve_forward_args(BUNDLE)
    plan = quant_hoist.read_plan(b.B)
    if plan:
        values = quant_hoist.read_values(b.B)
        args.extend(np.ascontiguousarray(values[entry.key]) for entry in plan)
    golden = np.load(BUNDLE / "golden.npy")
    output = np.zeros(golden.shape, golden.dtype, order="C")
    HostModel.load(str(host / "model.so"))(
        [(a.ctypes.data, a.shape) for a in args] + [(output.ctypes.data, output.shape)]
    )
    np.save(host / "output.npy", output)
    baseline = np.load(BASE_NATIVE / "output.npy")
    raw_sha = hashlib.sha256(output.astype("<f4").tobytes()).hexdigest()
    exact = bool(np.array_equal(output.view("u4"), baseline.view("u4")))
    close = bool(np.allclose(output, golden, atol=0.03125, rtol=0.02))
    record.update(
        finished_utc=now(),
        status="pass" if exact and close else "fail",
        outputs=output.size,
        shape=list(output.shape),
        dtype=str(output.dtype),
        allclose=close,
        original_compiled_bits_exact=exact,
        raw_output_sha256=raw_sha,
        reference_path=str(host / "output.npy"),
        reference_sha256=b.sha(host / "output.npy"),
        max_abs_error=float(np.max(np.abs(output - golden))),
        finite=bool(np.isfinite(output).all()),
        model_object_sha256=b.sha(host / "model.o"),
        shared_object_sha256=b.sha(host / "model.so"),
        normal_helper_fenv_path=str(host / "normal_helper_fenv.json"),
        normal_helper_fenv_sha256=b.sha(host / "normal_helper_fenv.json"),
    )
    save_progress(receipt, record)
    assert (
        output.shape == (1, 8, 32000)
        and output.dtype == np.float32
        and output.size == 256000
    )
    assert b.sha(BUNDLE / "golden.npy") == GOLD and exact and close and raw_sha == RAW
    print("FULL_ORIGINAL_NATIVE_PASS", output.size, raw_sha, flush=True)


def target():
    build = reclose()
    case = b.O / "target"
    native_path = b.O / "host/validation_v2.json"
    native_gate = json.loads(native_path.read_text())
    assert (
        native_gate["status"] == "pass"
        and native_gate["original_compiled_bits_exact"]
        and native_gate["allclose"]
    )
    elf = case / "model.elf"
    assert b.sha(elf) == build["candidate_elf_sha256"]
    audit = b.audit_elf(elf.read_bytes())
    assert audit["status"] == "pass"
    command = [
        str(b.GCC.with_name("spike")),
        "-g",
        "--extension=gemmini",
        "--isa=rv64gc",
        "-m0x80000000:0x400000000",
        str(elf),
    ]
    receipt = {
        "schema": "tiny_source_continuation_full_target_v1",
        "status": "running",
        "started_utc": now(),
        "argv": command,
        "elf_path": str(elf),
        "elf_sha256": b.sha(elf),
        "marker": build["inherited_marker"],
        "marker_contract": build["marker_contract"],
        "reference_path": str(b.O / "host/output.npy"),
        "reference_sha256": b.sha(b.O / "host/output.npy"),
        "torch_golden_path": str(BUNDLE / "golden.npy"),
        "torch_golden_sha256": b.sha(BUNDLE / "golden.npy"),
        "torch_atol": 0.03125,
        "torch_rtol": 0.02,
        "torch_allclose": True,
        "outputs": 256000,
        "original_compiled_raw_sha256": RAW,
        "nofsm_audit": audit,
        "normal_lower_recipe_path": str(case / "host_llvm/source_binding.json"),
        "normal_lower_recipe_sha256": b.sha(case / "host_llvm/source_binding.json"),
        "controlled_link_path": str(b.O / "build.json"),
        "controlled_link_sha256": b.sha(b.O / "build.json"),
        "native_qualification_path": str(native_path),
        "native_qualification_sha256": b.sha(native_path),
        "before_verified_stock_job": 2004,
        "before_verified_stock_cycles": 422018733,
        "actual_candidate_hardware_cycles": None,
        "hardware_status": "not submitted",
        "scope": build["scope"],
        "timing_scope": "Strict Spike cycles are retired instructions, not stock hardware cycles;2053 capsule gain is not a whole prediction.",
        "histogram_scope": "Same final ELF PC execution counts collected during the required numeric run. No chronological addresses or physical memory/cross-call latency are inferred.",
        "token_usage_available": False,
    }
    path = case / "spike_validation.json"
    assert not path.exists()
    save_progress(path, receipt)
    with (case / "spike.log").open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        receipt["spike_pid"] = process.pid
        save_progress(path, receipt)
        try:
            code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=20)
            receipt.update(status="timeout", finished_utc=now())
            save_progress(path, receipt)
            raise
    console = (case / "spike.log").read_text().replace("\r", "")
    metrics = {
        line.split()[1]: line.split()[2]
        for line in console.splitlines()
        if line.startswith("METRIC ")
    }
    passed = (
        code == 0
        and "DONE" in console
        and metrics.get("memref_rank_mismatch") == "0"
        and metrics.get("build_hash") == build["inherited_marker"]
        and console.count("OUT_SHA256 f32le 256000 1024000 " + RAW) == 1
    )
    receipt.update(
        status="pass" if passed else "fail",
        finished_utc=now(),
        exit_code=code,
        metrics=metrics,
        spike_console_path=str(case / "spike.log"),
        spike_console_sha256=b.sha(case / "spike.log"),
        functional_instructions_not_hardware_cycles=int(metrics.get("cycles", "0")),
        spike_full_output_match=passed,
        spike_output_sha256=RAW if passed else None,
    )
    assert b.sha(elf) == receipt["elf_sha256"]
    save_progress(path, receipt)
    assert passed
    adapter = {
        k: receipt[k]
        for k in (
            "reference_path",
            "reference_sha256",
            "torch_golden_path",
            "torch_golden_sha256",
            "torch_atol",
            "torch_rtol",
            "torch_allclose",
            "spike_console_path",
            "spike_console_sha256",
            "spike_output_sha256",
            "spike_full_output_match",
            "elf_sha256",
            "normal_lower_recipe_path",
            "normal_lower_recipe_sha256",
            "controlled_link_path",
            "controlled_link_sha256",
        )
    }
    adapter.update(
        schema="reference_validation_v1",
        all_original_compiled_words_exact=True,
        original_reference_elements=256000,
        original_reference_shape=[1, 8, 32000],
        source_qualification_path=str(path),
        source_qualification_sha256=b.sha(path),
        inherited_marker=receipt["marker"],
        marker_contract=receipt["marker_contract"],
        scope=receipt["scope"],
    )
    b.save(case / "reference_validation.json", adapter)
    print(
        "FULL_ORIGINAL_STRICT_PASS",
        receipt["elf_sha256"],
        receipt["functional_instructions_not_hardware_cycles"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("native", "native-resume", "target"))
    args = parser.parse_args()
    if args.action == "target":
        target()
    else:
        native(resume=args.action == "native-resume")
