"""Exact original PV/i8 observer screen; no model strategy or routing installed."""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from merlin.common.paths import runtime_dir
from merlin.llvmlower.balanced_radix_groups import plan_balanced_radix_groups
from merlin.llvmlower.scaled_integer_dot_enclosure import ScaledIntegerDotEffects
from merlin.llvmlower.scaled_integer_observer_codegen import (
    ScaledIntegerObserverPlan,
    emit_scaled_integer_observer,
)

HERE = Path(__file__).resolve().parents[1]
CORE = Path("/scratch/agustin/tmp/merlin-source-integer-pv-main-20261007")
OLD = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
REF = Path("/scratch/agustin/tmp/gemmini-residual-domain-20261007")
OUT = HERE / "out/artifacts/probes/source-integer-pv-native-20261007"
CLANG = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang")
PROBES = OLD / "out/artifacts/probes"
CAP = PROBES / "tiny-attention-original-v-projection-capture-v2-20261007"
LOCAL = PROBES / "dynamic-i8-attention-original-attribution-20261007"
OBS = PROBES / "dynamic-i8-attention-quant-observations-20261007"
PRODUCT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint,
                          ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class Counts(ctypes.Structure):
    _fields_ = [(field, ctypes.c_uint64) for field in
                ("certified", "replayed", "source_pairs", "unadmitted_rows", "product_calls", "readout_words")]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    captures = json.loads((CAP / "qualification.json").read_text())["source_contexts"]
    bindings_path = REF / "docs/perf_records/tiny_attention_projection_binding_audit.json"
    bindings = json.loads(bindings_path.read_text())["contexts"]
    source_quant = ctypes.CDLL(str(OBS / "original_quant_helpers.so"))
    llvm = (OBS / "original_quant_helpers.ll").read_text()
    commands, contexts, pins = [], [], {}
    for path in [Path(__file__), CLANG, CAP / "qualification.json", bindings_path,
                 OBS / "original_quant_helpers.so", OBS / "original_quant_helpers.ll",
                 OLD / "docs/perf_records/tiny_attention_source_integer_strategy.json"]:
        pins[str(path)] = sha(path)
    for i, (capture, binding) in enumerate(zip(captures, bindings, strict=True)):
        directory = OUT / f"context_{i:02d}"
        directory.mkdir()
        assert capture["activation_scale_word"] == binding["activation_scale_word"]
        start = llvm.index(f"define void @attention_quant_{i}(")
        end = llvm.index("\n}", start)
        factors = re.findall(r"fmul float %\d+, f0x([0-9A-F]{8})", llvm[start:end])
        assert len(factors) == 1
        plan = ScaledIntegerObserverPlan(4, 64, 8, 64, 31, capture["activation_scale_word"], int(factors[0], 16),
                plan_balanced_radix_groups(radix_bits=8, lhs_digits=4, rhs_digits=3, reduction_length=8,
                                          accumulator_bits=32, reconstruction_bits=64),
                ScaledIntegerDotEffects(True, True, True), row_factors=(8, 8), output_permutation=(2, 0, 1, 3))
        save(directory / "source_plan.json", asdict(plan))
        source = directory / "executor.c"
        source.write_text("#define MERLIN_SOURCE_BITCAST_COPY __builtin_memcpy\n" + emit_scaled_integer_observer(plan, symbol="integer_observer"))
        shared = directory / ("executor_" + sha(source)[:16] + ".so")
        argv = [str(CLANG), "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC",
                "-I", str(runtime_dir() / "c"), "-MD", "-MF", str(directory / "executor.d"),
                str(source), "-lm", "-o", str(shared)]
        commands.append(argv)
        complete = subprocess.run(argv, capture_output=True, text=True)
        (directory / "compile.stdout").write_text(complete.stdout)
        (directory / "compile.stderr").write_text(complete.stderr)
        assert complete.returncode == 0, complete.stderr
        lib = ctypes.CDLL(str(shared))
        lib.integer_observer_workspace_bytes.restype = ctypes.c_size_t
        lib.integer_observer_workspace_alignment.restype = ctypes.c_size_t
        lib.integer_observer.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t, PRODUCT, ctypes.c_void_p, ctypes.POINTER(Counts)]
        lib.integer_observer.restype = ctypes.c_int
        inputs = {}
        for name, path in {"p": LOCAL / f"context_{i:02d}_family1/a.npy", "code": CAP / f"context_{i:02d}/code.npy",
                "beta": CAP / f"context_{i:02d}/channel_scale.npy", "source_v": CAP / f"context_{i:02d}/v.npy",
                "source_endpoint": LOCAL / f"context_{i:02d}_family1/original.npy", "original_i8": OBS / f"context_{i:02d}/original.npy"}.items():
            pins[str(path)] = sha(path)
            inputs[name] = np.load(path)
        p = np.ascontiguousarray(inputs["p"].reshape(4, 64, 8))
        code = np.ascontiguousarray(inputs["code"].reshape(8, 4, 64).transpose(1, 0, 2))
        beta = np.ascontiguousarray(inputs["beta"].reshape(4, 64))
        for name, array in (("p", p), ("code", code), ("beta", beta)):
            np.save(directory / f"{name}.npy", array)
        size, alignment = lib.integer_observer_workspace_bytes(), lib.integer_observer_workspace_alignment()
        allocation = ctypes.create_string_buffer(size + alignment + 128)
        address = (ctypes.addressof(allocation) + 64 + alignment - 1) // alignment * alignment
        ctypes.memset(allocation, 0xA5, len(allocation))
        before, after = ctypes.string_at(address - 64, 64), ctypes.string_at(address + size, 64)
        output = np.full(16384 + 128, 73, "i1")
        input_words = tuple(array.tobytes() for array in (p, code, beta))
        calls = []

        @PRODUCT
        def product(opaque, diagonal, h, m, k, n, ap, bp, cp):
            a = np.ctypeslib.as_array((ctypes.c_int8 * (h*m*k)).from_address(ap)).reshape(h, m, k)
            b = np.ctypeslib.as_array((ctypes.c_int8 * (h*k*n)).from_address(bp)).reshape(h, k, n)
            c = np.ctypeslib.as_array((ctypes.c_int32 * (h*m*n)).from_address(cp)).reshape(h, m, n)
            c[:] = a.astype("i8") @ b.astype("i8")
            np.save(directory / f"group_{diagonal}_reference.npy", c)
            calls.append({"diagonal": diagonal, "batch": h, "m": m, "k": k, "n": n,
                          "reference_words": h*m*n, "native_exact_integer_standin": True})
            return 1

        counts = Counts()
        assert lib.integer_observer(p.ctypes.data, code.ctypes.data, beta.ctypes.data, output[64:-64].ctypes.data,
                                    address, size, product, None, ctypes.byref(counts)) == 1
        original_observed = np.empty((1, 8, 2048), "i1")
        original = getattr(source_quant, f"attention_quant_{i}")
        original.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        original.restype = None
        original(inputs["source_endpoint"].ctypes.data, original_observed.ctypes.data)
        assert np.array_equal(original_observed, inputs["original_i8"])
        assert np.array_equal(output[64:-64].reshape(1, 8, 2048), original_observed)
        assert np.all(output[:64] == 73) and np.all(output[-64:] == 73)
        assert input_words == tuple(array.tobytes() for array in (p, code, beta))
        assert ctypes.string_at(address - 64, 64) == before and ctypes.string_at(address + size, 64) == after
        record = {"context_for_evidence_only": i, "original_compiled_i8_words_exact": 16384,
                  "workspace_bytes": size, "workspace_alignment": alignment,
                  "counts": {field: getattr(counts, field) for field, _ in counts._fields_}, "calls": calls,
                  "source_plan": str(directory / "source_plan.json"), "source_join": binding,
                  "input_and_dirty_output_workspace_guards": True}
        contexts.append(record)
        np.save(directory / "candidate_i8.npy", output[64:-64].reshape(1, 8, 2048))
        save(directory / "qualification.json", record)
        print("INTEGER_PV_CONTEXT_PASS", i, record["counts"], flush=True)
    for path in [*OUT.rglob("*"), *CORE.glob("src/merlin/llvmlower/*integer*observer*.py"),
                 CORE / "src/merlin/llvmlower/balanced_radix_groups.py"]:
        if path.is_file():
            pins[str(path)] = sha(path)
    # Actual compiler-reported transitive header closure, rather than only local
    # direct includes. This loop reads filenames; it never executes dependency text.
    for depfile in OUT.glob("context_*/executor.d"):
        for name in depfile.read_text().replace("\\\n", " ").split(":", 1)[1].split():
            path = Path(name)
            if path.is_file(): pins[str(path.resolve())] = sha(path)
    save(OUT / "qualification.json", {"schema": "exact_source_integer_pv_all22_native_v1", "status": "pass",
         "source_contexts": contexts, "original_compiled_i8_words_exact": 360448,
         "whole_model_gate": "NOT RUN: no model routing/ABI installed", "target_products": "NOT RUN: native exact integer stand-in",
         "cycle_claim": "UNKNOWN: preparation/certificate/sourcefallback/readout/target costs unmeasured",
         "commands": commands, "ownership": {"representation_enclosure_executor": "Merlin", "target_products_ABI_ISA": "OOT"},
         "token_usage_available": False, "pins": pins})
    print("ALL22_ORIGINAL_COMPILED_I8_PASS", flush=True)


if __name__ == "__main__":
    main()
