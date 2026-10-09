"""Whole original-gate screen; native integer emulation, not target timing."""
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from merlin.llvmlower.abi import HostModel, make_descriptor
from merlin.runtime.dispatch_runtime import resolve_forward_args

w = Path(__file__).resolve().parent
out = w / "bounded_native"
run_out = Path(os.environ.get("ENDPOINT_SCREEN_OUTPUT", str(out)))
run_out.mkdir(parents=True, exist_ok=True)
steps = int(os.environ.get("ENDPOINT_SCREEN_STEPS", "1"))
assert steps in (0,1)
check_upstream = os.environ.get("ENDPOINT_SCREEN_VERIFY_UPSTREAM") == "1"
reuse_library = os.environ.get("ENDPOINT_SCREEN_REUSE_LIBRARY") == "1"
root = Path("/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005")
provider = root / "host_outlined/model.so"
sha = lambda p: hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest()
installation = json.loads((out / "installation.json").read_text())
numeric_witness = installation["native_numeric_witness"]
native_path = Path("/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/native_evaluator.py")
assert sha(native_path) == numeric_witness["native_evaluator_sha256"]
assert sha(native_path.parent / "screen_bounded.c") == numeric_witness["numeric_source_sha256"]
header = Path("/scratch/agustin/tmp/merlin-closed-bf16-certificate-20261005/merlin/runtime/c/f32_interval_endpoint.h")
runtime_witness = dict(numeric_witness, interval_header_sha256=sha(header))
# The host IR calls a registered bridge and contains no interval implementation.
# Keep initial compiler metadata, pin the actual registered code separately. The
# fixed header narrows signed-zero eligibility; it changes no source arithmetic,
# ABI or admitted endpoint policy. This is a native functional screen only.
runtime_witness["superseded_compiler_header_sha256"] = numeric_witness["interval_header_sha256"]
runtime_witness["qualification_scope"] = "Actual registered native evaluator; target provider remains unavailable"
runtime_witness["actual_max_bf16_steps"] = steps
(run_out / "registered_numeric_witness.json").write_text(json.dumps(runtime_witness, indent=2) + "\n")
spec = importlib.util.spec_from_file_location("native_endpoint_evaluator", native_path)
numeric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numeric)
assert numeric.SOURCE_CONTRACT == numeric_witness["source_contract"]
evaluator = numeric.NativeEndpointEvaluator(numeric.SOURCE_CONTRACT, max_bf16_steps=steps)

original = json.loads((root / "build_outlined/device_signatures.json").read_text())
selected = json.loads((out / "build/device_signatures.json").read_text())
assert {r["symbol"]: r["dtypes"] for r in original["routed"]} == {r["symbol"]: r["dtypes"] for r in selected["routed"]}
assert sha(provider) == "a3dda3fccafc08aff62e749223eda9a839bec3e2cd9a8fd91d926f4c65d9198e"
clang = "/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang"
if reuse_library:
    original_run = json.loads((out / "native_validation.json").read_text())
    assert sha(out / "native_model.o") == original_run["model_object_sha256"]
    assert sha(out / "native.so") == original_run["library_sha256"]
    assert sha(out / "build/lower/model.ll") == original_run["model_llvm_sha256"]
else:
    subprocess.run([clang, "-O2", "-march=native", "-ffp-contract=off", "-fPIC", "-c",
        str(out / "build/lower/model.ll"), "-o", str(out / "native_model.o")], check=True)
    subprocess.run(["cc", "-O2", "-fPIC", "-c", str(out / "native_bridge.c"),
        "-o", str(out / "native_bridge.o")], check=True)
    subprocess.run(["cc", "-shared", str(out / "native_model.o"), str(out / "native_bridge.o"),
        str(w / "native_wrapper.o"), "-Wl,--wrap=expf", str(provider), "-lm", "-o", str(out / "native.so")], check=True)
model = HostModel.load(str(out / "native.so"))
source_fallback = HostModel.load(str(w / "independent_group/group.so"), name="source_group_exact")
assert sha(w / "independent_group/group.so") == "dabc2a877b69aa3de5f21f2de11d111fc9658f570e79742f7997901bfa7afab9"
desc = make_descriptor(4)
read_shapes = [(1,12,256,64),(1,12,512,64),(1,1,256,512),
    (1,12,192,64),(1,12,192,64),(1,12,128,64),(1,12,512,64),(1,1,256,512),
    (1,12,192,64),(1,12,192,64),(1,12,128,64),(1,12,256,64)]
call_records = []
callback_errors = []

def view(pointer, index):
    descriptor = C.cast(pointer, C.POINTER(desc)).contents
    shape, strides = tuple(descriptor.sizes), tuple(descriptor.strides)
    if shape != read_shapes[index] or any(stride <= 0 for stride in strides) or descriptor.offset < 0:
        raise ValueError("source-bound descriptor dimensions/strides changed")
    dtype = np.dtype(np.uint8 if index in (2,7) else np.uint16)
    span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    byte_count = span * dtype.itemsize
    backing = (C.c_uint8 * byte_count).from_address(descriptor.aligned + descriptor.offset * dtype.itemsize)
    array = np.ndarray(shape, dtype=dtype, buffer=backing,
        strides=tuple(stride * dtype.itemsize for stride in strides))
    if index != 11:
        array.flags.writeable = False
    return array, dict(shape=shape, offset=descriptor.offset, strides=strides, dtype=str(dtype))

callback_type = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_void_p))

@callback_type
def callback(pointers):
    index = len(call_records)
    started = time.monotonic()
    try:
        values = [view(pointers[i], i) for i in range(12)]
        inputs, destination = [value[0] for value in values[:11]], values[11][0]
        # Dirty logical output verifies fresh full-writing semantics; padding is untouched.
        destination.fill(0xa55a)
        try:
            result, counts = evaluator.evaluate(inputs)
            original_source_fallback = None
        except Exception as error:
            copies = [np.ascontiguousarray(value) for value in inputs]
            result = np.empty(read_shapes[11], np.uint16)
            source_fallback([(value.ctypes.data, value.shape) for value in copies] + [(result.ctypes.data,result.shape)])
            counts = []
            original_source_fallback = repr(error)
        if result.dtype != np.uint16 or result.shape != destination.shape:
            raise ValueError("numeric evaluator does not fully define the typed source endpoint")
        np.copyto(destination, result, casting="no")
        np.save(run_out / ("group_" + str(index) + "_output.npy"), result)
        record = dict(index=index, descriptors=[value[1] for value in values],
            input_raw_sha256=[hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for value in inputs],
            output_raw_sha256=hashlib.sha256(result.tobytes()).hexdigest(), heads=counts,
            original_source_fallback=original_source_fallback,
            native_wall_seconds=time.monotonic()-started)
        if check_upstream:
            copies = [np.ascontiguousarray(value) for value in inputs]
            upstream = np.empty(read_shapes[11], np.uint16)
            source_fallback([(value.ctypes.data,value.shape) for value in copies] + [(upstream.ctypes.data,upstream.shape)])
            record["compiled_original_endpoint_mismatches"] = int(np.count_nonzero(result != upstream))
            record["compiled_original_endpoint_raw_sha256"] = hashlib.sha256(upstream.tobytes()).hexdigest()
            if record["compiled_original_endpoint_mismatches"]:
                np.save(run_out / ("group_" + str(index) + "_upstream.npy"), upstream)
                for i,value in enumerate(copies):
                    np.save(run_out / ("group_" + str(index) + "_input_" + str(i) + ".npy"),value)
        if index == 0:
            accepted = json.loads((w / "accepted_taps.json").read_text())
            expected = [r["raw_sha256"] for r in accepted["records"] if r["role"] == "input"]
            # Reference is read only after the result, for independent observation.
            record["original_live_inputs_equal"] = record["input_raw_sha256"] == expected
            endpoint = np.load(w / "endpoint.npy")
            record["group0_changed_words"] = int(np.count_nonzero(result != endpoint))
        call_records.append(record)
        (run_out / "runtime_group_calls.json").write_text(json.dumps(call_records, indent=2) + "\n")
        print("NATIVE_GROUP", index, "seconds", round(record["native_wall_seconds"],3),
            "source_fallback", original_source_fallback, flush=True)
        return 0
    except BaseException as error:
        callback_errors.append(repr(error))
        (run_out / "callback_error.json").write_text(json.dumps(callback_errors))
        print("CALLBACK_FAILURE", repr(error), flush=True)
        return 1

model.lib.native_endpoint_register.argtypes = [callback_type]
model.lib.native_endpoint_register(callback)
arguments = resolve_forward_args(root / "bundle")
golden = np.load(root / "bundle/golden.npy")
result = np.full_like(golden, np.nan)
start = time.monotonic()
model([(value.ctypes.data,value.shape) for value in arguments] + [(result.ctypes.data,result.shape)])
np.save(run_out / "native_output.npy", result)
receipt = dict(schema="closed_bf16_group_whole_native_policy_v1",
    scope="Full original source48groups/384contractions; native exact-integer NumPy products + C certificate/sourcefallback. No target coverage, hardware cycles or performance projection.",
    numerical_policy=("exact_endpoint_control" if steps == 0 else "fixed maximum1adjacentBF16endpointword; wider/special/unsupported use original source fallback"),
    actual_max_bf16_steps=steps, independent_compiled_original_endpoint_checks=check_upstream,
    original_endpoint_mismatch_groups=sum(r.get("compiled_original_endpoint_mismatches",0)>0 for r in call_records),
    original_atol=.03125, original_rtol=.02, elements=int(result.size),
    failed_count=int(np.count_nonzero(~np.isclose(result,golden,atol=.03125,rtol=.02))),
    allclose=bool(np.allclose(result,golden,atol=.03125,rtol=.02)), finite=bool(np.isfinite(result).all()),
    bitwise_mismatches=int(np.count_nonzero(result.view("u4")!=golden.view("u4"))),
    maxabs=float(np.max(np.abs(result-golden))),
    actual_runtime_calls=len(call_records), expected_source_groups=installation["source_groups"],
    full_group_source_fallback_calls=sum(r["original_source_fallback"] is not None for r in call_records),
    callback_errors=callback_errors, native_wall_seconds=time.monotonic()-start,
    model_llvm_sha256=sha(out / "build/lower/model.ll"), model_object_sha256=sha(out / "native_model.o"),
    library_sha256=sha(out / "native.so"), original_runtime_integer_provider_sha256=sha(provider),
    bridge_source_sha256=sha(out / "native_bridge.c"), installation_sha256=sha(out / "installation.json"),
    evaluator_source_sha256=sha(native_path), evaluator_shared_sha256=evaluator.shared_sha256,
    registered_numeric_witness_sha256=sha(run_out / "registered_numeric_witness.json"),
    actual_interval_header_sha256=sha(header),
    superseded_compiler_interval_header_sha256=numeric_witness["interval_header_sha256"],
    source_fallback_library_sha256=sha(w / "independent_group/group.so"),
    original_golden_sha256=sha(root / "bundle/golden.npy"), output_sha256=sha(run_out / "native_output.npy"),
    actual_call_record_sha256=sha(run_out / "runtime_group_calls.json"), token_usage_available=False)
(run_out / "native_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
assert len(call_records) == 48 and not callback_errors
assert call_records[0]["original_live_inputs_equal"]
assert receipt["allclose"] and receipt["finite"]
print("ACCEPTED_WHOLE_BOUNDED_ENDPOINT", flush=True)
