"""Single exact consumer-observation whole screen; native product stand-in only."""
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from merlin.llvmlower.abi import HostModel, make_descriptor
from merlin.runtime.dispatch_runtime import resolve_forward_args

w = Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005')
out = w / "bounded_native"
run_out = Path(os.environ.get("ENDPOINT_SCREEN_OUTPUT", '/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005/numeric_absolute_native_screen'))
run_out.mkdir(parents=True, exist_ok=True)
steps = None
check_upstream = os.environ.get("ENDPOINT_SCREEN_VERIFY_UPSTREAM") == "1"
reuse_library = True
root = Path("/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005")
provider = root / "host_outlined/model.so"
sha = lambda p: hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest()
installation = json.loads((out / "installation.json").read_text())
numeric_witness = installation["native_numeric_witness"]
native_dir=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/smol-absolute-values-20261006/native_frozen')
native_path=native_dir/'workspace_frontier_adapter.py'
frozen_path=native_dir/'manifest.json'
frozen_receipt=json.loads(frozen_path.read_text())
local_dependencies={str(native_dir/path):pin for path,pin in frozen_receipt['local_pins'].items()}
local_dependencies.update(frozen_receipt['transitive_compile_dependencies'])
local_dependencies[str(frozen_path)]=sha(frozen_path)
local_dependencies[frozen_receipt['compile'][0]]=frozen_receipt['compiler_sha256']
assert all(sha(Path(path))==pin for path,pin in local_dependencies.items())
proof_path=Path("/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/frontier_compiled_oracle.json")
proof=json.loads(proof_path.read_text())
assert proof["candidate_i8_mismatch"] == proof["candidate_scale_mismatch"] == proof["numpy_source_formula_vs_compiled_mismatch"] == 0
assert sha(w / "quant_frontier/source_quant_frontier.mlir") == proof["source_frontier_sha256"]
assert sha(w / "quant_frontier/frontier.so") == proof["source_library_sha256"]
closure_path=w / "quant_frontier/full_source_closure_semantic.json"
closure=json.loads(closure_path.read_text())
assert closure["source_group_calls"] == closure["covered_source_groups"] == 48
assert closure["quantized_consumer_frontiers"] == 12 and closure["unquantized_bf16_escapes"] == 0
expected_consumer="8d3422c82de7222db754860ac49f3940915d3fddfca349c8122964a92389c4f4"
assert {r["consumer_semantic_sha256"] for r in closure["records"]} == {expected_consumer}
for record in closure["records"]:
    assert record["observed_types"] == ["tensor<1x1024x768xi8>","tensor<1x1024xbf16>"]
    assert record["assembly_coordinates"] == [dict(source_ordinal=i,offsets=[0,0,i*256,0],sizes=[1,12,256,64]) for i in range(4)]
header=native_dir/'f32_interval_endpoint.h'
spec=importlib.util.spec_from_file_location('native_workspace_frontier_evaluator',native_path)
numeric=importlib.util.module_from_spec(spec);spec.loader.exec_module(numeric)
SOURCE_CONTRACT=frozen_receipt['source_contract']
QUANT_CONTRACT=frozen_receipt['quant_contract']
assert SOURCE_CONTRACT==numeric_witness['source_contract']
evaluator=numeric.NativeWorkspaceFrontierEvaluator(SOURCE_CONTRACT,QUANT_CONTRACT,directory=native_dir)
evaluator.shared_sha256=sha(native_dir/'provider.so')
assert evaluator.size==121963584 and evaluator.shared_sha256=='a29579875927c720053402d2c28213a19102687e2133ce3bb49b1350eedec53a'
runtime_witness=dict(source_contract=SOURCE_CONTRACT,quant_contract=QUANT_CONTRACT,
    numeric_dependency_pins=local_dependencies,native_shared_sha256=evaluator.shared_sha256,
    complete_compile_manifest_sha256=sha(frozen_path),workspace_bytes=evaluator.size,
    workspace_alignment=evaluator.alignment,workspace_effect_contract=frozen_receipt['effect_contract'],
    compiled_source_quant_proof_sha256=sha(proof_path),full_live_consumer_closure_sha256=sha(closure_path),
    complete_consumer_semantic_sha256=expected_consumer,
    numerical_policy='exact source integer words and escaping BF16scale; selectively refine ambiguous observations',
    complete_source_dag_semantic_sha256=numeric_witness['source_semantic_sha256'],
    superseded_compiler_numeric_witness=numeric_witness,
    qualification_scope='Registered native caller-workspace portable executor with exact integer-product stand-in; unchanged borrowed BF16 ABI and original typed quant consumer. No target products/cycles or normal target provider claimed.')
(run_out/'registered_numeric_witness.json').write_text(json.dumps(runtime_witness,indent=2)+'\n')

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
            output_raw_sha256=hashlib.sha256(result.tobytes()).hexdigest(), numeric_counts=counts,
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
assert all(sha(Path(path)) == pin for path,pin in runtime_witness["numeric_dependency_pins"].items())
np.save(run_out / "native_output.npy", result)
receipt = dict(schema="closed_bf16_group_whole_native_workspace_quant_frontier_policy_v1",
    scope="Full original48groups/384contractions; native caller-workspace C executor with radix3 integer-product stand-in, original scalar source DAG and exact observed rowquant i8+BF16scale certificate/selective source replay. No target coverage/cycles claim.",
    numerical_policy="exact_source_quant_consumer_observations",
    replay_source_fmas=sum(record["numeric_counts"].get("replay_total_fmas",0) for record in call_records),
    actual_max_bf16_steps=steps, independent_compiled_original_endpoint_checks=check_upstream,
    original_endpoint_mismatch_groups=(sum(r.get("compiled_original_endpoint_mismatches",0)>0 for r in call_records) if check_upstream else None),
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
assert receipt["bitwise_mismatches"] == 0
print("ACCEPTED_WHOLE_WORKSPACE_QUANT_FRONTIER_ENDPOINT", flush=True)
