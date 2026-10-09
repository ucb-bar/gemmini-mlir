"""Close already completed frozen artifacts; do not rebuild or rerun gates."""
from pathlib import Path
import hashlib
import json
import shutil

from mlir_oot.no_fsm_audit import audit_elf

root = Path.cwd()
work = root / "out/artifacts/probes/smol-absolute-values-20261006"
target = work / "absolute"
proof = work / "representation_proof"
native = work / "native_frozen"
records = root / "docs/perf_records"
snapshot = records / "source_snapshots/smol_standard_absolute_complete_group"
snapshot.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


pins = {}


def add_pin(path, expected=None):
    path = str(Path(path).resolve())
    digest = sha(path)
    if expected is not None:
        assert digest == expected, path
    if path in pins:
        assert pins[path] == digest, path
    pins[path] = digest


build = read(target / "build.json")
compile_receipt = read(target / "compile.json")
default = read(work / "default_control/compile.json")
representation = read(proof / "build.json")
spike = read(target / "spike_receipt.json")
manifest = read(native / "manifest.json")
journey = read(records / "smol_numeric_absolute_full48_journey.json")
validation = read(records / "smol_numeric_absolute_full48_validation.json")
observations = read(records / "smol_numeric_absolute_full48_observations.json")

assert default["default_object_byte_exact"] is True
control = Path(build["control_build"]).parent
add_pin(control / "build.json", build["control_build_sha256"])
assert sha(work / "default_control/provider.o") == sha(control / "provider.o")
assert spike["status"] == "pass" and spike["returncode"] == 0
add_pin(target / "model.elf", spike["elf_sha256"])
add_pin(target / "spike.log", spike["log_sha256"])
log = (target / "spike.log").read_text()
counter = int(next(line.split()[1] for line in log.splitlines()
                   if line.startswith("WORKSPACE_GROUP_CYCLES ")))
assert counter == 3370349620
assert "WORKSPACE_GROUP ALL196608 AND GUARDS PASS" in log
statistics = lambda text: [line for line in text.splitlines()
                           if line.startswith("WORKSPACE_STAT ")]
assert statistics(log) == statistics((control / "spike.log").read_text())
assert (proof / "spike.log").read_text().strip() == "ABSOLUTE PASS 565925"
for receipt in (build, compile_receipt, default, representation):
    for path, digest in receipt["pins"].items():
        add_pin(path, digest)
for path, digest in manifest["local_pins"].items():
    add_pin(native / path, digest)
for path, digest in manifest["transitive_compile_dependencies"].items():
    add_pin(path, digest)
add_pin(manifest["compile_commands"][0][0], manifest["compiler_sha256"])
add_pin(native / "manifest.json", journey["frozen_manifest_sha256"])
for path, digest in journey["independently_rehashed_frozen_pins"].items():
    add_pin(path, digest)

source_build = read(control / "build.json")
unchanged = lambda command, provider: [arg for arg in command
    if arg.endswith(".o") and arg != str(provider)]
assert unchanged(build["link"], target / "provider.o") == unchanged(
    source_build["link"], control / "provider.o")
audits = {}
for name, elf in (("candidate", target / "model.elf"),
                  ("representation", proof / "absolute.elf")):
    audits[name] = audit_elf(elf.read_bytes())
    assert audits[name]["status"] == "pass"

assert validation["bitwise_mismatches"] == 0 and validation["allclose"] is True
assert validation["elements"] == 1600
assert validation["actual_runtime_calls"] == 48
assert validation["full_group_source_fallback_calls"] == 0
assert validation["original_atol"] == 0.03125 and validation["original_rtol"] == 0.02
assert validation["evaluator_shared_sha256"] == sha(native / "provider.so")
gate = journey["gate"]
assert gate["original_input_equal_groups"] == 48
assert gate["compiled_source_integer_words"] == 9437184
assert gate["compiled_source_scale_words"] == 12288
assert gate["integer_mismatches"] == 0 and gate["bf16_scale_mismatches"] == 0
assert observations["schema"] == "complete_source_quant_observation_pair_v1"

files = {
    "archive.py": work / "archive.py",
    "prepare_absolute_values.py": root / "out/artifacts/probes/smol-classification-20261006/prepare_absolute_values.py",
    "prepare_native.py": work / "prepare_native.py",
    "target_build.json": target / "build.json",
    "target_compile.json": target / "compile.json",
    "target_spike_receipt.json": target / "spike_receipt.json",
    "target_spike.log": target / "spike.log",
    "default_compile.json": work / "default_control/compile.json",
    "representation_build.json": proof / "build.json",
    "representation.c": proof / "absolute.c",
    "representation.log": proof / "spike.log",
    "original_fixture.py": proof / "original_fixture_test_source_numeric_capability.py",
    "numeric_capability.h": target / "numeric_capability.h",
}
for name, path in files.items():
    add_pin(path)
    shutil.copy2(path, snapshot / name)
    add_pin(snapshot / name)
for name in ("smol_numeric_absolute_full48_validation.json",
             "smol_numeric_absolute_full48_observations.json",
             "smol_numeric_absolute_full48_journey.json"):
    add_pin(records / name)
add_pin(control / "spike.log")
add_pin(control / "spike_receipt.json")

receipt = {
    "schema": "standard_absolute_values_production_attention_v1",
    "scope": "Complete original12-head/256-query/1024-key attention group. Functional Spike retired instructions; no FireSim cycles or whole-model speedup prediction.",
    "core_commit": "c0624c4e1",
    "optimization": "Explicit standard F32/F64 absolute-value builtins under independent source obligations. Generic capability and arithmetic in Merlin; actual target ISA proof, xDSL products, resources and ranked ABI in OOT.",
    "default_object_byte_exact": True,
    "default_control_object_sha256": sha(control / "provider.o"),
    "control_instructions": build["control_instructions"],
    "candidate_instructions": counter,
    "reduction_percent": 100 * (build["control_instructions"] - counter) / build["control_instructions"],
    "unchanged_objects": "All15 xDSL product kernels, source/input data, driver, bridge, startup and linker remain unchanged. Only numeric provider.o changes.",
    "unchanged_stats": statistics(log),
    "complete_original_accepted_carriers_and_guards": 196608,
    "source_replay_fmas": 4461440,
    "product_calls": 480,
    "logical_readback_bytes": 86507520,
    "workspace_bytes": 121963584,
    "numeric_contract": build["contract"],
    "independent_target_representation_checks": 565925,
    "target_rounding_modes": 5,
    "nan_payload_policy": "Source payload observation explicitly waived; actual qualified platform additionally retains tested payload bits.",
    "candidate_elf_sha256": spike["elf_sha256"],
    "native_so_sha256": sha(native / "provider.so"),
    "full48_native": {
        "elements": 1600, "bitwise_mismatches": 0, "calls": 48,
        "source_fallback_calls": 0, "original_input_equal_groups": 48,
        "compiled_source_integer_words": 9437184,
        "compiled_source_scale_words": 12288,
        "consumer_mismatches": 0,
        "changed_unobserved_bf16_endpoint_words": 6007,
        "gate_atol": 0.03125, "gate_rtol": 0.02,
        "scope": "Frozen numeric executor plus original compiled consumers. Separate ordinary compiled physical wrapper/fallback and whole target qualifications required.",
    },
    "nofsm": audits,
    "pins": pins,
    "source_accuracy_gate_changed": False,
    "default_policy_changed": False,
    "hardware_admitted": False,
    "token_usage_available": False,
}
(records / "smol_standard_absolute_complete_group.json").write_text(
    json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"status": "closed", "pins": len(pins),
                  "instructions": counter, "reduction_percent": receipt["reduction_percent"],
                  "full48_original": "pass", "hardware": "not_admitted"}))
