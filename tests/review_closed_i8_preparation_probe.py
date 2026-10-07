"""Independently review the closed integer observation before cycle admission."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            state.update(chunk)
    return state.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def review(args):
    check(not args.output.exists(), "review output must be fresh")
    packet = json.loads(args.packet.read_text())
    for path, expected in packet["pins"].items():
        check(digest(path) == expected, "changed preparation artifact: " + path)
    core = Path(packet["generic_core_checkout"])
    actual_head = subprocess.check_output(
        ["git", "-C", str(core), "rev-parse", "HEAD"], text=True
    ).strip()
    check(actual_head == packet["generic_core_commit"], "generic source head differs")
    changed = subprocess.check_output(
        ["git", "-C", str(core), "diff", "--name-only", packet["dependency_base"], actual_head],
        text=True,
    ).splitlines()
    check(set(changed) == {
        "src/merlin/llvmlower/AGENT.md",
        "src/merlin/llvmlower/source_expression_interval.py",
        "src/merlin/llvmlower/source_expression_interval_llvm.py",
        "merlin/tests/ir/test_source_expression_interval_i8_llvm.py",
    }, "unexpected generic topic changes")
    check(re.search(r"^50 passed in [\d.]+s$", args.tests.read_text(), re.M),
          "independent source/native/refusal tests did not pass")
    probe = args.packet.parent.parent.parent / "out/artifacts/probes"
    native = probe / "closed-i8-interval-normal-2062-20261007"
    contexts_path = native / "actual22contexts/qualification.json"
    contexts = json.loads(contexts_path.read_text())
    check(contexts["status"] == "pass" and len(contexts["contexts"]) == 22,
          "actual source context coverage differs")
    for index, context in enumerate(contexts["contexts"]):
        check(context["context"] == index and context["original45056i8_exact"],
              "actual source observer words differ")
        cases = context["all4modes7sticky"]
        check({(c["mode"], c["preset"]) for c in cases} == {
            (mode, preset) for mode in (0, 1024, 2048, 3072)
            for preset in (0, 1, 4, 8, 16, 32, 61)
        } and len(cases) == 28, "native environment coverage differs")
        check(all(c["source_flags"] == c["candidate_flags"] for c in cases if c["mode"]),
              "unsupported-mode original effects differ")
        check(context["readonly_input_alias_case"] and context["inputs_unchanged"]
              and not context["destination_alias_permission"], "alias ownership changed")
    old_release = json.loads(args.current_release.read_text())
    adapter_path = Path(old_release["standard_adapter"])
    adapter = json.loads(adapter_path.read_text())
    reference_path = Path(adapter["reference_path"])
    torch_path = Path(adapter["torch_golden_path"])
    check(digest(reference_path) == adapter["reference_sha256"]
          and digest(torch_path) == adapter["torch_golden_sha256"], "original reference changed")
    reference, torch = np.load(reference_path), np.load(torch_path)
    check(reference.dtype == np.float32 and reference.shape == (1, 8, 32000)
          and torch.shape == reference.shape, "original whole-model type or extent differs")
    raw_sha = hashlib.sha256(reference.astype("<f4", copy=False).tobytes()).hexdigest()
    check(raw_sha == "ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3",
          "original whole reference words differ")
    for arm in ("control", "selected"):
        actual = np.load(native / arm / "native/host/output.npy")
        check(actual.dtype == reference.dtype and actual.shape == reference.shape
              and actual.tobytes() == reference.tobytes(), "whole native words differ: " + arm)
        check(np.allclose(actual, torch, atol=.03125, rtol=.02),
              "original Torch whole gate failed: " + arm)
    champion_model = adapter_path.parent / "model.o"
    check(digest(native / "control/target/model.o") == digest(champion_model),
          "normal control does not reproduce stock2062 object")
    functional = probe / "closed-i8-interval-result-20261007/target_v3"
    console = (functional / "spike.stdout").read_text()
    check(re.findall(r"^INTEGER_RESULT_SOURCE_GATE modes=5 presets=7 words=45056 flags guards PASS$",
                     console, re.M) == ["INTEGER_RESULT_SOURCE_GATE modes=5 presets=7 words=45056 flags guards PASS"],
          "complete target floating-environment gate missing")
    rows = re.findall(r"^INTEGER_RESULT_ROW id=(\d+) sample=(\d+) cycles=(\d+) instructions=(\d+) digest=([a-f0-9]+)$",
                      console, re.M)
    check([(int(r[0]), int(r[1])) for r in rows] == [(0, 0), (1, 1), (1, 2), (0, 3)]
          and len({r[4] for r in rows}) == 1, "matched complete target sample contract differs")
    check(console.count("INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0") == 1,
          "complete output/guard/input contract missing")
    audit = audit_elf((functional / "build/layer.elf").read_bytes())
    check(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
          "target executable zeroFSM audit failed")
    means = [float(np.mean([int(r[3]) for r in rows if int(r[0]) == arm])) for arm in (0, 1)]
    result = {
        "schema": "root_closed_integer_observer_preparation_review_v1",
        "status": "SOURCE_NATIVE_AND_COMPLETE_TARGET_NUMERICS_VERIFIED_CYCLE_ADMISSION_PENDING",
        "pins_reclosed": len(packet["pins"]), "independent_tests": 50,
        "actual_contexts": 22, "actual_original_integer_words": 22 * 45056,
        "original_whole_native_words": reference.size, "original_raw_sha256": raw_sha,
        "original_Torch_gate": {"atol": .03125, "rtol": .02, "passed": True},
        "normal_control_object_exact_stock2062": True, "final_target_noFSM": audit,
        "functional_instruction_means": means,
        "instruction_fraction_reduction": 1 - means[1] / means[0],
        "scope": "Direct publication of the existing certified integer observation; source/table/cold continuation unchanged. Float escapes and unsupported environments retain source refusal. Complete native whole evidence is not target whole evidence.",
        "candidate_whole_ELF_reviewed": False, "whole_target_reviewed": False,
        "stock_release": False, "whole_cycle_prediction": "UNKNOWN",
        "upstream": "Unpublished four-file topic on its explicit qualified source-interval dependency; no main/default promotion.",
        "pins": {str(p.resolve()): digest(p) for p in (
            args.packet, args.current_release, args.tests, contexts_path, adapter_path,
            reference_path, torch_path, Path(__file__),
        )},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "pins_reclosed", "independent_tests",
                                           "original_whole_native_words", "functional_instruction_means")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "current-release", "tests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
