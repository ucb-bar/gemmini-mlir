"""Reclose whole exact gates and an unchanged terminal residual replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from residual_output_whole_closure_probe import check as check_whole

from mlir_oot.no_fsm_audit import audit_elf


def check(receipt: dict) -> None:
    for name, expected in receipt["pins"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact changed: {name}")
    check_whole(json.loads(Path(receipt["whole_receipt"]).read_text()))
    control = json.loads(Path(receipt["capsule_control"]).read_text())
    replay = json.loads(Path(receipt["unchanged_candidate_replay"]).read_text())
    for arm, result in enumerate((control, replay)):
        if result["completed"] is not True or result["returncode"] != 0:
            raise ValueError("complete original capsule did not terminate correctly")
        if result["cycles"] != receipt["capsule_cycles"][arm]:
            raise ValueError("ROI timer changed")
        for marker in (
            "OUTPUT_GUARD_PREDICTOR PASS all802816 guards4096",
            f"OUTPUT_GUARD_PASS arm{arm} all802816 guards4096 inputs1605632",
            f"OUTPUT_GUARD_CYCLES {arm} {result['cycles']}",
        ):
            if result["stdout"].count(marker) != 1:
                raise ValueError("complete predictor/source/input/guard marker missing")
        if result["engine"]["binary_sha256"] != receipt["engine_sha256"]:
            raise ValueError("capsule engine changed")
        if (
            result["engine"]["receipt"]["receipt_sha256"]
            != receipt["engine_receipt_sha256"]
        ):
            raise ValueError("capsule memory/build regime changed")
    elf = Path(replay["elf"])
    if hashlib.sha256(elf.read_bytes()).hexdigest() != replay["elf_sha256"]:
        raise ValueError("candidate replay executable changed")
    if replay["elf_sha256"] != receipt["original_incomplete_candidate_elf_sha256"]:
        raise ValueError("longer replay rebuilt or changed the candidate")
    if replay["max_cycles"] != 30_000_000 or replay["wall_timeout_s"] != 5400:
        raise ValueError("replay budget differs from recorded intent")
    for path in receipt["audited_executables"]:
        if audit_elf(Path(path).read_bytes())["status"] != "pass":
            raise ValueError("final executable contains forbidden/unknown instructions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    check(receipt)
    print(f"RESIDUAL_OUTPUT_RELEASE_CLOSURE_PASS {len(receipt['pins'])} pins")


if __name__ == "__main__":
    main()
