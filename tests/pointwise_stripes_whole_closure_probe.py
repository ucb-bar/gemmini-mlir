"""Reclose source-bound pointwise accumulator stripe whole qualification artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def check(receipt):
    for name, digest in receipt["pins"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("artifact changed: " + name)
    golden = np.load(receipt["original_golden"]).reshape(-1)
    if golden.dtype != np.float32 or golden.size != 1000:
        raise ValueError("original source gate requires1,000binary32words")
    digest = hashlib.sha256(golden.astype("<f4").tobytes()).hexdigest()
    for arm in receipt["arms"].values():
        actual = np.load(arm["native_output"]).reshape(-1)
        if actual.dtype != np.float32 or not np.array_equal(
            actual.view(np.uint32), golden.view(np.uint32)
        ):
            raise ValueError("native whole output differs from immutable original")
        for name in ["native_validation", "strict_validation"]:
            verdict = json.loads(Path(arm[name]).read_text())
            if (
                not verdict["quality_pass"]
                or not verdict["exact_equal"]
                or verdict["elements"] != 1000
                or verdict["atol"]
                or verdict["rtol"]
            ):
                raise ValueError("original exact gate changed")
        target = json.loads(Path(arm["strict_validation"]).read_text())
        if (
            not target["spike_full_output_match"]
            or target["spike_output_sha256"] != digest
        ):
            raise ValueError("strict complete output differs")
        audit = audit_elf(Path(arm["elf"]).read_bytes())
        if audit != arm["fresh_final_audit"] or audit["status"] != "pass":
            raise ValueError("actual final executable instruction audit changed")
        if target["elf_sha256"] != audit["elf_sha256"]:
            raise ValueError("strict target ran another executable")
    catalog = json.loads(Path(receipt["normal_catalog"]).read_text())
    if (
        hashlib.sha256(Path(catalog["source_snapshot"]).read_bytes()).hexdigest()
        != catalog["source_sha256"]
    ):
        raise ValueError("normal source catalog seal changed")
    link = receipt["controlled_link"]
    closure = receipt["source_bundle_closure"]
    if link["control_elf_sha256"] != receipt.get(
        "expected_control_elf_sha256",
        "7ee47ff7c904053c680200bb2ef2bfa207c081ef43f6711ba38d8d07c3e30d97",
    ):
        raise ValueError("qualified control was not byte reproduced")
    changed = {Path(x["selected"]).parent.name for x in link["changed_leaves"]}
    if (
        changed != set(closure["changed_selected_kernels"])
        or len(changed) != closure["applications"]
    ):
        raise ValueError("selected source/object coverage changed")
    if receipt["default_enabled"] or receipt["whole_performance"] != "UNKNOWN":
        raise ValueError(
            "qualification does not imply automatic selection or performance"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    args = p.parse_args()
    receipt = json.loads(args.receipt.read_text())
    check(receipt)
    print(f"POINTWISE_STRIPE_WHOLE_CLOSURE_PASS {len(receipt['pins'])} pins")


if __name__ == "__main__":
    main()
