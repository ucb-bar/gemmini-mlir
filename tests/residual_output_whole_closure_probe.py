"""Reclose a frozen exact residual whole-model qualification receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def check(receipt: dict) -> None:
    for name, expected in receipt["pins"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact changed: {name}")
    golden = np.load(receipt["original_golden"]).reshape(-1)
    if golden.dtype != np.float32 or golden.size != 1000:
        raise ValueError("original golden must contain 1,000 binary32 words")
    output_digest = hashlib.sha256(golden.astype("<f4").tobytes()).hexdigest()
    for arm in receipt["arms"].values():
        output = np.load(arm["native_output"]).reshape(-1)
        if output.dtype != np.float32 or not np.array_equal(
            output.view(np.uint32), golden.view(np.uint32)
        ):
            raise ValueError("native output differs from original binary32 words")
        native = json.loads(Path(arm["native_validation"]).read_text())
        strict = json.loads(Path(arm["strict_validation"]).read_text())
        for verdict in (native, strict):
            if not all(verdict[k] is True for k in ("quality_pass", "exact_equal")):
                raise ValueError("original exact gate failed")
            if verdict["elements"] != 1000 or verdict["atol"] or verdict["rtol"]:
                raise ValueError("original gate changed")
        if not strict["spike_full_output_match"]:
            raise ValueError("strict target output was not compared completely")
        if strict["spike_output_sha256"] != output_digest:
            raise ValueError("strict target digest differs from original golden")
        audit = audit_elf(Path(arm["elf"]).read_bytes())
        if audit["status"] != "pass" or audit != arm["fresh_final_audit"]:
            raise ValueError("final executable audit changed")
        if strict["elf_sha256"] != audit["elf_sha256"]:
            raise ValueError("strict target log is bound to another executable")
    catalog = json.loads(Path(receipt["normal_catalog"]).read_text())
    source = Path(catalog["source_snapshot"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != catalog["source_sha256"]:
        raise ValueError("normal compiler catalog source seal changed")
    selected = [
        route
        for route in catalog["residual_additions"]
        if route.get("single_output_guard") is True
    ]
    if len(selected) != 1:
        raise ValueError("expected one source-derived selected relation")
    if receipt["hardware_release"] != "HELD; original complete capsule pending":
        raise ValueError("this receipt does not authorize hardware release")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    check(receipt)
    print(f"WHOLE_EXACT_CLOSURE_PASS {len(receipt['pins'])} pins")


if __name__ == "__main__":
    main()
