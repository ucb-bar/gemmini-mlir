"""Reclose the original whole gate and explicit serial-five current1992 route."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from merlin.llvmlower.quantized_affine_pair import derive

from mlir_oot.no_fsm_audit import audit_elf


def check(receipt):
    for name, digest in receipt["pins"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("artifact changed: " + name)
    catalog = json.loads(Path(receipt["normal_catalog"]).read_text())
    source = Path(catalog["source_snapshot"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != catalog["source_sha256"]:
        raise ValueError("normal compiler source seal changed")
    selected = [r for r in catalog["residual_additions"] if r.get("single_output_guard")]
    if len(selected) != 1:
        raise ValueError("source-certified selected relation coverage changed")
    route = selected[0]
    if route["proof"] != derive(**route["proof"]["source"], **route["proof"]["predictor"]):
        raise ValueError("complete source arithmetic certificate changed")
    if route["fully_written_arguments"] != [2] or route["result_argument"] != 2:
        raise ValueError("private single-writer ABI changed")
    for arm in receipt["arms"].values():
        native = json.loads(Path(arm["native_validation"]).read_text())
        strict = json.loads(Path(arm["strict_validation"]).read_text())
        golden = np.load(native["original_golden"]).reshape(-1)
        output = np.load(arm["native_output"]).reshape(-1)
        if (
            golden.dtype != np.float32
            or output.dtype != np.float32
            or golden.size != 1000
            or not np.array_equal(output.view(np.uint32), golden.view(np.uint32))
        ):
            raise ValueError("original1,000 binary32 words changed")
        for verdict in (native, strict):
            if (
                verdict["elements"] != 1000
                or not verdict["exact_equal"]
                or not verdict["quality_pass"]
                or verdict["atol"] != 0
                or verdict["rtol"] != 0
            ):
                raise ValueError("original exact gate changed")
        digest = hashlib.sha256(golden.astype("<f4").tobytes()).hexdigest()
        if not strict["spike_full_output_match"] or strict["spike_output_sha256"] != digest:
            raise ValueError("complete strict target output changed")
        audit = audit_elf(Path(arm["elf"]).read_bytes())
        if audit != arm["fresh_final_audit"] or audit["status"] != "pass":
            raise ValueError("final executable audit changed")
        if strict["elf_sha256"] != audit["elf_sha256"]:
            raise ValueError("strict output belongs to another executable")
    if receipt["streaming"].split(";")[0] != "DISABLED":
        raise ValueError("serial storage contract changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_text())
    check(record)
    print("SERIAL_FIVE_CURRENT1992_CLOSURE_PASS", len(record["pins"]), "pins")
