"""Reclose the original model and emitted segmented input composition."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(receipt):
    for filename, digest in receipt["pins"].items():
        if sha(filename) != digest:
            raise ValueError("artifact changed: " + filename)
    link = json.loads(Path(receipt["controlled_link"]).read_text())
    if link["control_elf_sha256"] != receipt["control"]["elf_sha256"]:
        raise ValueError("controlled baseline identity changed")
    for stage in link["partial_link_graph"]:
        if sha(stage["argv"][-1]) != stage["output_sha256"]:
            raise ValueError("partial-link closure changed")
        if "old_sha256" in stage and stage["old_sha256"] != stage["output_sha256"]:
            raise ValueError("existing target stages were not preserved")
    catalog = json.loads(Path(receipt["normal_catalog"]).read_text())
    if sha(catalog["source_snapshot"]) != catalog["source_sha256"]:
        raise ValueError("normal compiler source seal changed")
    acceptance = catalog["segmented_input_acceptance"]
    if len(acceptance["source_rewrites"]) != 3 or acceptance["refused"]:
        raise ValueError("accepted source geometry coverage changed")
    closure = json.loads(Path(receipt["emitted_host_closure"]).read_text())
    if len(closure["routes"]) != 3 or any(
        not row["same_actual_producer_result_descriptor"]
        or row["materialized_original_consumer_calls"]
        for row in closure["routes"]
    ):
        raise ValueError("actual producer/consumer descriptor closure changed")
    for source in acceptance["source_rewrites"]:
        if source["address"]["dtype"] != "i8" or source["address"]["element_bytes"] != 1:
            raise ValueError("typed physical source storage changed")
    for arm in receipt["arms"].values():
        native = json.loads(Path(arm["native_validation"]).read_text())
        strict = json.loads(Path(arm["strict_validation"]).read_text())
        golden = np.load(native["original_golden"]).reshape(-1)
        output = np.load(arm["native_output"]).reshape(-1)
        if golden.dtype != np.float32 or golden.size != 1000 or not np.array_equal(
            output.view(np.uint32), golden.view(np.uint32)
        ):
            raise ValueError("original1,000 binary32 words changed")
        for verdict in (native, strict):
            if verdict["elements"] != 1000 or not verdict["exact_equal"] or not verdict[
                "quality_pass"
            ] or verdict["atol"] != 0 or verdict["rtol"] != 0:
                raise ValueError("original zero-tolerance gate changed")
        digest = hashlib.sha256(golden.astype("<f4").tobytes()).hexdigest()
        if not strict["spike_full_output_match"] or strict["spike_output_sha256"] != digest:
            raise ValueError("complete strict target output changed")
        audit = audit_elf(Path(arm["elf"]).read_bytes())
        if audit != arm["fresh_final_audit"] or audit["status"] != "pass":
            raise ValueError("final executable audit changed")
        if strict["elf_sha256"] != audit["elf_sha256"]:
            raise ValueError("strict output belongs to another executable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_text())
    check(record)
    print("SEGMENTED_CURRENT1992_CLOSURE_PASS", len(record["pins"]), "pins")
