"""Reclose exact source arithmetic, selected components, and whole outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from merlin.llvmlower.guarded_quantized_mean import derive

from mlir_oot.device_mean_bundle import IntegerSumPlan, merlin_callbacks
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
            raise ValueError("actual partial-link closure changed")
        if (
            stage["arm"] == "control"
            or not stage["argv"][-1].endswith("/guarded_mean_mixed.o")
        ) and stage["old_sha256"] != stage["output_sha256"]:
            raise ValueError("nonselected target stages changed")
    catalog_path = Path(receipt["normal_catalog"])
    catalog = json.loads(catalog_path.read_text())
    if sha(catalog["source_snapshot"]) != catalog["source_sha256"]:
        raise ValueError("normal compiler source seal changed")
    routes = catalog["guarded_mean_additions"]
    if len(routes) != 1 or routes[0]["implementation"] != "gemmini_integer_sum":
        raise ValueError("exact source reduction coverage changed")
    route = routes[0]
    proof = route["proof"]
    if proof != derive(proof["count"], proof["input_scale"], proof["output_scale"]):
        raise ValueError("complete original source certificate changed")
    if (
        route["producer_contract"]
        != IntegerSumPlan(*route["output_shape"], proof["count"]).proof()
    ):
        raise ValueError("integer producer resource/effect contract changed")
    if route["fully_written_arguments"] != [1, 2] or route["result_argument"] != 2:
        raise ValueError("private scratch/result writer ABI changed")
    prepare, _ = merlin_callbacks(
        Path(receipt["llvm_bin"]),
        Path(receipt["normal_mean_bundle"]),
        (lambda path, work: path, None),
    )
    prepare(Path(catalog["source_snapshot"]), catalog_path.parent)
    for capsule in receipt["capsules"]:
        measured = json.loads(Path(capsule).read_text())
        if not measured["passed"] or len(measured["records"]) != 2:
            raise ValueError("complete target capsule did not finish")
        for row in measured["records"]:
            if (
                sha(row["elf"]) != row["elf_sha256"]
                or not row["passed"]
                or row["marker"] not in row["stdout"]
            ):
                raise ValueError("complete target capsule identity or output changed")
            if audit_elf(Path(row["elf"]).read_bytes())["status"] != "pass":
                raise ValueError("capsule contains forbidden instructions")
    for arm in receipt["arms"].values():
        native = json.loads(Path(arm["native_validation"]).read_text())
        strict = json.loads(Path(arm["strict_validation"]).read_text())
        golden = np.load(native["original_golden"]).reshape(-1)
        output = np.load(arm["native_output"]).reshape(-1)
        if (
            golden.dtype != np.float32
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
                raise ValueError("original zero-tolerance whole gate changed")
        if (
            not strict["spike_full_output_match"]
            or strict["spike_output_sha256"]
            != hashlib.sha256(golden.astype("<f4").tobytes()).hexdigest()
        ):
            raise ValueError("complete strict target output changed")
        audit = audit_elf(Path(arm["elf"]).read_bytes())
        if (
            audit != arm["fresh_final_audit"]
            or audit["status"] != "pass"
            or strict["elf_sha256"] != audit["elf_sha256"]
        ):
            raise ValueError("final executable or instruction audit changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_text())
    check(record)
    print("INTEGER_MEAN_CURRENT1992_CLOSURE_PASS", len(record["pins"]), "pins")
