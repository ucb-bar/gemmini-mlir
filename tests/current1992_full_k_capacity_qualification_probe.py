"""Reclose the exact whole-source composition and measured kernel lineage."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(record):
    for filename, digest in record["pins"].items():
        if sha(filename) != digest:
            raise ValueError("artifact changed: " + filename)
    link = json.loads(Path(record["controlled_link"]).read_text())
    if link["control_reproduced_sha256"] != record["control"]["elf_sha256"]:
        raise ValueError("whole baseline was not reproduced")
    for stage in link["partial_link_graph"]:
        if sha(stage["argv"][-1]) != stage["output_sha256"]:
            raise ValueError("actual partial-link closure changed")
        if (
            stage["arm"] == "control"
            and stage["output_sha256"] != stage["source_sha256"]
        ):
            raise ValueError("frozen control object changed")
    catalog = json.loads(Path(record["normal_catalog"]).read_text())
    if sha(catalog["source_snapshot"]) != catalog["source_sha256"]:
        raise ValueError("normal compiler source seal changed")
    if catalog["compilation"]["object_sha256"] != sha(
        Path(record["arms"]["controlled1992"]["elf"]).parent / "guarded_mean_mixed.o"
    ):
        raise ValueError("controlled device aggregate differs from normal catalog")
    control = json.loads(Path(record["control_manifest"]).read_text())
    candidate = json.loads(Path(record["candidate_manifest"]).read_text())
    historical = json.loads(Path(record["historical_manifest"]).read_text())
    before = {r["symbol"]: r for r in control["routes"]}
    history = {r["symbol"]: r for r in historical["routes"]}
    changes = []
    for route in candidate["routes"]:
        old = before[route["symbol"]]
        for key in (
            "region",
            "kernel",
            "proof",
            "numeric_contract",
            "integer_readout",
            "virtual_padding_proof",
            "paired_readout",
        ):
            if route.get(key) != old.get(key):
                raise ValueError("source/numeric binding changed")
        if (
            route["adapter_compilation"]["object_sha256"]
            != old["adapter_compilation"]["object_sha256"]
        ):
            raise ValueError("ranked adapter changed")
        selected = route.get("cached_b_capacity_decision", {}).get("applied", False)
        if selected:
            if (
                route["compilation"]["object_sha256"]
                != history[route["symbol"]]["compilation"]["object_sha256"]
            ):
                raise ValueError("selected implementation differs from measured family")
            changes.append(route["symbol"])
        elif (
            route["schedule"] != old["schedule"]
            or route["compilation"]["object_sha256"]
            != old["compilation"]["object_sha256"]
        ):
            raise ValueError("unselected implementation changed")
    if sorted(changes) != sorted(record["changed_kernels"]) or len(changes) != 4:
        raise ValueError("selected source coverage changed")
    for arm in record["arms"].values():
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
                raise ValueError("original zero-tolerance gate changed")
        if (
            not strict["spike_full_output_match"]
            or strict["spike_output_sha256"]
            != hashlib.sha256(golden.astype("<f4").tobytes()).hexdigest()
        ):
            raise ValueError("strict target output differs")
        audit = audit_elf(Path(arm["elf"]).read_bytes())
        if (
            audit != arm["fresh_final_audit"]
            or audit["status"] != "pass"
            or strict["elf_sha256"] != audit["elf_sha256"]
        ):
            raise ValueError("final executable audit changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_text())
    check(record)
    print("CURRENT1992_FULL_K_CAPACITY_CLOSURE_PASS", len(record["pins"]), "pins")
