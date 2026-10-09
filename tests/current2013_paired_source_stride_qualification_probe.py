"""Reclose the paired source-stride producer and its whole-source admission."""

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
    if (
        link["control_elf_sha256"] != record["control"]["elf_sha256"]
        or sha(link["control_elf"]) != record["control"]["elf_sha256"]
        or link["candidate_elf_sha256"] != record["candidate"]["elf_sha256"]
    ):
        raise ValueError("controlled executable identity changed")
    for stage in link["partial_link_graph"]:
        if sha(stage["argv"][-1]) != stage["output_sha256"]:
            raise ValueError("partial-link closure changed")
        if stage["arm"] == "control" and (
            stage["output_sha256"] != stage["old_sha256"]
        ):
            raise ValueError("frozen control partial object changed")
    for filename, digest in link["retained_input_pins"].items():
        if sha(filename) != digest:
            raise ValueError("retained controlled input changed")
    catalog = json.loads(Path(record["normal_catalog"]).read_text())
    if sha(catalog["source_snapshot"]) != catalog["source_sha256"]:
        raise ValueError("normal compiler source seal changed")
    final_aggregate = Path(record["candidate"]["elf"]).parent / "segmented_mixed.o"
    if catalog["compilation"]["object_sha256"] != sha(final_aggregate):
        raise ValueError("normal selected device aggregate differs from control")
    control = json.loads(Path(record["control_manifest"]).read_text())
    candidate = json.loads(Path(record["candidate_manifest"]).read_text())
    before = {r["symbol"]: r for r in control["routes"]}
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
                raise ValueError("source or numeric binding changed")
        producer = route.get("readout_producer_domain")
        old_producer = old.get("readout_producer_domain")
        if producer is not None:
            producer = json.loads(json.dumps(producer))
            old_producer = json.loads(json.dumps(old_producer))
            bound = producer["proof"]
            if bound["kernel_object_sha256"] != route["compilation"]["object_sha256"]:
                raise ValueError("producer certificate is not bound to selected code")
            for key in ("kernel_object", "kernel_object_sha256", "kernel_ir_sha256"):
                del bound[key]
                del old_producer["proof"][key]
            if producer != old_producer:
                raise ValueError("producer semantic domain changed")
        if (
            route["adapter_compilation"]["object_sha256"]
            != (old["adapter_compilation"]["object_sha256"])
        ):
            raise ValueError("ranked adapter changed")
        if (
            route["compilation"]["object_sha256"]
            != (old["compilation"]["object_sha256"])
        ):
            if not route["source_stride_policy_decision"]["applied"]:
                raise ValueError("changed kernel lacks resource-bound admission")
            changes.append(route["symbol"])
        elif route["schedule"] != old["schedule"]:
            raise ValueError("unchanged kernel schedule binding changed")
    if changes != record["changed_kernels"] or len(changes) != 1:
        raise ValueError("selected source coverage changed")
    selected = next(r for r in candidate["routes"] if r["symbol"] in changes)
    if selected["compilation"]["object_sha256"] != sha(record["measured_kernel"]):
        raise ValueError("normal kernel differs from measured capsule")
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
            raise ValueError("original 1,000 binary32 words changed")
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
    capsules = json.loads(Path(record["capsule_receipt"]).read_text())
    for pair in capsules["pairs"]:
        a, b = (json.loads(Path(p).read_text()) for p in pair["result_paths"])
        if a["common_operand_addresses"] != b["common_operand_addresses"]:
            raise ValueError("capsule operand placement changed")
        for result in (a, b):
            if (
                result["status"] != "pass"
                or result["nofsm_status"] != "pass"
                or result["strict_spike"]["status"] != "pass"
            ):
                raise ValueError("complete capsule qualification changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_text())
    check(record)
    print("CURRENT2013_PAIRED_SOURCE_STRIDE_CLOSURE_PASS", len(record["pins"]), "pins")
