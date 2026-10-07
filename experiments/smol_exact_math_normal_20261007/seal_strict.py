"""Close the completed original whole-source gate; retain native-only evidence."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REFERENCE = Path(
    "/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/"
    "root_smol_endpoint_whole_spike_20261007_qualification.json"
)
CORE = Path(
    "/scratch/agustin/tmp/merlin-host-llvm-helper-main-20261007/out/artifacts/"
    "host_llvm_helper_merged_delivery/qualification.json"
)


def identity(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def entries(receipt):
    if "pins" in receipt:
        return list(receipt["pins"].items())
    return [(row["path"], row["sha256"]) for row in receipt["file_pins"]]


def only_fields(text, prefix):
    rows = [line.split() for line in text.splitlines() if line.startswith(prefix)]
    assert len(rows) == 1, (prefix, rows)
    return rows[0]


def main():
    native_path = ROOT / "native_qualification.json"
    admission_path = ROOT / "strict_admission.json"
    terminal_path = ROOT / "strict_terminal.json"
    native = json.loads(native_path.read_text())
    admission = json.loads(admission_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    core = json.loads(CORE.read_text())
    inherited_path = Path(native["inherited_source_packet"]["path"])
    inherited = json.loads(inherited_path.read_text())

    packets = [native, admission, reference, core, inherited]
    expected = {}
    for packet in packets:
        for path, digest in entries(packet):
            if path in expected:
                assert expected[path] == digest, path
            expected[path] = digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(identity, sorted(expected)))
    for path, row in zip(sorted(expected), actual, strict=True):
        assert expected[path] == row["sha256"], path

    assert terminal["returncode"] == 0
    assert terminal["argv"] == admission["argv"]
    assert terminal["elf_sha256"] == native["target_elf"]["sha256"]
    assert terminal["stdout_sha256"] == identity(ROOT / "spike.stdout")["sha256"]
    assert terminal["histogram_sha256"] == identity(ROOT / "spike.stderr")["sha256"]
    assert reference["status"] == "PASS" and reference["bitwise_exact"]
    assert core["head"] == native["core_head"]
    audit = native["final_nofsm"]
    assert audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"]
    assert audit["elf_sha256"] == terminal["elf_sha256"]

    output = (ROOT / "spike.stdout").read_text()
    digest = only_fields(output, "OUT_SHA256 ")
    assert digest[:4] == ["OUT_SHA256", "f32le", "1600", "6400"]
    assert len(digest) == 5
    assert digest[4] == reference["original_raw_output_sha256"]
    assert only_fields(output, "METRIC memref_rank_mismatch ")[-1] == "0"
    assert output.splitlines().count("DONE") == 1
    functional = int(only_fields(output, "METRIC cycles ")[-1])
    assert functional > 0
    build_hash = only_fields(output, "METRIC build_hash ")[-1]
    assert build_hash == "d06abe401c50"

    paths = [
        Path(__file__), native_path, admission_path, terminal_path, REFERENCE,
        CORE, inherited_path, ROOT / "spike.stdout", ROOT / "spike.stderr",
    ]
    receipt = {
        "schema": "root.smol_exact_math.normal_whole_target_qualification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ORIGINAL_WHOLE_TARGET_GATE",
        "feature": native["feature"],
        "generic_implementation_preexisted_on_main": True,
        "core_head": native["core_head"],
        "target_elf": native["target_elf"],
        "original_gate": reference["original_gate"],
        "original_raw_output_sha256": digest[4],
        "bitwise_exact": True,
        "rank_mismatch": 0,
        "build_hash": build_hash,
        "terminal": terminal,
        "final_nofsm": audit,
        "spike_functional_cycles": functional,
        "whole_firesim_cycles": "UNKNOWN",
        "stock_job": None,
        "numeric_policy": native["numeric_policy"],
        "native_receipt_unchanged": identity(native_path),
        "source_capture_pins_reclosed": len(entries(inherited)),
        "native_build_pins_reclosed": len(entries(native)),
        "admission_pins_reclosed": len(entries(admission)),
        "reference_whole_pins_reclosed": len(entries(reference)),
        "core_package_pins_reclosed": len(entries(core)),
        "unique_inherited_pins_reclosed": len(expected),
        "scope": (
            "Complete actual target executable and all1600 original f32 words. "
            "Functional count is not FPGA timing; different core revisions "
            "prevent isolated pass savings attribution. No physical memory "
            "capacity, accelerator percentage or automatic selection promotion."
        ),
        "file_pins": [identity(path) for path in paths],
    }
    destination = ROOT / "whole_target_qualification.json"
    assert not destination.exists(), "Preserve immutable evidence; make a successor."
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "qualification": identity(destination),
        "unique_inherited_pins_reclosed": len(expected),
        "spike_functional_cycles": functional,
    }))


if __name__ == "__main__":
    main()
