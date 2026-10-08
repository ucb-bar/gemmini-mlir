"""Migration inventories commit actual relocated bytes, not runtime qualification."""

import hashlib
import json
from pathlib import Path

import pytest

TRANSFORMATIONS = (
    "resource_migration.json",
    "build_source_identity_migration.json",
    "rocc_semantics_migration.json",
    "readout_encoding_migration.json",
    "pinned_runtime_migration.json",
    "runtime_environment_migration.json",
    "rtl_checks_migration.json",
    "portfolio_capabilities_migration.json",
    "completion_capability_migration.json",
    "required_extraction_migration.json",
    "schedule_vocabulary_migration.json",
    "contract_alignment_migration.json",
    "whole_model_backend_migration.json",
    "simulator_identity_migration.json",
    "primitive_helper_migration.json",
    "transport_alignment_migration.json",
    "standalone_elementwise_review.json",
)
OUTPUT_SCOPES = {
    "standalone_elementwise_review.json": {"contracts/target_contract.yaml", "contracts/residual.yaml"},
    "transport_alignment_migration.json": {
        "gemmini_conformance/kernel_slot.py",
        "tests/test_conformance_policy.py",
        "tests/test_provider_resources.py",
        "tests/test_program_build_sources.py",
    },
    "primitive_helper_migration.json": {"backend/gemmini_primitive_probe.py", "backend/primitive_program.py"},
    "contract_alignment_migration.json": {"contracts/target_contract.yaml", "contracts/residual.yaml"},
    "rtl_checks_migration.json": {"backend/rtl_checks.py", "backend/rocc_semantics.py", "tests/test_rtl_checks.py"},
    "rocc_semantics_migration.json": {"backend/rocc_semantics.py", "backend/__init__.py"},
    "readout_encoding_migration.json": {
        "backend/rocc_semantics.py",
        "backend/gemmini_codegen_mlir.py",
        "backend/gemmini_loop_conv.py",
    },
}


def evolved_identity(root, path, expected, *, after=None):
    """Replay recorded chains, then explicitly scoped output supersessions.

    The later extraction records commit outputs only; they do NOT prove a complete
    input-hash chain. Keep that limitation rather than inventing predecessor hashes.
    """
    active = after not in TRANSFORMATIONS
    for inventory in TRANSFORMATIONS:
        if not active:
            active = inventory == after
            continue
        document = json.loads((root / inventory).read_text())
        if inventory in OUTPUT_SCOPES:
            outputs = document["current_files"]
            assert set(outputs) == OUTPUT_SCOPES[inventory], "unexpected supersession scope"
            assert document["supersedes"], "output replacement must declare supersession"
            expected = outputs.get(path, expected)
        else:
            changes = {row["destination"]: row for row in document["files"]}
            assert len(changes) == len(document["files"]), "conflicting destination declarations"
            if path in changes:
                row = changes[path]
                assert row["source_sha256"] == expected, "discontinuous source identity"
                expected = row["destination_sha256"]
    return expected


@pytest.mark.parametrize(
    "inventory",
    [
        "conformance_migration.json",
        "build_support_migration.json",
        "resource_migration.json",
        "build_source_identity_migration.json",
        "pinned_runtime_migration.json",
        "runtime_environment_migration.json",
        "portfolio_capabilities_migration.json",
        "completion_capability_migration.json",
        "required_extraction_migration.json",
        "schedule_vocabulary_migration.json",
    ],
)
def test_relocated_owner_inventory(inventory):
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / inventory).read_text())
    assert document["files"]
    for row in document["files"]:
        path = root / row["destination"]
        expected = row.get("destination_sha256", row.get("sha256"))
        expected = evolved_identity(root, row["destination"], expected, after=inventory)
        assert expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


@pytest.mark.parametrize("inventory", list(OUTPUT_SCOPES))
def test_explicit_output_supersessions(inventory):
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / inventory).read_text())
    assert set(document["current_files"]) == OUTPUT_SCOPES[inventory]
    for path, digest in document["current_files"].items():
        expected = evolved_identity(root, path, digest, after=inventory)
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("mutation", ["source", "duplicate", "scope"])
def test_changed_or_conflicting_migration_records_refuse(tmp_path, mutation):
    root = Path(__file__).resolve().parents[1]
    for inventory in TRANSFORMATIONS:
        (tmp_path / inventory).write_bytes((root / inventory).read_bytes())
    record = json.loads((tmp_path / TRANSFORMATIONS[0]).read_text())
    row = record["files"][0]
    expected = row["source_sha256"]
    if mutation == "source":
        row["source_sha256"] = "0" * 64
    elif mutation == "duplicate":
        record["files"].append(dict(row))
    else:
        output = json.loads((tmp_path / TRANSFORMATIONS[2]).read_text())
        output["current_files"]["unexpected.py"] = "0" * 64
        (tmp_path / TRANSFORMATIONS[2]).write_text(json.dumps(output))
    (tmp_path / TRANSFORMATIONS[0]).write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        evolved_identity(tmp_path, row["destination"], expected)
