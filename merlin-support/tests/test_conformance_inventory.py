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
    "support_integration_review.json",
    "rocc_contract_key_import_review.json",
)
OUTPUT_SCOPES = {
    "rocc_contract_key_import_review.json": {"backend/rocc_semantics.py"},
    "support_integration_review.json": {
        "backend/gemmini.py",
        "backend/gemmini_codegen_mlir.py",
        "backend/gemmini_loop_matmul_decode.py",
        "backend/rtl_checks.py",
        "contracts/target_contract.yaml",
        "contracts/residual.yaml",
        "tests/test_provider_resources.py",
    },
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
            if path in outputs and "previous_files" in document:
                assert document["previous_files"][path] == expected, "discontinuous integration identity"
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


def test_integration_refuses_relabelled_predecessor(tmp_path):
    root = Path(__file__).resolve().parents[1]
    name = "support_integration_review.json"
    record = json.loads((root / name).read_text())
    record["previous_files"]["backend/gemmini.py"] = "0" * 64
    (tmp_path / name).write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="discontinuous integration identity"):
        evolved_identity(
            tmp_path,
            "backend/gemmini.py",
            "8ec831a02f779620389484bfa5f9e4e71b7b2f6600fd3b3e2918f557f03df955",
            after="standalone_elementwise_review.json",
        )


@pytest.mark.parametrize("mutation", ["predecessor", "scope"])
def test_rocc_key_import_review_refuses_relabelled_predecessor_or_scope(tmp_path, mutation):
    root = Path(__file__).resolve().parents[1]
    name = "rocc_contract_key_import_review.json"
    record = json.loads((root / name).read_text())
    if mutation == "predecessor":
        record["previous_files"]["backend/rocc_semantics.py"] = "0" * 64
    else:
        record["current_files"]["backend/unrelated.py"] = "0" * 64
    (tmp_path / name).write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        evolved_identity(
            tmp_path,
            "backend/rocc_semantics.py",
            "7c1e3e3714b08c70d587513c9ea0ad149d63cd16ead03d58494c1c6c14312b6f",
            after="support_integration_review.json",
        )
