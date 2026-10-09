"""Offline relocation and public-interface checks; never generate goldens or export capsules."""

import hashlib
import json
from pathlib import Path

import pytest
from gemmini_conformance import model_slices as M

from merlin.targetgen.contract import matmul_interface


def test_relocation_changes_only_the_two_shared_imports():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "model_slices_migration.json").read_text())["files"][0]
    payload = (root / record["destination"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record["destination_sha256"]
    original = payload.replace(b"from merlin.targetgen.contract.matmul_interface import",
                               b"from .contract.matmul_interface import")
    assert hashlib.sha256(original).hexdigest() == record["source_sha256"]


def test_seven_recipes_preserve_instruction_policy_without_golden_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recipe inspection must not generate goldens or export files")

    monkeypatch.setattr(M.CG, "golden", forbidden)
    monkeypatch.setattr(M, "export_capsule_dir", forbidden)
    recipes = M.standard_model_slices(label="held_out")
    assert [row["name"].split("_")[0] for row in recipes] == [f"C{i}" for i in range(7)]
    assert all(row["label"] == "held_out" for row in recipes)
    assert all(row["numeric_policy"]["compare"] == "exact_int" for row in recipes)
    assert all(row["required_oracle_tiers"] == ["L0", "L1", "L2", "L3"] for row in recipes)
    assert recipes[0]["inputs"] == [
        {"name": "W1", "role": "weight", "shape": [64, 64], "dtype": "i8"},
        {"name": "X", "role": "input", "shape": [16, 64], "dtype": "i8"},
    ]
    assert recipes[0]["expected"]["instruction_classes"] == [
        "FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST", "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"
    ]
    assert recipes[1]["operation"]["attributes"]["epilogue"] == ["relu"]


@pytest.mark.parametrize("overrides", [
    {},
    {"target": "another_device", "operand_dtype": "bf16", "acc_dtype": "f32", "output_dtype": "f32"},
    {"epilogue": ["bias_add", "requant"], "bias": "bias0", "bias_dtype": "i32", "requant_shift": 3},
    {"epilogue": ["acc_scale"], "acc_scale": 0.5, "comment": "explicit scales"},
    {"scale_block": 4, "scale_dtype": "i8"},
    {"epilogue": ["maxpool"], "pool_attrs": {
        "pool_in_dims": [4, 4], "pool_size": [2, 2], "pool_stride": [2, 2]}, "commit_rows": 4},
])
def test_legacy_interface_default_matches_shared_explicit_target_emitter(overrides):
    arguments = dict(lhs="A", weight="W", out="Y", M=16, K=32, N=16,
                     epilogue=[], output_dtype="i32", comment="offline parity")
    arguments.update(overrides)
    expected = matmul_interface.emit_interface_mlir(**{"target": "gemmini", **arguments})
    assert M.emit_interface_mlir(**arguments) == expected
    assert M._dt is matmul_interface._dt
    assert M._header is matmul_interface._header
