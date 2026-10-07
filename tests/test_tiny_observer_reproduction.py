"""Refuse stale reproduction dependencies before any target compiler runs."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "experiments/tiny_closed_observer/build.py"
)
spec = importlib.util.spec_from_file_location("tiny_observer_reproduction", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_stale_source_refused(tmp_path):
    source = tmp_path / "source.ll"
    source.write_bytes(b"accepted")
    receipt = {"pins": {str(source): module.digest(source)}}
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Qualification input changed"):
        module.reclose(receipt, {})


def test_snapshot_override_must_reproduce_original_bytes(tmp_path):
    snapshot = tmp_path / "snapshot.md"
    snapshot.write_bytes(b"original")
    name = str(tmp_path / "historical.md")
    receipt = {"pins": {name: module.digest(snapshot)}}
    module.reclose(receipt, {name: snapshot})
    snapshot.write_bytes(b"updated")
    with pytest.raises(ValueError, match="Qualification input changed"):
        module.reclose(receipt, {name: snapshot})


def test_missing_source_refused(tmp_path):
    with pytest.raises(ValueError, match="Qualification input changed"):
        module.reclose({"pins": {str(tmp_path / "missing.ll"): "0" * 64}}, {})


def test_existing_output_refused_before_input_loading(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SOURCE), "--inputs", "absent", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Output must not already exist" in result.stderr
    assert not list(tmp_path.iterdir())
