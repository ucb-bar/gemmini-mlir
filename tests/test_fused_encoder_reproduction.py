"""Authentication failures must precede compiler execution."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "experiments/fused_encoder_radix/build.py"
)
spec = importlib.util.spec_from_file_location("encoder_reproduction", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_changed_dependency_refused(tmp_path):
    source = tmp_path / "provider.c"
    source.write_text("original")
    receipt = {"pins": {str(source): module.digest(source)}}
    module.reclose(receipt)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Qualification input changed"):
        module.reclose(receipt)


def test_missing_dependency_refused(tmp_path):
    with pytest.raises(ValueError, match="Qualification input changed"):
        module.reclose({"pins": {str(tmp_path / "absent"): "0" * 64}})


def test_output_identity_checks_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"first")
    b.write_bytes(b"first")
    module.require_identity(a, b)
    b.write_bytes(b"other")
    with pytest.raises(ValueError, match="Reproduction differs"):
        module.require_identity(a, b)


def test_existing_output_refused_before_inputs_or_compiler(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE),
            "--baseline",
            "absent",
            "--qualification",
            "absent",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Output must not already exist" in result.stderr
    assert not list(tmp_path.iterdir())
