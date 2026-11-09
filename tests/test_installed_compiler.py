"""Actual source lowering and output identity through the installed SDK boundary."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from gemmini import compile_source
from xdsl.utils.exceptions import ParseError


def test_real_integer_matmul_uses_selected_sdk(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "examples/matmul.mlir").read_text(encoding="utf-8")
    result = compile_source(source, output=tmp_path / "build")
    manifest = json.loads(result.manifest.read_text())
    assert manifest["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert manifest["execution"] == "not_run"
    assert manifest["toolchain"]["packages"]["xdsl"]
    assert manifest["toolchain"]["complete_dependency_identity"] is False
    assert 'kernel_entry' in result.artifact.read_text()
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256((result.directory / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert all("compiler_library" not in path for path in sys.path)
    with pytest.raises(FileExistsError):
        compile_source(source, output=result.directory)


def test_invalid_source_never_publishes_completed_compilation(tmp_path):
    destination = tmp_path / "invalid"
    with pytest.raises((ValueError, ParseError)):
        compile_source("not valid MLIR", output=destination)
    assert not (destination / "compilation.json").exists()
    assert not (destination / "gemmini.json").exists()
    assert json.loads((destination / "failure.json").read_text())["status"] == "failed"


def test_public_service_resolves_installed_provider_and_inspects_products(tmp_path):
    from merlin import api

    source = Path(__file__).resolve().parents[1] / "examples/matmul.mlir"
    product = api.compile(source, backend="gemmini", output=tmp_path / "public")
    inspected = api.inspect(product.directory)
    assert inspected["backend"] == "gemmini"
    assert {item["kind"] for item in inspected["artifacts"]} == {"llvm-dialect", "json"}
    assert inspected["metadata"]["execution"] == "not_run"
    assert (product.directory / "gemmini.json").exists()
    assert (product.directory / "compilation.json").exists()
    with pytest.raises(ValueError, match="does not build"):
        api.compile(source, backend="gemmini", output=tmp_path / "unsupported",
                    options=api.CompileOptions(platform="zephyr"))
    assert not (tmp_path / "unsupported").exists()
