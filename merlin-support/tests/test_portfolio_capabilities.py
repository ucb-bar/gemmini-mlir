"""Public provider capabilities preserve implementation identity without native execution."""

import importlib
from pathlib import Path

import pytest

from merlin.runtime.backends.base import get_backend


@pytest.mark.parametrize(
    "capability,implementation,methods",
    [
        ("host_witness_abi", "gemmini_host_witness_abi", ("derive_native_witness_abi",)),
        ("completion_contract", "gemmini_completion_contract", ("derive_completion_contract",)),
        (
            "primitive_probe",
            "gemmini_primitive_probe",
            (
                "prepare_primitive_probe",
                "execute_prepared_primitive",
                "isolated_primitive_signature",
                "runtime_elf_digest",
            ),
        ),
    ],
)
def test_explicit_capability_is_original_provider_module(monkeypatch, capability, implementation, methods):
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(Path(__file__).resolve().parents[1]))
    backend = get_backend("gemmini")
    adapter = getattr(backend, capability)
    assert adapter is importlib.import_module(backend.__name__ + "." + implementation)
    assert all(callable(getattr(adapter, method)) for method in methods)
    assert Path(adapter.__file__).resolve().is_relative_to(Path(backend.__file__).resolve().parent)
