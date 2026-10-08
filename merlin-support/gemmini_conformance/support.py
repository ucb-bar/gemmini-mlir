"""Bind conformance to this explicitly selected support provider, never a namesake."""

from pathlib import Path

from merlin.runtime.backends import base
from merlin.targetgen import target_registry

ROOT = Path(__file__).resolve().parents[1]
TARGET = "gemmini"


def backend():
    selected = target_registry.resolve(TARGET)
    if selected.base.resolve() != ROOT:
        raise RuntimeError(
            f"Gemmini conformance requires this support provider; set MERLIN_TARGET_PATH={ROOT}"
        )
    module = base.get_backend(TARGET)
    if Path(module.__file__).resolve() != ROOT / "backend" / "__init__.py":
        raise RuntimeError("Gemmini conformance backend differs from the selected support provider")
    return module
