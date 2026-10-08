"""Explicit legacy fixture dependency; never a support-provider fallback."""

import os
from pathlib import Path


def fixture_root() -> Path:
    value = os.environ.get("GEMMINI_LEGACY_FIXTURE_ROOT")
    if not value:
        raise RuntimeError("set GEMMINI_LEGACY_FIXTURE_ROOT to the historical Merlin fixture checkout")
    root = Path(value)
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError("GEMMINI_LEGACY_FIXTURE_ROOT must be an existing absolute directory")
    return root
