"""Resolve the public host solver without changing other Python dependencies.

GEMMINI_Z3_PACKAGE may select a caller-supplied z3 package directory. Otherwise
the selected public tool runtime supplies the fallback. Only that package is
loaded: its site-packages directory is never placed on sys.path.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import ModuleType


PUBLIC_Z3_PACKAGE = Path(
    "/work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/"
    "phase1-runtime-v1/venv/lib/python3.12/site-packages/z3"
)
_LOCK = threading.RLock()


def ensure_z3() -> ModuleType:
    """Import ordinary z3, or the exact public/caller-selected package.

    Missing packages fail explicitly. Errors inside an installed package or
    while loading its native solver propagate; no verification is skipped.
    """
    with _LOCK:
        try:
            return importlib.import_module("z3")
        except ModuleNotFoundError as error:
            if error.name != "z3":
                raise

        package = Path(os.environ.get("GEMMINI_Z3_PACKAGE", str(PUBLIC_Z3_PACKAGE)))
        initializer = package / "__init__.py"
        if not initializer.is_file():
            raise ModuleNotFoundError(
                f"z3-solver is unavailable: ordinary import failed and the selected "
                f"package has no initializer: {initializer}; GEMMINI_Z3_PACKAGE "
                "may select an available public host z3 package directory",
                name="z3",
            )
        spec = importlib.util.spec_from_file_location(
            "z3", initializer, submodule_search_locations=[str(package)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot construct the z3 package loader for {initializer}")
        before = set(sys.modules)
        module = importlib.util.module_from_spec(spec)
        sys.modules["z3"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # A failed initialization must not leave a partially imported
            # package available to a subsequent compiler invocation.
            for name in set(sys.modules) - before:
                if name == "z3" or name.startswith("z3."):
                    sys.modules.pop(name, None)
            raise
        return module
