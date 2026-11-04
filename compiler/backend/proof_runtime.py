"""Resolve the mandatory solver from the selected public compiler runtime.

Some launchers use their own Python rather than the selected runtime's Python.
The shared scheduling verifiers still require Z3; failure to resolve it must
remain a compiler error, never permission to skip a proof.
"""
from importlib import import_module
from pathlib import Path
import sys


PUBLIC_RUNTIME = Path(
    "/work/merlin/tasks/gemmini-shared-phase1-20261021/"
    ".tmp/phase1-runtime-v1/venv"
)


def require_proof_runtime():
    try:
        return import_module("z3")
    except ModuleNotFoundError as exc:
        if exc.name != "z3":
            raise
    # This location is a granted tool input in the selected launch contract.
    # Append it so existing launcher dependencies retain their precedence.
    for site in sorted((PUBLIC_RUNTIME / "lib").glob("python*/site-packages")):
        if not (site / "z3" / "__init__.py").is_file():
            continue
        entry = str(site)
        if entry not in sys.path:
            sys.path.append(entry)
        return import_module("z3")
    raise ModuleNotFoundError(
        "mandatory scheduling proof dependency z3 is unavailable in the "
        "launcher and selected public runtime", name="z3"
    )
