"""Resolve proof dependencies from the task's declared compiler runtime.

The model-check driver may launch python3 outside that runtime. Load only the
missing package namespace, preserving the caller's interpreter and every other
import. No proof, placement or arithmetic fallback is supplied here.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

# TASK.md's selected shared runtime, also used by this compiler's toolchain.
# This is a tool installation, never a workspace, model artifact or cache path.
SELECTED_RUNTIME = Path('/work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/phase1-runtime-v1/venv')
BUNDLED_Z3 = Path(__file__).with_name('vendor') / 'z3' / '__init__.py'


def ensure_proof_dependencies():
    """Return real Z3, or fail explicitly when its installation is unavailable.

    An existing import always wins. Only a missing top-level ``z3`` permits the
    selected-runtime adapter; broken installed packages and native libraries
    remain errors rather than silently changing solver implementations.
    """
    try:
        return importlib.import_module('z3')
    except ModuleNotFoundError as error:
        if error.name != 'z3':
            raise
    # Isolated model builds need the same dependency inside their package:
    # the selected tool installation is not mounted in every child process.
    # These are unchanged selected-runtime Python files and its native library.
    packages = ([BUNDLED_Z3] if BUNDLED_Z3.is_file() else
        sorted((SELECTED_RUNTIME / 'lib').glob('python*/site-packages/z3/__init__.py')))
    if len(packages) != 1:
        raise ModuleNotFoundError(
            'Z3 proof dependency unavailable: install z3-solver in the invoking Python '
            'or provide the declared selected compiler runtime at ' + str(SELECTED_RUNTIME), name='z3')
    package = packages[0]
    spec = importlib.util.spec_from_file_location('z3', package,
        submodule_search_locations=[str(package.parent)])
    if spec is None or spec.loader is None:
        raise ImportError('cannot load selected runtime Z3 package: ' + str(package))
    module = importlib.util.module_from_spec(spec)
    sys.modules['z3'] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Do not leave a partial package behind in callers that catch failure.
        for name in tuple(sys.modules):
            if name == 'z3' or name.startswith('z3.'):
                del sys.modules[name]
        raise
    return module
