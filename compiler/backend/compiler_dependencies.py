"""Select the disclosed compiler runtime when caller dependencies are missing.

The caller's xDSL and JSON Schema installation remains authoritative. Optional
proof dependencies belong to their consumers and are not entrypoint requirements.
MERLIN_COMPILER_PYTHON may name an explicit absolute interpreter path. Otherwise
the launch's selected phase1 runtime is a fallback for missing core dependencies,
when mounted. Never mix another interpreter's site-packages or change compiler/tool
arguments.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


SELECTED_PYTHON = Path(
    '/work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/'
    'phase1-runtime-v1/venv/bin/python3'
)
_REEXEC_MARKER = '_MERLIN_COMPILER_DEPENDENCY_REEXEC'


def _dependency_problem():
    """Check core parsing/schema APIs, without loading optional proof tooling."""
    try:
        from xdsl.ir import Operation
        from xdsl.dialects import llvm
        from jsonschema import Draft202012Validator
        # Imports must expose the actual required core APIs; a package's mere
        # presence is insufficient. Solver imports stay in proof consumers.
        if not callable(Operation.verify) or not hasattr(llvm, 'FuncOp'):
            raise ImportError('incompatible xdsl API')
        if not callable(Draft202012Validator.check_schema):
            raise ImportError('incompatible jsonschema API')
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'
    return None


def ensure_compiler_dependencies(entrypoint):
    """Keep a working caller interpreter; otherwise exec the declared runtime."""
    explicit = os.environ.get('MERLIN_COMPILER_PYTHON')
    target = Path(explicit) if explicit is not None else SELECTED_PYTHON
    if not target.is_absolute():
        raise RuntimeError('MERLIN_COMPILER_PYTHON must be an absolute interpreter path')
    # Compare invocation paths, not resolved binaries: two virtualenv Python
    # symlinks may resolve to the same binary but have different dependencies.
    current = Path(os.path.abspath(sys.executable))
    selected = Path(os.path.abspath(target))
    problem = _dependency_problem()
    # An explicit launcher may exec Python under a different invocation path.
    # Its successful child has already honored the authoritative selection.
    if problem is None and (explicit is None or current == selected
                            or os.environ.get(_REEXEC_MARKER) == str(selected)):
        return
    reason = problem or 'explicit MERLIN_COMPILER_PYTHON selection'
    if os.environ.get(_REEXEC_MARKER) == str(selected) or current == selected:
        raise RuntimeError(f'compiler dependencies unavailable in {selected}: {reason}')
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise RuntimeError(f'compiler runtime is not executable: {selected}; {reason}')
    environment = os.environ.copy()
    environment[_REEXEC_MARKER] = str(selected)
    os.execve(str(selected), [str(selected), str(entrypoint), *sys.argv[1:]], environment)
