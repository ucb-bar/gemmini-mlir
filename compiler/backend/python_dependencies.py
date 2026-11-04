"""Resolve the public compiler's solver from the selected Python tool runtime.

Some package-build workers invoke a stock Python instead of the selected venv.
Load only the missing solver package; never put another Merlin installation or
an entire site-packages tree ahead of the submitted compiler library.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys


# This is the public tool runtime explicitly selected for this launch. It is a
# build dependency, not a target runtime, simulator, reference or answer source.
SELECTED_PYTHON_RUNTIME = Path(
    '/work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/phase1-runtime-v1/venv')


def ensure_selected_solver():
    """Return actual solver provenance, preserving every other import failure."""
    try:
        solver = importlib.import_module('z3')
        resolution = 'active_python_environment'
    except ModuleNotFoundError as error:
        if error.name != 'z3':
            raise
        packaged_entry = Path(__file__).resolve().parents[1] / 'compiler_dependencies/z3/__init__.py'
        packages = ([packaged_entry] if packaged_entry.is_file() else
            sorted(SELECTED_PYTHON_RUNTIME.glob(
                'lib/python*/site-packages/z3/__init__.py')))
        if len(packages) != 1:
            raise RuntimeError('the selected public Python tool runtime must supply '
                'one z3-solver package; found ' + str(len(packages))) from error
        entry = packages[0]
        spec = importlib.util.spec_from_file_location('z3', entry,
            submodule_search_locations=[str(entry.parent)])
        if spec is None or spec.loader is None:
            raise RuntimeError('cannot load the selected public z3-solver package')
        solver = importlib.util.module_from_spec(spec)
        sys.modules['z3'] = solver
        try:
            spec.loader.exec_module(solver)
        except BaseException:
            # A partial package must never mask a real dependency/load failure.
            for name in tuple(sys.modules):
                if name == 'z3' or name.startswith('z3.'):
                    sys.modules.pop(name, None)
            raise
        resolution = ('packaged_selected_public_solver' if entry == packaged_entry
            else 'selected_public_python_runtime')
    entry = Path(solver.__file__).resolve()
    return dict(package='z3-solver', version=solver.get_version_string(),
        resolution=resolution, entry=str(entry),
        entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        role='compiler source/storage/order proofs; no target execution')
