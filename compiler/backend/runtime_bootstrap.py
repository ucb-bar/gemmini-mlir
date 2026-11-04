"""Load the installed proof dependency without assuming a Python executable path."""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys

_PUBLIC_RUNTIME = Path('/work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/phase1-runtime-v1/venv')


def prepare_z3():
    """Keep normal imports; recover only from a missing top-level z3 package.

    Explicit MERLIN_Z3_PYTHONPATH entries name installed package parents (or a
    z3 package itself). Otherwise inspect the active distribution metadata and
    the single supplied public runtime. Load only z3, leaving other dependency
    resolution and the caller's interpreter untouched. Missing installations leave
    normal resolution unchanged: proof sites still require the actual solver.
    """
    try:
        return importlib.import_module('z3')
    except ModuleNotFoundError as exc:
        if exc.name != 'z3':
            raise
    packages = []
    for entry in os.environ.get('MERLIN_Z3_PYTHONPATH', '').split(os.pathsep):
        if entry:
            path = Path(entry)
            packages.append(path if path.name == 'z3' else path / 'z3')
    from importlib import metadata
    try:
        distribution = metadata.distribution('z3-solver')
    except metadata.PackageNotFoundError:
        pass
    else:
        packages.append(Path(distribution.locate_file('z3')))
    if _PUBLIC_RUNTIME.is_dir():
        packages.extend(sorted((_PUBLIC_RUNTIME / 'lib').glob('python*/site-packages/z3')))
    seen = set()
    for package in packages:
        initializer = package / '__init__.py'
        if not initializer.is_file() or initializer.resolve() in seen:
            continue
        seen.add(initializer.resolve())
        spec = importlib.util.spec_from_file_location('z3', initializer,
            submodule_search_locations=[str(package)])
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules['z3'] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # A broken installed solver is an error, never an admission bypass.
            for name in tuple(sys.modules):
                if name == 'z3' or name.startswith('z3.'):
                    del sys.modules[name]
            raise
        return module
    # Preserve proof-free compilation when no solver is supplied. This does not
    # install a stub or suppress errors at any checker: its ordinary import z3
    # still raises if a selected proof requires an unavailable solver.
    return None


def missing_z3_error(error):
    """Recognize an actual unavailable solver, including chained diagnostics."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ModuleNotFoundError) and error.name == 'z3':
            return True
        error = error.__cause__ or error.__context__
    return False


def dependency_diagnostic():
    """Summarize runtime visibility without disclosing filesystem locations.

    Only active Python metadata, explicit dependency configuration and the one
    supplied public runtime are inspected. This does not search other roots or
    change dependency/proof admission.
    """
    from importlib import metadata
    try:
        metadata.distribution('z3-solver')
    except metadata.PackageNotFoundError:
        installed_distribution = False
    else:
        installed_distribution = True
    try:
        visible_package = importlib.util.find_spec('z3') is not None
    except (ImportError, ValueError):
        visible_package = False
    public_exists = _PUBLIC_RUNTIME.is_dir()
    candidates = (list((_PUBLIC_RUNTIME / 'lib').glob('python*/site-packages/z3'))
                  if public_exists else [])
    return {
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'executable_basename': Path(sys.executable).name,
        'isolated': bool(sys.flags.isolated),
        'no_site': bool(sys.flags.no_site),
        'normal_z3_package_visible': visible_package,
        'active_z3_distribution_visible': installed_distribution,
        'explicit_z3_dependency_entries': sum(bool(entry) for entry in
            os.environ.get('MERLIN_Z3_PYTHONPATH', '').split(os.pathsep)),
        'public_runtime_exists': public_exists,
        'public_z3_package_candidates': sum((package / '__init__.py').is_file()
                                            for package in candidates),
        'python_search_path_entries': len(sys.path),
        'site_package_search_entries': sum(Path(entry).name in
            ('site-packages', 'dist-packages') for entry in sys.path),
    }
