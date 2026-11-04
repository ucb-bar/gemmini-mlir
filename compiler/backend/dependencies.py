"""Select the frozen compiler-only solver in relocated compiler packages.

Model build workers need not share the author's Python environment.  Keep the
same solver bindings and native library with the compiler rather than silently
weakening source/storage proofs when an ambient package is absent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def select_proof_solver(root: Path) -> None:
    directory = root / 'compiler_dependencies'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('package') != 'z3-solver' or not manifest.get('files'):
        raise ValueError('compiler proof-solver dependency manifest is incomplete')
    for relative, digest in manifest['files'].items():
        path = directory / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError(f'compiler proof-solver dependency changed: {relative}')
    # Entry-point invocation is a fresh process. Never reuse another imported
    # solver and accidentally combine bindings from two native-library versions.
    existing = sys.modules.get('z3')
    if existing is not None and Path(existing.__file__).resolve().parent != (directory / 'z3').resolve():
        raise ValueError('compiler proof solver was imported before dependency selection')
    sys.path.insert(0, str(directory))
