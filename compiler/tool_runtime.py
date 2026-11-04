"""Load the public SMT tool, including its documented snapshot-contained package.

Only the selected z3 package is loaded; no other site-packages are added to
sys.path. The retained archive and unpacked bytes are fully listed in the tool
manifest. Temporary extraction persists until process exit for lazy imports.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import threading
from types import ModuleType

PUBLIC_Z3_PACKAGE = Path(
    "/work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/"
    "phase1-runtime-v1/venv/lib/python3.12/site-packages/z3"
)
DEPENDENCY_ROOT = Path(__file__).resolve().parent / "tool_dependencies"
_LOCK = threading.RLock()
_TEMPORARIES = []


def _load_package(package: Path) -> ModuleType:
    initializer = package / "__init__.py"
    if not initializer.is_file():
        raise ModuleNotFoundError(
            f"z3-solver is unavailable: selected package has no initializer: {initializer}",
            name="z3",
        )
    spec = importlib.util.spec_from_file_location(
        "z3", initializer, submodule_search_locations=[str(package)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot construct z3 package loader for {initializer}")
    before = set(sys.modules)
    module = importlib.util.module_from_spec(spec)
    sys.modules["z3"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in set(sys.modules) - before:
            if name == "z3" or name.startswith("z3."):
                sys.modules.pop(name, None)
        raise
    return module


def _bundled_package() -> Path | None:
    """Verify the exact regular-file allowlist, then extract to private storage."""
    manifest_path = DEPENDENCY_ROOT / "z3-solver.manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "gemmini.packaged_tool_dependency.v1"
            or manifest.get("package") != "z3-solver"
            or manifest.get("version") != "4.16.0.0"):
        raise ImportError("unsupported packaged z3-solver manifest")
    archive_name = manifest.get("archive")
    if not isinstance(archive_name, str) or Path(archive_name).name != archive_name:
        raise ImportError("invalid packaged z3-solver archive name")
    archive = DEPENDENCY_ROOT / archive_name
    data = archive.read_bytes()
    if (len(data) != manifest["archive_bytes"]
            or hashlib.sha256(data).hexdigest() != manifest["archive_sha256"]):
        raise ImportError("packaged z3-solver archive size or SHA256 mismatch")
    expected = manifest["files"]
    if not isinstance(expected, dict) or not expected:
        raise ImportError("packaged z3-solver has no file allowlist")
    for name, record in expected.items():
        parts = PurePosixPath(name).parts
        if (not parts or parts[0] != "z3" or len(parts) < 2
                or any(part in (".", "..") for part in parts)
                or PurePosixPath(name).is_absolute()
                or str(PurePosixPath(name)) != name
                or not isinstance(record.get("bytes"), int) or record["bytes"] < 0):
            raise ImportError(f"invalid packaged z3-solver file record: {name}")
    if sum(row["bytes"] for row in expected.values()) != manifest["unpacked_bytes"]:
        raise ImportError("packaged z3-solver unpacked byte count mismatch")
    temporary = tempfile.TemporaryDirectory(prefix="gemmini-z3-")
    destination = Path(temporary.name)
    try:
        # Read the already verified bytes rather than reopening an archive that
        # could change between verification and extraction.
        import io
        seen = set()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as packed:
            for member in packed:
                name = member.name
                if name not in expected or name in seen or not member.isfile():
                    raise ImportError(f"unapproved packaged z3-solver member: {name}")
                record = expected[name]
                if member.size != record["bytes"]:
                    raise ImportError(f"packaged z3-solver file size mismatch: {name}")
                stream = packed.extractfile(member)
                if stream is None:
                    raise ImportError(f"cannot read packaged z3-solver file: {name}")
                with stream:
                    content = stream.read(record["bytes"] + 1)
                if (len(content) != record["bytes"]
                        or hashlib.sha256(content).hexdigest() != record["sha256"]):
                    raise ImportError(f"packaged z3-solver file SHA256 mismatch: {name}")
                target = destination.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(content)
                seen.add(name)
        if seen != set(expected):
            raise ImportError("packaged z3-solver archive is missing approved files")
        package = destination / "z3"
        if not (package / "__init__.py").is_file():
            raise ImportError("packaged z3-solver has no initializer")
    except BaseException:
        temporary.cleanup()
        raise
    _TEMPORARIES.append(temporary)
    atexit.register(temporary.cleanup)
    return package


def ensure_z3() -> ModuleType:
    """Resolve installed, explicitly selected, bundled, then public-runtime z3.

    Errors inside an installed/selected package propagate. Corrupt bundled bytes
    fail explicitly. Exact schedule verification always uses the real solver.
    """
    with _LOCK:
        try:
            return importlib.import_module("z3")
        except ModuleNotFoundError as error:
            if error.name != "z3":
                raise
        selected = os.environ.get("GEMMINI_Z3_PACKAGE")
        if selected is not None:
            return _load_package(Path(selected))
        package = _bundled_package()
        if package is None:
            package = PUBLIC_Z3_PACKAGE
        return _load_package(package)
