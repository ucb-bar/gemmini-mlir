"""Compile admitted source using the installed SDK and explicit output ownership.

The archived compiler.py and its embedded library remain reproducible separately.
This installed entry never changes sys.path or silently selects the frozen SDK.
It emits LLVM dialect and a command plan, not a linked or executed image.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass(frozen=True)
class Compilation:
    directory: Path
    artifact: Path
    commands: Path
    manifest: Path


def compile_source(source: str, *, output: str | Path, selected_buffer: dict | None = None) -> Compilation:
    """Compile exact source text into a fresh directory without historical cache reuse.

    The old cache includes the embedded SDK. An installed SDK is external to
    that tree, so reuse stays disabled until its complete dependency identity
    is admitted. Numerical, ABI and scheduling checks in the backend still run.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("compilation requires nonempty source text")
    directory = Path(output).resolve()
    if directory.exists():
        raise FileExistsError(f"compilation requires a fresh output directory: {directory}")
    # Resolve required components before publishing any files. Importing the
    # public API itself does not require a compiler or solver installation.
    import z3  # noqa: F401 -- mandatory scheduling proof dependency

    import merlin

    from .backend.program import _compile

    directory.mkdir(parents=True)
    try:
        command_buffer, artifact_text = _compile(source, directory, selected_buffer=selected_buffer)
        if command_buffer.get("declined"):
            raise ValueError(f"backend declined the source: {command_buffer['declined']}")
        artifact = directory / "module.mlir"
        commands = directory / "commands.json"
        artifact.write_text(artifact_text, encoding="utf-8")
        commands.write_text(json.dumps(command_buffer, indent=2) + "\n", encoding="utf-8")
        manifest = directory / "gemmini.json"
        packages = {}
        for distribution in ("gemmini-mlir", "merlin", "xdsl", "z3-solver"):
            try:
                packages[distribution] = version(distribution)
            except PackageNotFoundError:
                packages[distribution] = None
        products = [
            {"path": path.name, "kind": kind,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path, kind in ((artifact, "llvm-dialect"), (commands, "command-plan"))
        ]
        manifest.write_text(json.dumps({
            "schema": "gemmini.compilation.v1", "status": "compiled",
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "artifacts": products,
            "shared_library": {"module": "merlin", "source": str(Path(merlin.__file__).resolve())},
            "toolchain": {"python": platform.python_version(), "packages": packages,
                          "complete_dependency_identity": False},
            "execution": "not_run", "qualification": "not_established",
        }, indent=2) + "\n", encoding="utf-8")
        return Compilation(directory, artifact, commands, manifest)
    except Exception as exc:
        (directory / "failure.json").write_text(json.dumps({
            "status": "failed", "type": type(exc).__name__, "reason": str(exc),
        }, indent=2) + "\n", encoding="utf-8")
        raise
