"""The LLVM data layout of a cross target, read from the compiler that compiles for it.

MLIR translates a module to LLVM IR under the module's ``llvm.data_layout``, or under LLVM's default
layout when it carries none -- and the default aligns 64-bit integers and doubles to 4 bytes. Every
load and store the translation emits then states ``align 4``, which the cross compiler keeps even after
it substitutes its own layout; on a vector host an under-aligned 64-bit access has no legal vector form,
so a loop over int64 or f64 data is not vectorized at all. The layout is a property of the target
(triple and ISA flags), so it is ASKED of the compiler with the build's own cross flags -- never written
here. Nothing here names a target.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_file

_PREFIX = 'target datalayout = "'
_KNOWN: dict[tuple[str, ...], str] = {}
_LOCK = threading.Lock()


def parse(llvm_ir: str) -> str | None:
    """The layout string of a ``target datalayout = "..."`` line in ``llvm_ir``, or ``None``."""
    for line in llvm_ir.splitlines():
        stripped = line.strip()
        if stripped.startswith(_PREFIX) and stripped.endswith('"'):
            return stripped[len(_PREFIX) : -1]
    return None


def default_index_bits(layout: str) -> int:
    """The default pointer's index width when it agrees with its representation width.

    LLVM spells a default pointer as ``p:size:abi[:preferred[:index]]`` or
    ``p0:...``. Omitted index width equals size. MLIR index lowering also
    determines memref descriptor field widths, so a differing GEP index width
    cannot alone authorize this producer's C-facing ABI. An absent default
    pointer declaration has no checked width, even if LLVM has a default.
    """
    selected = []
    for field in layout.split("-"):
        parts = field.split(":")
        if parts[0] not in {"p", "p0"}:
            continue
        if len(parts) not in {3, 4, 5} or any(not part.isdecimal() for part in parts[1:]):
            raise ValueError("compiler data layout has malformed default pointer fields")
        size, abi = int(parts[1]), int(parts[2])
        preferred = int(parts[3]) if len(parts) >= 4 else abi
        index = int(parts[4]) if len(parts) == 5 else size
        if min(size, abi, preferred, index) <= 0 or index > size:
            raise ValueError("compiler data layout has invalid default pointer/index widths")
        if index != size:
            raise ValueError("compiler pointer and GEP index widths differ; memref ABI is unverified")
        selected.append(index)
    if len(selected) != 1:
        raise ValueError("compiler data layout has no unique declared default pointer index width")
    return selected[0]


def selected_index_bits(observed: object) -> int | None:
    """Validate a complete compiler-owned index observation without rerunning it.

    Consumers must separately bind the selected observation to their actual
    compiler invocation and source/capture identities.
    """
    if not isinstance(observed, Mapping) or set(observed) != {
        "schema",
        "compiler_requested",
        "compiler_resolved",
        "compiler_sha256",
        "cross_flags",
        "data_layout",
        "index_bits",
        "scope",
    }:
        return None
    if (
        observed.get("schema") != "merlin.selected-index-lowering.v1"
        or type(observed.get("index_bits")) is not int
        or not all(
            type(observed.get(key)) is str and bool(observed[key])
            for key in ("compiler_requested", "compiler_resolved", "data_layout", "scope")
        )
        or not is_sha256(observed.get("compiler_sha256"))
        or type(observed.get("cross_flags")) is not list
        or not observed["cross_flags"]
        or any(type(flag) is not str or not flag for flag in observed["cross_flags"])
    ):
        return None
    try:
        return observed["index_bits"] if default_index_bits(observed["data_layout"]) == observed["index_bits"] else None
    except ValueError:
        return None


def _query(clang: str | Path, cross_flags: Sequence[str]) -> str:
    flags = [str(flag) for flag in cross_flags if str(flag) != "-c"]
    done = subprocess.run(
        [str(clang), *flags, "-S", "-emit-llvm", "-x", "c", "-", "-o", "-"],
        input="",
        capture_output=True,
        text=True,
        timeout=120,
    )
    layout = parse(done.stdout) if done.returncode == 0 else None
    if not layout:
        raise RuntimeError(f"{clang} states no data layout for {flags}: {done.stderr[-500:]}")
    return layout


def observe_index_width(clang: str | Path, cross_flags: Sequence[str]) -> dict:
    """Fresh compiler-owned lowering selection for exact cross flags.

    The compiler executable's bytes are pinned before and after observation.
    This does not authenticate its dynamic libraries or the later MLIR runtime.
    """
    import shutil

    requested = str(clang)
    executable = Path(requested)
    if not executable.is_file() and executable.name == requested:
        found = shutil.which(requested)
        if found:
            executable = Path(found)
    resolved = executable.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("selected cross compiler is not a regular file")
    before = sha256_file(resolved)
    flags = [str(flag) for flag in cross_flags]
    if "-c" in flags:
        raise ValueError("selected cross flags cannot suppress LLVM data-layout observation")
    layout = _query(resolved, flags)
    if sha256_file(resolved) != before:
        raise ValueError("selected cross compiler changed during layout observation")
    return {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": requested,
        "compiler_resolved": str(resolved),
        "compiler_sha256": before,
        "cross_flags": flags,
        "data_layout": layout,
        "index_bits": default_index_bits(layout),
        "scope": "compiler binary and exact cross flags only; no transitive toolchain closure",
    }


def of(clang: str | Path, cross_flags: Sequence[str]) -> str:
    """The data layout ``clang`` uses for ``cross_flags``; raises when it states none."""
    key = (str(clang), *map(str, cross_flags))
    with _LOCK:
        if key in _KNOWN:
            return _KNOWN[key]
    layout = _query(clang, cross_flags)
    with _LOCK:
        _KNOWN[key] = layout
    return layout
