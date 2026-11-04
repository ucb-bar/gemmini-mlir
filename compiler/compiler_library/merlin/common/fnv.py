"""FNV-1a over 64-bit words, natively when this host can build the one-loop helper.

The digest is inherently serial (each step multiplies the previous state), so numpy cannot vectorize
it and the interpreter loop costs ~0.2 us a word: an open whole model's oracle digests every device
dispatch's output, about two minutes of a SmolVLA build in this loop alone. The same loop in C is
three orders of magnitude faster.

The helper is compiled once per (source, compiler) with the host's own C compiler into the
regenerable cache, CHECKED against the interpreter loop on a fixed vector before it is ever used,
and skipped -- the interpreter loop answers -- whenever any of that is unavailable or disagrees. The
offset and prime are the caller's arguments, so no constant lives here. Nothing here names a target.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

#: Set to ``0`` to force the interpreter loop (for a comparison, or a host whose compiler misbehaves).
NATIVE_ENV = "MERLIN_NATIVE_FNV"

_SOURCE = r"""
#include <stddef.h>
#include <stdint.h>
uint64_t merlin_fnv1a64_words(const uint64_t *words, size_t count, uint64_t state, uint64_t prime) {
  for (size_t i = 0; i < count; i++) {
    state ^= words[i];
    state *= prime;
  }
  return state;
}
"""

_M64 = (1 << 64) - 1
_LOCK = threading.Lock()
_LOADED: dict[str, object] = {}


def python_words(words, offset: int, prime: int) -> int:
    """The interpreter loop: the definition the native helper is checked against."""
    state = int(offset) & _M64
    for word in words.tolist():
        state ^= word
        state = (state * prime) & _M64
    return state


def _compiler() -> str | None:
    return shutil.which(os.environ.get("CC") or "cc")


def _build(directory: Path, compiler: str) -> Path | None:
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=60).stdout
    key = hashlib.sha256((_SOURCE + "\0" + compiler + "\0" + version).encode()).hexdigest()[:24]
    library = directory / f"fnv1a64-{key}.so"
    if library.is_file():
        return library
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as work:
        source = Path(work) / "fnv.c"
        source.write_text(_SOURCE, encoding="utf-8")
        built = Path(work) / library.name
        done = subprocess.run(
            [compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(built)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if done.returncode != 0 or not built.is_file():
            return None
        os.replace(built, library)  # atomic: a concurrent builder's identical library is simply replaced
    return library


def _native():
    """The checked native function, or ``None`` (then the interpreter loop answers)."""
    if os.environ.get(NATIVE_ENV, "1") == "0" or sys.byteorder != "little":
        return None
    with _LOCK:
        if "fn" in _LOADED:
            return _LOADED["fn"]
        _LOADED["fn"] = None
        try:
            import numpy as np

            from merlin.common.artifacts import cache_dir

            compiler = _compiler()
            library = _build(Path(cache_dir("native")), compiler) if compiler else None
            if library is None:
                return None
            fn = ctypes.CDLL(str(library)).merlin_fnv1a64_words
            fn.restype = ctypes.c_uint64
            fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64]
            # CHECKED BEFORE USE, on a vector with every bit pattern class (zero, all-ones, sign bits).
            rng = np.random.default_rng(0)
            probe = rng.integers(0, 1 << 63, 4096, dtype=np.uint64) * np.uint64(3)
            probe[:3] = [0, _M64, 1 << 63]
            offset, prime = (int(v) | 1 for v in rng.integers(1 << 62, 1 << 63, 2, dtype=np.uint64) * np.uint64(2))
            if fn(probe.ctypes.data, probe.size, offset, prime) != python_words(probe, offset, prime):
                return None
            _LOADED["fn"] = fn
        except Exception:  # noqa: BLE001 -- an unavailable helper is the interpreter loop, never a failure
            return None
        return _LOADED["fn"]


def fnv1a64_words(words, offset: int, prime: int) -> int:
    """FNV-1a of ``words`` (a 1-D little-endian uint64 array) from ``offset`` with ``prime``."""
    import numpy as np

    words = np.ascontiguousarray(words, dtype="<u8")
    fn = _native()
    if fn is None or words.size == 0:
        return python_words(words, offset, prime)
    return int(fn(words.ctypes.data, words.size, int(offset) & _M64, int(prime) & _M64))
