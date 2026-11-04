"""Serialize proofs that share Z3's process-global default context.

Z3Py expressions and solvers below use implicit ``main_ctx()``. Independent
Solver instances still share that native context and may not operate on it
concurrently. Hold this lock from expression construction through solver use
and proof-result extraction. It is independent of the xDSL parser's IR_LOCK.
Scheduling, code emission, compiler subprocesses and execution stay outside it.
"""
from __future__ import annotations

from functools import wraps
from threading import RLock

SYMBOLIC_LOCK = RLock()


def serialized_symbolic_check(function):
    """Protect one complete symbolic proof, allowing nested proof helpers."""
    @wraps(function)
    def checked(*args, **kwargs):
        with SYMBOLIC_LOCK:
            return function(*args, **kwargs)
    return checked
