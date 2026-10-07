"""Structural reader for the ``linalg-on-tensors`` input grammar.

Parsed with xDSL's own builtin/func/linalg/tensor/arith/math dialects (the producer's ``quant_ext``
and ``prov.*`` extensions are read as unregistered ops, which is still structural parsing -- the
operand graph, the types and the attributes are all real IR). No text scraping.
"""

from __future__ import annotations

from ..ir.workload import Workload

MARKER = "linalg-on-tensors"


def is_linalg_on_tensors(text: str) -> bool:
    """True when the module header declares the ``prov.level = "linalg-on-tensors"`` grammar.

    The two grammars are told apart by the module's OWN attribute dictionary, which is on the first
    non-comment line of either: ``merlin_iface.version`` for one, ``prov.level`` for the other.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        return MARKER in stripped
    return False


def read(text: str) -> Workload:
    from .linalg_model import build_workload

    return build_workload(text)
