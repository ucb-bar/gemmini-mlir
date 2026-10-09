"""Grammar-routing parse: one xDSL Context that admits BOTH input grammars.

`merlin_iface` v0.1 and the `linalg-on-tensors` capsule form are both real MLIR, so both are
parsed by the same structural parser; the module's own attributes decide which reader runs.
"""
from __future__ import annotations

import re

from xdsl.context import Context
from xdsl.dialects import arith, bufferization, builtin, cf, func, math, memref, scf, tensor
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser

from .iface_dialect import GRAMMAR_VERSION, MERLIN_IFACE
from .mixed_matmul import LINALG_WITH_MIXED_MATMUL


class GrammarError(Exception):
    """The module is well-formed MLIR but not a grammar version this backend implements."""

_DIALECTS = (builtin.Builtin, func.Func, LINALG_WITH_MIXED_MATMUL, tensor.Tensor, arith.Arith,
             math.Math, scf.Scf, memref.MemRef, cf.Cf, bufferization.Bufferization)

# xDSL's current linalg.generic printer emits a parenthesized type list for
# multiple results, but its parser consumes an unparenthesized list.  Large
# upstream captures use the printer spelling for arg-reductions.  Normalize
# only this equivalent result-type spelling before structural parsing; no
# operation, operand, attribute, or result type is altered.
_MULTI_GENERIC_RESULTS = re.compile(
    r"(?m)^(\s*\}\s*->\s*)\((tensor<[^()\n]+>,\s*tensor<[^()\n]+>(?:,\s*tensor<[^()\n]+>)*)\)(\s*)$"
)


def _normalize_multi_generic_results(text: str) -> str:
    return _MULTI_GENERIC_RESULTS.sub(r"\1\2\3", text)


def context() -> Context:
    ctx = Context(allow_unregistered=True)
    for dialect in _DIALECTS:
        ctx.load_dialect(dialect)
    ctx.load_dialect(MERLIN_IFACE)
    return ctx


def parse_module(text: str) -> ModuleOp:
    """Parse + verify, and reject a grammar version this backend does not implement.

    The version gate belongs HERE, in `parse`: the contract says a consumer must reject a version
    it does not implement, and a module that only fails later has already been accepted.
    """
    module = Parser(context(), _normalize_multi_generic_results(text)).parse_module()
    module.verify()
    attr = module.attributes.get("merlin_iface.version")
    if attr is not None:
        version = getattr(attr, "data", None)
        if version != GRAMMAR_VERSION:
            raise GrammarError(
                f"merlin_iface version {version!r} is not implemented (this backend implements "
                f"{GRAMMAR_VERSION!r})")
    return module


def is_merlin_iface(module: ModuleOp) -> bool:
    return "merlin_iface.version" in module.attributes
