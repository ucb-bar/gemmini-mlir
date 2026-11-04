"""Explicit out-of-line placement of a proved pure source continuation.

The caller supplies exact typed arithmetic and existing floating effect policy.
This changes only function placement attributes. It adds no arithmetic, alias,
exception or target permission and does not decide profitability.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .late_quant_rne import _functions, _tokens
from .source_expression_interval import IntervalEffectContract, ScalarExpression
from .source_expression_interval_llvm import _compiled_tree, _extended_instruction, _typed_tree


@dataclass(frozen=True)
class SourceContinuationBinding:
    function: str
    expression: ScalarExpression


def outline_source_continuations(source, *, bindings=(), expected_source_sha256=None, effects=None):
    """Place complete source-exact scalar continuations out of line explicitly.

    Only one-input binary32 pure straight-line source functions are supported.
    Every operation must belong to the exact typed arithmetic DAG, with no dead
    operations, unknown calls, memory, control flow, fastmath or strict context.
    Unrecognized function attributes refuse. The unchanged body/signature and
    original uses remain authoritative; all selections validate transactionally.
    The normal build must verify the returned complete LLVM and reclose its ABI.
    Empty selection preserves any source bytes without parsing or policy.
    """
    digest = hashlib.sha256(source.encode()).hexdigest()
    report = {"schema": "source_continuation_outline_v1", "source_sha256": digest, "routes": []}
    if not bindings:
        return source, report
    if expected_source_sha256 != digest:
        raise ValueError("source witness changed")
    if not isinstance(effects, IntervalEffectContract):
        raise ValueError("explicit source floating effect contract required")
    effects.validate()
    tokens = _tokens(source)
    if any(t.text == "strictfp" or t.text.startswith("@llvm.experimental.constrained.") for t in tokens):
        raise ValueError("strict or constrained source context unsupported")
    extents = {}
    for body in _functions(tokens):
        start = max(t.start for t in tokens if t.text == "define" and t.start < body[0].start)
        opening = max((t for t in tokens if t.text == "{" and t.start < body[0].start), key=lambda t: t.start)
        header = [t for t in tokens if start <= t.start < opening.start]
        names = [t.text for t in header if t.text.startswith("@")]
        if len(names) != 1 or names[0] in extents:
            raise ValueError("ambiguous source function definition")
        extents[names[0]] = (header, opening, body)
    edits, selected = [], set()
    for binding in bindings:
        if not isinstance(binding, SourceContinuationBinding) or not isinstance(binding.expression, ScalarExpression):
            raise ValueError("typed source continuation binding required")
        name = binding.function
        if (
            not isinstance(name, str)
            or not name
            or name[0].isdigit()
            or not all(c.isascii() and (c.isalnum() or c in "_.$-") for c in name)
        ):
            raise ValueError("simple explicit source symbol required")
        symbol = "@" + name
        if symbol in selected or symbol not in extents:
            raise ValueError("duplicate or unavailable source definition")
        selected.add(symbol)
        binding.expression.validate()
        header, opening, body = extents[symbol]
        h = [t.text for t in header]
        if (
            len(h) < 7
            or h[:4] != ["define", "float", symbol, "("]
            or h[4] != "float"
            or not h[5].startswith("%")
            or h[6] != ")"
        ):
            raise ValueError("plain single-input binary32 source function required")
        attributes = h[7:]
        if any(attribute not in ("alwaysinline", "noinline", "cold") for attribute in attributes):
            raise ValueError("unsupported source function attribute/context")
        definitions, returned = {}, []
        lines = source[opening.end : body[-1].end].splitlines()
        for line in lines:
            statement = _tokens(line)
            if not statement:
                continue
            words = [t.text for t in statement]
            if words[:2] == ["ret", "float"] and len(words) == 3:
                returned.append(words[2])
                continue
            operation = _extended_instruction(statement, 0)
            if operation is None or operation.result in definitions:
                raise ValueError("unsupported or duplicate source operation")
            definitions[operation.result] = operation
        if len(returned) != 1:
            raise ValueError("single binary32 source return required")
        tree, nodes = _compiled_tree(returned[0], definitions, h[5])
        if tree != _typed_tree(binding.expression) or nodes != set(definitions):
            raise ValueError("complete source arithmetic DAG changed")
        # Original instructions and their textual ordering remain untouched.
        edits.append((header[6].end, opening.start, " noinline cold "))
        report["routes"].append(
            {
                "function": name,
                "expression_sha256": binding.expression.canonical_sha256,
                "original_header_sha256": hashlib.sha256(source[header[0].start : opening.start].encode()).hexdigest(),
                "original_body_sha256": hashlib.sha256(source[opening.end : body[-1].end].encode()).hexdigest(),
                "complete_pure_source_DAG": True,
                "source_operations_order_signature_uses_unchanged": True,
                "numeric_effect_alias_permissions_added": False,
            }
        )
    changed = source
    for left, right, replacement in sorted(edits, reverse=True):
        changed = changed[:left] + replacement + changed[right:]
    report["rewritten_sha256"] = hashlib.sha256(changed.encode()).hexdigest()
    return changed, report
