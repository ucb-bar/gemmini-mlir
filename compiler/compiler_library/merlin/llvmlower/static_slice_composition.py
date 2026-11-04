"""Exact adjacent private slice composition; no arithmetic or policy grants.

The surviving result and every observer retain their SSA identity. Admission
uses the shared immutable tensor coordinate proof, with no layout guessing.
Callers own source-stage permission and source identity binding.
"""

from __future__ import annotations

import hashlib

from xdsl.dialects import tensor
from xdsl.dialects.builtin import DenseArrayBase, i64

from .tensor_preparation_identity import canonical_tensor_read_view, validate_tensor_read_view

ENABLED = False


def compose_static_slices(module):
    """Mutate only checked adjacent same-rank slice pairs; return source indices.

    Indices refer to the original module walk, before any erasure. Unknown
    attributes, rank reductions, dynamic operands and visible intermediates
    refuse. A returned final slice is allowed: its exact result is preserved.
    """
    original = tuple(module.walk())
    positions = {op: index for index, op in enumerate(original)}
    records = []
    for second in original:
        if not isinstance(second, tensor.ExtractSliceOp) or second.parent is None:
            continue
        first = second.source.owner
        if not isinstance(first, tensor.ExtractSliceOp) or first.parent is not second.parent:
            continue
        # Restrict to exactly two views, with no crossed operation/effect.
        if isinstance(first.source.owner, tensor.ExtractSliceOp) or first.next_op is not second:
            continue
        uses = tuple(first.results[0].uses)
        if len(uses) != 1 or uses[0].operation is not second:
            continue
        if any(not name.startswith("prov.") for op in (first, second) for name in op.attributes):
            continue
        if any(set(op.properties) - {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
               for op in (first, second)):
            continue
        try:
            before = canonical_tensor_read_view(second.results[0])
            validate_tensor_read_view(before)
        except ValueError:
            continue
        if before.root is not first.source:
            continue
        result = second.results[0]
        observers = tuple((use.operation, use.index) for use in result.uses)
        second.operands = (before.root,)
        second.properties["static_offsets"] = DenseArrayBase.from_list(i64, before.offsets)
        second.properties["static_strides"] = DenseArrayBase.from_list(i64, before.strides)
        after = canonical_tensor_read_view(result)
        if after.identity != before.identity or tuple((u.operation, u.index) for u in result.uses) != observers:
            raise ValueError("slice composition failed exact coordinate/observer preservation")
        second.parent.erase_op(first)
        records.append({
            "removed_source_op_index": positions[first],
            "retained_source_op_index": positions[second],
            "offsets": list(before.offsets), "sizes": list(before.sizes),
            "strides": list(before.strides),
            "proof": "shared_exact_tensor_read_view_and_unchanged_result_observers",
        })
    module.verify()
    return {"schema": "merlin.static_slice_composition.v1", "compositions": records}


def compose_source(source):
    """Explicit source adapter; unchanged cases retain exact input bytes."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    module = parse_mlir_text(source)
    module.verify()
    report = compose_static_slices(module)
    selected = text(module, generic=True) if report["compositions"] else source
    report.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  transformed_sha256=hashlib.sha256(selected.encode()).hexdigest())
    return selected, report
