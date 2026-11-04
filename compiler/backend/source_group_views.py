"""Private contiguous source-view admission for existing device region storage.

The boundary tensor remains the materialized storage owner. In particular, a
slice or transpose producing that boundary is never reinterpreted as an alias of
its own input. Only proved collapse/expand operations inside the region forward
the boundary address, without moving arithmetic or inferring pointer adjacency.
"""
from __future__ import annotations

from .source_views import SourceViewError, prove_contiguous_reassociation


def private_input_view(op, owned, block):
    """Return exact view evidence, or None while retaining the original boundary.

    Every observer must belong to this one region. This also prevents two device
    regions from claiming a shared view or replacing an independently returned
    view's output writer. Tensor purity grants no mutable-memory lifetime claim.
    """
    from xdsl.ir import BlockArgument
    from xdsl.traits import is_side_effect_free
    from merlin.llvmlower.tensor_preparation_identity import canonical_tensor_read_view, validate_tensor_read_view

    if (getattr(op, 'name', '') not in ('tensor.collapse_shape', 'tensor.expand_shape')
            or op.parent is not block or not is_side_effect_free(op)
            or any(not key.startswith('prov.') for key in op.attributes)):
        return None
    try:
        proof = prove_contiguous_reassociation(op)
        if any(use.operation not in owned for result in op.results for use in result.uses):
            return None
        source = op.operands[0]
        positions = {operation: index for index, operation in enumerate(block.ops)}
        if isinstance(source, BlockArgument):
            if source.owner is not block:
                return None
        elif source.owner.parent is not block or positions[source.owner] >= positions[op]:
            return None
        # The shared identity snapshots types, source operations and ownership.
        # It deliberately does not claim physical aliasing through a producer.
        identity = canonical_tensor_read_view(op.results[0])
        validate_tensor_read_view(identity)
    except (SourceViewError, ValueError):
        return None
    return {**proof, 'source_ids': {key: str(value) for key, value in op.attributes.items()
                                  if key.startswith('prov.')},
            'lifetime': 'every result observer belongs to this device region',
            'effects': 'pure immutable tensor reassociation',
            'address': 'same dense materialized boundary pointer, zero byte offset'}


def input_view_chain(value, boundaries, owned):
    """Return boundary/root and ordered, revalidated private view operations."""
    steps = []
    while value not in boundaries and getattr(value.owner, 'name', '') in ('tensor.collapse_shape', 'tensor.expand_shape'):
        operation = value.owner
        proof = private_input_view(operation, owned, operation.parent)
        if operation not in owned or proof is None:
            raise SourceViewError('input view is not a proved private region dependency')
        steps.append((operation, proof))
        value = operation.operands[0]
    return value, tuple(reversed(steps))


def private_reassociation_successor(value):
    """Trace a private ordered view path; return the original on any refusal.

    Consumers can fuse only if the final type exactly equals the initial type.
    This helper grants no axis interpretation to an intervening arithmetic op.
    """
    original = value
    operations = []
    while len(uses := list(value.uses)) == 1:
        operation = uses[0].operation
        if operation.name not in ('tensor.collapse_shape', 'tensor.expand_shape'):
            break
        if not operation.operands or operation.operands[0] is not value or len(operation.results) != 1:
            return original, (), ()
        operations.append(operation)
        value = operation.results[0]
    if not operations or value.type != original.type or len(list(value.uses)) != 1:
        return original, (), ()
    owned = {*operations, next(iter(value.uses)).operation}
    proofs = []
    for operation in operations:
        proof = private_input_view(operation, owned, original.owner.parent)
        if proof is None:
            return original, (), ()
        proofs.append(proof)
    return value, tuple(operations), tuple(proofs)
