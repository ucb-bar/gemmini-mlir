"""Exact row-major tensor reassociation proofs, independent of storage allocation.

A proof permits the same element address and zero byte offset at both types.
It does not permit an independent output ABI pointer to alias an input pointer.
"""
from __future__ import annotations

from math import prod


class SourceViewError(ValueError):
    pass


def prove_contiguous_reassociation(op):
    """Return JSON-compatible shape/axis evidence for a static tensor view.

    Check each reassociation product ourselves: the supplied collapse operation
    verifier checks the index attribute but does not check per-group extents.
    Ordinary tensors have canonical contiguous logical element order; encoded,
    dynamic and non-reassociation operations cannot establish this proof.
    """
    from xdsl.dialects.builtin import TensorType, NoneAttr, ArrayAttr, IntegerAttr

    if op.name not in ('tensor.collapse_shape', 'tensor.expand_shape'):
        raise SourceViewError('only contiguous collapse/expand reassociation is an alias')
    if len(op.operands) != 1 or len(op.results) != 1:
        raise SourceViewError('view requires one static tensor source and result')

    def typ(value):
        t = value.type
        if not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr):
            raise SourceViewError('view requires ordinary contiguous ranked tensors')
        shape = list(t.get_shape())
        if any(d <= 0 for d in shape):
            raise SourceViewError('view requires positive static extents')
        return shape, str(t.element_type)

    source_shape, dtype = typ(op.operands[0])
    shape, result_dtype = typ(op.results[0])
    if dtype != result_dtype or prod(source_shape) != prod(shape):
        raise SourceViewError('view changes element count or dtype')
    attr = op.properties.get('reassociation')
    if not isinstance(attr, ArrayAttr) or any(
        not isinstance(group, ArrayAttr) or any(not isinstance(i, IntegerAttr) for i in group)
        for group in attr
    ):
        raise SourceViewError('view lacks typed reassociation indices')
    groups = [[i.value.data for i in group] for group in attr]
    collapse = op.name == 'tensor.collapse_shape'
    expanded, collapsed = (source_shape, shape) if collapse else (shape, source_shape)
    if not collapsed:
        # MLIR permits collapsing a unit tensor to a scalar with no groups.
        if groups or any(d != 1 for d in expanded):
            raise SourceViewError('rank-zero reassociation requires only unit extents')
    elif (len(groups) != len(collapsed) or any(not group for group in groups)
          or [i for group in groups for i in group] != list(range(len(expanded)))):
        raise SourceViewError('reassociation must partition every expanded axis in order')
    elif any(prod(expanded[i] for i in group) != extent for group, extent in zip(groups, collapsed, strict=True)):
        raise SourceViewError('reassociation group product differs from collapsed extent')
    if not collapse:
        static = op.properties.get('static_output_shape')
        if static is None or list(static.get_values()) != shape:
            raise SourceViewError('expand output_shape differs from its static result')
    try:
        op.verify()
    except Exception as exc:
        raise SourceViewError(f'invalid tensor reassociation: {exc}') from exc
    return {'operation': op.name, 'source_shape': source_shape, 'shape': shape,
            'dtype': dtype, 'reassociation': groups, 'layout': 'row_major',
            'byte_offset': 0}
