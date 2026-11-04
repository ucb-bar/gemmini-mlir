"""Exact source addresses for readout across contiguous static tensor views.

This adapter changes only the command's storage geometry. Source tensor ABI and
SSA ownership stay intact; numerical admission belongs to quantized_boundary.
A live floating intermediate is a refusal, so its host observer is preserved.
"""
from __future__ import annotations

from math import prod

from merlin.common.mlir_query import op_name


def static_view_source(op, result):
    """Prove a dense, static expand/collapse retains row-major linear order.

    Validate reassociation explicitly: equal element counts alone cannot prove
    an address mapping (and never admit a transpose, slice or arbitrary cast).
    """
    from .quantized_boundary import QuantizedBoundaryError, _type
    name = op_name(op)
    if (name not in ('tensor.expand_shape', 'tensor.collapse_shape')
            or len(op.operands) != 1 or len(op.results) != 1
            or op.results[0] is not result or op.regions):
        raise QuantizedBoundaryError('readout view is not a static single-source expand/collapse')
    source = op.operands[0]
    a, b = _type(source), _type(result)
    if (not a['dense'] or not b['dense'] or a['dtype'] != b['dtype']
            or a['dtype'] not in ('i32', 'f32') or not a['shape'] or not b['shape']
            or any(n <= 0 for n in (*a['shape'], *b['shape']))):
        raise QuantizedBoundaryError('readout view needs equal-type static non-scalar dense tensors')
    reassociation = op.properties.get('reassociation', op.attributes.get('reassociation'))
    try:
        groups = [[int(axis.value.data) for axis in group] for group in reassociation]
    except (TypeError, AttributeError, ValueError) as exc:
        raise QuantizedBoundaryError('readout view has no typed static reassociation') from exc
    expanded, collapsed = (b['shape'], a['shape']) if name == 'tensor.expand_shape' else (a['shape'], b['shape'])
    if (len(expanded) <= len(collapsed) or len(groups) != len(collapsed)
            or any(not group for group in groups)
            or [axis for group in groups for axis in group] != list(range(len(expanded)))
            or any(prod(expanded[axis] for axis in group) != size
                   for group, size in zip(groups, collapsed, strict=True))):
        raise QuantizedBoundaryError('readout view reassociation changes axes, volume or contiguous order')
    if name == 'tensor.expand_shape':
        shape = op.properties.get('static_output_shape', op.attributes.get('static_output_shape'))
        try:
            declared = list(shape.get_values())
        except (AttributeError, TypeError) as exc:
            raise QuantizedBoundaryError('readout expand has no complete static output shape') from exc
        if declared != b['shape']:
            raise QuantizedBoundaryError('readout expand static output shape differs from result type')
    return source


def quantized_view_candidate(op):
    """Return the existing standalone plan type, or None on an unproved domain.

    Intended as a source_groups hook after its ordinary quantizer matcher. The
    proof requires a view, exact existing full-i32-domain scale equivalence, a
    single function block and no externally observed member result. A refusal
    leaves every original source operation and output available to the host.
    """
    from .quantized_boundary import QuantizedBoundaryError, _type, prove_accumulator_quantize
    from .source_groups import _standalone_member, _standalone_capability
    try:
        proof = prove_accumulator_quantize(op, allow_static_views=True)
        views = [member for member in proof.operations
                 if op_name(member) in ('tensor.expand_shape', 'tensor.collapse_shape')]
        if not views:
            return None
        block = op.parent
        if block is None or getattr(block.parent_op(), 'name', None) != 'func.func':
            return None
        if any(member.parent is not block for member in proof.operations):
            return None
        proof.revalidate()
        count = prod(_type(proof.source)['shape'])
        shape = (1, count)
        diagnostic = proof.record()
        diagnostic['source_address_proof'] = {
            'schema': 'submission.quantized_contiguous_views.v1',
            'mapping': 'identical row-major linear element index',
            'storage_shape': list(shape),
            'views': [{'operation': op_name(view),
                       'source_shape': _type(view.operands[0])['shape'],
                       'result_shape': _type(view.results[0])['shape'],
                       'reassociation': str(view.properties.get('reassociation', view.attributes.get('reassociation'))),
                       'source_ids': {k: str(v) for k, v in view.attributes.items()
                                      if k.startswith(('prov.', 'merlin.source'))}}
                      for view in views],
            'source_tensor_abi': 'unchanged; command storage_shapes only',
        }
        candidate = _standalone_member(op, proof.operations, 'ACCUMULATOR_READOUT',
            {'src': proof.source}, proof.attributes, diagnostic,
            storage_shapes={proof.source: shape, op.results[0]: shape})
        if candidate is None:
            return None
        _standalone_capability(candidate)
        return candidate
    except (QuantizedBoundaryError, ValueError, TypeError):
        return None
