"""Exact static tensor-layout proofs for value-preserving device DMA copies.

These rules materialize distinct SSA results: they do not infer pointer aliases,
ignore dynamic index operands, or fold an observable producer out of the plan.
"""
from __future__ import annotations
from math import prod

from merlin.targetgen.contract import linalg_iface as li


def _shape(value):
    shape = li._shape(value.type)
    return shape if shape and all(type(d) is int and d > 0 for d in shape) else None


def _integers(attribute):
    if attribute is None:
        return None
    if hasattr(attribute, 'get_values'):
        return [int(value) for value in attribute.get_values()]
    try:
        return [int(value.value.data) for value in attribute]
    except (TypeError, AttributeError, ValueError):
        return None


def _row_strides(shape):
    return [prod(shape[i + 1:]) for i in range(len(shape))]


def _reshape_proof(kind, attrs, source_shape, result_shape):
    groups_attr = attrs.get('reassociation')
    if groups_attr is None:
        return False
    groups = [_integers(group) for group in groups_attr]
    if any(not group for group in groups):
        return False
    expanded, collapsed = ((source_shape, result_shape) if kind == 'tensor.collapse_shape'
                           else (result_shape, source_shape))
    if len(groups) != len(collapsed) or [i for group in groups for i in group] != list(range(len(expanded))):
        return False
    return all(prod(expanded[i] for i in group) == extent for group, extent in zip(groups, collapsed))


def layout_device_semantics(op):
    """Return MOVEMENT inputs/attrs only with an exact contiguous-layout proof."""
    kind = li._op_name(op)
    if len(op.results) != 1 or not op.operands:
        return None
    source, result = op.operands[0], op.results[0]
    source_shape, result_shape = _shape(source), _shape(result)
    if source_shape is None or result_shape is None:
        return None
    dtype = li._dtype(source.type)
    # Full-width accumulator DMA copies f32 as an opaque 32-bit payload. No
    # arithmetic conversion is involved, including for NaNs and signed zero.
    if dtype not in ('i8', 'i32', 'f32') or li._dtype(result.type) != dtype:
        return None
    attrs = {**op.attributes, **op.properties}
    offset = 0
    proof = None
    source_strides = None
    if kind in ('tensor.collapse_shape', 'tensor.expand_shape'):
        if len(op.operands) != 1 or not _reshape_proof(kind, attrs, source_shape, result_shape):
            return None
        proof = 'static ordered reassociation preserves contiguous row-major element order'
    elif kind == 'tensor.cast':
        if len(op.operands) != 1 or source_shape != result_shape:
            return None
        proof = 'static tensor cast preserves exact shape, dtype and element order'
    elif kind == 'tensor.extract_slice':
        if len(op.operands) != 1:
            return None
        offsets = _integers(attrs.get('static_offsets'))
        sizes = _integers(attrs.get('static_sizes'))
        strides = _integers(attrs.get('static_strides'))
        rank = len(source_shape)
        if any(values is None or len(values) != rank for values in (offsets, sizes, strides)):
            return None
        if any(off < 0 or size <= 0 or stride <= 0 or off + (size - 1) * stride >= dim
               for off, size, stride, dim in zip(offsets, sizes, strides, source_shape)):
            return None
        # Tensor rank reduction can drop only unit-size axes. Removing units
        # from both shapes gives a canonical, ambiguity-free order check.
        if (len(result_shape) > rank or [d for d in sizes if d != 1] != [d for d in result_shape if d != 1]
            or prod(sizes) != prod(result_shape)):
            return None
        physical = _row_strides(source_shape)
        span = 1 + sum((size - 1) * stride * step for size, stride, step in zip(sizes, strides, physical))
        offset = sum(off * step for off, step in zip(offsets, physical))
        if span == prod(sizes):
            proof = 'static in-bounds slice visits one contiguous row-major interval'
        else:
            # Rank-reducing slices remove only unit axes. Nonunit axes retain
            # their order; coordinates of any retained unit axes are always0.
            live_steps = iter(stride * step for size, stride, step in zip(sizes, strides, physical) if size != 1)
            source_strides = [next(live_steps) if size != 1 else 1 for size in result_shape]
            proof = 'static in-bounds positive-stride slice maps exact result axes to source strides'
    elif kind == 'linalg.copy':
        inputs, outputs = li._ins_outs(op)
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0] is not source or source_shape != result_shape:
            return None
        proof = 'named linalg copy preserves shape, dtype and order'
    elif kind == 'linalg.transpose':
        inputs, outputs = li._ins_outs(op)
        perm = _integers(attrs.get('permutation'))
        if (len(inputs) != 1 or len(outputs) != 1 or perm is None
            or sorted(perm) != list(range(len(source_shape)))
            or [source_shape[i] for i in perm] != result_shape):
            return None
        if [i for i in perm if source_shape[i] != 1] == [i for i, d in enumerate(source_shape) if d != 1]:
            proof = 'permutation moves unit axes only and preserves contiguous element order'
        else:
            physical = _row_strides(source_shape)
            source_strides = [physical[i] for i in perm]
            proof = 'static axis permutation maps result coordinates exactly to source row-major strides'
    else:
        return None
    count = prod(result_shape)
    maximum_index = (offset + count - 1 if source_strides is None else
                     offset + sum((dim - 1) * stride for dim, stride in zip(result_shape, source_strides)))
    if maximum_index >= prod(source_shape):
        return None
    descriptor = {'schema': 'contiguous_layout_copy_v1' if source_strides is None else 'strided_layout_copy_v1',
                  'source_offset_elements': offset, 'elements': count,
                  'source_shape': source_shape, 'result_shape': result_shape}
    if source_strides is not None:
        descriptor['source_strides_elements'] = source_strides
    return 'MOVEMENT', {'src': source}, {'output_dtype': dtype, 'layout_copy': descriptor, 'layout_proof': proof}


def layout_device_program(op):
    """A sequence of real DMA copies for exact fill, concat and insert semantics.

    Each sequence belongs to one source task and materializes one distinct result.
    Later commands may fill disjoint parts or patch an initialized destination.
    """
    kind = li._op_name(op)
    if len(op.results) != 1:
        return None
    result = op.results[0]
    result_shape = _shape(result)
    dtype = li._dtype(result.type)
    if result_shape is None or dtype not in ('i8', 'i32', 'f32'):
        return None
    attrs = {**op.attributes, **op.properties}
    if kind == 'tensor.splat' and len(op.operands) == 1:
        scalar = op.operands[0]
        if li._dtype(scalar.type) != dtype:
            return None
        value = li._scalar_const(scalar)
        if dtype == 'f32':
            from math import isfinite
            import struct
            # Finite f32 literals round-trip through binary32 exactly, including
            # negative zero. Leave NaN literals on the host: a numeric parser
            # need not preserve their payload bits.
            if type(value) is not float or not isfinite(value):
                return None
            value = struct.unpack('<i', struct.pack('<f', value))[0]
        else:
            if type(value) is not int:
                return None
            bits = 8 if dtype == 'i8' else 32
            value = (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
        return [({}, {'output_dtype': dtype,
                      'layout_fill': {'schema': 'constant_layout_fill_v1', 'value': value,
                                      'result_shape': result_shape, 'dtype': dtype},
                      'layout_proof': 'typed source scalar constant is replicated exactly into every result element'})]
    if kind == 'tensor.concat':
        dim_attr = attrs.get('dim')
        axis = getattr(getattr(dim_attr, 'value', None), 'data', None)
        rank = len(result_shape)
        if type(axis) is not int or axis < 0 or axis >= rank or not op.operands:
            return None
        shapes = [_shape(value) for value in op.operands]
        if any(shape is None or len(shape) != rank or li._dtype(value.type) != dtype
               for shape, value in zip(shapes, op.operands)):
            return None
        if any(shape[i] != result_shape[i] for shape in shapes for i in range(rank) if i != axis):
            return None
        if sum(shape[axis] for shape in shapes) != result_shape[axis]:
            return None
        target_strides = _row_strides(result_shape)
        offset, commands = 0, []
        for value, shape in zip(op.operands, shapes):
            descriptor = {'schema': 'indexed_layout_copy_v1', 'source_shape': shape,
                          'result_shape': result_shape, 'iteration_shape': shape,
                          'source_offset_elements': 0, 'destination_offset_elements': offset,
                          'source_strides_elements': _row_strides(shape),
                          'destination_strides_elements': target_strides}
            commands.append(({'src': value}, {'output_dtype': dtype, 'layout_copy': descriptor,
                                              'layout_proof': 'static concat partitions the destination into exact ordered source regions'}))
            offset += shape[axis] * target_strides[axis]
        return commands
    if kind == 'tensor.insert_slice' and len(op.operands) == 2:
        source, destination = op.operands
        source_shape, destination_shape = _shape(source), _shape(destination)
        if (source_shape is None or destination_shape != result_shape
            or any(li._dtype(value.type) != dtype for value in (source, destination))):
            return None
        offsets = _integers(attrs.get('static_offsets'))
        sizes = _integers(attrs.get('static_sizes'))
        strides = _integers(attrs.get('static_strides'))
        rank = len(result_shape)
        if any(values is None or len(values) != rank for values in (offsets, sizes, strides)):
            return None
        if any(off < 0 or size <= 0 or stride <= 0 or off + (size - 1) * stride >= dim
               for off, size, stride, dim in zip(offsets, sizes, strides, result_shape)):
            return None
        if (len(source_shape) > rank or prod(source_shape) != prod(sizes)
            or [d for d in source_shape if d != 1] != [d for d in sizes if d != 1]):
            return None
        target_strides = _row_strides(result_shape)
        live_steps = iter(step * stride for size, step, stride in zip(sizes, target_strides, strides) if size != 1)
        destination_steps = [next(live_steps) if size != 1 else 1 for size in source_shape]
        initialize = {'schema': 'contiguous_layout_copy_v1', 'source_shape': result_shape,
                      'result_shape': result_shape, 'source_offset_elements': 0, 'elements': prod(result_shape)}
        patch = {'schema': 'indexed_layout_copy_v1', 'source_shape': source_shape,
                 'result_shape': result_shape, 'iteration_shape': source_shape,
                 'source_offset_elements': 0,
                 'destination_offset_elements': sum(off * step for off, step in zip(offsets, target_strides)),
                 'source_strides_elements': _row_strides(source_shape),
                 'destination_strides_elements': destination_steps}
        proof = 'copy original destination then replace exactly the statically indexed insertion region'
        return [({'src': destination}, {'output_dtype': dtype, 'layout_copy': initialize, 'layout_proof': proof}),
                ({'src': source}, {'output_dtype': dtype, 'layout_copy': patch, 'layout_proof': proof})]
    return None
