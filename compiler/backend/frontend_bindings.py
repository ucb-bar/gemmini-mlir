"""Exact read-only storage bindings for source tensor matrix reshapes."""
from __future__ import annotations

from math import prod


def _readonly_matrix_reshape(value, views, entry_arguments, tensor_type, *, dtype, role, require_entry=True):
    """Resolve verified contiguous reshapes to an original immutable input.

    Tensor collapse/expand preserves linear element order. The selected logical
    pointer ABI stores unencoded static tensors contiguously, so this binding
    changes only the rank/shape used for read-only matrix loads. No
    output storage or earlier computed tensor receives an alias permission.
    """
    from xdsl.dialects import tensor

    result_type = tensor_type(value)
    if result_type['dtype'] != dtype or len(result_type['shape']) != 2:
        raise ValueError(f'{role} reshape requires a rank-two {dtype} result')
    current, seen = value, set()
    while current in views:
        if current in seen:
            raise ValueError(f'{role} reshape contains an alias cycle')
        seen.add(current)
        operation = current.owner
        if type(operation) not in (tensor.CollapseShapeOp, tensor.ExpandShapeOp):
            raise ValueError(f'{role} alias is not an ordinary source reshape')
        allowed = ({'reassociation'} if type(operation) is tensor.CollapseShapeOp
                   else {'reassociation', 'static_output_shape'})
        if (set(operation.properties) != allowed
                or any(not key.startswith('prov.') for key in operation.attributes)
                or len(operation.operands) != 1 or len(operation.results) != 1
                or operation.results[0] is not current
                or operation.operands[0] is not views[current]):
            raise ValueError(f'{role} reshape has unsupported metadata or dynamic operands')
        operation.verify()
        source_type, target_type = tensor_type(operation.operands[0]), tensor_type(current)
        if (source_type['dtype'] != dtype or target_type['dtype'] != dtype
                or prod(source_type['shape']) != prod(target_type['shape'])):
            raise ValueError(f'{role} reshape changes element type or count')
        expanding = type(operation) is tensor.ExpandShapeOp
        lower = source_type['shape'] if expanding else target_type['shape']
        higher = target_type['shape'] if expanding else source_type['shape']
        groups = tuple(tuple(index.value.data for index in group)
                       for group in operation.reassociation)
        if (not lower or len(groups) != len(lower) or any(not group for group in groups)
                or tuple(index for group in groups for index in group) != tuple(range(len(higher)))
                or any(prod(higher[index] for index in group) != extent
                       for group, extent in zip(groups, lower, strict=True))):
            raise ValueError(f'{role} reshape reassociation must preserve contiguous source axes')
        if expanding and tuple(operation.static_output_shape.get_values()) != tuple(target_type['shape']):
            raise ValueError(f'{role} reshape static dimensions differ from its result type')
        current = operation.operands[0]
    if not seen or (require_entry and current not in entry_arguments):
        raise ValueError(f'{role} reshape requires original read-only entry storage')
    return current, result_type


def readonly_initializer_reshape(value, views, entry_arguments, tensor_type):
    """Retain the exact i32 initializer admission and its existing diagnostics."""
    return _readonly_matrix_reshape(value, views, entry_arguments, tensor_type,
                                    dtype='i32', role='initializer')


def legacy_returned_matrix_view(value, views, entry_arguments, tensor_type,
                                returned_values, *, dtype):
    """Keep an existing same-shaped computed return's ordinary materialization.

    This grants no alias: the caller uses its established computed-storage and
    output writers. Every reshape is still verified before the legacy path.
    """
    if value not in returned_values:
        return False
    root, spec = _readonly_matrix_reshape(value, views, entry_arguments, tensor_type,
        dtype=dtype, role='returned matrix', require_entry=False)
    return root not in entry_arguments and tensor_type(root) == spec


def bind_matmul_operand(value, role, *, views, entry_arguments, tensor_type,
                        entry_bindings, source_owners, tensors, aliases,
                        output_names, returned_values, materialize):
    """Bind only an ordinary i8 matmul operand view to original input storage.

    Existing non-view materialization and its order remain caller-owned. A
    dedicated role/view binding never reads or changes ordinary view caches,
    returned-output aliases, initializer aliases or the original pointer ABI.
    """
    if role not in ('lhs', 'rhs'):
        raise ValueError('matrix operand requires an explicit lhs or rhs role')
    if value not in views:
        return materialize(value)
    if legacy_returned_matrix_view(value, views, entry_arguments, tensor_type,
                                   returned_values, dtype='i8'):
        return materialize(value)
    root, spec = _readonly_matrix_reshape(value, views, entry_arguments, tensor_type,
                                         dtype='i8', role=f'matmul {role}')
    index, result_index = source_owners[value]
    name = f'_matmul_{role}_view_{index}_{result_index}'
    binding = {'source': entry_bindings[root], **spec}
    if name in aliases:
        if aliases[name] != binding:
            raise ValueError('matrix operand view storage binding changed')
        return name
    if name in tensors or name in output_names:
        raise ValueError('matrix operand view storage name collides with a source tensor')
    aliases[name] = binding
    return name
