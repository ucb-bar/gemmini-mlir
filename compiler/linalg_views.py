"""Storage-preserving tensor views for the source-owned linalg driver.

The driver chooses which results may share storage and publishes those aliases
in its source/ABI plan. Observable outputs retain their own destination buffers.
"""
from math import prod

from merlin.llvmlower.host_tensor_codegen import tensor_shape, attr_of


_VIEW_OPS = frozenset((
    'tensor.collapse_shape', 'tensor.expand_shape', 'tensor.reshape', 'tensor.cast',
    'tensor.extract_slice',
))


def _slice_offset(op):
    """Return an element offset only for a contiguous static rectangular view."""
    if len(op.operands) != 1:
        return None
    source_shape = tensor_shape(op.operands[0].type)
    result_shape = tensor_shape(op.results[0].type)
    attrs = [attr_of(op, name) for name in ('static_offsets', 'static_sizes', 'static_strides')]
    if any(attr is None for attr in attrs):
        return None
    offsets, sizes, strides = [tuple(map(int, attr.get_values())) for attr in attrs]
    if (any(len(items) != len(source_shape) for items in (offsets, sizes, strides))
            or any(size <= 0 or offset < 0 or stride <= 0
                   or offset+(size-1)*stride >= extent
                   for offset,size,stride,extent in zip(offsets,sizes,strides,source_shape))):
        return None
    retained = 0
    for size in sizes:
        if retained < len(result_shape) and size == result_shape[retained]:
            retained += 1
        elif size != 1:
            return None
    if retained != len(result_shape):
        return None
    # If an axis varies, each inner axis must cover its complete extent so
    # adjacent output elements are adjacent in source storage.
    varying = False
    for offset, size, stride, extent in zip(offsets, sizes, strides, source_shape):
        if varying and (offset != 0 or size != extent):
            return None
        if size > 1:
            if stride != 1:
                return None
            varying = True
    return sum(offset*prod(source_shape[i+1:]) for i,offset in enumerate(offsets))


def is_aliasable(op):
    """Prove a packed, immutable intermediate view needs no physical buffer."""
    if op.name not in _VIEW_OPS or len(op.results) != 1 or not op.operands:
        return False
    source, result = op.operands[0], op.results[0]
    if (not hasattr(source.type, 'element_type')
            or not hasattr(result.type, 'element_type')
            or source.type.element_type != result.type.element_type):
        return False
    # Ranked tensor values have immutable, packed row-major logical elements.
    # Encoded tensor layouts need their own layout proof, not an element count.
    for value in (source, result):
        encoding = getattr(value.type, 'encoding', None)
        if encoding is not None and str(encoding) not in ('none', '#builtin.none'):
            return False
    source_shape, result_shape = tensor_shape(source.type), tensor_shape(result.type)
    if any(d < 0 for d in (*source_shape, *result_shape)):
        return False
    if op.name == 'tensor.extract_slice':
        if _slice_offset(op) is None:
            return False
    elif prod(source_shape) != prod(result_shape):
        return False
    # A direct return is observable even if mistakenly admitted by the driver.
    if any(use.operation.name == 'func.return' for use in result.uses):
        return False
    return True


def try_lower(host, op):
    """Alias a view admitted by the driver's matching ABI declaration scan.

    ``host.aliasable_results`` contains only intermediate SSA values omitted
    from physical source/ABI bindings. The lazy value carries the result shape;
    accelerator consumers use the unchanged pointer through ``host.ptrs``.
    """
    if not is_aliasable(op) or getattr(host, 'row_alignment', 1) != 1:
        return False
    source, result = op.operands[0], op.results[0]
    if (result not in getattr(host, 'aliasable_results', ())
            or source not in host.ptrs):
        return False
    ptr = host.ptrs[source]
    if op.name == 'tensor.extract_slice':
        offset = _slice_offset(op)
        if offset:
            ptr = host.fp.fb.gep(ptr, host.fp.fb.const(offset), source.type.element_type)
    host.ptrs[result] = ptr
    host.vals[result] = host.load_arg(ptr, result.type)
    host.materialized_results.add(result)
    aliases = getattr(host, 'storage_aliases', None)
    if aliases is None:
        host.storage_aliases = aliases = {}
    aliases[result] = source
    return True
