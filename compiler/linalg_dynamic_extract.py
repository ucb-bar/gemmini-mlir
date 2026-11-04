"""Source-proven runtime tensor.extract adapter for pointer-backed host tensors."""
from xdsl.dialects.builtin import TensorType, IndexType
from merlin.llvmlower.host_tensor_codegen import TensorVal, elem_name, tensor_shape
from merlin.llvmlower.dynamic_tensor_extract import load_element


def try_lower_scalar(host, op, ins, *, allow_static=False):
    """Return (handled, scalar), preserving the ordinary fallback on other ops."""
    if op.name != 'tensor.extract' or len(op.results) != 1 or not op.operands:
        return False, None
    value = op.operands[0]
    if not isinstance(value.type, TensorType) or value not in host.ptrs:
        return False, None
    shape = tensor_shape(value.type)
    if (len(op.operands) != len(shape)+1 or len(ins) != len(op.operands)
            or any(d < 0 for d in shape)
            or any(not isinstance(v.type, IndexType) for v in op.operands[1:])
            or op.results[0].type != value.type.element_type
            or not isinstance(ins[0], TensorVal)):
        return False, None
    # Leave fully static extracts on the established scalar route. In particular
    # a scalar value in an SSA-only computed tensor need not be spilled.
    if not allow_static and all(host.static_int(v) is not None for v in op.operands[1:]):
        return False, None
    source = ins[0]
    ptr = host.ptrs[value]
    # Lazy shared loads and completed counted/device results already reside at
    # their ABI address. An eager SSA-only producer must first publish its own
    # tensor buffer; never read its as-yet uninitialized result allocation.
    if (value not in host.materialized_results
            and getattr(source.elems, 'ptr', None) != ptr):
        host.store_result(ptr, source)
        host.materialized_results.add(value)
    result = load_element(host.fp, ptr, shape, elem_name(value.type.element_type),
                          ins[1:], row_alignment=host.row_alignment)
    return True, result


def try_lower(host, op):
    """Lower a tensor.extract used directly in the enclosing tensor program."""
    if op.name != 'tensor.extract':
        return False
    handled, value = try_lower_scalar(host, op, [host.get(v) for v in op.operands],
                                      allow_static=True)
    if handled:
        host.vals[op.results[0]] = value
    return handled
