"""Typed runtime scalar insertion into a fresh immutable packed tensor."""
from math import prod
from xdsl.dialects.builtin import TensorType, NoneAttr, IndexType, IntegerType
from merlin.llvmlower.host_tensor_codegen import TensorVal, tensor_shape, elem_name
from merlin.llvmlower.dynamic_tensor_insert import insert_element


def plan(op):
    if (op.name != 'tensor.insert' or len(op.results) != 1 or len(op.operands) < 2
            or op.regions or op.successors or op.properties
            or any(not k.startswith('prov.') for k in op.attributes)):
        return None
    scalar, source, *indices = op.operands
    result = op.results[0]
    if (not isinstance(source.type, TensorType) or source.type != result.type
            or not isinstance(source.type.encoding, NoneAttr)
            or scalar.type != source.type.element_type
            or any(not isinstance(v.type, IndexType) for v in indices)):
        return None
    shape = tuple(tensor_shape(source.type)); dtype = elem_name(source.type.element_type)
    sizes = {'i8': 1, 'i32': 4, 'i64': 8, 'f32': 4}
    if (len(indices) != len(shape) or dtype not in sizes
            or any(d <= 0 for d in shape) or prod(shape) * sizes[dtype] >= 1 << 63):
        return None
    return shape, dtype


def try_lower_index_cast(host, op):
    """Use existing scalar semantics for index preparation at tensor level."""
    if (op.name != 'arith.index_cast' or len(op.operands) != 1 or len(op.results) != 1
            or op.regions or op.successors or op.properties
            or any(not k.startswith('prov.') for k in op.attributes)
            or not isinstance(op.results[0].type, IndexType)
            or not isinstance(op.operands[0].type, IntegerType)
            or op.operands[0].type.width.data not in (8, 32, 64)
            or op.operands[0] not in host.vals):
        return False
    host.vals[op.results[0]] = host._scalar_op(op, [host.get(op.operands[0])])
    host.last_lowering = ('host', 'shared_scalar_index_preparation', 'runtime_index')
    return True


def try_lower(host, op):
    if try_lower_index_cast(host, op):
        return True
    geometry = plan(op)
    if geometry is None or host.row_alignment != 1:
        return False
    scalar, source, *indices = op.operands; result = op.results[0]
    shape, dtype = geometry
    if (source not in host.ptrs or result not in host.ptrs
            or any(v not in host.vals for v in op.operands)
            or result in getattr(host, 'aliasable_results', ())
            or result in getattr(host, 'storage_aliases', {})
            or any(v is not result and p is host.ptrs[result] for v, p in host.ptrs.items())):
        return False
    for index, extent in zip(indices, shape):
        known = host.static_int(index)
        if known is not None and not 0 <= known < extent:
            return False
    tensor = host.get(source)
    if not isinstance(tensor, TensorVal):
        return False
    ptr = host.ptrs[source]
    if (source not in host.materialized_results
            and getattr(tensor.elems, 'ptr', None) is not ptr):
        host.store_result(ptr, tensor)
        host.materialized_results.add(source)
    insert_element(host.fp, ptr, host.ptrs[result], shape, dtype,
                   [host.get(v) for v in indices], host.get(scalar))
    host.vals[result] = host.load_arg(host.ptrs[result], result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result, try_lower)
    host.last_lowering = ('host', 'shared_dynamic_scalar_insert', 'runtime_indexed_update')
    return True
