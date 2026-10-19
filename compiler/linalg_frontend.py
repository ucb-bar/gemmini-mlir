"""Structural frontend for tensor contractions and storage-preserving views."""
from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, StringAttr, TensorType
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.math import Math
from xdsl.dialects.scf import Scf
from xdsl.dialects.tensor import Tensor
from xdsl.ir.affine import AffineDimExpr
from xdsl.parser import Parser


def make_context():
    context = Context(allow_unregistered=True)
    for dialect in (Builtin, Func, Arith, Linalg, Tensor, Math, Scf, Cf):
        context.load_dialect(dialect)
    return context


def parse_module(text):
    return Parser(make_context(), text).parse_module()


def op_name(operation):
    if getattr(operation, 'name', '') == 'builtin.unregistered':
        name = operation.attributes.get('op_name__')
        if isinstance(name, StringAttr):
            return name.data
    return getattr(operation, 'name', '')


def contraction_kind(operation):
    """Check indexing semantics and scalar arithmetic, independent of provenance tags."""
    name = op_name(operation)
    if name in ('linalg.matmul', 'linalg.quantized_matmul'):
        return 'MATMUL'
    if name == 'linalg.matmul_transpose_b':
        return 'ATTENTION_QK'
    if name == 'linalg.batch_matmul':
        return 'BATCHED_MATMUL'
    if name != 'linalg.generic':
        return None
    maps_attr = operation.properties.get('indexing_maps')
    if maps_attr is None or len(maps_attr) != 3:
        return None
    maps = [item.data for item in maps_attr]
    dimensions = []
    for mapping in maps:
        if any(not isinstance(expr, AffineDimExpr) for expr in mapping.results):
            return None
        dimensions.append(tuple(expr.position for expr in mapping.results))
    iterator_types = operation.properties.get('iterator_types')
    iterators = [item.data.value for item in iterator_types] if iterator_types is not None else []
    if len(dimensions[0]) < 2 or len(dimensions[0]) != len(dimensions[1]):
        return None
    rank = len(dimensions[0])
    batch = tuple(range(rank - 2))
    m, n, k = rank - 2, rank - 1, rank
    ordinary = [batch + (m, k), batch + (k, n), batch + (m, n)]
    transposed = [batch + (m, k), batch + (n, k), batch + (m, n)]
    if iterators != ['parallel'] * rank + ['reduction']:
        return None
    if dimensions != ordinary and dimensions != transposed:
        return None
    block = operation.regions[0].block
    inner = list(block.ops)
    arithmetic = [op_name(item) for item in inner]
    accepted = {'arith.extsi', 'arith.extui', 'arith.muli', 'arith.addi', 'linalg.yield'}
    if any(item not in accepted for item in arithmetic):
        return None
    if arithmetic.count('arith.muli') != 1 or arithmetic.count('arith.addi') != 1:
        return None
    multiply = next(item for item in inner if op_name(item) == 'arith.muli')
    addition = next(item for item in inner if op_name(item) == 'arith.addi')
    terminator = inner[-1]
    if multiply.results[0] not in addition.operands or block.args[2] not in addition.operands:
        return None
    if op_name(terminator) != 'linalg.yield' or tuple(terminator.operands) != tuple(addition.results):
        return None
    for actual, expected in zip(multiply.operands, block.args[:2]):
        owner = actual.owner
        if op_name(owner) in ('arith.extsi', 'arith.extui'):
            actual = owner.operands[0]
        if actual != expected:
            return None
    if rank > 2:
        if dimensions != ordinary:
            return None
        return 'BATCHED_MATMUL'
    return 'MATMUL' if dimensions == ordinary else 'ATTENTION_QK'


def parse_linalg(text):
    try:
        from .frontend import add_pointer_abi, tensor_spec
    except ImportError:
        from frontend import add_pointer_abi, tensor_spec
    module = parse_module(text)
    functions = [operation for operation in module.body.block.ops if isinstance(operation, FuncOp)]
    if len(functions) != 1:
        raise ValueError('Expected exactly one tensor entry function')
    function = functions[0]
    block = function.body.block
    buffer = {'abi_version': '0.1', 'target': 'gemmini', 'tensors': {}, 'commands': []}
    names = {}
    for ordinal, argument in enumerate(block.args):
        name = argument.name_hint or ('arg' + str(ordinal))
        names[argument] = name
        buffer['tensors'][name] = tensor_spec(argument.type, 'input')
    operations = list(block.ops)
    returns = [operation for operation in operations if op_name(operation) == 'func.return']
    if len(returns) != 1:
        raise ValueError('Tensor entry requires one return')
    final_values = list(returns[0].operands)
    aliases = {}
    ignored = {'tensor.empty', 'arith.constant', 'tensor.splat', 'linalg.fill'}
    for operation in operations:
        if op_name(operation) in ('tensor.collapse_shape', 'tensor.expand_shape'):
            source, result = operation.operands[0], operation.results[0]
            if source.type.element_type != result.type.element_type:
                raise ValueError('A storage view cannot change element dtype')
            from math import prod
            if prod(source.type.get_shape()) != prod(result.type.get_shape()):
                raise ValueError('A storage view cannot change element count')
            aliases[result] = source
    def origin(value):
        while value in aliases:
            value = aliases[value]
        return value
    finals = {origin(value): ('Y' + str(index), value.type)
              for index, value in enumerate(final_values)}
    for ordinal, operation in enumerate(operations):
        name = op_name(operation)
        if name in ignored or name in ('tensor.collapse_shape', 'tensor.expand_shape', 'func.return'):
            continue
        kind = contraction_kind(operation)
        if kind is None:
            raise ValueError('Unsupported typed tensor operation ' + name)
        inputs = list(operation.inputs)
        if len(inputs) < 2 or len(operation.results) != 1:
            raise ValueError('A contraction requires two data tensors and one result')
        lhs, rhs = inputs[:2]
        if str(lhs.type.element_type) != 'i8' or str(rhs.type.element_type) != 'i8':
            raise ValueError('Accelerator contraction inputs must be i8')
        if len(inputs) != 2:
            raise ValueError('Nonzero-point quantized matmul is not supported')
        lhs_name, rhs_name = names[origin(lhs)], names[origin(rhs)]
        result = operation.results[0]
        dst_name, dst_type = finals.get(result, (result.name_hint or ('result' + str(ordinal)), result.type))
        names[result] = dst_name
        spec = tensor_spec(dst_type, 'output')
        buffer['tensors'][dst_name] = spec
        if kind == 'MATMUL':
            resident, accumulator = 'resident_' + str(ordinal), 'acc_' + str(ordinal)
            buffer['commands'].extend([
                {'opcode': 'RES_PACK', 'operands': {'src': rhs_name, 'dst': resident},
                 'attributes': {'layout': 'packed_rhs'}},
                {'opcode': 'MATMUL_RESIDENT', 'operands': {'lhs': lhs_name, 'rhs': resident, 'dst': accumulator}},
                {'opcode': 'COMMIT', 'operands': {'src': accumulator, 'dst': dst_name},
                 'attributes': {'epilogue': [], 'output_dtype': spec['dtype']}},
                {'opcode': 'EVICT', 'operands': {'handle': resident}},
            ])
        else:
            keys = ('q', 'k') if kind == 'ATTENTION_QK' else ('a', 'w')
            buffer['commands'].append({'opcode': kind,
                'operands': {keys[0]: lhs_name, keys[1]: rhs_name, 'dst': dst_name},
                'attributes': {'epilogue': [], 'output_dtype': spec['dtype']}})
    if not buffer['commands']:
        raise ValueError('Tensor program contains no supported accelerator contraction')
    return add_pointer_abi(buffer)
