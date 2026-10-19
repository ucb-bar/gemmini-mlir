"""Typed xDSL reader for the versioned accelerator interface."""
from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr, Builtin, FloatAttr, IntegerAttr, ModuleOp, StringAttr, TensorType,
)
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, var_operand_def, var_result_def
from xdsl.parser import Parser


class InterfaceSyntax:
    @classmethod
    def parse(cls, parser):
        position = parser.pos
        unresolved = []
        first = parser.parse_optional_unresolved_operand()
        if first is not None:
            unresolved.append(first)
            while parser.parse_optional_punctuation(',') is not None:
                unresolved.append(parser.parse_unresolved_operand())
        attributes = parser.parse_optional_attr_dict()
        parser.parse_punctuation(':')
        if cls.name == 'merlin_iface.tensor':
            inputs, outputs = [], [parser.parse_type()]
        else:
            signature = parser.parse_function_type()
            inputs, outputs = signature.inputs.data, signature.outputs.data
        operands = parser.resolve_operands(unresolved, inputs, position)
        return cls.create(operands=operands, result_types=outputs, attributes=attributes)


OPCODES = {
    'resident_pack': ('RES_PACK', ['src']),
    'matmul': ('MATMUL_RESIDENT', ['lhs', 'rhs']),
    'commit': ('COMMIT', ['src']),
    'evict': ('EVICT', ['handle']),
    'movement': ('MOVEMENT', ['src']),
    'attention_qk': ('ATTENTION_QK', ['q', 'k']),
    'attention_pv': ('ATTENTION_PV', ['p', 'v']),
    'bias_add': ('BIAS_ADD', ['src', 'bias']),
    'residual_add': ('RESIDUAL_ADD', ['lhs', 'rhs']),
    'conv2d': ('CONV2D', ['ifm', 'weight']),
    'matmul_batched': ('BATCHED_MATMUL', ['a', 'w']),
    'rmsnorm': ('RMSNORM', ['src', 'gamma']),
    'rope': ('ROPE', ['src']),
    'softmax': ('SOFTMAX', ['src']),
}

INTERFACE_OPS = []
for _mnemonic in ['tensor', *OPCODES]:
    _operation = type(
        'Interface_' + _mnemonic,
        (InterfaceSyntax, IRDLOperation),
        {'name': 'merlin_iface.' + _mnemonic,
         'arguments': var_operand_def(), 'values': var_result_def()},
    )
    INTERFACE_OPS.append(irdl_op_definition(_operation))
INTERFACE = Dialect('merlin_iface', INTERFACE_OPS, [])


def scalar_attribute(value):
    if isinstance(value, StringAttr):
        return value.data
    if isinstance(value, (IntegerAttr, FloatAttr)):
        return value.value.data
    if isinstance(value, ArrayAttr):
        return [scalar_attribute(item) for item in value.data]
    raise ValueError('Unsupported interface attribute: ' + str(value))


def tensor_spec(value_type, role):
    if not isinstance(value_type, TensorType):
        raise ValueError('Expected a ranked tensor result, got ' + str(value_type))
    shape = list(value_type.get_shape())
    if any(d <= 0 for d in shape):
        raise ValueError('Only positive static tensor extents are supported')
    return {'shape': shape, 'dtype': str(value_type.element_type), 'role': role}


def add_pointer_abi(buffer):
    if buffer.get('kernel_abi', {}).get('kind') == 'whole_program':
        arguments = [arg['tensor'] for arg in buffer['kernel_abi']['args']]
    else:
        produced = {command.get('operands', {}).get('dst') for command in buffer['commands']}
        derived = set(buffer.get('params', {}).get('derivations', {}))
        arguments = [name for name, spec in buffer['tensors'].items()
                     if name not in produced and name not in derived
                     and spec.get('role') not in ('output', 'intermediate')]
        publishing = {'COMMIT', 'BIAS_ADD', 'RESIDUAL_ADD', 'ATTENTION_QK',
                      'ATTENTION_PV', 'CONV2D', 'BATCHED_MATMUL', 'MOVEMENT'}
        outputs = []
        for command in buffer['commands']:
            dst = command.get('operands', {}).get('dst')
            if command['opcode'] in publishing and dst not in outputs:
                outputs.append(dst)
        outputs.extend(name for name, spec in buffer['tensors'].items()
                       if spec.get('role') == 'output' and name not in outputs)
        selected = buffer.get('outputs')
        if selected is not None:
            outputs = [name for name in outputs if name in selected]
        arguments.extend(outputs)
    buffer['compiler_pointer_abi'] = {'version': 1, 'arguments': arguments}
    return buffer


def parse(text):
    if 'merlin_iface.version' not in text:
        try:
            from .linalg_frontend import parse_linalg
        except ImportError:
            from linalg_frontend import parse_linalg
        return parse_linalg(text)
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(INTERFACE)
    module = Parser(context, text).parse_module()
    version = module.attributes.get('merlin_iface.version')
    if not isinstance(version, StringAttr) or version.data != '0.1':
        raise ValueError('Unsupported merlin_iface.version: ' + str(version))
    attributes = {key: scalar_attribute(value) for key, value in module.attributes.items()}
    buffer = {'abi_version': attributes.get('merlin_iface.abi_version', '0.1'),
              'target': attributes.get('merlin_iface.target', 'gemmini'),
              'tensors': {}, 'commands': []}
    names = {}
    for ordinal, operation in enumerate(module.body.block.ops):
        mnemonic = operation.name.removeprefix('merlin_iface.')
        attrs = {key: scalar_attribute(value) for key, value in operation.attributes.items()}
        if mnemonic == 'tensor':
            if len(operation.results) != 1 or operation.operands:
                raise ValueError('merlin_iface.tensor requires one result and no operands')
            result = operation.results[0]
            name = attrs.pop('name', result.name_hint or ('tensor_' + str(ordinal)))
            spec = tensor_spec(result.type, attrs.pop('role', 'input'))
            for key in ('scale_of', 'block'):
                if key in attrs:
                    spec[key] = attrs.pop(key)
            if name in buffer['tensors']:
                raise ValueError('Duplicate tensor name ' + name)
            buffer['tensors'][name] = spec
            names[result] = name
            continue
        if mnemonic not in OPCODES:
            raise ValueError('Unknown interface operation ' + operation.name)
        opcode, keys = OPCODES[mnemonic]
        if len(operation.operands) != len(keys):
            raise ValueError(operation.name + ' has an incorrect operand count')
        operands = {key: names[value] for key, value in zip(keys, operation.operands)}
        if mnemonic != 'evict':
            if len(operation.results) != 1:
                raise ValueError(operation.name + ' requires exactly one result')
            result = operation.results[0]
            name = attrs.pop('name', result.name_hint or ('value_' + str(ordinal)))
            names[result] = name
            operands['dst'] = name
            if isinstance(result.type, TensorType):
                spec = tensor_spec(result.type, 'output')
                prior = buffer['tensors'].get(name)
                if prior is not None and (prior['shape'], prior['dtype']) != (spec['shape'], spec['dtype']):
                    raise ValueError('Inconsistent tensor redeclaration ' + name)
                buffer['tensors'][name] = spec
        elif operation.results:
            raise ValueError('merlin_iface.evict cannot have results')
        if mnemonic == 'resident_pack':
            attrs.setdefault('layout', 'packed_rhs')
        command = {'opcode': opcode, 'operands': operands}
        if attrs:
            command['attributes'] = attrs
        buffer['commands'].append(command)
    if not buffer['commands']:
        raise ValueError('Interface contains no accelerator commands')
    return add_pointer_abi(buffer)
