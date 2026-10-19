"""Source-bound metadata and command inventory for composed tensor programs."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from xdsl.dialects.builtin import FloatAttr, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.func import FuncOp

try:
    from .frontend import add_pointer_abi, tensor_spec
    from .linalg_frontend import contraction_kind, op_name, parse_module
except ImportError:
    from frontend import add_pointer_abi, tensor_spec
    from linalg_frontend import contraction_kind, op_name, parse_module


@dataclass
class SourceProgram:
    module: object
    function: object
    operations: list
    returned: object
    names: dict
    op_indices: dict
    device_kinds: dict
    text: str


def attribute_string(operation, key):
    value = operation.attributes.get(key, operation.properties.get(key))
    return value.data if isinstance(value, StringAttr) else None


def zero_initializer(value):
    owner = value.owner
    name = op_name(owner)
    if name == 'arith.constant':
        attribute = owner.properties.get('value', owner.attributes.get('value'))
        if isinstance(attribute, (FloatAttr, IntegerAttr)):
            return attribute.value.data == 0
        get_values = getattr(attribute, 'get_values', None)
        return get_values is not None and all(number == 0 for number in get_values())
    if name in ('tensor.splat', 'linalg.fill'):
        return bool(owner.operands) and zero_initializer(owner.operands[0])
    return False


def device_kind(operation):
    kind = contraction_kind(operation)
    if kind is None:
        return None
    inputs = list(getattr(operation, 'inputs', []))
    outputs = list(getattr(operation, 'outputs', []))
    if len(inputs) != 2 or len(outputs) != 1 or len(operation.results) != 1:
        return None
    if not all(isinstance(value.type, TensorType) for value in [*inputs, *outputs, *operation.results]):
        return None
    if any(str(value.type.element_type) != 'i8' for value in inputs):
        return None
    if str(operation.results[0].type.element_type) != 'i32' or not zero_initializer(outputs[0]):
        return None
    return kind


def parsed_source(text):
    module = parse_module(text)
    functions = [operation for operation in module.body.block.ops
                 if isinstance(operation, FuncOp) and operation.body.blocks]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError('Composed source requires one single-block entry function')
    function = functions[0]
    block = function.body.block
    returns = [operation for operation in block.ops if op_name(operation) == 'func.return']
    if len(returns) != 1 or block.last_op != returns[0]:
        raise ValueError('Composed source requires one final return operation')
    returned = returns[0]
    returned.verify()
    operations = [operation for operation in block.ops if operation != returned]
    indices = {operation: index for index, operation in enumerate(operations)}
    names = {}
    for index, argument in enumerate(block.args):
        if not isinstance(argument.type, TensorType):
            raise ValueError('Composed entry arguments must be ranked tensors')
        names[argument] = 'arg' + str(index)
    final_names = {value: 'Y' + str(index) for index, value in enumerate(returned.operands)}
    if len(final_names) != len(returned.operands):
        raise ValueError('Repeated source returns need an explicit output-copy source operation')
    for index, operation in enumerate(operations):
        for result_index, result in enumerate(operation.results):
            if isinstance(result.type, TensorType):
                names[result] = final_names.get(result, 't' + str(index) + '_' + str(result_index))
    if any(value not in names or value in block.args for value in returned.operands):
        raise ValueError('Composed outputs must be materialized tensor operation results')
    kinds = {operation: device_kind(operation) for operation in operations}
    return SourceProgram(module, function, operations, returned, names, indices, kinds, text)


def parse_mixed(text, program=None):
    program = parsed_source(text) if program is None else program
    block = program.function.body.block
    final_values = set(program.returned.operands)
    tensors = {}
    for value, name in program.names.items():
        role = 'input' if value in block.args else 'output' if value in final_values else 'intermediate'
        tensors[name] = tensor_spec(value.type, role)
    arguments = [{'tensor': name, 'access': 'read' if spec['role'] == 'input' else 'write'}
                 for name, spec in tensors.items()]
    outputs = [program.names[value] for value in program.returned.operands]
    tasks, bindings, commands, placements = [], [], [], []
    for index, operation in enumerate(program.operations):
        region = attribute_string(operation, 'prov.region_id')
        kind = program.device_kinds[operation]
        reads = list(dict.fromkeys(program.names[value] for value in operation.operands if value in program.names))
        writes = [program.names[value] for value in operation.results if value in program.names]
        task = {'task_index': index, 'kind': 'accelerator' if kind else 'host',
                'source_op_indices': [index], 'source_region_ids': [region] if region else [],
                'instruction_start': index, 'instruction_end': index + 1,
                'reads': reads, 'writes': writes}
        tasks.append(task)
        for result_index, result in enumerate(operation.results):
            if result in program.names:
                bindings.append({'op_index': index, 'result_index': result_index,
                                 'tensor': program.names[result]})
        reason = ('Integer contraction runs on the accelerator' if kind else
                  'Source initialization, layout, or floating point computation uses the selected host lane')
        placements.append({'source_op_index': index, 'region_id': region,
                           'operation': op_name(operation), 'lane': 'accelerator' if kind else 'host',
                           'reason': reason})
        if kind is None:
            continue
        lhs, rhs = list(operation.inputs)
        lhs_name, rhs_name = program.names[lhs], program.names[rhs]
        dst = program.names[operation.results[0]]
        attrs = {'merlin.source_op_index': index, 'epilogue': [], 'output_dtype': 'i32'}
        if region:
            attrs['prov.region_id'] = region
        if kind == 'MATMUL':
            resident, accumulator = 'resident_' + str(index), 'acc_' + str(index)
            source_attrs = {'merlin.source_op_index': index}
            if region:
                source_attrs['prov.region_id'] = region
            commands.extend([
                {'opcode': 'RES_PACK', 'operands': {'src': rhs_name, 'dst': resident},
                 'attributes': {**source_attrs, 'layout': 'packed_rhs'}},
                {'opcode': 'MATMUL_RESIDENT', 'operands': {'lhs': lhs_name, 'rhs': resident, 'dst': accumulator},
                 'attributes': source_attrs},
                {'opcode': 'COMMIT', 'operands': {'src': accumulator, 'dst': dst}, 'attributes': attrs},
                {'opcode': 'EVICT', 'operands': {'handle': resident}, 'attributes': source_attrs},
            ])
        else:
            keys = ('q', 'k') if kind == 'ATTENTION_QK' else ('a', 'w')
            commands.append({'opcode': kind,
                             'operands': {keys[0]: lhs_name, keys[1]: rhs_name, 'dst': dst},
                             'attributes': attrs})
    count = len(program.operations)
    plan = {'schema': 'mixed_program_plan_v1', 'source_sha256': sha256(text.encode('utf-8')).hexdigest(),
            'source_op_count': count, 'tasks': tasks, 'schedule_instruction_count': count,
            'prologue_instruction_range': [0, 0], 'epilogue_instruction_range': [count, count],
            'entry_bindings': [program.names[value] for value in block.args],
            'source_values': bindings, 'output_bindings': outputs, 'compiler_temporaries': []}
    buffer = {'abi_version': '0.1', 'target': 'gemmini', 'tensors': tensors, 'commands': commands,
              'kernel_abi': {'kind': 'whole_program', 'args': arguments, 'outputs': outputs},
              'params': {'global_program_plan': plan, 'lane_placement': placements}}
    return add_pointer_abi(buffer)
