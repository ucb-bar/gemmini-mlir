"""Source readers and typed semantic command construction for Gemmini.

The interface reader is the public grammar implementation.  Tensor-linalg records
retain their scalar SSA DAG and indexing maps; target lowering must explicitly
implement those records rather than infer arithmetic from provenance labels.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

_LIBRARY = Path(__file__).resolve().parent / 'compiler_library'
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))

from merlin.targetgen.oot_starterkit.iface import parse_interface, parse_linalg
from merlin.targetgen.contract import interface_emit as interface
from merlin.targetgen.oot_starterkit import cmdbuf
from merlin.targetgen.oot_starterkit.cmdbuf import CommandBufferBuilder

# The editable source distribution stages contract data beside the workspace.
if not cmdbuf._SCHEMA.exists():
    for _root in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
        _candidate = _root / "schemas" / "command_buffer.schema.json"
        if _candidate.is_file():
            cmdbuf._SCHEMA = _candidate
            break


def _interface_source(text: str) -> dict[str, Any]:
    cb = parse_interface(text)
    if cb['target'] != 'gemmini':
        raise ValueError(f"expected Gemmini source, received target {cb['target']!r}")
    module_attrs = interface._parse_attr_block(interface._module_attrs(text))
    if module_attrs.get('merlin_iface.version', '0.1') != '0.1':
        raise ValueError('unsupported merlin_iface version')
    aliases = {}
    outputs = []
    inputs = []
    # Reuse the canonical operation/type/attribute grammar. The public reader
    # leaves commit outputs undeclared; their explicit source type is authoritative.
    for raw in text.splitlines():
        op = interface._op_line(raw.strip())
        if op is None or not op['mnemonic']:
            continue
        attrs = interface._parse_attr_block(op['attrs_body'])
        kind, result = op['mnemonic'], op['result']
        if kind == 'tensor':
            aliases[result] = attrs.get('name', result)
            if attrs.get('role', 'input') not in ('output', 'intermediate'):
                inputs.append(aliases[result])
        elif kind == 'commit' or kind in interface._NAMED_OP_OPERAND_KEYS:
            name = attrs.get('name', result)
            shape, dtype = interface._shape_dtype(interface._last_type(op['tail']))
            declared = cb['tensors'].get(name)
            if declared is not None and (declared['shape'] != shape or declared['dtype'] != dtype):
                raise ValueError(f'inconsistent source types for {name}')
            cb['tensors'][name] = {'shape': shape, 'dtype': dtype, 'role': 'output'}
            aliases[result] = name
            outputs.append(name)
        elif result:
            aliases[result] = result
    for command in cb['commands']:
        operands = command.get('operands', {})
        for key, value in operands.items():
            if isinstance(value, str):
                operands[key] = aliases.get(value, value)
    outputs = list(dict.fromkeys(outputs))
    cb['params'] = {'source_format': 'merlin_iface', 'input_order': inputs, 'output_order': outputs}
    # JSON object key order is not an ABI. The public whole-program declaration
    # carries the source pointer order through serializers that sort object keys.
    cb['kernel_abi'] = {'kind': 'whole_program',
                        'args': [{'tensor': name, 'access': 'read'} for name in inputs]
                                + [{'tensor': name, 'access': 'write'} for name in outputs],
                        'outputs': outputs}
    return cb


def _is_integer_matmul(op: dict[str, Any]) -> bool:
    """Match arithmetic and maps, independent of provenance or capsule identity."""
    if op['kind'] != 'linalg.generic' or op['iterator_types'] != ['parallel', 'parallel', 'reduction']:
        return False
    maps = [''.join(m.split()) for m in (op['indexing_maps'] or [])]
    if maps != ['affine_map<(d0,d1,d2)->(d0,d2)>',
                'affine_map<(d0,d1,d2)->(d2,d1)>',
                'affine_map<(d0,d1,d2)->(d0,d1)>']:
        return False
    if [r['dtype'] for r in op['ins']] != ['i8', 'i8'] or len(op['outs']) != 1:
        return False
    if [r['dtype'] for r in op['results']] != ['i32']:
        return False
    body = op['scalar_body'] or {}
    if body.get('arguments') != ['i8', 'i8', 'i32'] or body.get('captures'):
        return False
    if body.get('yields') != ['op:3:0']:
        return False
    expected = [('arith.extsi', ['arg:0']), ('arith.extsi', ['arg:1']),
                ('arith.muli', ['op:0:0', 'op:1:0']), ('arith.addi', ['arg:2', 'op:2:0'])]
    operations = body.get('operations', [])
    if len(operations) != len(expected):
        return False
    for operation, (kind, operands) in zip(operations, expected):
        if (operation['op'] != kind or operation['operands'] != operands or
                operation['results'] != ['i32'] or operation['regions']):
            return False
        if any(key != 'overflowFlags' or value != '#arith.overflow<none>'
               for key, value in operation['attributes'].items()):
            return False
    return True


def _linalg_source(text: str) -> dict[str, Any]:
    from source_parser import register_named_products, parse_source_module
    register_named_products()
    graph = parse_linalg(text)
    if graph['returns'] is None:
        raise ValueError('Gemmini source frontend requires a single-block return join')
    builder = CommandBufferBuilder('gemmini')
    # Names are positional unless the source explicitly carries logical ABI names.
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import StringAttr, ArrayAttr
    module = parse_source_module(text)
    fn = next(op for op in module.walk() if op.name == 'func.func' and op.sym_name.data == graph['entry'])
    # Complete destination initializer values omitted by the structural reader.
    from merlin.targetgen.contract import linalg_iface as li
    payload = [op for op in fn.body.blocks[0].ops
               if op.name not in ('func.return', 'linalg.yield') and not li._initialization_only(op)]
    for rec, source_op in zip(graph['ops'], payload, strict=True):
        if rec['indexing_maps'] is None and hasattr(source_op, 'get_indexing_maps'):
            rec['indexing_maps'] = [str(value) for value in source_op.get_indexing_maps()]
        if not rec['iterator_types'] and hasattr(source_op, 'get_iterator_types'):
            rec['iterator_types'] = [value.data.value for value in source_op.get_iterator_types()]
            rec['reduction_dims'] = [i for i, kind in enumerate(rec['iterator_types']) if kind == 'reduction']
        _, source_outs = li._ins_outs(source_op)
        for out_rec, source_out in zip(rec['outs'], source_outs, strict=True):
            owner = source_out.owner
            if getattr(owner, 'name', '') == 'linalg.fill':
                value = li._scalar_const(owner.operands[0])
                if value is not None:
                    out_rec['const_value'] = value
            elif getattr(owner, 'name', '') == 'tensor.splat':
                value = li._scalar_const(source_out)
                if value is not None:
                    out_rec['const_value'] = value
    def string_list(key):
        value = fn.attributes.get(key)
        if value is None:
            return None
        if not isinstance(value, ArrayAttr) or any(not isinstance(v, StringAttr) for v in value):
            raise ValueError(f'{key} must be an array of strings')
        return [v.data for v in value]
    inputs = string_list('merlin.logical_inputs') or [f'arg{i}' for i in range(len(graph['args']))]
    roles = string_list('merlin.logical_input_roles') or ['input'] * len(inputs)
    outputs = string_list('merlin.logical_outputs') or [f'output{i}' for i in range(len(graph['results']))]
    if len(inputs) != len(graph['args']) or len(roles) != len(inputs) or len(outputs) != len(graph['results']):
        raise ValueError('logical ABI arity differs from source signature')
    if len(set(inputs + outputs)) != len(inputs + outputs):
        raise ValueError('logical ABI names must be distinct')
    def declare(name, record, role):
        shape = record['shape']
        if any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f'dynamic source shape for {name}')
        builder.tensor(name, shape, record['dtype'], role)
    for name, role, spec in zip(inputs, roles, graph['args']):
        declare(name, spec, role)
    for op in graph['ops']:
        for k, spec in enumerate(op['results']):
            declare(f'op{op["id"]}_{k}', spec, 'intermediate')
    constant_ids = {}
    def edge(record):
        origin, index = record['source']
        if origin == 'arg' and index is not None:
            return inputs[index]
        if origin == 'op':
            return f'op{index}_{record.get("result_index", 0)}'
        if origin in ('const', 'init'):
            # Retain exact init kind/value in the command record. These names
            # denote compiler-created values, never external input buffers.
            key = repr(record)
            if key not in constant_ids:
                name = f'init{len(constant_ids)}'
                constant_ids[key] = name
                declare(name, record, 'intermediate')
            return constant_ids[key]
        raise ValueError(f'unsupported source dataflow edge: {record!r}')
    for op in graph['ops']:
        ins = [edge(rec) for rec in op['ins']]
        outs = [edge(rec) for rec in op['outs']]
        results = [f'op{op["id"]}_{k}' for k in range(len(op['results']))]
        attrs = {'source_operation': op['kind'], 'semantic_record': copy.deepcopy(op)}
        operands = {f'in{i}': value for i, value in enumerate(ins)}
        operands.update({f'init{i}': value for i, value in enumerate(outs)})
        operands.update({f'dst{i}': value for i, value in enumerate(results)})
        # MATMUL carries initialization explicitly. Generic contractions are
        # retained as scalar DAGs until their maps and arithmetic are matched.
        if (op['kind'] in ('linalg.matmul', 'linalg.matmul_transpose_b') or _is_integer_matmul(op)) and len(ins) == 2 and len(results) == 1:
            operands.update(lhs=ins[0], rhs=ins[1], dst=results[0])
            attrs['transpose_rhs'] = op['kind'] == 'linalg.matmul_transpose_b'
            attrs['accumulator_init'] = copy.deepcopy(op['outs'])
            builder.command('MATMUL', operands, attrs)
        elif op['kind'] == 'linalg.batch_matmul' and len(ins) == 2 and len(results) == 1:
            operands.update(a=ins[0], w=ins[1], dst=results[0])
            builder.command('BATCHED_MATMUL', operands, attrs)
        else:
            builder.command('VREDUCE' if op['reduction_dims'] else 'VECTOR_MAP', operands, attrs)
    for name, result in zip(outputs, graph['returns']):
        declare(name, result, 'output')
        builder.command('ROW_MAJOR_VIEW', {'src': edge(result), 'dst': name}, {'identity': True})
    builder.params(source_format='linalg-on-tensors', entry=graph['entry'], input_order=inputs,
                   output_order=outputs, source_graph=graph)
    return builder.to_dict()


def parse_source(text: str) -> dict[str, Any]:
    """Parse source into a complete typed semantic command buffer.

    Unknown source syntax or unresolved edges raise; operations are never dropped.
    Target lowering must reject any semantic command it does not implement.
    """
    if 'merlin_iface.' in text and 'merlin_iface.tensor' in text:
        cb = _interface_source(text)
        builder = CommandBufferBuilder('gemmini')
        builder._cb = cb
        return builder.to_dict()
    return _linalg_source(text)
