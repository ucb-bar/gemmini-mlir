#!/usr/bin/env python3
"""Read public semantic input; the C++ tool performs all target lowering."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent / 'compiler_library'))
from merlin.targetgen.oot_starterkit.iface import parse_interface, parse_linalg

def inventory(text):
    if 'merlin_iface.' in text:
        import merlin.targetgen.contract.interface_emit as ie
        module_attrs = ie._parse_attr_block(ie._module_attrs(text))
        if module_attrs.get('merlin_iface.version', ie.GRAMMAR_VERSION) != ie.GRAMMAR_VERSION:
            raise ValueError('unsupported merlin_iface.version: ' + str(module_attrs['merlin_iface.version']))
        model = parse_interface(text)
        aliases = {}
        tensor_order = []
        command_sources = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            op = ie._op_line(line)
            if not op or op['mnemonic'] not in ie.defined_mnemonics():
                continue
            attrs = ie._parse_attr_block(op['attrs_body'])
            if op['mnemonic'] != 'tensor':
                command_sources.append({
                    'source_op_index': len(command_sources),
                    'prov': {key[len('prov.'):]: value for key, value in attrs.items()
                             if key.startswith('prov.') and isinstance(value, str)},
                })
            if not op['result']:
                continue
            name = attrs.get('name', op['result'])
            if op['mnemonic'] == 'tensor' or op['mnemonic'] in ie._NAMED_OP_OPERAND_KEYS or op['mnemonic'] == 'commit':
                aliases[op['result']] = name
                if name not in tensor_order:
                    tensor_order.append(name)
            if op['mnemonic'] != 'commit':
                continue
            shape, dtype = ie._shape_dtype(ie._last_type(op['tail']))
            existing = model['tensors'].get(name)
            if existing and (existing['shape'], existing['dtype']) != (shape, dtype):
                raise ValueError('commit result type differs from declared tensor ' + name)
            if existing is None:
                model['tensors'][name] = {'shape': shape, 'dtype': dtype, 'role': 'output'}
        for command in model['commands']:
            command['operands'] = {key: aliases.get(value, value) for key, value in command.get('operands', {}).items()}
            attrs = command.get('attributes', {})
            if isinstance(attrs.get('bias'), str):
                bias = aliases.get(attrs['bias'], attrs['bias'])
                if command['opcode'] == 'COMMIT':
                    command['operands']['bias'] = bias
                    attrs['bias'] = bias
                else:
                    attrs['bias'] = bias
        # The semantic reader adds named whole-op outputs during parsing, while
        # COMMIT outputs are supplied above. Restore their shared source order.
        model['tensors'] = {name: model['tensors'][name] for name in
                            tensor_order + [name for name in model['tensors'] if name not in tensor_order]}
        if len(command_sources) != len(model['commands']):
            raise ValueError('interface source operation inventory differs from semantic reader commands')
        params = model.setdefault('params', {})
        params['compiler_tensor_order'] = list(model.get('tensors', {}))
        params['compiler_command_sources'] = command_sources
        params['compiler_source_op_count'] = len(command_sources)
        return {'format': 'interface', 'model': model}
    model = parse_linalg(text)
    import merlin.targetgen.contract.linalg_iface as li
    from xdsl.parser import Parser
    module = Parser(li.make_linalg_context(), text).parse_module()
    definitions = [o for o in module.walk() if o.name == 'func.func' and o.body.blocks]
    fn = next((o for o in definitions if li._fn_name(o) == model['entry']), definitions[0])
    for k in ('logical_inputs', 'logical_outputs', 'logical_input_roles'):
        v = fn.attributes.get('merlin.' + k)
        if v is not None:
            model[k] = [x.data for x in v]
    block = fn.body.blocks[0]
    direct = [o for o in block.ops if li._op_name(o) != 'func.return']
    payload = [o for o in direct if not li._initialization_only(o) and li._op_name(o) != 'linalg.yield']
    if len(payload) != len(model['ops']):
        raise ValueError('semantic reader payload differs from direct source operations')
    source_owners = {value: (index, result_index)
                     for index, op in enumerate(direct)
                     for result_index, value in enumerate(op.results)}
    payload_owners = {value: (index, result_index)
                      for index, op in enumerate(payload)
                      for result_index, value in enumerate(op.results)}
    arguments = {value: index for index, value in enumerate(block.args)}

    def constant_fields(value):
        owner = getattr(value, 'owner', None)
        kind = li._op_name(owner) if getattr(owner, 'name', '') else ''
        scalar = li._scalar_const(value)
        if scalar is not None:
            return {'const_value': scalar}
        if kind in ('linalg.fill', 'tensor.splat') and owner.operands:
            return constant_fields(owner.operands[0])
        if kind == 'arith.constant':
            attrs = {**owner.attributes, **(getattr(owner, 'properties', {}) or {})}
            constant = attrs.get('value')
            if callable(getattr(constant, 'get_values', None)):
                # Dense values are read from the typed attribute, never source text.
                values = list(constant.get_values())
                # Numeric equality merges +0.0 and -0.0, whose signs are
                # observable in source floating arithmetic. NaNs retain their
                # existing nonuniform handling because NaN != NaN.
                if values and all(
                    x == values[0] and
                    (not isinstance(x, float) or x != 0.0 or
                     math.copysign(1.0, x) == math.copysign(1.0, values[0]))
                    for x in values
                ):
                    return {'const_value': values[0]}
                return {'const_values': values}
        return {}

    def value_fields(value):
        fields = constant_fields(value)
        if value in source_owners:
            fields['source_op_index'], fields['result_index'] = source_owners[value]
        elif value in arguments:
            fields['arg_index'] = arguments[value]
        owner = getattr(value, 'owner', None)
        kind = li._op_name(owner) if getattr(owner, 'name', '') else ''
        if kind in ('tensor.empty', 'linalg.fill', 'tensor.splat'):
            fields['init_kind'] = kind
        return fields

    def operand_record(value):
        rec = {'shape': li._shape(value.type), 'dtype': li._dtype(value.type)}
        if value in arguments:
            rec['source'] = ('arg', arguments[value])
        elif value in payload_owners:
            rec['source'] = ('op', payload_owners[value][0])
        else:
            owner = getattr(value, 'owner', None)
            kind = li._op_name(owner) if getattr(owner, 'name', '') else ''
            rec['source'] = ('init', kind.split('.')[-1]) if kind in ('tensor.empty', 'linalg.fill', 'tensor.splat') else ('const', None) if kind == 'arith.constant' else ('other', kind)
        rec.update(value_fields(value))
        return rec

    def enrich_payload(rec, op, source_index):
        rec['source_op_index'] = source_index
        rec.setdefault('source_provenance', {})['source_op_index'] = source_index
        ins, outs = li._ins_outs(op)
        for key, values in (('ins', ins), ('outs', outs)):
            if len(rec[key]) != len(values):
                raise ValueError('semantic reader operand arity differs from source')
            for operand_rec, value in zip(rec[key], values):
                operand_rec.update(value_fields(value))
        body = rec.get('scalar_body')
        if body is not None:
            # Rebuild only the structural scalar inventory with the same reader so
            # enclosing scalar constants and tensor initializers retain SSA owners.
            rec['scalar_body'] = li._scalar_body(op, operand_record)
        for result_index, result in enumerate(op.results):
            rec['results'][result_index].update(value_fields(result))

    records = []
    payload_indices = {op: index for index, op in enumerate(payload)}
    for source_index, op in enumerate(direct):
        kind = li._op_name(op)
        ins, outs = li._ins_outs(op)
        rec = {'index': source_index, 'source_op_index': source_index,
               'kind': kind, 'operation': kind,
               'initialization_only': li._initialization_only(op),
               'operands': [operand_record(value) for value in op.operands],
               'ins': [operand_record(value) for value in ins],
               'outs': [operand_record(value) for value in outs],
               'results': [dict(shape=li._shape(value.type), dtype=li._dtype(value.type),
                                **value_fields(value)) for value in op.results],
               'attributes': li._semantic_attributes(op), 'prov': li._prov(op)}
        if len(op.results) == 1:
            rec.update(constant_fields(op.results[0]))
        if op in payload_indices:
            index = payload_indices[op]
            rec['payload_index'] = index
            enrich_payload(model['ops'][index], op, source_index)
            rec.update(copy.deepcopy(model['ops'][index]))
        records.append(rec)
    terminator = block.ops.last
    if model.get('returns') is not None and terminator is not None and li._op_name(terminator) == 'func.return':
        for rec, value in zip(model['returns'], terminator.operands):
            rec.update(value_fields(value))
    model['source_operations'] = records
    model['source_op_count'] = len(records)
    return {'format': 'linalg', 'model': model}

def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return '-inf' if value < 0 else ('nan' if math.isnan(value) else 'inf')
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def main():
    argv = sys.argv[1:]
    paths = [x for x in argv if not x.startswith('-')]
    if not paths:
        raise ValueError('an input MLIR file is required')
    source = Path(paths[-1])
    data = source.read_bytes()
    doc = inventory(data.decode())
    doc['source_sha256'] = hashlib.sha256(data).hexdigest()
    doc['source_path'] = str(source.resolve())
    dump = next((x.split('=', 1)[1] for x in argv if x.startswith('--dump-semantic=')), None)
    if dump:
        Path(dump).write_text(json.dumps(json_safe(doc), allow_nan=False))
        return 0
    with tempfile.TemporaryDirectory(prefix='merlin-input-') as directory:
        p = Path(directory) / 'input.json'
        p.write_text(json.dumps(json_safe(doc), allow_nan=False))
        return subprocess.call([str(Path(__file__).resolve().parent / 'build' / 'merlin-opt'), '--semantic-input=' + str(p)] + argv)
if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print('compiler: ' + str(e), file=sys.stderr)
        sys.exit(1)
