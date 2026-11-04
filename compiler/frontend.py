"""Source-semantic normalization for the two public compiler input grammars.

The returned dictionary retains the contract command representation.  ``LINALG_OP``
is an explicitly unlowered structural operation, never an execution claim; consumers
must lower its source record or reject it.  No source operation is silently omitted.
"""
from __future__ import annotations

import copy
from typing import Any

from merlin.targetgen.oot_starterkit.iface import (
    is_linalg_on_tensors, parse_interface, parse_linalg,
)
from merlin.targetgen.contract import interface_emit as _ie
from merlin.targetgen.contract import linalg_iface as _li


def parse_source(text: str) -> dict[str, Any]:
    """Read source text into a semantic command program, preserving ordered I/O."""
    # Public typed-source fragments may omit the optional provenance level tag.
    # The actual parse remains structural; no source text is rewritten.
    linalg = is_linalg_on_tensors(text) or ('func.func' in text and not _ie.op_mnemonics(text))
    cb = _linalg(text) if linalg else _legacy(text)
    cb['source_text'] = text
    return cb


def _legacy(text: str) -> dict[str, Any]:
    cb = parse_interface(text)
    module_attrs = _ie._parse_attr_block(_ie._module_attrs(text))
    version = module_attrs.get('merlin_iface.version')
    if version != '0.1':
        raise ValueError(f'unsupported merlin_iface.version {version!r}')
    cb.setdefault('params', {})
    cb['source_format'] = 'merlin_iface'
    cb['source_attributes'] = module_attrs
    # Canonical parser exposes logical tensor names but its operand strings are SSA
    # names. Resolve the explicit name attributes before handing off to scheduling.
    aliases: dict[str, str] = {}
    tensor_order: list[str] = []
    source_ops = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('//'):
            continue
        rec = _ie._op_line(line)
        if rec is None or not rec['mnemonic'] or not _ie._is_op_statement(line, rec['mnemonic']):
            continue
        attrs = _ie._parse_attr_block(rec['attrs_body'])
        result = rec['result']
        name = attrs.get('name', result)
        if result:
            aliases[result] = name
        if rec['mnemonic'] == 'tensor':
            cb['tensors'][name]['source_attributes'] = attrs
            if name not in tensor_order:
                tensor_order.append(name)
            continue
        record = copy.deepcopy(rec)
        record['attributes'] = attrs
        source_ops.append(record)
        # COMMIT outputs are not tensor declarations in the supplied parser.
        if result:
            try:
                shape, dtype = _ie._shape_dtype(_ie._last_type(rec['tail']))
            except ValueError:
                pass  # resident and accumulator handles have no host tensor type
            else:
                spec = {'shape': shape, 'dtype': dtype, 'role': 'output'}
                if name in cb['tensors'] and any(cb['tensors'][name][k] != spec[k] for k in ('shape', 'dtype')):
                    raise ValueError(f'inconsistent declared result type for {name!r}')
                cb['tensors'].setdefault(name, spec)
                if name not in tensor_order:
                    tensor_order.append(name)
    if len(source_ops) != len(cb['commands']):
        raise ValueError('canonical interface parser did not preserve every command')
    for i, (cmd, rec) in enumerate(zip(cb['commands'], source_ops, strict=True)):
        cmd['operands'] = {k: aliases.get(v, v) for k, v in cmd.get('operands', {}).items()}
        cmd.setdefault('attributes', {})
        cmd['source_index'] = i
        cmd['source_record'] = rec
    # The supplied parser declares named whole-op results during its first pass,
    # but omits COMMIT tensors.  Appending those afterward changes result order
    # in mixed programs (e.g. a COMMIT before a CONV2D). Preserve source order
    # for both the tensor inventory and explicit output publication list.
    cb['tensors'] = {name: cb['tensors'][name] for name in tensor_order}
    cb['inputs'] = [n for n, t in cb['tensors'].items() if t['role'] != 'output']
    cb['outputs'] = [n for n, t in cb['tensors'].items() if t['role'] == 'output']
    return cb


def _typed_linalg_details(text: str, entry: str):
    """Read ABI attributes and initializer constants from the same typed grammar."""
    from xdsl.parser import Parser
    from merlin.common.ir_lock import IR_LOCK
    with IR_LOCK:
        module = Parser(_li.make_linalg_context(), text).parse_module()
    fn = next(o for o in module.walk() if o.name == 'func.func' and _li._fn_name(o) == entry)
    attrs = fn.attributes
    def names(key):
        value = attrs.get(key)
        return [v.data for v in value] if value is not None else None
    payload = [o for o in fn.body.blocks[0].ops if o.name not in ('func.return', 'linalg.yield') and not _li._initialization_only(o)]
    initializers = {}
    for i, op in enumerate(payload):
        _, outs = _li._ins_outs(op)
        vals = []
        for out in outs:
            owner = out.owner
            kind = _li._op_name(owner) if hasattr(owner, 'name') else 'block_argument'
            value = None
            if kind in ('linalg.fill', 'tensor.splat') and owner.operands:
                value = _li._scalar_const(owner.operands[0])
            elif kind == 'arith.constant':
                value = _li._scalar_const(out)
            vals.append({'kind': kind, 'value': value})
        initializers[i] = vals
    return names('merlin.logical_inputs'), names('merlin.logical_outputs'), names('merlin.logical_input_roles'), initializers


def _linalg(text: str) -> dict[str, Any]:
    inventory = parse_linalg(text)
    if inventory['returns'] is None:
        raise ValueError('linalg normalization requires a complete single-block return join')
    in_names, out_names, roles, inits = _typed_linalg_details(text, inventory['entry'])
    in_names = in_names if in_names is not None else [f'arg{i}' for i in range(len(inventory['args']))]
    out_names = out_names if out_names is not None else [f'out{i}' for i in range(len(inventory['results']))]
    roles = roles if roles is not None else ['input'] * len(in_names)
    if len(in_names) != len(inventory['args']) or len(roles) != len(in_names) or len(out_names) != len(inventory['results']):
        raise ValueError('logical ABI arity differs from typed entry signature')
    if any(not isinstance(n, str) or not n for n in in_names + out_names) or len(set(in_names + out_names)) != len(in_names + out_names):
        raise ValueError('logical ABI names must be nonempty and distinct')
    cb: dict[str, Any] = {'abi_version': '0.1', 'target': 'gemmini', 'params': {},
        'source_format': 'linalg-on-tensors', 'linalg': inventory,
        'tensors': {}, 'commands': [], 'inputs': in_names, 'outputs': out_names}
    for name, role, spec in zip(in_names, roles, inventory['args'], strict=True):
        cb['tensors'][name] = {'shape': spec['shape'], 'dtype': spec['dtype'], 'role': role, 'source_argument': spec['index']}
    result_names: dict[tuple[int, int], str] = {}
    # Bind function results before internal values so complete calls retain public
    # names even when the returned value is consumed by another operation as well.
    return_aliases = []
    for name, edge, spec in zip(out_names, inventory['returns'], inventory['results'], strict=True):
        cb['tensors'][name] = {**spec, 'role': 'output'}
        if edge['source'][0] == 'op':
            key = (edge['source'][1], edge.get('result_index', 0))
            if key not in result_names:
                result_names[key] = name
            else:
                return_aliases.append((name, edge))
        else:
            return_aliases.append((name, edge))
    for rec in inventory['ops']:
        for k, spec in enumerate(rec['results']):
            key = (rec['id'], k)
            if key not in result_names:
                name = f'__value_{rec["id"]}_{k}'
                while name in cb['tensors']:
                    name = '_' + name
                result_names[key] = name
            name = result_names[key]
            cb['tensors'].setdefault(name, {**spec, 'role': 'intermediate'})
    def edge_name(edge):
        kind, index = edge['source']
        if kind == 'arg':
            if not isinstance(index, int) or not 0 <= index < len(in_names):
                raise ValueError(f'invalid function argument edge {edge!r}')
            return in_names[index]
        if kind == 'op':
            return result_names[(index, edge.get('result_index', 0))]
        return None
    for rec in inventory['ops']:
        rec['initializers'] = inits[rec['id']]
        srcs = [edge_name(e) for e in rec['ins']]
        dsts = [result_names[(rec['id'], k)] for k in range(len(rec['results']))]
        common = {'source_index': rec['id'], 'source_record': rec}
        if _is_integer_matmul(rec) and all(srcs[:2]):
            lhs, rhs = srcs[:2]
            resident = f'__resident_{rec["id"]}'
            acc = f'__acc_{rec["id"]}'
            cb['commands'].extend([
                {'opcode': 'RES_PACK', 'operands': {'src': rhs, 'dst': resident}, 'attributes': {'layout': 'packed_rhs'}, **common},
                {'opcode': 'MATMUL_RESIDENT', 'operands': {'lhs': lhs, 'rhs': resident, 'dst': acc}, 'attributes': {f'prov.{k}': v for k, v in rec['prov'].items()}, **common},
                {'opcode': 'COMMIT', 'operands': {'src': acc, 'dst': dsts[0]}, 'attributes': {'epilogue': [], 'output_dtype': 'i32'}, **common},
                {'opcode': 'EVICT', 'operands': {'handle': resident}, 'attributes': {}, **common},
            ])
        else:
            operands = {f'ins{i}': name for i, name in enumerate(srcs) if name is not None}
            operands.update({f'dst{i}': name for i, name in enumerate(dsts)})
            cb['commands'].append({'opcode': 'LINALG_OP', 'operands': operands,
                'attributes': {'operation': rec['kind']}, **common})
    for name, edge in return_aliases:
        src = edge_name(edge)
        if src is None:
            raise ValueError(f'cannot resolve returned initializer/constant {edge!r}')
        cb['commands'].append({'opcode': 'LINALG_RETURN', 'operands': {'src': src, 'dst': name},
            'attributes': {}, 'source_record': edge})
    return cb


def _is_integer_matmul(rec: dict[str, Any]) -> bool:
    """Conservatively prove the canonical signed i8/i8 -> i32 rank-2 MAC."""
    ins, outs, results = rec['ins'], rec['outs'], rec['results']
    if len(ins) != 2 or len(outs) != 1 or len(results) != 1:
        return False
    a, b = ins
    if a['dtype'] != 'i8' or b['dtype'] != 'i8' or results[0]['dtype'] != 'i32':
        return False
    if len(a['shape']) != 2 or len(b['shape']) != 2:
        return False
    m, k = a['shape']; k2, n = b['shape']
    if k != k2 or results[0]['shape'] != [m, n] or outs[0]['shape'] != [m, n]:
        return False
    # An output init participates in the reduction.  A nonzero fill must never
    # disappear just because its shape resembles an ordinary matmul destination.
    if rec.get('initializers') != [{'kind': 'linalg.fill', 'value': 0}]:
        return False
    if rec['kind'] == 'linalg.matmul':
        return True
    if rec['kind'] != 'linalg.generic':
        return False
    expected_maps = ['affine_map<(d0,d1,d2)->(d0,d2)>',
                     'affine_map<(d0,d1,d2)->(d2,d1)>',
                     'affine_map<(d0,d1,d2)->(d0,d1)>']
    if [''.join(m.split()) for m in rec['indexing_maps']] != expected_maps:
        return False
    if rec['iterator_types'] != ['parallel', 'parallel', 'reduction']:
        return False
    body = rec.get('scalar_body') or {}
    if body.get('captures') or body.get('yields') != ['op:3:0']:
        return False
    ops = body.get('operations', [])
    expected = [('arith.extsi', ['arg:0']), ('arith.extsi', ['arg:1']),
                ('arith.muli', ['op:0:0', 'op:1:0']), ('arith.addi', ['arg:2', 'op:2:0'])]
    if len(ops) != len(expected):
        return False
    return all(o['op'] == kind and o['operands'] == args and o['results'] == ['i32']
               and o.get('regions', 0) == 0 and o.get('effects') in (None, [])
               for o, (kind, args) in zip(ops, expected, strict=True))
