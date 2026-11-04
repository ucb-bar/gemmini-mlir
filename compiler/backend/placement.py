"""Declare the placement of already lowered logical operations.

This is a declaration adapter, not an eligibility oracle or traffic model. Call
it on the final logical buffer returned by ``compile_buffer``. Nothing here
changes commands, storage, the pointer ABI, or target emission.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import prod

from merlin.perf.lane_cost import dtype_bits


class PlacementError(ValueError):
    """A logical buffer cannot support an honest complete declaration."""


# Operand names are the ones consumed by compiler.py, not source spelling.
_SEMANTICS = {
    'CONV2D': ('convolution', ('ifm', 'weight')),
    'BATCHED_MATMUL': ('batched_matmul', ('a', 'w')),
    'ATTENTION_QK': ('attention', ('q', 'k')),
    'ATTENTION_PV': ('attention', ('p', 'v')),
    'BIAS_ADD': ('bias', ('src', 'bias')),
    'RESIDUAL_ADD': ('residual', ('lhs', 'rhs')),
    'MOVEMENT': ('movement', ('src',)),
    'STRIDED_MOVEMENT': ('movement', ('src',)),
    'SLICE_COPY': ('movement', ('src',)),
    'INSERT_COPY': ('movement', ('src', 'initial')),
    'ACCUMULATOR_READOUT': ('readout', ('src',)),
    'MAXPOOL2D': ('pooling', ('src',)),
}


def annotate_placement(cb):
    """Return a deep copy with one Gemmini declaration per semantic result.

    MATMUL and its COMMIT(s) are joined by accumulator def-use; residency
    handles resolve at the producer, so later eviction cannot lose provenance.
    A product committed to two outputs produces two rows. Existing placement
    records, including an outer mixed host plan, are preserved verbatim after
    logical-command validation. Malformed/unknown streams raise PlacementError
    atomically. This helper does not make unsupported hardware claims.
    """
    if not isinstance(cb, Mapping):
        raise PlacementError('logical command buffer must be a mapping')
    result = deepcopy(cb)
    commands = result.get('commands')
    params = result.get('params', {})
    if not isinstance(commands, list) or not isinstance(params, Mapping):
        raise PlacementError('commands must be a list and params a mapping')
    existing = params.get('lane_placement')
    if existing is not None and (not isinstance(existing, list) or any(
            not isinstance(row, Mapping) or not row.get('lane') or not row.get('family')
            for row in existing)):
        raise PlacementError('existing lane_placement must contain lane/family mappings')
    if 'declined' in result:
        decline = result['declined']
        if commands or not isinstance(decline, Mapping) or not isinstance(decline.get('reason'), str) or not decline['reason'].strip():
            raise PlacementError('decline requires an explicit reason and no commands')
        if existing:
            raise PlacementError('decline cannot also declare accepted/host placement')
        return result
    if not commands:
        raise PlacementError('empty non-declined stream has no accepted operations')
    # Reuse the compiler's own geometry/def-use validation; do not maintain a
    # second, weaker definition of a well-formed convolution or contraction.
    from .frontend import infer_tensors
    try:
        infer_tensors(result)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        raise PlacementError(f'malformed logical command stream: {exc}') from exc
    tensors = result.get('tensors', {})
    aliases = params.get('tensor_aliases', {})
    constants = result.get('constants', {})
    if any(not isinstance(x, Mapping) for x in (tensors, aliases, constants)):
        raise PlacementError('tensor declarations must be mappings')

    def boundary(name):
        if not isinstance(name, str) or not name:
            raise PlacementError('tensor reference must be a nonempty string')
        spec = aliases.get(name, tensors.get(name, constants.get(name)))
        if not isinstance(spec, Mapping):
            raise PlacementError(f'no typed boundary for {name!r}')
        shape, dtype = spec.get('shape'), spec.get('dtype')
        bits = dtype_bits(dtype)
        if (not isinstance(shape, (list, tuple)) or any(type(d) is not int or d <= 0 for d in shape)
                or bits is None or bits % 8):
            raise PlacementError(f'nonstatic or non-byte boundary for {name!r}')
        row = dict(tensor=name, shape=list(shape), dtype=dtype, bytes=prod(shape) * (bits // 8))
        if name in aliases:
            source = spec.get('source')
            if not isinstance(source, str) or source not in tensors:
                raise PlacementError(f'unbound storage alias {name!r}')
            row['storage_source'] = source
        return row

    residents, products, rows = {}, {}, []
    all_products = []

    def operands(a, keys):
        if set(a) != set(keys) or any(not isinstance(a[k], str) or not a[k] for k in keys):
            raise PlacementError(f'expected tensor/handle operands {keys}, got {a!r}')

    def emit(family, indices, names, dst, attributes):
        # These attributes are actual additional device input pointers. Column
        # scales are immediate arrays, not independent logical tensor buffers.
        pointer_attributes = ('bias', 'accumulator_init') if family == 'matmul' else ('bias',) if family == 'convolution' else ()
        for key in pointer_attributes:
            if attributes.get(key) is not None:
                names.append(attributes[key])
        inputs = [boundary(n) for n in dict.fromkeys(names)]
        outputs = [boundary(dst)]
        source = []
        for index in indices:
            attrs = commands[index].get('attributes', {})
            metadata = {k: deepcopy(v) for k, v in attrs.items()
                        if k.startswith('prov.') or k in ('merlin.source_op_index', 'merlin.global_task')}
            if metadata:
                source.append(dict(command_index=index, attributes=metadata))
        row = dict(lane='unit:gemmini', unit='gemmini', family=family,
                   command_indices=indices, opcodes=[commands[i]['opcode'] for i in indices],
                   inputs=inputs, outputs=outputs,
                   boundary_bytes=dict(inputs=sum(x['bytes'] for x in inputs),
                                       outputs=sum(x['bytes'] for x in outputs)),
                   byte_basis='static logical tensor boundaries; not measured transfers')
        if source:
            row['source'] = source
        bindings = params.get('frontend_source_values', [])
        if isinstance(bindings, list):
            values = [deepcopy(b) for b in bindings if isinstance(b, Mapping) and b.get('tensor') == dst]
            if values:
                row['source_results'] = values
        rows.append(row)

    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            raise PlacementError(f'command {index} is not a mapping')
        code = command.get('opcode')
        a, attrs = command.get('operands', {}), command.get('attributes', {})
        if (not isinstance(code, str) or not isinstance(a, Mapping) or not isinstance(attrs, Mapping)
                or any(not isinstance(k, str) for k in attrs)):
            raise PlacementError(f'command {index} operands/attributes are not mappings')
        if code == 'RES_PACK':
            operands(a, ('src', 'dst'))
            src = residents.get(a['src'], a['src'])
            boundary(src)
            residents[a['dst']] = src
        elif code == 'EVICT':
            operands(a, ('handle',))
            if a['handle'] not in residents:
                raise PlacementError(f'eviction of unknown resident at {index}')
            del residents[a['handle']]
        elif code in ('MATMUL', 'MATMUL_RESIDENT'):
            operands(a, ('lhs', 'rhs', 'dst'))
            names = [a['lhs'], residents.get(a['rhs'], a['rhs'])]
            for name in names:
                boundary(name)
            product = dict(index=index, names=names, attributes=attrs, commits=0)
            products[a['dst']] = product
            all_products.append(product)
        elif code == 'COMMIT':
            operands(a, ('src', 'dst'))
            if a['src'] not in products:
                raise PlacementError(f'commit of unknown accumulator at {index}')
            p = products[a['src']]
            emit('matmul', [p['index'], index], list(p['names']), a['dst'], {**p['attributes'], **attrs})
            p['commits'] += 1
        elif code == 'CONCAT_COPY':
            keys=tuple(f'src_{i}' for i in range(len(attrs.get('input_shapes',()))))
            operands(a,(*keys,'dst'))
            emit('movement',[index],[a[k] for k in keys],a['dst'],attrs)
        elif code in _SEMANTICS:
            family, keys = _SEMANTICS[code]
            operands(a, (*keys, 'dst'))
            names = [residents.get(a[k], a[k]) if code == 'CONV2D' and k == 'weight' else a[k] for k in keys]
            emit(family, [index], names, a['dst'], attrs)
        else:
            raise PlacementError(f'unknown semantic opcode {code!r} at {index}')
    if not rows or any(not p['commits'] for p in all_products):
        raise PlacementError('stream contains no semantic results or an uncommitted product')
    if existing is None:
        result.setdefault('params', {})['lane_placement'] = rows
    return result
