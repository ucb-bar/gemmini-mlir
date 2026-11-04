"""Conservative static storage intervals for the emitted sequential CB semantics.

This analysis neither changes the CB nor emits LLVM. The caller supplies the
complete logical ABI names, runs this after graph rewrites, and creates one
64-byte-aligned typed global per returned slot. Only listed tensor names may
share that slot; lowering-created scratch absent from the CB remains private.
No device completion is inferred from a scheduler's dependency record.
"""
from __future__ import annotations

from math import prod


_WIDTH = {'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
          'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8, 'i1': 1}
# Audited implementation boundaries in matmul, batched, elementwise,
# layout_movement, convolution/pooling and siblings. Deferred declarations
# deliberately do not appear here. Host regions are synchronous CPU code, but
# are not device completion fences and therefore cannot alone authorize reuse.
_END_FENCE = {'COMMIT', 'CONV2D', 'MOVEMENT', 'RESIDUAL_ADD', 'BIAS_ADD',
              'BATCHED_MATMUL', 'ATTENTION_QK', 'ATTENTION_PV'}
_FENCES = {'FENCE', 'BARRIER', 'SYNC'}
_ALIAS_KEYS = {'alias', 'aliases', 'alias_of', 'view_of', 'storage',
               'storage_id', 'storage_offset', 'strides', 'byte_offset',
               'alias_set', 'base_tensor', 'base_buffer', 'offset_elements'}


def _strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        return [s for v in value for s in _strings(v)]
    return []


def _layout_full(attrs, spec):
    fill, copy = attrs.get('layout_fill'), attrs.get('layout_copy')
    if fill is not None:
        return (isinstance(fill, dict) and fill.get('schema') == 'constant_layout_fill_v1'
                and fill.get('result_shape') == spec.get('shape'))
    if copy is None:
        return True
    if not isinstance(copy, dict) or copy.get('result_shape') != spec.get('shape'):
        return False
    if copy.get('schema') in ('contiguous_layout_copy_v1', 'strided_layout_copy_v1'):
        return True
    # Indexed insert/concat commands may write only a part of the destination.
    # Even a complete sequence is not assumed to initialize every byte unless
    # an earlier ordinary full writer has already done so.
    return False


def _events(cb):
    tensors = cb.get('tensors', {})
    residents, deferred, events, reasons = {}, {}, [], []
    def resolve(name):
        seen = set()
        while name in residents and name not in seen:
            seen.add(name)
            name = residents[name]
        if name in seen:
            reasons.append('cyclic resident alias')
        return name
    def known(names):
        return set(n for n in _strings(names) if n in tensors)
    for index, command in enumerate(cb.get('commands', ())):
        opcode = command.get('opcode')
        op, attrs = command.get('operands') or {}, command.get('attributes') or {}
        reads, writes, full = set(), set(), set()
        boundary = None
        if any(k in attrs for k in _ALIAS_KEYS):
            reasons.append(f'command {index} carries unproved storage aliases')
        if opcode == 'RES_PACK':
            if isinstance(op.get('src'), str) and isinstance(op.get('dst'), str):
                residents[op['dst']] = resolve(op['src'])
                # The emitted RES_PACK itself only records a handle. Retaining
                # this source at declaration too is conservative if a writer
                # changes it before the eventual COMMIT.
                reads = known(op['src'])
            else:
                reasons.append(f'malformed resident pack at {index}')
        elif opcode in ('MATMUL_RESIDENT', 'MATMUL'):
            if all(isinstance(op.get(k), str) for k in ('lhs', 'rhs', 'dst')):
                deferred[op['dst']] = (op.copy(), attrs.copy())
                reads = known([op['lhs'], resolve(op['rhs'])])
            else:
                reasons.append(f'malformed deferred matmul at {index}')
        elif opcode == 'COMMIT':
            operation = deferred.get(op.get('src'))
            if operation is None:
                reasons.append(f'unresolved deferred matmul at {index}')
            else:
                args, matmul_attrs = operation
                reads = known([args['lhs'], resolve(args['rhs'])])
                combined = {**matmul_attrs, **attrs}
                reads |= known(op.get('bias', combined.get('bias')))
            writes = known(op.get('dst'))
            full = writes.copy()
        elif opcode == 'EVICT':
            residents.pop(op.get('handle', op.get('src')), None)
        elif opcode in _FENCES:
            boundary = 'explicit completion fence'
        elif opcode == 'VECTOR_MAP' and attrs.get('op') == 'sibling_matmul':
            members = attrs.get('members')
            if not isinstance(members, list) or not members:
                reasons.append(f'missing sibling effects at {index}')
            else:
                for member in members:
                    reads |= known([member.get('lhs'), member.get('rhs'),
                                    member.get('attributes', {}).get('bias')])
                    writes |= known(member.get('dst'))
                full = writes.copy()
                boundary = 'sibling lowerer end fence'
        elif opcode == 'HOST_REGION' or (opcode == 'VECTOR_MAP' and attrs.get('op') == 'host_region'):
            inputs = attrs.get('input_tensors', op.get('inputs'))
            outputs = attrs.get('output_tensors', op.get('outputs'))
            if (not isinstance(inputs, (list, tuple)) or not isinstance(outputs, (list, tuple))
                    or not attrs.get('host_module_ir')
                    or len(known(inputs)) != len(set(_strings(inputs)))
                    or len(known(outputs)) != len(set(_strings(outputs)))):
                reasons.append(f'unproved closed host effects at {index}')
            reads, writes = known(inputs), known(outputs)
            full = writes.copy()
        elif opcode in _END_FENCE:
            writes = known(op.get('dst'))
            reads = known([resolve(v) for key, value in op.items() if key != 'dst'
                           for v in _strings(value)])
            reads |= known(attrs.get('bias'))
            full = writes.copy()
            if opcode == 'MOVEMENT':
                full = {n for n in writes if _layout_full(attrs, tensors[n])}
        else:
            # Unknown instructions could capture an address, alias another
            # tensor or leave device work outstanding: do not reuse anything.
            reasons.append(f'unknown emitted effects for {opcode!r} at {index}')
            reads = known(list(op.values()))
        if opcode in _END_FENCE:
            boundary = opcode + ' lowerer end fence'
        events.append({'index': index, 'opcode': opcode, 'reads': reads,
                       'writes': writes, 'full_writes': full, 'completion': boundary})
    return events, reasons


def plan_storage(cb, external_names):
    """Return a JSON-safe typed-slot plan without mutating input dictionaries.

    `tensors[name]` contains slot, offset (always zero), bytes, first/last CB
    instruction indices and proof. `slots` gives dtype, allocated bytes,
    alignment and occupants. External/constant tensors are excluded. A private
    tensor gets its own slot, so callers can use this result unconditionally.
    Reuse has strict interval separation and a completion boundary between
    occupants. Unknown aliases/effects disable all reuse, never compilation.
    """
    external = set(external_names)
    tensors = cb.get('tensors', {})
    # The explicit ABI remains authoritative even if a caller's name set was
    # incomplete or an output carries a generic intermediate role in the CB.
    abi = cb.get('kernel_abi') or {}
    external.update(_strings(abi.get('outputs')))
    for arg in abi.get('args', ()):
        if isinstance(arg, dict):
            external.update(_strings(arg.get('tensor')))
    constants = set()
    declarations = cb.get('constants', {})
    if isinstance(declarations, dict):
        constants.update(declarations)
    for command in cb.get('commands', ()):
        attrs = command.get('attributes') or {}
        if command.get('opcode') == 'MOVEMENT' and attrs.get('layout_fill'):
            constants.update(_strings((command.get('operands') or {}).get('dst')))
    events, reasons = _events(cb)
    for name, spec in tensors.items():
        if any(k in spec for k in _ALIAS_KEYS):
            reasons.append(f'tensor {name!r} carries unproved storage aliases')
    boundaries = {e['index']: e['completion'] for e in events if e['completion']}
    rows, excluded = {}, {}
    for name, spec in tensors.items():
        if (name in external or name in constants
                or spec.get('role') in ('input', 'output', 'constant', 'const', 'weight', 'bias', 'parameter')
                or any(k in spec for k in ('data', 'values', 'value', 'initializer'))):
            excluded[name] = 'logical ABI or immutable constant storage'
            continue
        uses = [e for e in events if name in e['reads'] or name in e['writes']]
        if not uses:
            excluded[name] = 'not materialized by declared CB effects'
            continue
        shape, dtype = spec.get('shape'), spec.get('dtype')
        if (not isinstance(shape, (tuple, list)) or any(type(n) is not int or n <= 0 for n in shape)
                or dtype not in _WIDTH):
            excluded[name] = 'unknown static storage geometry; retain original private allocation'
            continue
        first, last = uses[0]['index'], uses[-1]['index']
        private = []
        if name in uses[0]['reads']:
            private.append('read before a proven complete writer')
        if name not in uses[0]['full_writes']:
            private.append('first writer does not prove full initialization')
        # A zero-initialized global may supply untouched bytes for partial
        # writes. Sharing it would change those bytes on subsequent uses.
        if any(name in e['reads'] and name in e['writes'] for e in uses):
            private.append('in-place storage effects')
        rows[name] = {'slot': None, 'offset': 0, 'bytes': prod(shape) * _WIDTH[dtype],
                      'dtype': dtype, 'first': first, 'last': last,
                      'reusable': not private and not reasons,
                      'proof': private or ['full first writer; bounded named storage; sequential uses'],
                      'reuse_after': None}
    slots = []
    for name, row in sorted(rows.items(), key=lambda item: (item[1]['first'], item[0])):
        candidates = []
        if row['reusable']:
            for slot in slots:
                old = rows[slot['occupants'][-1]]
                fence = next((b for b in boundaries if old['last'] <= b < row['first']), None)
                if (slot['reusable'] and slot['dtype'] == row['dtype']
                        and old['last'] < row['first'] and fence is not None):
                    candidates.append((max(slot['bytes'], row['bytes']), slot['slot'], fence))
        if candidates:
            _, index, fence = min(candidates)
            slot = slots[index]
            previous = slot['occupants'][-1]
            row['reuse_after'] = {'tensor': previous, 'completion_index': fence,
                                  'completion_proof': boundaries[fence]}
            row['proof'].append('strictly disjoint lifetimes separated by audited completion boundary')
            slot['bytes'] = max(slot['bytes'], row['bytes'])
            slot['occupants'].append(name)
        else:
            index = len(slots)
            slot = {'slot': index, 'dtype': row['dtype'], 'bytes': row['bytes'],
                    'alignment': 64, 'reusable': row['reusable'], 'occupants': [name]}
            slots.append(slot)
        row['slot'] = index
    aligned = lambda size: (size + 63) // 64 * 64
    baseline = sum(aligned(r['bytes']) for r in rows.values())
    planned = sum(aligned(s['bytes']) for s in slots)
    return {'schema': 'sequential_storage_plan_v1', 'tensors': rows, 'slots': slots,
            'excluded': excluded, 'disabled_reasons': list(dict.fromkeys(reasons)),
            'completion_boundaries': [{'index': i, 'proof': proof} for i, proof in boundaries.items()],
            'totals': {'materialized_tensors': len(rows), 'slots': len(slots),
                       'private_aligned_bytes': baseline, 'planned_aligned_bytes': planned,
                       'saved_aligned_bytes': baseline - planned},
            'scope': 'CB-declared internal storage only; hidden lowering scratch remains private'}
