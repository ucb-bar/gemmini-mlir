"""Compact source-SSA work/placement profiles from original graphs and plans.

Counts describe one entry evaluation by default. They are static logical work,
never observed time, physical traffic, occupancy, or accelerator eligibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from math import prod


def _name(op):
    value = getattr(op, 'op_name', None)
    return value.data if op.name == 'builtin.unregistered' and value is not None else op.name


def _type(value):
    from xdsl.dialects.builtin import TensorType
    ty = value.type
    if not isinstance(ty, TensorType):
        return {'type': str(ty), 'shape': None, 'dtype': str(ty), 'logical_bytes': None}
    shape = list(ty.get_shape())
    dtype = str(ty.element_type)
    width = getattr(ty.element_type, 'bitwidth', None)
    if callable(width):
        width = width()
    if not isinstance(width, int):
        digits = dtype[1:] if dtype.startswith(('i', 'f')) else '16' if dtype == 'bf16' else ''
        width = int(digits) if digits.isdigit() else None
    size = ((prod(shape) * width + 7) // 8
            if width is not None and all(d >= 0 for d in shape) else None)
    return {'type': str(ty), 'shape': shape, 'dtype': dtype,
            'encoding': str(ty.encoding), 'logical_bytes': size}


def _literal(value):
    op = value.owner
    if getattr(op, 'name', '') != 'arith.constant':
        return None
    attr = {**op.attributes, **op.properties}.get('value')
    raw = getattr(getattr(attr, 'value', None), 'data', None)
    return raw if type(raw) is int else None


def _loop_count(op):
    if op.name == 'scf.for':
        bounds = [_literal(v) for v in op.operands[:3]]
        if len(bounds) == 3 and all(v is not None for v in bounds) and bounds[2] > 0:
            lower, upper, step = bounds
            return max(0, (upper - lower + step - 1) // step)
    return None


def _domain(op):
    """Infer iteration extents only from consistent projected source axes."""
    from xdsl.ir.affine import AffineDimExpr
    maps = getattr(op, 'indexing_maps', None)
    if maps is None:
        return None
    domains = list(maps)
    if len(domains) != len(op.operands) or not domains:
        return None
    count = domains[0].data.num_dims
    extents = [None] * count
    for mapping, value in zip(domains, op.operands):
        shape = _type(value)['shape']
        if shape is None or len(shape) != len(mapping.data.results):
            return None
        for expr, size in zip(mapping.data.results, shape):
            if not isinstance(expr, AffineDimExpr) or size < 0:
                return None
            old = extents[expr.position]
            if old is not None and old != size:
                return None
            extents[expr.position] = size
    return extents if all(size is not None for size in extents) else None


def _macs(op, domain):
    """Recognize exact scalar sum-products; provenance is not work proof."""
    inputs = tuple(getattr(op, 'inputs', ()))
    if len(inputs) != 2 or len(op.results) != 1:
        return None
    if op.name in ('linalg.matmul', 'linalg.batch_matmul', 'linalg.matmul_transpose_b'):
        a, b = (_type(v)['shape'] for v in inputs)
        out = _type(op.results[0])['shape']
        if a and b and out and all(d >= 0 for d in (*a, *b, *out)):
            return prod(out) * a[-1]
        return None
    if op.name != 'linalg.generic' or domain is None or len(op.regions) != 1:
        return None
    block = op.regions[0].block
    term = block.last_op
    if len(block.args) != 3 or term is None or term.name != 'linalg.yield' or len(term.operands) != 1:
        return None
    def strip(value):
        while getattr(value.owner, 'name', '') in ('arith.extsi', 'arith.extui', 'arith.extf'):
            value = value.owner.operands[0]
        return value
    addition = term.operands[0].owner
    if getattr(addition, 'name', '') not in ('arith.addi', 'arith.addf'):
        return None
    for accumulator, product in (tuple(addition.operands), tuple(reversed(addition.operands))):
        multiply = product.owner
        if (accumulator is block.args[2] and getattr(multiply, 'name', '') in ('arith.muli', 'arith.mulf')
                and {strip(v) for v in multiply.operands} == set(block.args[:2])):
            iterators = getattr(op, 'iterator_types', ())
            if any('reduction' in str(it) for it in iterators):
                return prod(domain)
    return None


def _work(op, functions, active=()):
    """Exact known work subtotal, plus explicit unsupported/dynamic observations."""
    name = _name(op)
    domain = _domain(op)
    macs = _macs(op, domain)
    scalar = [nested for nested in op.walk() if nested is not op
              and (nested.name.startswith('arith.') or nested.name.startswith('math.'))]
    iterations = prod(domain) if domain is not None else None
    work = {'known_macs': macs or 0, 'known_scalar_evaluations':
            iterations * len(scalar) if iterations is not None else 0,
            'unknown_work': [], 'iteration_extents': domain,
            'contraction': macs is not None}
    if name.startswith('linalg.') and scalar and iterations is None:
        work['unknown_work'].append('iteration domain not proved from projected static source axes')
    if name in ('scf.for', 'scf.while', 'scf.if', 'scf.parallel'):
        count = _loop_count(op)
        for region in op.regions:
            for block in region.blocks:
                for child in block.ops:
                    if child.name.startswith('scf.') and child.name.endswith('yield'):
                        continue
                    nested = _work(child, functions, active)
                    if count is not None:
                        work['known_macs'] += count * nested['known_macs']
                        work['known_scalar_evaluations'] += count * nested['known_scalar_evaluations']
                    work['unknown_work'].extend(nested['unknown_work'])
        if count is None:
            work['unknown_work'].append('runtime control/trip count unknown: ' + name)
        work['literal_trip_count'] = count
    if name == 'func.call':
        callee = str(op.callee.root_reference.data)
        if (callee not in functions or callee in active
                or len(functions[callee].body.blocks) != 1):
            work['unknown_work'].append('unavailable or recursive callee: ' + callee)
        else:
            for child in functions[callee].body.block.ops:
                nested = _work(child, functions, (*active, callee))
                work['known_macs'] += nested['known_macs']
                work['known_scalar_evaluations'] += nested['known_scalar_evaluations']
                work['unknown_work'].extend(nested['unknown_work'])
    if name.startswith(('arith.', 'math.')) and not op.regions:
        # Scalar operations only; tensor arithmetic needs an explicit domain.
        if all(_type(v)['shape'] is None for v in (*op.operands, *op.results)):
            work['known_scalar_evaluations'] += 1
    return work


def _invocations(op, functions, count=1, active=(), path=()):
    """Control/call multiplicities, without interpreting arbitrary annotations."""
    records = []
    if op.name == 'func.call':
        callee = op.callee.root_reference.data
        records.append({'path': list(path), 'operation': op.name, 'callee': callee,
                        'invocations_per_entry': count})
        if callee in functions and callee not in active and len(functions[callee].body.blocks) == 1:
            for index, child in enumerate(functions[callee].body.block.ops):
                records.extend(_invocations(child, functions, count, (*active, callee), (*path, index)))
        return records
    child_count = count
    if op.name.startswith('scf.') and op.regions:
        trip = _loop_count(op)
        records.append({'path': list(path), 'operation': op.name,
                        'invocations_per_entry': count, 'literal_trip_count': trip})
        child_count = count * trip if count is not None and trip is not None else None
    for region_index, region in enumerate(op.regions):
        for block_index, block in enumerate(region.blocks):
            for index, child in enumerate(block.ops):
                records.extend(_invocations(child, functions, child_count, active,
                                             (*path, region_index, block_index, index)))
    return records


def build_source_profile(source, command_buffer, *, entry_invocations=1, placement_report=None):
    """Profile original SSA and actual source-owning tasks, without mutating IR.

    ``entry_invocations`` is a caller-declared static multiplicity; ``None`` means
    unknown. Default counts are per single entry evaluation. It is never inferred
    from capsule names, source IDs, or frontend trace statistics.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
    from source_attribution import task_definitions
    if entry_invocations is not None and (type(entry_invocations) is not int or entry_invocations < 0):
        raise ValueError('entry_invocations must be a nonnegative static integer or None')
    source = source.decode() if isinstance(source, bytes) else source
    inventory = source_operation_inventory(source)
    declared_digest = command_buffer.get('params', {}).get('source_sha256')
    if declared_digest is not None and declared_digest != inventory['source_sha256']:
        raise ValueError('task plan digest differs from original profiled source')
    module = parse_mlir_text(source)
    functions = {op.sym_name.data: op for op in module.body.block.ops if op.name == 'func.func'}
    owner = functions[inventory['entry']]
    operations = tuple(op for op in owner.body.block.ops if op.name != 'func.return')
    ordinals = {op: i for i, op in enumerate(operations)}
    works = [_work(op, functions, (inventory['entry'],)) for op in operations]
    gaps = []
    if declared_digest is None:
        gaps.append('plan has no source digest; ownership join uses supplied original source ordinals')
    try:
        tasks = task_definitions(command_buffer)
    except ValueError as exc:
        tasks = []
        gaps.append(str(exc))
    placements = (placement_report or command_buffer.get('params', {}).get('source_placement_diagnostics') or {})
    placement_rows = {r['source_op_index']: r for r in placements.get('operations', [])}
    tasks_by_op = defaultdict(list)
    for task in tasks:
        for index in task['source_op_indices']:
            if type(index) is not int or not 0 <= index < len(operations):
                gaps.append('task references absent original operation: ' + str(index))
                continue
            tasks_by_op[index].append(task['task_index'])
    def placement(index):
        owned = [task for task in tasks if task['task_index'] in tasks_by_op[index]]
        kinds = {task['kind'] == 'host_source' for task in owned}
        return 'mixed' if len(kinds) > 1 else 'host' if kinds == {True} else 'accelerator' if kinds == {False} else 'unassigned'
    def signature(index):
        op = operations[index]
        return {'operation': _name(op), 'inputs': [_type(v) for v in op.operands],
                'outputs': [_type(v) for v in op.results],
                'layout_properties': {k: str(v) for k, v in op.properties.items()
                    if k in ('indexing_maps', 'permutation', 'reassociation', 'static_offsets',
                             'static_sizes', 'static_strides', 'static_output_shape')},
                'scalar_body': dict(Counter(_name(child) for child in op.walk() if child is not op)),
                'iteration_extents': works[index]['iteration_extents']}
    signatures = {}
    for index in range(len(operations)):
        row = signature(index)
        key = json.dumps(row, sort_keys=True)
        if key not in signatures:
            signatures[key] = {**row, 'source_op_indices': [], 'placement_counts': Counter()}
        signatures[key]['source_op_indices'].append(index)
        signatures[key]['placement_counts'][placement(index)] += 1
    for row in signatures.values():
        row['static_occurrences_per_entry'] = len(row['source_op_indices'])
        row['declared_invocations'] = (len(row['source_op_indices']) * entry_invocations
                                       if entry_invocations is not None else None)
        row['placement_counts'] = dict(row['placement_counts'])
        row['known_macs_per_entry'] = sum(works[i]['known_macs'] for i in row['source_op_indices'])
        row['known_scalar_evaluations_per_entry'] = sum(works[i]['known_scalar_evaluations'] for i in row['source_op_indices'])
        row['source_region_ids'] = sorted({str(operations[i].attributes['prov.region_id'])
            for i in row['source_op_indices'] if 'prov.region_id' in operations[i].attributes})
    quant = []
    for index, op in enumerate(operations):
        before = {_type(v)['dtype'] for v in op.operands if _type(v)['shape'] is not None}
        after = {_type(v)['dtype'] for v in op.results if _type(v)['shape'] is not None}
        conversions = [child for child in op.walk() if child.name in
                       ('arith.sitofp', 'arith.uitofp', 'arith.fptosi', 'arith.fptoui', 'arith.trunci')]
        if _name(op).startswith('quant_ext.') or conversions or (before and after and before != after):
            quant.append({'source_op_index': index, 'operation': _name(op),
                          'input_dtypes': sorted(before), 'output_dtypes': sorted(after),
                          'attributes': {k: str(v) for k, v in op.attributes.items()},
                          'properties': {k: str(v) for k, v in op.properties.items()},
                          'numeric_body': [{'operation': _name(child),
                              'operand_types': [str(v.type) for v in child.operands],
                              'result_types': [str(v.type) for v in child.results],
                              'attributes': {k: str(v) for k, v in child.attributes.items() if not k.startswith('prov.')},
                              'properties': {k: str(v) for k, v in child.properties.items()}}
                              for child in op.walk() if child is not op],
                          'placement': placement(index),
                          'observation': 'typed conversion boundary; quantization intent not inferred'})
    returned = {v for op in owner.body.block.ops if op.name == 'func.return' for v in op.operands}
    edges = []
    for index, op in enumerate(operations):
        for result_index, value in enumerate(op.results):
            consumers = set()
            for use in value.uses:
                user = use.operation
                while user not in ordinals and user.parent_op() is not None:
                    user = user.parent_op()
                if user in ordinals:
                    consumers.add(ordinals[user])
            for consumer in sorted(consumers):
                src_tasks, dst_tasks = tasks_by_op[index], tasks_by_op[consumer]
                if src_tasks != dst_tasks:
                    edges.append({'producer_op': index, 'result_index': result_index,
                                  'consumer_op': consumer, 'producer_tasks': src_tasks,
                                  'consumer_tasks': dst_tasks, 'logical_bytes': _type(value)['logical_bytes'],
                                  'private_intermediate': value not in returned,
                                  'producer_placement': placement(index), 'consumer_placement': placement(consumer)})
    regions = []
    for task in tasks:
        indices = sorted(set(i for i in task['source_op_indices'] if type(i) is int and 0 <= i < len(operations)))
        unknown = sorted({reason for i in indices for reason in works[i]['unknown_work']})
        forms = {form for i in indices for form in placement_rows.get(i, {}).get('selected_device_forms', [])}
        commands = command_buffer.get('commands', [])
        forms.update(commands[i]['opcode'] for i in task.get('command_indices', [])
                     if type(i) is int and 0 <= i < len(commands))
        forms.update(str(t.get('kind', t.get('operation', 'source_tensor_transport')))
            for t in command_buffer.get('params', {}).get('source_transport_tasks', [])
            if t.get('source_task_index') == task['task_index'])
        region = {'task_index': task['task_index'], 'kind': task['kind'], 'placement':
                  'host' if task['kind'] == 'host_source' else 'accelerator',
                  'source_op_indices': indices, 'operator_counts': dict(Counter(_name(operations[i]) for i in indices)),
                  'source_region_ids': sorted({str(operations[i].attributes['prov.region_id'])
                      for i in indices if 'prov.region_id' in operations[i].attributes}),
                  'contraction_indices': [i for i in indices if works[i]['contraction']],
                  'known_macs_per_entry': sum(works[i]['known_macs'] for i in indices),
                  'known_scalar_evaluations_per_entry': sum(works[i]['known_scalar_evaluations'] for i in indices),
                  'static_output_elements_per_entry': sum(prod(_type(v)['shape'])
                      for i in indices for v in operations[i].results
                      if _type(v)['shape'] is not None and all(d >= 0 for d in _type(v)['shape'])),
                  'unknown_work': unknown, 'entry_invocations': entry_invocations,
                  'selected_device_forms': sorted(forms),
                  'host_reasons': sorted({placement_rows[i]['host_reason'] for i in indices
                                          if placement_rows.get(i, {}).get('host_reason')})}
        private = {(e['producer_op'], e['result_index']): e['logical_bytes'] for e in edges
                   if task['task_index'] in e['producer_tasks'] and e['private_intermediate']}
        region['private_outgoing_tensor_count'] = len(private)
        region['private_outgoing_logical_bytes'] = (sum(private.values())
            if all(size is not None for size in private.values()) else None)
        region['logical_byte_scope'] = 'distinct source SSA tensors leaving task, excluding program returns'
        regions.append(region)
    # Nearest contraction ancestors through actual SSA edges; no name-based stage inference.
    ancestors = {}
    interstage = {}
    for index, op in enumerate(operations):
        upstream = defaultdict(set)
        for value in op.operands:
            parent = ordinals.get(value.owner)
            if parent is None:
                continue
            if works[parent]['contraction']:
                upstream[parent].add(value)
            else:
                for contraction, path in ancestors.get(parent, {}).items():
                    upstream[contraction].update((*path, value))
        if works[index]['contraction']:
            for contraction, path in upstream.items():
                private_values = [v for v in path if v not in returned]
                sizes = [_type(v)['logical_bytes'] for v in private_values if _type(v)['shape'] is not None]
                interstage[(contraction, index)] = {'producer_contraction': contraction,
                    'consumer_contraction': index,
                    'intermediate_op_indices': sorted({ordinals[v.owner] for v in path} - {contraction}),
                    'private_tensor_count': sum(_type(v)['shape'] is not None for v in private_values),
                    'byte_scope': 'distinct connecting source SSA tensor values, including producer result, excluding program returns',
                    'private_intermediate_logical_bytes': sum(sizes) if all(s is not None for s in sizes) else None}
        ancestors[index] = upstream
    ranked_host = sorted((r for r in regions if r['placement'] == 'host'),
        key=lambda r: (r['known_scalar_evaluations_per_entry'], r['known_macs_per_entry'],
                       r['static_output_elements_per_entry']), reverse=True)
    return {'schema': 'source_bound_work_profile_v1', 'source_sha256': inventory['source_sha256'],
            'entry': inventory['entry'], 'entry_invocations': entry_invocations,
            'program_inputs': inventory['arguments'], 'program_outputs': inventory['returns'],
            'scope': 'static logical source work per entry; no measured time, cycles, occupancy, or physical traffic',
            'counts': {'source_operations': len(operations), 'tasks': len(tasks),
                       'contractions': sum(w['contraction'] for w in works),
                       'known_macs_per_entry': sum(w['known_macs'] for w in works),
                       'known_macs_declared_invocations': sum(w['known_macs'] for w in works) * entry_invocations if entry_invocations is not None else None,
                       'operations_with_unknown_work': sum(bool(w['unknown_work']) for w in works),
                       'unassigned_operations': sum(not tasks_by_op[i] for i in range(len(operations)))},
            'operator_shape_groups': list(signatures.values()), 'regions': regions,
            'control_and_call_multiplicities': [record for i, op in enumerate(operations)
                for record in _invocations(op, functions, active=(inventory['entry'],), path=(i,))],
            'unknown_work': [{'source_op_index': i, 'reasons': work['unknown_work']}
                             for i, work in enumerate(works) if work['unknown_work']],
            'contraction_stages': [{'source_op_index': i, 'task_indices': tasks_by_op[i],
                'placement': placement(i), 'known_macs_per_entry': work['known_macs'],
                'invocations_per_entry': 1, 'source_attributes':
                {k: str(v) for k, v in operations[i].attributes.items()},
                'iteration_extents': work['iteration_extents']}
                for i, work in enumerate(works) if work['contraction']],
            'ranked_host_task_indices': [r['task_index'] for r in ranked_host],
            'host_tasks_with_unknown_work': [r['task_index'] for r in ranked_host if r['unknown_work']],
            'ranking_basis': 'descending proved scalar evaluations, then proved MACs and output elements; unknown work is not zero cost',
            'quantization_and_conversion_boundaries': quant, 'task_graph_edges': edges,
            'contraction_interstage_edges': list(interstage.values()), 'gaps': gaps}


def attach_source_profile(source, command_buffer, **options):
    profile = build_source_profile(source, command_buffer, **options)
    command_buffer.setdefault('params', {})['source_work_profile'] = profile
    return profile
