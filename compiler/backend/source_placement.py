"""Placement evidence from the live source and actual compiled callbacks.

These records are static source attribution, not execution timing, MAC accounting
or an accelerator eligibility oracle. Unknown host proofs remain unknown.
"""
from math import prod


def _attr(op, name):
    value = op.attributes.get(name)
    return getattr(value, 'data', None)


def capture(entry, inventory):
    """Bind original direct operation identities before routing mutates source."""
    operations = tuple(op for op in entry.body.block.ops if op.name != 'func.return')
    if len(operations) != inventory['source_op_count']:
        raise ValueError('source placement inventory differs from live original entry')
    records = []
    for op, row in zip(operations, inventory['operations']):
        name = (_attr(op, 'op_name__') or op.name) if op.name == 'builtin.unregistered' else op.name
        if name != row['operation']:
            raise ValueError('source placement operation differs from exact inventory')
        records.append(dict(row, family=_attr(op, 'prov.family') or op.name,
                            source_operation=_attr(op, 'prov.op') or op.name))
    return operations, records


def tensor_record(value):
    from merlin.perf.lane_cost import dtype_bits
    shape = list(value.type.get_shape())
    dtype = str(value.type.element_type)
    bits = dtype_bits(dtype)
    return dict(shape=shape, dtype=dtype,
                bytes=None if bits is None or any(d < 0 for d in shape)
                else prod(shape) * ((bits + 7) // 8))


def selected_region(region, originals, artifact):
    """Claim a device region only when its callback supplied actual commands."""
    cb = artifact.command_buffer
    if not isinstance(cb, dict) or not isinstance(cb.get('commands'), list) or not cb['commands']:
        raise ValueError('device placement needs a nonempty actual callback command buffer')
    if cb.get('target') != 'gemmini':
        raise ValueError('device placement callback does not identify the selected Gemmini target')
    indices = {op: index for index, op in enumerate(originals)}
    inputs = [tensor_record(value) for value in region.inputs]
    outputs = [tensor_record(value) for value in region.outputs]
    sizes = [row['bytes'] for row in (*inputs, *outputs)]
    return dict(source_op_indices=sorted({indices[op] for op in region.operations if op in indices}),
                input_types=inputs, output_types=outputs,
                static_boundary_bytes=None if None in sizes else sum(sizes),
                device_forms=list(dict.fromkeys(command['opcode'] for command in cb['commands'])),
                callback_command_count=len(cb['commands']), callback_target=cb.get('target'),
                source_reason=region.reason)


def lane_placement(inventory, records, groups, artifacts):
    """Expose compiled groups and explicit source host seams without fake commands."""
    if len(groups) != len(artifacts):
        raise ValueError('source placement lost compiled callback correspondence')
    result, covered = [], set()
    for group, artifact in zip(groups, artifacts):
        cb = artifact.command_buffer
        if not isinstance(cb, dict) or not cb.get('commands'):
            raise ValueError('source placement cannot claim an empty device route')
        selected = group['source_op_indices']
        if any(index < 0 or index >= len(records) for index in selected):
            raise ValueError('source placement references an absent original operation')
        covered.update(selected)
        families = list(dict.fromkeys(records[index]['family'] for index in selected
                    if records[index]['operation'] not in ('tensor.empty','tensor.splat','arith.constant','linalg.fill')))
        result.append(dict(lane='accelerator', family=families[0] if len(families) == 1 else 'composed',
            reason='actual compiled Gemmini callback with nonempty target commands; ' + group['source_reason'],
            unit='gemmini', symbol=artifact.symbol, source_sha256=inventory['source_sha256'],
            device_source_sha256=group['source_sha256'],
            source_op_indices=selected, source_families=families,
            source_operations=[records[index] for index in selected],
            numerical_contract='exact typed extracted source supplied to device callback',
            input_types=group['input_types'], output_types=group['output_types'],
            static_boundary_bytes=group['static_boundary_bytes'], device_forms=group['device_forms'],
            callback_command_count=len(cb['commands']), artifact=str(artifact.path)))
    bookkeeping = []
    for index, row in enumerate(records):
        if index in covered:
            continue
        if row['operation'] in ('tensor.empty','tensor.splat','arith.constant','linalg.fill'):
            bookkeeping.append(index)
            continue
        result.append(dict(lane='host', family=row['family'],
            reason='original source tensor operation has no selected compiled device callback; '
                   'shared host lowering preserves its typed semantics; device composition proof remains unresolved',
            status='unresolved_device_proof', source_sha256=inventory['source_sha256'],
            source_op_indices=[index], source_operations=[row],
            numerical_contract='original typed source arithmetic; no hardware-unsupported claim'))
    return result, dict(schema='submission.source_placement.v1', source_sha256=inventory['source_sha256'],
        source_op_count=inventory['source_op_count'], compiled_device_regions=len(groups),
        host_or_unresolved_regions=sum(row['lane'] == 'host' for row in result),
        source_bookkeeping_indices=bookkeeping,
        scope='actual compiled callbacks plus original tensor seams; source bookkeeping excluded from compute regions',
        static_boundary_bytes=sum(g['static_boundary_bytes'] for g in groups)
            if all(g['static_boundary_bytes'] is not None for g in groups) else None,
        boundary_bytes_kind='static sum of callback input/output footprints; not measured traffic',
        macs='not_derived', timing='not_measured')
