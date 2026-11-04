"""Source-bound descriptions of actual compiler placement, never eligibility.

These records describe the completed lowering. Static tensor extents are not
executed traffic, and a host placement does not establish hardware incapability.
"""
from __future__ import annotations

from math import prod


def _bytes(spec):
    shape, dtype = spec.get('shape'), spec.get('dtype', '')
    if not isinstance(shape, (tuple, list)) or any(type(d) is not int or d < 0 for d in shape):
        return None
    if dtype not in ('i8', 'i32', 'i64', 'f16', 'f32', 'f64', 'bf16'):
        return None
    return prod(shape) * int(dtype[2:] if dtype.startswith('bf') else dtype[1:]) // 8


def command_placement(buffer):
    """Record accepted logical effects after their target lowering succeeds."""
    tensors = buffer.get('tensors', {})
    rows, bindings = [], []
    families = {'MATMUL': 'dense', 'MATMUL_RESIDENT': 'dense', 'COMMIT': 'accumulator_readout',
        'CONV2D': 'conv', 'BATCHED_MATMUL': 'dense', 'ATTENTION_QK': 'attention',
        'ATTENTION_PV': 'attention', 'MOVEMENT': 'movement', 'STRIDED_MOVEMENT': 'movement',
        'ACCUMULATOR_READOUT': 'accumulator_readout', 'MAXPOOL2D': 'pool',
        'RESIDUAL_ADD': 'elementwise', 'BIAS_ADD': 'elementwise'}
    for index, command in enumerate(buffer.get('commands', ())):
        opcode = command['opcode']
        attrs = command.get('attributes', {})
        row = dict(command_index=index, opcode=opcode,
            source_op_index=attrs.get('merlin.source_op_index'),
            source_region_id=attrs.get('prov.region_id'),
            tensors={key: {**tensors.get(name, {}), 'name': name,
                'static_bytes': _bytes(tensors.get(name, {}))}
                for key, name in command.get('operands', {}).items() if isinstance(name, str)})
        if opcode in ('RES_PACK', 'EVICT'):
            bindings.append(dict(row, effect='logical operand binding; physical lifetime owned by target schedule'))
            continue
        if opcode not in families:
            raise ValueError('completed lowering has an unclassified placement effect: ' + opcode)
        rows.append(dict(row, lane='systolic_mesh', target='gemmini', family=families[opcode],
            form='contraction_scoped' if opcode in ('COMMIT', 'MATMUL', 'MATMUL_RESIDENT') else 'standalone_or_composed',
            reason='accepted logical effect consumed by target lowering and common terminal encoder',
            epilogue=list(attrs.get('epilogue', ())), eligibility='not independently adjudicated'))
    params = buffer.setdefault('params', {})
    params['lane_placement'] = rows
    params['placement_report'] = dict(schema='submission.actual_placement.v1',
        population='accepted logical command effects; binding-only commands separately recorded',
        device_effects=len(rows), host_arithmetic_effects=0, binding_effects=bindings,
        required_region_count=None, eligible_but_host_count=None,
        eligibility='UNKNOWN: actual lowering report is not the target eligibility census',
        static_bytes='per logical tensor extent; not deduplicated or executed memory traffic',
        measured_host_cycles=None, measured_transfer_bytes=None)
    return buffer


def source_placement(inventory, device_members):
    """Join original direct source indices to the partition actually compiled.

    ``device_members`` is captured from source operation objects before routing,
    not reconstructed from names or a saved report. Shared dependencies can be
    present in a device function while remaining live on the original host.
    """
    rows = []
    for source in inventory['operations']:
        index = source['source_op_index']
        memberships = list(device_members.get(index, ()))
        rows.append(dict(source, lane='systolic_mesh' if memberships else 'host', target='gemmini' if memberships else None,
            family=source['operation'], device_groups=memberships,
            reason=('included in checked source region compiled by target callback; shared dependencies may remain host-live'
                if memberships else 'retained by source region partition and lowered by shared Merlin host codegen; device eligibility unresolved'),
            eligibility='UNKNOWN',
            static_input_bytes=[_bytes(s) for s in source['operands']],
            static_output_bytes=[_bytes(s) for s in source['results']]))
    return dict(schema='submission.source_placement.v1', source_sha256=inventory['source_sha256'],
        population='every original direct entry operation, including initialization and views',
        source_op_count=inventory['source_op_count'], operations=rows,
        required_region_count=None, eligible_but_host_count=None,
        eligibility='UNKNOWN: this partition is not an independent coverage admission',
        static_boundary_bytes_are_traffic=False, measured_host_cycles=None)
