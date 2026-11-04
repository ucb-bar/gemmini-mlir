"""Describe realized logical lanes without asserting an all-source eligibility census.

This projection runs after code generation. It cannot select placement or change
an instruction, and deliberately separates static interface bytes from traffic.
"""
from __future__ import annotations

from math import prod


_FAMILIES = {
    'COMMIT': 'matmul',
    'CONV2D': 'conv2d',
    'BATCHED_MATMUL': 'matmul',
    'ATTENTION_QK': 'matmul',
    'ATTENTION_PV': 'matmul',
    'MOVEMENT': 'movement',
    'STRIDED_MOVEMENT': 'movement',
    'ACCUMULATOR_READOUT': 'readout',
    'RESIDUAL_ADD': 'add',
    'BIAS_ADD': 'add',
    'MAXPOOL2D': 'maxpool',
}


def annotate(command_buffer):
    """Attach source-command/region identities to lanes that actually lowered."""
    params = command_buffer.setdefault('params', {})
    if command_buffer.get('declined'):
        return command_buffer
    lanes = []
    for index, command in enumerate(command_buffer.get('commands', ())):
        opcode = command['opcode']
        if opcode not in _FAMILIES:
            # RES_PACK/EVICT are logical handles. MATMUL's real work occurs at
            # its COMMIT; counting both would duplicate one source contraction.
            continue
        attributes = command.get('attributes', {})
        lanes.append(dict(lane='accelerator.gemmini', family=_FAMILIES[opcode],
            command_index=index, opcode=opcode,
            source_op_index=attributes.get('merlin.source_op_index'),
            inputs={key: value for key, value in command.get('operands', {}).items()
                    if key not in ('dst',)},
            output=command.get('operands', {}).get('dst'),
            epilogue=list(attributes.get('epilogue', ())),
            reason='consumed by target lowering and terminal command encoder'))
    regions = params.get('device_regions', ())
    for region_index, region in enumerate(regions):
        child = region.get('command_buffer')
        if child is None:
            continue
        annotate(child)
        for lane in child.get('params', {}).get('lane_placement', ()):
            lanes.append(dict(lane, device_region_index=region_index,
                device_symbol=region.get('symbol'),
                source_index_namespace='extracted device source'))
    if command_buffer.get('backend') == 'shared_host_target_callbacks':
        lanes.append(dict(lane='host', family='complete_program_host_seams',
            reason='shared host lowering after explicit device-region extraction; '
                   'remaining per-operation hardware eligibility is not determined by this projection',
            eligibility='unknown',
            source_op_indices=params.get('global_program_plan', {}).get('tasks', [{}])[0].get('source_op_indices', [])))
    params['lane_placement'] = lanes
    boundary_bytes = 0
    for tensor in command_buffer.get('tensors', {}).values():
        if tensor.get('role') not in ('input', 'weight', 'bias', 'output'):
            continue
        dtype = tensor.get('dtype', '')
        if len(dtype) < 2 or not dtype[1:].isdigit():
            boundary_bytes = None
            break
        boundary_bytes += prod(tensor['shape']) * ((int(dtype[1:]) + 7) // 8)
    params['placement_scope'] = dict(
        schema='submission.realized_logical_lanes.v1',
        population='lowered terminal logical operations and extracted device regions',
        all_source_eligibility_counts=None,
        static_interface_tensor_bytes=boundary_bytes,
        static_bytes_are_runtime_traffic=False,
        measured_host_cycles=None)
    return command_buffer
