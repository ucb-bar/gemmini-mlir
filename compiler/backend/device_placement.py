"""Declare the device route actually emitted by the target compiler.

This does not select placement or derive hardware capabilities. The dispatcher
has already lowered every original command successfully before this is called.
Host tensor regions use the separate source-bound mixed-program declaration.
"""
from __future__ import annotations


def declare_emitted_device_program(command_buffer, *, symbol, emitted_operations):
    commands = command_buffer.get('commands', ())
    if command_buffer.get('declined') or not commands:
        raise ValueError('a device route requires a successful nonempty original command program')
    if emitted_operations <= 0:
        raise ValueError('a device route requires an actual emitted target product')
    params = command_buffer.setdefault('params', {})
    params['lane_placement'] = [dict(
        lane='accelerator', unit=command_buffer['target'], family='compiled_device_program',
        reason='all original commands lowered through the selected target dispatcher; host work is command issue, addressing and completion',
        symbol=symbol, source_command_indices=list(range(len(commands))),
        operations=[command['opcode'] for command in commands],
        emitted_operation_count=emitted_operations,
        outputs=list(command_buffer.get('kernel_abi', {}).get('outputs', ())),
        evidence='actual emitted target program; no elapsed-time or engine-occupancy claim')]
    return command_buffer
