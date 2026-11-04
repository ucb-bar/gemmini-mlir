"""Optional explicit publication of the existing legacy logical pointer boundary.

This declaration contains no numerical values and changes neither the source nor
its call order. It is not a mixed-program attribution plan.
"""
from pathlib import Path

from merlin.compile.logical_abi import (kernel_arg_order, logical_abi,
                                        logical_inputs, logical_outputs)
from merlin.compile.tuning_knobs import knob


def bind_explicit_legacy_abi(program):
    """Return a whole-program pointer record, or None for the retained/typed route.

    The caller attaches the returned record as kernel_abi before serialization.
    Only already typed tensors are admitted: a missing COMMIT declaration is a
    frontend error, never an opportunity to guess its shape or readout dtype.
    """
    if program.get('source_format') != 'merlin_iface':
        return None
    if knob('legacy_abi', default='retained', values=('retained', 'explicit'),
            root=Path(__file__).parent) == 'retained':
        return None
    abi = logical_abi()
    order = kernel_arg_order(program, abi)
    tensors = program.get('tensors') or {}
    inputs = logical_inputs(program)
    outputs = logical_outputs(program, abi.published_by)
    if len(set(order)) != len(order) or set(order) != set(tensors):
        raise ValueError('explicit legacy ABI requires the existing logical argument order '
                         'to cover every declared tensor exactly once')
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError('explicit legacy ABI requires distinct published outputs')
    if set(inputs) & set(outputs) or set(inputs) | set(outputs) != set(order):
        raise ValueError('explicit legacy ABI requires each logical tensor to be '
                         'an initialized input or a published write-only output')
    declared_outputs = {name for name, spec in tensors.items() if spec.get('role') == 'output'}
    if declared_outputs != set(outputs):
        raise ValueError('explicit legacy ABI outputs must match every declared output role')
    for name in order:
        spec = tensors[name]
        shape = spec.get('shape')
        if not isinstance(shape, (tuple, list)) or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f'explicit legacy ABI tensor {name!r} lacks a static logical shape')
        if not spec.get('dtype'):
            raise ValueError(f'explicit legacy ABI tensor {name!r} lacks its source dtype')
    result = {'kind': 'whole_program',
              'args': [{'tensor': name, 'access': 'write' if name in outputs else 'read'}
                       for name in order],
              'outputs': list(outputs)}
    # Resolve through the public reader as well, so an explicit declaration can
    # never silently change the call order whose preservation was requested.
    if kernel_arg_order({**program, 'kernel_abi': result}, abi) != order:
        raise ValueError('explicit legacy ABI changed the logical pointer order')
    return result
