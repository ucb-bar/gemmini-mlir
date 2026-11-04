"""Make a typed interface program's existing pointer boundary explicit.

The logical-v2 contract allows an equivalent whole_program declaration. Writing
out that boundary lets storage/readback owners use every result's declared type,
including results produced before the last operation in a connected interface.
This does not change tensor semantics, data, shapes, types, or pointer order.
"""
from __future__ import annotations

import copy
from pathlib import Path

from merlin.compile.logical_abi import logical_abi, logical_interface


def declare_interface_abi(program: dict, abi=None) -> dict:
    """Return a copy with the default logical pointer ABI declared explicitly.

    A caller-supplied ABI is preserved unchanged. Compiler temporaries and
    resident/accumulator handles remain internal, as specified by logical-v2.
    Every published result remains a separate write-only output, even when a
    later operation also consumes it inside the same submitted function.
    """
    if program.get('kernel_abi'):
        return program
    if abi is None:
        abi = logical_abi(Path(__file__).resolve().parent / 'contracts' /
                          'mlir_oot_backend_contract.yaml')
    before = logical_interface(program, abi)
    out = copy.deepcopy(program)
    tensors = out.setdefault('tensors', {})
    for buf in before:
        original = tensors.get(buf.name) or {}
        tensors[buf.name] = {
            **original,
            'shape': list(buf.shape),
            'dtype': buf.dtype,
            **({'role': 'output'} if buf.kind == 'output' else {}),
        }
    out['kernel_abi'] = {
        'kind': 'whole_program',
        'args': [{'tensor': buf.name,
                  'access': 'write' if buf.kind == 'output' else 'read'}
                 for buf in before],
        'outputs': [buf.name for buf in before if buf.kind == 'output'],
    }
    after = logical_interface(out, abi)
    if before != after:
        raise ValueError('explicit interface ABI changed the typed pointer boundary')
    return out
