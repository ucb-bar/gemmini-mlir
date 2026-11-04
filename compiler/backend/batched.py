"""Shape-driven batched and attention contractions on the Gemmini array.

Batch slices follow source row-major order. QK's row-per-key operand is copied
into a transposed temporary before invoking the ordinary device contraction.
The host preparation performs no multiplication or accumulation of tensor data.
"""
from __future__ import annotations

from xdsl.dialects.builtin import i8, i32

from merlin.runtime.commandbuffer import batched_matmul_geometry
from .matmul import lower_matmul
from .attention_prepare import prepare_key_transpose


def _shape(spec, label, rank=None):
    shape = tuple(spec['shape'])
    if (rank is not None and len(shape) != rank) or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(f'{label} requires positive static dimensions' +
                         (f' of rank {rank}' if rank is not None else ''))
    return shape


def lower_batched(ctx, command, residents):
    """Lower canonical BATCHED_MATMUL, ATTENTION_QK and ATTENTION_PV commands."""
    opcode = command['opcode']
    operands = command['operands']
    attrs = dict(command.get('attributes', {}))
    tensors = ctx.cb['tensors']
    keys = {'BATCHED_MATMUL': ('a', 'w'),
            'ATTENTION_QK': ('q', 'k'), 'ATTENTION_PV': ('p', 'v')}
    if opcode not in keys:
        raise ValueError(f'unsupported contraction {opcode}')
    left_key, right_key = keys[opcode]
    lhs = residents.get(operands[left_key], operands[left_key])
    rhs = residents.get(operands[right_key], operands[right_key])
    dst = operands['dst']
    a_spec, b_spec, c_spec = (tensors[name] for name in (lhs, rhs, dst))
    if (a_spec['dtype'], b_spec['dtype']) != ('i8', 'i8'):
        raise ValueError(f'{opcode} requires signed i8 operands')
    dtype = attrs.get('output_dtype', c_spec['dtype'])
    if dtype != c_spec['dtype'] or dtype not in ('i8', 'i32'):
        raise ValueError(f'{opcode} output dtype must agree with its tensor and be i8/i32')
    attrs['output_dtype'] = dtype
    f = ctx.f
    a, b, c = ctx.ptr(lhs), ctx.ptr(rhs), ctx.ptr(dst)

    if opcode == 'BATCHED_MATMUL':
        geometry = batched_matmul_geometry(a_spec['shape'], b_spec['shape'], c_spec['shape'])
        if dtype != 'i32':
            raise ValueError('BATCHED_MATMUL signed i8 operands require an i32 destination')
        # Shape prefixes are authoritative; the optional batch hint cannot
        # shorten execution or introduce shared-weight broadcasting.
        m, k, n = geometry.m, geometry.k, geometry.n
        def batch_slice(batch):
            def at(base, stride, ty):
                return f.gep(base, f.mul_i(batch, f.const(stride)), ty)
            lower_matmul(ctx, at(a, m * k, i8), at(b, k * n, i8),
                         at(c, m * n, i32), m, k, n, attrs)
        f.for_loop(0, geometry.batch_count, 1, batch_slice, retain_loop=True)
        return

    ash = _shape(a_spec, opcode + ' lhs', 2)
    bsh = _shape(b_spec, opcode + ' rhs', 2)
    csh = _shape(c_spec, opcode + ' output', 2)
    m, k = ash
    if opcode == 'ATTENTION_QK':
        n, kb = bsh
    else:
        kb, n = bsh
    if k != kb or csh != (m, n):
        raise ValueError(f'{opcode} incompatible operand/output shapes')
    if opcode == 'ATTENTION_QK':
        b = prepare_key_transpose(ctx, command, rhs, n, k)
    lower_matmul(ctx, a, b, c, m, k, n, attrs)
