"""Exact raw i32 addition through full-width accumulator DMA.

Every tile is seeded, completed, accumulated, completed, and read out raw before
its rows are reused. Full-width loads bypass i8 scaling/saturation; accumulator
addition and raw readout preserve the source's modular i32 words.
"""
from __future__ import annotations

import math
from . import isa


def lower_i32_residual(ctx, operands, attributes):
    from .elementwise import ElementwiseError, _shape, _dtype, _tiles
    lhs, rhs, dst = (operands[key] for key in ('lhs', 'rhs', 'dst'))
    shape = _shape(ctx, lhs)
    if any(_shape(ctx, name) != shape for name in (rhs, dst)):
        raise ElementwiseError('raw i32 sum requires identical contiguous tensor shapes')
    if any(_dtype(ctx, name) != 'i32' for name in (lhs, rhs, dst)):
        raise ElementwiseError('raw i32 sum requires i32 operands and output')
    allowed = {'lhs_scale', 'rhs_scale', 'bound_lsb', 'epilogue', 'output_dtype',
               'acc_scale', 'relu', 'semantic'}
    if set(attributes) - allowed:
        raise ElementwiseError('raw i32 sum has unsupported semantic attributes')
    if (attributes.get('epilogue', []) or attributes.get('relu', False)
            or attributes.get('output_dtype', 'i32') != 'i32'
            or attributes.get('semantic', 'residual_add') not in ('residual_add', 'add')
            or attributes.get('bound_lsb', 0) != 0
            or any(type(attributes.get(key, 1.0)) not in (int, float)
                   or attributes.get(key, 1.0) != 1.0
                   for key in ('lhs_scale', 'rhs_scale', 'acc_scale'))):
        raise ElementwiseError('raw i32 sum requires identity scales and no epilogue')
    # The selected ABI declares fresh output ownership. Reject explicit aliases;
    # no in-place or partially overlapping entry-argument contract is inferred.
    aliases = getattr(ctx, 'cb', {}).get('params', {}).get('tensor_aliases', {})
    def root(name):
        seen = set()
        while name in aliases:
            if name in seen:
                raise ElementwiseError('raw i32 sum rejects cyclic tensor aliases')
            seen.add(name)
            name = aliases[name]['source']
        return name
    if root(dst) in (root(lhs), root(rhs)):
        raise ElementwiseError('raw i32 sum requires a fresh nonaliasing output')
    ctx.fence()
    ctx.emit(*isa.config_ld(isa.DIM * 4, channel=0, shrink=False))
    ctx.emit(*isa.config_ld(isa.DIM * 4, channel=1, shrink=False))
    ctx.emit(*isa.config_st(isa.DIM * 4))
    def tile(offset, rows, cols):
        ctx.emit(*isa.mvin(ctx.ptr(lhs, offset), isa.acc_addr(0),
                          rows=rows, cols=cols, channel=0))
        ctx.fence()
        ctx.emit(*isa.mvin(ctx.ptr(rhs, offset), isa.acc_addr(0, accumulate=True),
                          rows=rows, cols=cols, channel=1))
        ctx.fence()
        ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0, full=True),
                           rows=rows, cols=cols))
        ctx.fence()
    _tiles(ctx, math.prod(shape), tile)
    ctx.fence()
    receipt = {'schema': 'gemmini.raw_i32_sum.v1', 'shape': list(shape),
               'arithmetic': 'i32 addition modulo 2**32',
               'load_path': 'full-width overwrite then full-width accumulate',
               'completion': 'seed -> drain -> accumulate -> drain -> raw readout -> drain',
               'output_aliases': 'refused; selected ABI requires fresh output'}
    ctx.metadata.setdefault('raw_i32_sums', []).append(receipt)
    return receipt
