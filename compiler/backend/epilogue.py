"""Compose contraction readout with the device pooling epilogue."""
from __future__ import annotations

from merlin.compile.logical_abi import pool_params, pool_out_dims
from .matmul import lower_matmul
from .pooling import device_pool


def lower_matmul_epilogue(ctx, lhs, rhs, dst, m, k, n, attrs=None):
    """Lower ordinary readout, or i8 readout followed by device maxpool.

    Pooling is split after the source-ordered pointwise readout. Its source
    matrix is private scratch; only the pooled matrix is externally observable.
    """
    attrs = dict(attrs or {})
    epilogue = attrs.get('epilogue', [])
    epilogue = list(epilogue) if isinstance(epilogue, (tuple, list)) else [epilogue]
    if 'maxpool' not in epilogue:
        return lower_matmul(ctx, lhs, rhs, dst, m, k, n, attrs)
    if epilogue[-1] != 'maxpool' or epilogue.count('maxpool') != 1:
        raise ValueError('device maxpool must be the last epilogue')
    if attrs.get('output_dtype', 'i32') != 'i8':
        raise ValueError('device maxpool currently requires i8 readout')
    pool = pool_params(attrs, op='COMMIT maxpool')
    h, w = pool['pool_in_dims']
    if m % (h * w):
        raise ValueError('COMMIT pool_in_dims must divide contraction rows')
    hp, wp = pool_out_dims(h, w, pool['pool_size'], pool['pool_stride'], pool['pool_padding'])
    if hp <= 0 or wp <= 0:
        raise ValueError('COMMIT pooling has empty output geometry')
    if isinstance(dst, str):
        shape = list(ctx.cb['tensors'][dst]['shape'])
        batch = m // (h * w)
        if shape not in ([batch * hp * wp, n], [batch, hp, wp, n]):
            raise ValueError('COMMIT destination does not match pooled geometry')
        identity = dst
    else:
        identity = str(len(ctx.cb['tensors']))
    scratch = ctx.alloc('__matmul_prepool_' + identity, [m, n], 'i8')
    readout = dict(attrs)
    readout['epilogue'] = epilogue[:-1]
    lower_matmul(ctx, lhs, rhs, scratch, m, k, n, readout)
    return device_pool(ctx, scratch, dst, m, n, attrs)
