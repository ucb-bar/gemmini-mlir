"""Checked full-spatial producer for strided 1x1 column-scaled convolution.

This adapter closes an existing route gap: conv_column's raw producer bypasses
compiler dispatch. Reuse the common full-spatial gather, residency and qualified
readout epoch schedule; original column-scale narrowing remains in conv_column.
No new target instruction, storage map, source numeric stage or ordering law.
"""
from __future__ import annotations


def lower(ctx, ifm, rhs, dst, attributes):
    """Apply the common producer or decline before changing target state."""
    attrs = dict(attributes or {})
    kernel = tuple(attrs.get('kernel', ()))
    stride = tuple(attrs.get('stride', (1, 1)))
    if (len(kernel) != 4 or kernel[:2] != (1, 1) or len(stride) != 2
            or any(type(v) is not int or v < 1 for v in stride)
            or max(stride) <= 1 or attrs.get('accumulator_init') is not None
            or ctx.spec(dst)['dtype'] != 'i32'
            or attrs.get('output_dtype', 'i32') != 'i32'
            or attrs.get('acc_scale_by_column') is not None
            or float(attrs.get('acc_scale', 1.0)) != 1.0
            or attrs.get('relu')
            or any(stage not in ('bias_add',) for stage in attrs.get('epilogue', ()))):
        return False
    from .conv_spatial_panels import lower as lower_spatial
    # Shared admission checks exact source axes, complete output extent,
    # nonaliasing, numeric seed, capacities and selected ordering facts before
    # any emission. Its bounded affine runs include scanline/image boundaries.
    receipt = lower_spatial(ctx, ifm, rhs, dst, attrs, allow_wide=True)
    if not receipt:
        return False
    receipt = dict(receipt, adapter='strided_pointwise_raw_full_spatial',
                   numeric_boundary='original biased i32 source; original caller owns per-column narrowing')
    ctx.metadata.setdefault('pointwise_spatial_raw_producers', []).append(receipt)
    return receipt
