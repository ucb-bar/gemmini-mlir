"""Explicit, recorded direct-convolution schedule selection.

Source binding proves the NHWC/HWIO contract before this policy runs. The
optional spatial schedule changes command ordering only. Shape and accumulator
capacity select eligibility; model names and region names are never consulted.
"""
from dataclasses import replace
from .golden_conv import GoldenConv
from .golden_flat_conv import GoldenFlatConv, eligible
from .golden_gemm import _ceil_div
from .tables import rtl_facts as F


def select_kernel(shape, *, flat_spatial=False):
    if flat_spatial and eligible(shape):
        # Keep at least the existing channel block. Prefer a multiple of four
        # tiles so all full B DMA commands transfer 64 adjacent channels.
        mt = _ceil_div(shape.oh * shape.ow, F.DIM)
        capacity = F.ACC_ROWS // (mt * F.DIM)
        nt = _ceil_div(shape.cout, F.DIM)
        bn = min(capacity, max(4, _ceil_div(nt, 4) * 4))
        if bn >= 4:
            bn = bn // 4 * 4
        shape = replace(shape, bn=bn, wide_b=True)
        return GoldenFlatConv(shape,wide_a=True,separate_b_bank=True), 'spatial_flat_wide_a_separate_b'
    return GoldenConv(shape), 'output_row'
