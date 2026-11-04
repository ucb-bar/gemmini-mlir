"""Compact retained-A preparation for explicitly selected grouped GEMV.

The disclosed multi-block MVIN layout separates successive 16-column blocks by
CONFIG_LD.block_stride rows. A one-row A permits block_stride1: one contiguous
64-byte load prepares four reduction tiles at rows256..259. This changes only
preparation, not product dimensions, K ordering, readout or completion boundaries.
"""
from __future__ import annotations

from pathlib import Path

from merlin.compile.tuning_knobs import knob
from xdsl.dialects.builtin import i8

from . import isa
from .activation_prepare import (BASE_ROW, ROW_BUDGETS, ROW_CAPACITY,
                                 PreparedActivation, eligible)
from .gemv_policy import select_grouped_gemv


def prepare_gemv(ctx, lhs, m, k, n, attrs=None):
    """Prepare one-row A in four-block DMA chunks, or emit nothing and return None.

    Caller has admitted the ordinary shared-column schedule and configured LD0
    for a contiguous row-major signed-i8 A. Explicit grouped columns2/4 with
    activation_residency=True, or the pinned automatic GEMV rule, select this
    alternative. The ordinary retained-A budget
    proves storage legality; at least64 K elements amortize two configuration
    commands. Selected source layout and the final LD0 configuration are the
    same as the incumbent. Caller uses PreparedActivation.address for compute.
    """
    if (any(type(d) is not int or d <= 0 for d in (m, k, n))
            or m != 1 or k < 64 or n <= isa.DIM
            or (attrs or {}).get('_lhs_row_stride', k) != k):
        return None
    root = Path(__file__).resolve().parents[1]
    columns = knob('matmul_column_tiles', default=1,
                   values=(1, 2, 4, 'auto', 'permute'), root=root)
    mode = knob('activation_residency', default=False,
                values=(False, True, 'auto'), root=root)
    selected = select_grouped_gemv(m, k, n)
    grouped = (type(columns) is int and columns in (2, 4)
               or columns == 'auto' and selected)
    retained = mode is True or mode == 'auto' and selected
    if not grouped or not retained:
        return None
    capacity = knob('activation_rows', default=ROW_CAPACITY,
                    values=ROW_BUDGETS, root=root)
    if not eligible(m, k, n, capacity):
        return None
    f = ctx.f
    pointer = ctx.ptr(lhs) if isinstance(lhs, str) else lhs
    descriptor = PreparedActivation(m, k)
    isa.config_ld(f, k, state=0, block_stride=1)
    full = k // 64 * 64

    def load(kk):
        isa.mvin(f, f.gep(pointer, kk, i8), descriptor.address(f, kk),
                 64, 1, state=0)

    f.for_loop(0, full, 64, load, retain_loop=True)
    if full < k:
        isa.mvin(f, f.gep(pointer, f.const(full), i8),
                 BASE_ROW + full // isa.DIM, k - full, 1, state=0)
    isa.config_ld(f, k, state=0)
    return descriptor
