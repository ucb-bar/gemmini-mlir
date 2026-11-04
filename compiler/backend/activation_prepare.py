"""Bounded full-LHS preparation for matmuls with many columns.

Reserve scratchpad rows starting at256, disjoint from streamed A0:16/B16:32
and resident weights128:256. Selected RTL facts provide four4096-row banks,
16384 addressable rows. The selectable budget ends no later than row8448.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from xdsl.dialects.builtin import i8
from xdsl.ir import SSAValue

from . import isa
from .gemv_policy import select_grouped_gemv

BASE_ROW = 256
ROW_CAPACITY = 256
ROW_BUDGETS = (256, 1024, 8192)
PHYSICAL_ROWS = 4 * 4096
assert BASE_ROW >= 256 and BASE_ROW + max(ROW_BUDGETS) <= PHYSICAL_ROWS
ROOT = Path(__file__).resolve().parents[1]


def select_residency(mode: bool | str, m: int, n: int, k: int | None = None) -> bool:
    """Select a performance alternative independently of resource legality.

    Auto combines bounded multirow reuse and the pinned large-GEMV rule.
    Optional K preserves existing callers without broadening their admission.
    Explicit True continues to permit every semantically eligible shape.
    """
    if type(mode) is bool:
        return mode
    if mode == 'auto':
        return (2 <= m <= isa.DIM and n >= 32
                or m > isa.DIM and n >= 128
                or k is not None and select_grouped_gemv(m, k, n))
    raise ValueError('activation_residency must be false, true, or auto')


def eligible(m: int, k: int, n: int, capacity: int) -> bool:
    """Check this layout's shape and scratchpad-capacity requirements."""
    return (m > 0 and k > 0 and n > isa.DIM
            and ((k + isa.DIM - 1) // isa.DIM) * m <= capacity)


@dataclass(frozen=True)
class PreparedActivation:
    """Scratchpad layout valid until this matmul's completion boundary.

    Callers must skip streamed A loads and use address for each reduction
    tile. The descriptor does not establish residency across matmul calls.
    """
    m: int
    k: int

    def address(self, f, kk: int | SSAValue):
        """Return the first local row for a DIM-aligned reduction offset."""
        if isinstance(kk, int):
            if kk < 0 or kk >= self.k or kk % isa.DIM:
                raise ValueError('activation reduction offset must name a valid tile')
            return BASE_ROW + (kk // isa.DIM) * self.m
        tile = f.udiv_i(kk, f.const(isa.DIM))
        return f.add_i(f.const(BASE_ROW), f.mul_i(tile, f.const(self.m)))


def prepare(ctx, lhs, m: int, k: int, n: int, attrs=None):
    """Load A once after caller configures DMA state0 with stride K.

    Eligible row-major i8 A[M,K] has N>16 and fits the selected activation_rows
    budget over its K tiles. Explicit True permits multiple M tiles; auto admits
    them when N>=128, with capacity and contiguity checked separately below.
    DMA gathers at most16 rows per command directly
    from the original source; no CPU tensor-value preparation is performed.
    Caller retains the existing common matmul completion fence and must not
    write its reserved rows before the final use of this descriptor. Other
    lowerers may reuse those rows after the common completion fence.
    """
    mode = knob('activation_residency', default=False, values=(False, True, 'auto'), root=ROOT)
    if not select_residency(mode, m, n, k=k):
        return None
    if not (m > 0 and k > 0 and n > isa.DIM):
        return None
    if (attrs or {}).get('_lhs_row_stride', k) != k:
        return None
    capacity = knob('activation_rows', default=ROW_CAPACITY, values=ROW_BUDGETS, root=ROOT)
    if not eligible(m, k, n, capacity):
        return None
    f = ctx.f
    pointer = ctx.ptr(lhs) if isinstance(lhs, str) else lhs
    descriptor = PreparedActivation(m, k)
    full_end = k // isa.DIM * isa.DIM

    def load_rows(kk, cols):
        def row(mi):
            index = f.add_i(f.mul_i(mi, f.const(k)), kk)
            local = f.add_i(descriptor.address(f, kk), mi)
            rows = f.smin(f.const(isa.DIM), f.sub_i(f.const(m), mi))
            isa.mvin(f, f.gep(pointer, index, i8), local, cols, rows, state=0)
        f.for_loop(0, m, isa.DIM, row, retain_loop=True)

    wide = (m > 1 and knob('column_dma', default='tile',
                           values=('tile', 'wide'), root=ROOT) == 'wide')
    if wide:
        # A 16-column block occupies M rows. For kk aligned to64, the
        # multiblock DMA maps column cc to BASE+(kk//16+cc//16)*M+mi.
        # Source bounds: mi+rows<=M and kk+cols<=K. Local bounds:
        # ceil(K/16)*M<=capacity, checked before emitting any preparation.
        isa.config_ld(f, k, state=0, block_stride=m)
        wide_end = k // 64 * 64
        if wide_end:
            f.for_loop(0, wide_end, 64,
                       lambda kk: load_rows(kk, 64), retain_loop=True)
        if wide_end < k:
            load_rows(f.const(wide_end), k - wide_end)
        # Common consumers and subsequent streaming expect the usual layout.
        isa.config_ld(f, k, state=0)
        return descriptor

    def load_full(kk):
        if m <= isa.DIM:
            isa.mvin(f, f.gep(pointer, kk, i8), descriptor.address(f, kk),
                     isa.DIM, m, state=0)
        else:
            load_rows(kk, isa.DIM)

    if full_end:
        f.for_loop(0, full_end, isa.DIM, load_full, retain_loop=True)
    if full_end < k:
        if m <= isa.DIM:
            isa.mvin(f, f.gep(pointer, f.const(full_end), i8),
                     descriptor.address(f, full_end), k - full_end, m, state=0)
        else:
            load_rows(f.const(full_end), k - full_end)
    return descriptor
