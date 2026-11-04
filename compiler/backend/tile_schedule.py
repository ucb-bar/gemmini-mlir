"""Adjacent-column GEMM groups sharing A across N and B across M.

`lower_tiled` is an optional complete lowering, called before the incumbent's
configuration/preparation emission. The caller selects the two-column knob and
passes its dimension-selected row limit. False emits nothing; True configures
EX/LD0/LD1/ST, computes all outputs, and emits a completion fence. Existing task
tags on ctx.f are retained. Zero-initialized signed-i8 products admit raw i32
or uniform scaled i8 readout; prepared/resident and column-scale paths retain
their incumbent lowering.

Scratch rows [0,16) hold one A tile, followed by two or four B tiles.
Accumulator rows column*row_tiles*16 + row*16 reserve at most512 rows.
Uniform signed-i8 readout uses the incumbent device scale/saturation/ReLU path.
All loops are ordinary CPU CFG loops; no loop-FSM instruction is emitted.
"""
from __future__ import annotations

from pathlib import Path

from merlin.compile.tuning_knobs import knob
from xdsl.dialects.builtin import i8

from . import isa
from .burst_load import try_load_columns
from .tile_epilogue import select_readout
from .tile_bias import has_bias, select_bias
from .activation_prepare import prepare as prepare_activation
from .row_policy import unroll_full_rows
from .gemv_policy import select_grouped_gemv


def select_column_tiles(m, k, n, row_tiles, choice="auto"):
    """Conservative shape-only policy for an optional caller-owned auto choice.

    Keep this profitability hypothesis separate from `_eligible`: a forced
    two-column choice may legally test shapes that this policy declines.
    Require two full output columns, a half-full input row tile, and two reduction
    tiles to amortize adjacent-column dispatch. Four full output columns use
    four-way sharing; smaller outputs retain two-way sharing. Preserve larger
    incumbent groups beyond the supported sixteen-row policy.
    The caller retains its declared default and applies semantic legality.
    Explicit integer choices bypass profitability, while unsupported choices
    raise rather than silently changing the requested schedule.
    """
    if type(choice) is int and choice in (1, 2, 4):
        return choice
    if choice != "auto":
        raise ValueError("column tile choice must be 1, 2, 4, or 'auto'")
    if any(type(d) is not int or d <= 0 for d in (m, k, n, row_tiles)):
        return 1
    if select_grouped_gemv(m, k, n):
        return 4
    if m < isa.DIM // 2 or k < 2 * isa.DIM or n < 2 * isa.DIM or row_tiles > 16:
        return 1
    return 4 if n >= 4 * isa.DIM else 2


def _eligible(m, k, n, attrs, row_tiles, bias=None):
    if (any(type(d) is not int or d <= 0 for d in (m, k, n, row_tiles))
            or n <= isa.DIM or select_readout(attrs, n, bias=bias) is None):
        return False
    if (attrs.get('_resident_weight') is not None
            or attrs.get('_activation') is not None
            or attrs.get('_compact_columns') is not None):
        return False
    boundary = attrs.get('source_numeric_boundary')
    if isinstance(boundary, dict) and boundary.get('destination_constants', [0]) != [0]:
        return False
    stride = attrs.get('_lhs_row_stride', k)
    if type(stride) is not int or stride < k:
        return False
    return True


def lower_tiled(ctx, a, b, c, m, k, n, attrs, row_tiles, column_tiles=2):
    """Emit complete two/four-column GEMM, or return False without effects.

    A/B/C are original row-major tensor pointers. A may have a larger physical
    row stride via attrs._lhs_row_stride; B and C have leading dimension N.
    The caller must not have emitted resident/activation preparation. An
    eligible activation-residency choice prepares full A once for all N groups.
    Row limits are capped to32/column_tiles for the512-row accumulator window.
    """
    attrs = dict(attrs or {})
    root = Path(__file__).resolve().parents[1]
    if attrs.get('acc_scale_by_column') is not None:
        # Keep automatic profitability and all existing uniform products intact.
        # An explicit column choice admits the new scale-run sharing experiment.
        choice = knob('matmul_column_tiles', default=1,
                      values=(1, 2, 4, 'auto', 'permute'), root=root)
        if type(choice) is not int or choice not in (2, 4):
            return False
        from .tile_column_readout import lower_column_tiled
        return lower_column_tiled(ctx, a, b, c, m, k, n, attrs,
                                  row_tiles, column_tiles)
    bias = None
    if has_bias(attrs):
        if not knob('tiled_bias', default=False, values=(False, True), root=root):
            return False
        bias = select_bias(attrs, m, n, ctx.cb['tensors'])
        if bias is None:
            return False
    if (type(column_tiles) is not int or column_tiles not in (2, 4)
            or not _eligible(m, k, n, attrs, row_tiles, bias)):
        return False
    row_tiles = min(row_tiles, 32 // column_tiles)
    row_stride = attrs.get('_lhs_row_stride', k)
    readout = select_readout(attrs, n, bias=bias)
    assert readout is not None
    f = ctx.f
    isa.config_ex(f, dataflow=1, activation=0, scale=1.0)
    isa.config_ld(f, row_stride, state=0)
    activation = None
    if row_stride == k:
        from .gemv_schedule import prepare_gemv
        activation = prepare_gemv(ctx, a, m, k, n, attrs)
        if activation is None:
            activation = prepare_activation(ctx, a, m, k, n, attrs)
    isa.config_ld(f, n, state=1)
    bias_pointer = None
    if bias is not None:
        bias_pointer = ctx.ptr(bias.source) if isinstance(bias.source, str) else bias.source
        bias.configure(f)
    readout.configure(f)

    unroll = unroll_full_rows(root=Path(__file__).resolve().parents[1])
    def group(mi, count, full_group=False):
        def each_row(body):
            def row(ri):
                start = f.add_i(mi, f.mul_i(ri, f.const(16)))
                rows = f.smin(f.const(16), f.sub_i(f.const(m), start))
                local = f.mul_i(ri, f.const(16))
                body(start, rows, local)
            if unroll and full_group and type(count) is int and count <= 16:
                for ri in range(count):
                    row(f.const(ri))
            else:
                f.for_loop(0, count, 1, row, retain_loop=True)

        def columns(nj, widths):
            if bias is not None:
                def initialize(start, rows, local):
                    for column, width in enumerate(widths):
                        col = f.add_i(nj, f.const(column * 16))
                        acc = f.add_i(local, f.const(column * row_tiles * 16))
                        bias.preload(f, bias_pointer, start, col, acc, width, rows)
                each_row(initialize)
            def reduction(kk):
                tk = f.smin(f.const(16), f.sub_i(f.const(k), kk))
                # LD1's block_stride16 places each adjacent 16-column block
                # into the same scratch rows consumed by the tiled path.
                # Default/unsupported wide DMA emits nothing before fallback.
                if not try_load_columns(f, b, n, kk, nj, widths, tk):
                    for column, width in enumerate(widths):
                        col = f.add_i(nj, f.const(column * 16))
                        index = f.add_i(f.mul_i(kk, f.const(n)), col)
                        isa.mvin(f, f.gep(b, index, i8), 16 + column * 16,
                                 width, tk, state=1)
                flags = (f.const(bias.compute_flags) if bias is not None else
                         f.or_i(f.const(0x80000000),
                                f.shl_i(f.smin(kk, f.const(1)), 30)))
                def compute_row(start, rows, local):
                    if activation is None:
                        ai = f.add_i(f.mul_i(start, f.const(row_stride)), kk)
                        isa.mvin(f, f.gep(a, ai, i8), 0, tk, rows, state=0)
                        a_local = 0
                    else:
                        a_local = f.add_i(activation.address(f, kk), start)
                    for column, width in enumerate(widths):
                        acc = f.add_i(local, f.const(column * row_tiles * 16))
                        isa.preload(f, 16 + column * 16, f.or_i(flags, acc),
                                    width, tk, width, rows)
                        isa.compute(f, a_local, isa.GARBAGE_ADDR, tk, rows,
                                    width, rows, flip=True)
                each_row(compute_row)
            f.for_loop(0, k, 16, reduction, retain_loop=True)
            def store_row(start, rows, local):
                for column, width in enumerate(widths):
                    col = f.add_i(nj, f.const(column * 16))
                    index = f.add_i(f.mul_i(start, f.const(n)), col)
                    acc = f.add_i(local, f.const(column * row_tiles * 16))
                    readout.store(f, c, index, acc, width, rows)
            each_row(store_row)

        column_span = column_tiles * 16
        full_columns = n // column_span * column_span
        f.for_loop(0, full_columns, column_span,
                   lambda nj: columns(nj, (16,) * column_tiles), retain_loop=True)
        remainder = n - full_columns
        if remainder:
            widths = tuple(min(16, remainder - start)
                           for start in range(0, remainder, 16))
            columns(f.const(full_columns), widths)

    group_rows = row_tiles * 16
    full_rows = m // group_rows * group_rows
    f.for_loop(0, full_rows, group_rows,
               lambda mi: group(mi, row_tiles, full_group=True), retain_loop=True)
    if full_rows < m:
        group(f.const(full_rows), (m - full_rows + 15) // 16)
    isa.fence(f)
    return True
