"""Exact blockwise column scales for adjacent-column accumulator readout.

MVOUT addresses an accumulator row, not an arbitrary column inside that row.
Consequently a 16-column contraction tile must have one common f32 scale.
Varying lanes retain the incumbent narrower-contraction schedule. Admission is
pure; command parameters use integer constants and register-only lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from . import isa
from .column_schedule import _lookup
from .tile_bias import BiasPreparation
from .tile_epilogue import Readout, select_readout


@dataclass(frozen=True)
class ColumnReadout:
    """One exact source scale for each aligned 16-column output tile.

    The caller owns contractions, accumulator addresses and final completion.
    Every store must cover only its indicated aligned tile, including its tail.
    Dynamic columns must be aligned and in [0, n); these are schedule invariants,
    not payload data. Literal columns specialize to literal CONFIG_ST bits.
    """

    uniform: Readout
    n: int
    scale_bits: tuple[int, ...]

    def configure(self, f):
        """No global configuration: each consumed store configures its scale."""
        return None

    def peel_group_start(self, column_tiles: int) -> int:
        """Choose a real group whose literal readout exposes a source scale.

        A static command decoder may not resolve a dynamic register lookup.
        The caller can compute/store this whole group once with literal column
        arguments, then loop over the groups before and after it. This is an
        actual output group, never an unused configuration witness.
        """
        if type(column_tiles) is not int or column_tiles not in (2, 4):
            raise ValueError('column group must contain two or four tiles')
        tile = next((i for i, bits in enumerate(self.scale_bits)
                     if bits != isa.f32_bits(1.0)), 0)
        return tile // column_tiles * column_tiles * isa.DIM

    def bits_for_column(self, f, column):
        if type(column) is int:
            if column < 0 or column >= self.n or column % isa.DIM:
                raise ValueError('readout column must start an aligned live tile')
            return self.scale_bits[column // isa.DIM]
        return _lookup(f, f.udiv_i(column, f.const(isa.DIM)), self.scale_bits)

    def store_column(self, f, destination, index, local, cols, rows, column):
        """Configure exact f32 scale, then read the corresponding live tile.

        `index` is the original row-major destination element index; `local`
        is the scheduled accumulator row. Neither is shifted to emulate an
        unsupported accumulator-column address. No scalar memory is accessed.
        """
        bits = self.bits_for_column(f, column)
        isa.config_st(f, self.uniform.stride_bytes, scale_bits=bits,
                      activation=self.uniform.activation)
        return self.uniform.store(f, destination, index, local, cols, rows)


def select_column_readout(attrs: Mapping, n: int, *,
                          bias: BiasPreparation | None = None) -> ColumnReadout | None:
    """Admit only finite f32, per-16-column uniform signed-i8 readout.

    Reuse the existing uniform selector for bias typing and exact epilogue order
    (bias -> scale -> saturation -> optional ReLU). Scale vectors override the
    scalar scale as in common matmul. No IR or context mutation precedes None.
    Equal f32 encodings are sufficient: the source readout contract is f32.
    """
    if type(n) is not int or n <= 0 or attrs.get('output_dtype', 'i32') != 'i8':
        return None
    scales = attrs.get('acc_scale_by_column')
    if not isinstance(scales, (list, tuple)) or len(scales) != n:
        return None
    scalar_attrs = dict(attrs)
    scalar_attrs.pop('acc_scale_by_column')
    uniform = select_readout(scalar_attrs, n, bias=bias)
    if uniform is None:
        return None
    try:
        values = tuple(float(value) for value in scales)
        bits = tuple(isa.f32_bits(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(value) or encoded & 0x7F800000 == 0x7F800000
           for value, encoded in zip(values, bits)):
        return None
    tiles = []
    for start in range(0, n, isa.DIM):
        tile = bits[start:start + isa.DIM]
        if any(value != tile[0] for value in tile):
            return None
        tiles.append(tile[0])
    return ColumnReadout(uniform, n, tuple(tiles))


@dataclass(frozen=True)
class RunReadout:
    """Exact consecutive scale runs, each split into products at most16 wide."""

    uniform: Readout
    columns: tuple[int, ...]
    widths: tuple[int, ...]
    scale_bits: tuple[int, ...]

    def groups(self, f, column_tiles, body):
        """Emit actual peeled output group plus at most two counted loops.

        `body(records)` consumes tuples (column, width, exact_scale_bits).
        Dynamic records are integer command parameters, never payload values.
        """
        count = len(self.columns)
        full = count // column_tiles
        tail = count % column_tiles
        peeled = next((i for i, bits in enumerate(self.scale_bits)
                       if bits != isa.f32_bits(1.0)), 0) // column_tiles

        def literal(group, size):
            start = group * column_tiles
            body(tuple((f.const(self.columns[i]), f.const(self.widths[i]),
                        self.scale_bits[i]) for i in range(start, start + size)))

        def dynamic(group):
            start = f.mul_i(group, f.const(column_tiles))
            records = []
            for offset in range(column_tiles):
                index = f.add_i(start, f.const(offset))
                records.append(tuple(_lookup(f, index, values) for values in
                                     (self.columns, self.widths, self.scale_bits)))
            body(tuple(records))

        literal(peeled, column_tiles if peeled < full else tail)
        f.for_loop(0, min(peeled, full), 1, dynamic, retain_loop=True)
        f.for_loop(min(peeled + 1, full), full, 1, dynamic, retain_loop=True)
        if tail and peeled != full:
            literal(full, tail)

    def store(self, f, destination, index, local, cols, rows, bits):
        isa.config_st(f, self.uniform.stride_bytes, scale_bits=bits,
                      activation=self.uniform.activation)
        return self.uniform.store(f, destination, index, local, cols, rows)


def select_run_readout(attrs: Mapping, n: int, *,
                       bias: BiasPreparation | None = None) -> RunReadout | None:
    """Admit exact source runs, retaining their narrow product/readout widths."""
    if type(n) is not int or n <= 0 or attrs.get('output_dtype', 'i32') != 'i8':
        return None
    scales = attrs.get('acc_scale_by_column')
    if not isinstance(scales, (list, tuple)) or len(scales) != n:
        return None
    scalar_attrs = dict(attrs)
    scalar_attrs.pop('acc_scale_by_column')
    uniform = select_readout(scalar_attrs, n, bias=bias)
    if uniform is None:
        return None
    try:
        values = tuple(float(value) for value in scales)
        bits = tuple(isa.f32_bits(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(value) or encoded & 0x7F800000 == 0x7F800000
           for value, encoded in zip(values, bits)):
        return None
    records = []
    start = 0
    while start < n:
        end = start + 1
        # Match the incumbent's source scale-run boundaries, even if distinct
        # Python values round to the same f32 representation.
        while end < n and values[end] == values[start]:
            end += 1
        for column in range(start, end, isa.DIM):
            records.append((column, min(isa.DIM, end - column), bits[start]))
        start = end
    return RunReadout(uniform, *(tuple(r[field] for r in records) for field in range(3)))


def lower_column_tiled(ctx, a, b, c, m, k, n, attrs, row_tiles, column_tiles):
    """Optional complete device lowering with shared A across exact scale runs.

    Caller admits only an explicit column-tiles choice. All semantic/type/local
    storage checks precede emission. A None/False path leaves incumbent intact.
    """
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    from xdsl.dialects.builtin import i8
    from .activation_prepare import prepare as prepare_activation
    from .tile_bias import has_bias, select_bias

    if (any(type(d) is not int or d <= 0 for d in (m, k, n, row_tiles))
            or type(column_tiles) is not int or column_tiles not in (2, 4)):
        return False
    if any(attrs.get(key) is not None for key in
           ('_resident_weight', '_activation', '_compact_columns')):
        return False
    boundary = attrs.get('source_numeric_boundary')
    if isinstance(boundary, dict) and boundary.get('destination_constants', [0]) != [0]:
        return False
    stride = attrs.get('_lhs_row_stride', k)
    if type(stride) is not int or stride < k:
        return False
    bias = None
    if has_bias(attrs):
        if not knob('tiled_bias', default=False, values=(False, True),
                    root=Path(__file__).resolve().parents[1]):
            return False
        bias = select_bias(attrs, m, n, ctx.cb['tensors'])
        if bias is None:
            return False
    readout = select_run_readout(attrs, n, bias=bias)
    if readout is None:
        return False
    row_tiles = min(row_tiles, 32 // column_tiles)
    f = ctx.f
    isa.config_ex(f, dataflow=1, activation=0, scale=1.0)
    isa.config_ld(f, stride, state=0)
    activation = prepare_activation(ctx, a, m, k, n, attrs) if stride == k else None
    isa.config_ld(f, n, state=1)
    pointer = None
    if bias is not None:
        pointer = ctx.ptr(bias.source) if isinstance(bias.source, str) else bias.source
        bias.configure(f)

    def group(mi, count):
        def each_row(body):
            def row(ri):
                start = f.add_i(mi, f.mul_i(ri, f.const(isa.DIM)))
                rows = f.smin(f.const(isa.DIM), f.sub_i(f.const(m), start))
                local = f.mul_i(ri, f.const(isa.DIM))
                body(start, rows, local)
            f.for_loop(0, count, 1, row, retain_loop=True)

        def columns(records):
            if bias is not None:
                def initialize(start, rows, local):
                    for tile, (column, width, bits) in enumerate(records):
                        acc = f.add_i(local, f.const(tile * row_tiles * isa.DIM))
                        bias.preload(f, pointer, start, column, acc, width, rows)
                each_row(initialize)

            def reduction(kk):
                tk = f.smin(f.const(isa.DIM), f.sub_i(f.const(k), kk))
                for tile, (column, width, bits) in enumerate(records):
                    index = f.add_i(f.mul_i(kk, f.const(n)), column)
                    isa.mvin(f, f.gep(b, index, i8), (tile + 1) * isa.DIM,
                             width, tk, state=1)
                flags = (f.const(bias.compute_flags) if bias is not None else
                         f.or_i(f.const(0x80000000),
                                f.shl_i(f.smin(kk, f.const(1)), 30)))

                def compute(start, rows, local):
                    if activation is None:
                        index = f.add_i(f.mul_i(start, f.const(stride)), kk)
                        isa.mvin(f, f.gep(a, index, i8), 0, tk, rows, state=0)
                        a_local = 0
                    else:
                        a_local = f.add_i(activation.address(f, kk), start)
                    for tile, (column, width, bits) in enumerate(records):
                        acc = f.add_i(local, f.const(tile * row_tiles * isa.DIM))
                        isa.preload(f, (tile + 1) * isa.DIM, f.or_i(flags, acc),
                                    width, tk, width, rows)
                        isa.compute(f, a_local, isa.GARBAGE_ADDR, tk, rows,
                                    width, rows, flip=True)
                each_row(compute)
            f.for_loop(0, k, isa.DIM, reduction, retain_loop=True)

            def store(start, rows, local):
                for tile, (column, width, bits) in enumerate(records):
                    index = f.add_i(f.mul_i(start, f.const(n)), column)
                    acc = f.add_i(local, f.const(tile * row_tiles * isa.DIM))
                    readout.store(f, c, index, acc, width, rows, bits)
            each_row(store)
        readout.groups(f, column_tiles, columns)

    span = row_tiles * isa.DIM
    full = m // span * span
    f.for_loop(0, full, span, lambda mi: group(mi, row_tiles), retain_loop=True)
    if full < m:
        group(f.const(full), (m - full + isa.DIM - 1) // isa.DIM)
    isa.fence(f)
    return True
