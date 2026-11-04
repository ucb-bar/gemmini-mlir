"""Exact column readout epochs spanning several original spatial row tiles.

The original i32 producer completes before this consumer. Each DMA writes one
private ACC row interval, all those loads complete, and stores share an unchanged
complete CONFIG_ST word until the epoch drains. Only address order changes;
rounding, bias, activation and source/output storage remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from . import isa


@dataclass(frozen=True)
class ColumnSpatialPlan:
    rows: int
    columns: int
    tile_rows: int = isa.DIM
    accumulator_rows: int = isa.ACC_ROWS

    def __post_init__(self):
        if (any(type(v) is not int or v <= 0 for v in
                (self.rows, self.columns, self.tile_rows, self.accumulator_rows))
                or self.tile_rows != isa.DIM
                or self.accumulator_rows != isa.ACC_ROWS):
            raise ValueError('column spatial epochs require the selected physical row geometry')

    def domains(self, column_count):
        """(number of tiles, rows per tile, starting source row)."""
        if type(column_count) is not int or not 0 < column_count <= self.columns:
            raise ValueError('column epoch needs a nonempty subset of original columns')
        full, tail = divmod(self.rows, self.tile_rows)
        result = []
        if full:
            result.append((full, self.tile_rows, 0))
        if tail:
            result.append((1, tail, full * self.tile_rows))
        return tuple(result)


def emit_spatial_column_readout(ctx, raw, dst, m, n, attributes):
    """Emit spatially packed readout groups with the existing terminal encoder.

    Small palettes use the identical complete words and original column lists
    from column_batches. Large palettes retain counted column selection, then
    collect spatial tiles for that one column, avoiding static N-way expansion
    of the whole consumer. No source tensor value is computed on the host.
    """
    from .column_batches import _admission, _palette, _column_at
    from .dense_readout import _emit_column_config, _c
    plan = ColumnSpatialPlan(m, n)
    if (ctx.spec(raw)['dtype'] != 'i32'
            or tuple(ctx.spec(raw)['shape']) != (m, n)
            or ctx.spec(dst)['dtype'] != 'i8'
            or prod(ctx.spec(dst)['shape']) != m * n
            or ctx.buffer(raw) is ctx.buffer(dst)):
        raise ValueError('spatial column epochs require disjoint original i32/i8 matrices')
    groups = _palette(n, attributes)
    admission = _admission()
    admission['store_configuration_snapshot'] = 'not assumed; identical configuration through epoch drain'
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n * 4, channel=2, shrink=False))

    def emit_domain(tiles, rows, row_base, columns, configure):
        count_columns = len(columns) if isinstance(columns, tuple) else 1
        capacity = isa.ACC_ROWS // rows
        total = tiles * count_columns

        def coordinate(index):
            tile = fb.udiv_i(index, fb.const(count_columns)) if count_columns != 1 else index
            row = fb.add_i(fb.const(row_base), fb.mul_i(tile, fb.const(rows)))
            if isinstance(columns, tuple):
                position = fb.urem_i(index, fb.const(count_columns)) if count_columns != 1 else fb.const(0)
                column = _column_at(fb, position, columns)
            else:
                column = columns
            return row, column

        def batch(first, count):
            def transfer(slot, store):
                row, column = coordinate(fb.add_i(first, slot))
                offset = fb.add_i(fb.mul_i(row, fb.const(n)), column)
                local_row = fb.mul_i(slot, fb.const(rows))
                if store:
                    ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(local_row, fb=fb),
                                       rows, 1, fb=fb))
                else:
                    ctx.emit(*isa.mvin(ctx.ptr(raw, offset), isa.acc_addr(local_row, fb=fb),
                                      rows, 1, channel=2, fb=fb))
            fb.for_loop(0, count, 1, lambda slot: transfer(slot, False), retain_loop=True)
            ctx.fence()
            configure()
            fb.for_loop(0, count, 1, lambda slot: transfer(slot, True), retain_loop=True)
            ctx.fence()

        complete = total // capacity * capacity
        if complete:
            fb.for_loop(0, complete, capacity,
                        lambda first: batch(first, capacity), retain_loop=True)
        if complete < total:
            batch(fb.const(complete), total - complete)

    if len(groups) <= 32:
        for command, columns in groups:
            for tiles, rows, base in plan.domains(len(columns)):
                emit_domain(tiles, rows, base, columns, lambda command=command: ctx.emit(*command))
        route = 'spatial_tiles_by_exact_store_palette'
    else:
        # Select a literal complete word once per epoch. The selected column
        # and all its original spatial tiles remain in one configuration domain.
        words = [None] * n
        for command, columns in groups:
            for column in columns:
                words[column] = command[2]
        funct, rs1, _ = groups[0][0]
        if any(command[:2] != (funct, rs1) for command, _ in groups):
            raise ValueError('column configurations differ outside the original scale field')
        def column_body(column):
            def configure():
                _emit_column_config(ctx, _c(fb, column), funct, rs1, words)
            for tiles, rows, base in plan.domains(1):
                emit_domain(tiles, rows, base, column, configure)
        fb.for_loop(0, n, 1, column_body, retain_loop=True)
        route = 'spatial_tiles_by_counted_original_column'
    ctx.fence()
    receipt = dict(selection='spatial', route=route,
        palette_groups=len(groups), palette_column_counts=[len(columns) for _, columns in groups],
        spatial_rows=m, output_columns=n, primitive_rows=isa.DIM,
        maximum_live_accumulator_rows=max(min(tiles * len(columns), isa.ACC_ROWS // rows) * rows
            for _, columns in groups for tiles, rows, _ in plan.domains(len(columns)))
            if len(groups) <= 32 else max(min(tiles, isa.ACC_ROWS // rows) * rows
                for tiles, rows, _ in plan.domains(1)),
        raw_row_stride_bytes=n * 4, output_row_stride_bytes=n,
        load_drain='all disjoint ACC slots before any store',
        store_drain='all unchanged-configuration stores before config change or slot reuse',
        partition='cartesian product of every original row tile and original column exactly once',
        admission=admission)
    ctx.metadata.setdefault('column_readouts', []).append(receipt)
    return receipt
