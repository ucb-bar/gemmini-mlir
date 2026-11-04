"""Bounded contiguous rectangles inside original fixed-configuration batches.

This changes transfer width only. Every original palette batch keeps its real
load and store drains, and every rectangle keeps the first original column's
private ACC interval. No configuration snapshot or asynchronous reuse is used.
"""
from __future__ import annotations
from . import isa
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout


def plan_column_runs(groups, n, rows, attributes):
    """Project width<=DIM rectangles back to every original source column."""
    if type(n) is not int or n <= 0 or type(rows) is not int or not 0 < rows <= isa.DIM:
        raise ValueError('column runs require static positive source geometry')
    if sorted(c for _, columns in groups for c in columns) != list(range(n)):
        raise ValueError('column runs require an exact nonaliasing source partition')
    scales = attributes.get('acc_scale_by_column', [attributes.get('acc_scale', 1.0)] * n)
    if len(scales) != n:
        raise ValueError('column run scales must retain the complete source vector')
    relu = bool(attributes.get('relu', False) or 'relu' in attributes.get('epilogue', ()))
    capacity = isa.ACC_ROWS // rows
    result = []; wide = False; rectangles = 0
    for command, columns in groups:
        batches = []
        for first in range(0, len(columns), capacity):
            original = columns[first:first + capacity]
            runs = []; slot = 0
            while slot < len(original):
                width = 1
                while (width < isa.DIM and slot + width < len(original)
                       and original[slot + width] == original[slot] + width):
                    width += 1
                column = original[slot]
                # Consume the shared exact integer readout contract, with the
                # actual source coefficients for this selected primitive.
                try:
                    readout = DenseTiledIntegerReadout.from_source_scales(
                        list(scales[column:column + width]), columns=width,
                        tile_columns=width, relu=relu, instruction_name='mvout')
                except ValueError:
                    return None
                if any(isa.config_st(n, scale=scales[c], activation=int(relu)) != command
                       for c in range(column, column + width)):
                    raise ValueError('column rectangle changed complete original CONFIG_ST')
                runs.append((column, width, slot * rows, readout.identity))
                wide |= width > 1
                slot += width
            projected = tuple(c for c, w, _, _ in runs for c in range(c, c + w))
            if projected != tuple(original):
                raise ValueError('column rectangles differ from original source coordinates')
            # Original slots reserve rows even when a rectangle leaves holes.
            windows = [(local, local + rows) for _, _, local, _ in runs]
            if any(a < d and c < b for i, (a, b) in enumerate(windows)
                   for c, d in windows[i + 1:]) or any(b > isa.ACC_ROWS for _, b in windows):
                raise ValueError('column rectangle accumulator ownership overlaps')
            rectangles += len(runs)
            if rectangles > 256:  # Static compiler expansion budget, not a shape selector.
                return None
            batches.append(tuple(runs))
        result.append((command, tuple(batches)))
    return tuple(result) if wide else None


def emit_column_runs(ctx, raw, dst, m, n, attributes, groups):
    """Return None before emission when no actual rectangle is profitable."""
    full = m // isa.DIM * isa.DIM
    extents = ([isa.DIM] if full else []) + ([m - full] if full < m else [])
    plans = {rows: plan_column_runs(groups, n, rows, attributes) for rows in extents}
    if any(plan is None for plan in plans.values()):
        return None
    from .column_batches import _admission
    admission = _admission()
    admission['store_configuration_snapshot'] = 'not assumed; identical configuration through original batch drain'
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n * 4, channel=2, shrink=False))
    def row_tile(row, rows):
        for command, batches in plans[rows]:
            for runs in batches:
                for column, width, local, _ in runs:
                    offset = fb.add_i(fb.mul_i(row, fb.const(n)), fb.const(column))
                    ctx.emit(*isa.mvin(ctx.ptr(raw, offset), isa.acc_addr(local),
                                      rows, width, channel=2, fb=fb))
                ctx.fence()
                ctx.emit(*command)
                for column, width, local, _ in runs:
                    offset = fb.add_i(fb.mul_i(row, fb.const(n)), fb.const(column))
                    ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(local),
                                       rows, width, fb=fb))
                ctx.fence()
    if full:
        fb.for_loop(0, full, isa.DIM, lambda row: row_tile(row, isa.DIM), retain_loop=True)
    if full < m:
        row_tile(fb.const(full), m - full)
    ctx.fence()
    return dict(selection='palette', route='identical_config_contiguous_rectangles',
        run_selection='contiguous', palette_groups=len(groups),
        rectangles_by_row_extent={str(rows): sum(len(runs) for _, batches in plan for runs in batches)
                                  for rows, plan in plans.items()},
        original_batch_boundaries='unchanged; real load and store drains retained',
        raw_row_stride_bytes=4*n, output_row_stride_bytes=n,
        source_partition='exact original columns, exact complete CONFIG_ST, DIM-bounded rectangles',
        admission=admission)
