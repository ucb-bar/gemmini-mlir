"""Exact preparation of repeated convolution palette store configurations.

Transfers, source coordinates and all completion boundaries retain the ordinary
palette schedule. Only an identical CONFIG_ST after a completed batch is elided:
no intervening operation changes the store configuration. This needs neither a
varying-configuration snapshot nor a new completion-order law.
"""
from dataclasses import dataclass
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from . import isa
from .column_batches import _palette, _column_at, _admission


@dataclass(frozen=True)
class PreparedPalette:
    rows: int
    columns: int
    groups: tuple
    row_extents: tuple

    def validate(self, attributes):
        if self.groups != _palette(self.columns, attributes):
            raise ValueError('prepared palette differs from original complete store words')
        if (self.rows <= 0 or self.columns <= 0 or not self.row_extents
                or any(not 0 < rows <= isa.DIM for rows in self.row_extents)):
            raise ValueError('prepared palette has invalid original row partition')
        full, tail = divmod(self.rows, isa.DIM)
        expected = ((isa.DIM,) if full else ()) + ((tail,) if tail else ())
        if expected != self.row_extents:
            raise ValueError('prepared palette changed original row partition')
        if any(isa.ACC_ROWS // rows < 1 for rows in self.row_extents):
            raise ValueError('prepared palette exceeds accumulator capacity')


def prepare_palette(m, n, attributes):
    if type(m) is not int or type(n) is not int or min(m, n) <= 0:
        raise ValueError('prepared palette requires positive static matrix geometry')
    groups = _palette(n, attributes)
    if len(groups) > 32:
        return None
    full, tail = divmod(m, isa.DIM)
    extents = ((isa.DIM,) if full else ()) + ((tail,) if tail else ())
    # Retain the incumbent rectangle consumer whenever it actually applies.
    run_selection = attributes.get('column_readout_runs', knob(
        'COLUMN_READOUT_RUNS', default='off', values=('off', 'contiguous'),
        root=Path(__file__).resolve().parents[1]))
    if run_selection not in ('off', 'contiguous'):
        raise ValueError('COLUMN_READOUT_RUNS must be off or contiguous')
    if run_selection == 'contiguous':
        from .column_runs import plan_column_runs
        if all(plan_column_runs(groups, n, rows, attributes) is not None for rows in extents):
            return None
    if not any(len(columns) > isa.ACC_ROWS // rows
               for _, columns in groups for rows in extents):
        return None
    plan = PreparedPalette(m, n, groups, extents)
    plan.validate(attributes)
    return plan


def emit_prepared_column_readout(ctx, raw, dst, m, n, attributes):
    selection = knob('CONV_COLUMN_PREPARATION', default='off',
        values=('off', 'palette_once'), root=Path(__file__).resolve().parents[1])
    if selection == 'off':
        return None
    batching = attributes.get('column_readout_batching', knob(
        'COLUMN_READOUT_BATCHING', default='off', values=('off', 'on', 'palette'),
        root=Path(__file__).resolve().parents[1]))
    if batching != 'palette':
        return None
    plan = prepare_palette(m, n, attributes)
    if plan is None:
        return None
    if (ctx.spec(raw)['dtype'] != 'i32' or ctx.spec(dst)['dtype'] != 'i8'
            or tuple(ctx.spec(raw)['shape']) != (m, n)
            or ctx.buffer(raw) is ctx.buffer(dst)):
        raise ValueError('prepared palette requires disjoint original raw i32 and output i8 storage')
    plan.validate(attributes)
    admission = _admission()
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n * 4, channel=2, shrink=False))

    def row_tile(row, rows):
        capacity = isa.ACC_ROWS // rows
        for command, columns in plan.groups:
            def batch(first, count, configure=False):
                def transfer(slot, store):
                    column = _column_at(fb, fb.add_i(first, slot), columns)
                    offset = fb.add_i(fb.mul_i(row, fb.const(n)), column)
                    local = isa.acc_addr(fb.mul_i(slot, fb.const(rows)), fb=fb)
                    if store:
                        ctx.emit(*isa.mvout(ctx.ptr(dst, offset), local, rows, 1, fb=fb))
                    else:
                        ctx.emit(*isa.mvin(ctx.ptr(raw, offset), local, rows, 1, channel=2, fb=fb))
                fb.for_loop(0, count, 1, lambda slot: transfer(slot, False), retain_loop=True)
                ctx.fence()
                if configure:
                    ctx.emit(*command)
                fb.for_loop(0, count, 1, lambda slot: transfer(slot, True), retain_loop=True)
                ctx.fence()
            # Peeling leaves the original batch/transfer order unchanged. After
            # the first store drain, only MVIN3, MVOUT and drains intervene;
            # none writes CONFIG_ST, and every store still uses its exact word.
            first_count = min(capacity, len(columns))
            batch(fb.const(0), first_count, True)
            complete = len(columns) // capacity * capacity
            if complete > capacity:
                fb.for_loop(capacity, complete, capacity,
                    lambda first: batch(first, capacity), retain_loop=True)
            if complete < len(columns) and complete:
                batch(fb.const(complete), len(columns) - complete)
    complete = m // isa.DIM * isa.DIM
    if complete:
        fb.for_loop(0, complete, isa.DIM, lambda row: row_tile(row, isa.DIM), retain_loop=True)
    if complete < m:
        row_tile(fb.const(complete), m - complete)
    ctx.fence()
    return dict(selection='palette', route='prepared_identical_config_batches',
        preparation=selection, palette_groups=len(plan.groups),
        palette_column_counts=[len(columns) for _, columns in plan.groups],
        palette_commands=[list(command) for command, _ in plan.groups],
        raw_row_stride_bytes=n * 4, output_row_stride_bytes=n,
        partition='unchanged original row/palette/batch order and scalar transfers',
        load_drain='original full drain before every batch store group',
        store_drain='original full drain before every ACC reuse or config change',
        configuration='first batch exact CONFIG_ST persists through identical later batches',
        admission=admission)
