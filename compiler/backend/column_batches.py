"""Fact-bound i32 column batches with synchronous configuration boundaries.

Load responses may complete out of order, so each original column owns a
disjoint ACC row interval and all loads drain before readout. Store completion
remains real for every column in ``on`` mode. ``palette`` mode holds one exact
CONFIG_ST word unchanged through each store group and drains before changing
it. Selected identical-MVOUT ordering observations do not independently
establish snapshots across changing CONFIG_ST words.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from . import isa
from .dense_readout import store_tile


def _admission():
    record = json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering = record['ordering']
    digest = hashlib.sha256(json.dumps(ordering, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if (digest != 'ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or ordering.get('status') != 'derived'
            or record['base_facts_sha256'] != 'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'):
        raise ValueError('column load batching requires selected public ordering facts')
    queues = {}
    for row in ordering['funct_queue']:
        if 'by_code' in row:
            queues.update({('config', int(k)): v for k, v in row['by_code'].items()})
        elif 'funct' in row:
            queues[row['funct']] = row['queue']
    if (queues.get(isa.FUNCT['mvin3']) != 'load'
            or queues.get(('config', 1)) != 'load'
            or queues.get(isa.FUNCT['mvout']) != 'store'
            or queues.get(('config', 2)) != 'store'
            or ordering['same_queue']['load']['release'] != 'issue'
            or ordering['same_queue']['store']['release'] != 'completion'
            or ordering['completion_order']['store']['order'] != 'in_order'):
        raise ValueError('column load batching lacks selected lane ordering')
    for lane in ('load', 'store'):
        entry = record['finite_entries'][lane]
        if entry.get('status') != 'derived' or type(entry.get('value')) is not int or entry['value'] < 1:
            raise ValueError('column load batching lacks finite queue progress')
    return dict(ordering_sha256=digest, base_facts_sha256=record['base_facts_sha256'],
        load_completion=ordering['completion_order']['load']['order'],
        store_completion=ordering['completion_order']['store']['order'],
        finite_load_entries=record['finite_entries']['load']['value'],
        finite_store_entries=record['finite_entries']['store']['value'],
        accumulator_capacity_rows=isa.ACC_ROWS,
        store_configuration_snapshot='not assumed; each store fully completes')


def emit_batched_column_readout(ctx, raw, dst, m, n, attributes):
    """Emit the optional load-batched route, or return None without emitting.

    Each batch is one spatial row tile across original columns. Its local row
    for column slot j is j*rows, and rows*columns never exceeds ACC_ROWS. Loads
    and stores use the original flattened row-major M*N layout. Batch completion
    precedes every row-window reuse. Raw producer completion is retained by the
    caller and by this function's initial fence.
    """
    selection = attributes.get('column_readout_batching', knob(
        'COLUMN_READOUT_BATCHING', default='off', values=('off', 'on', 'palette'),
        root=Path(__file__).resolve().parents[1]))
    if selection not in ('off', 'on', 'palette'):
        raise ValueError('COLUMN_READOUT_BATCHING must be off, on or palette')
    if selection == 'off':
        return None
    if type(m) is not int or type(n) is not int or min(m, n) <= 0:
        raise ValueError('column batching requires positive static matrix geometry')
    if (ctx.spec(raw)['dtype'] != 'i32' or ctx.spec(dst)['dtype'] != 'i8'
            or tuple(ctx.spec(raw)['shape']) != (m, n)):
        raise ValueError('column batching requires original i32 M*N raw storage and i8 output')
    if ctx.buffer(raw) is ctx.buffer(dst):
        raise ValueError('column batching requires disjoint raw and output storage')
    if selection == 'palette':
        return _emit_palette_readout(ctx, raw, dst, m, n, attributes)
    admission = _admission()
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n * 4, channel=2, shrink=False))

    def row_tile(row, rows):
        capacity = isa.ACC_ROWS // rows
        def batch(first_column, columns):
            def load(slot):
                column = fb.add_i(first_column, slot)
                offset = fb.add_i(fb.mul_i(row, fb.const(n)), column)
                local_row = fb.mul_i(slot, fb.const(rows))
                ctx.emit(*isa.mvin(ctx.ptr(raw, offset), isa.acc_addr(local_row, fb=fb),
                                  rows, 1, channel=2, fb=fb))
            fb.for_loop(0, columns, 1, load, retain_loop=True)
            ctx.fence()  # All disjoint load responses are now visible.
            def store(slot):
                column = fb.add_i(first_column, slot)
                local_row = fb.mul_i(slot, fb.const(rows))
                # No readiness fence is needed after the common load drain or
                # after the preceding store drain. Changing scales still has
                # the original real after-store completion fence.
                store_tile(ctx, dst, local_row, rows, 1, n, row, column, attributes,
                    completion=lambda phase: ctx.fence() if phase == 'after_store' else None)
            fb.for_loop(0, columns, 1, store, retain_loop=True)
        complete = n // capacity * capacity
        if complete:
            fb.for_loop(0, complete, capacity,
                lambda first: batch(first, capacity), retain_loop=True)
        if complete < n:
            batch(fb.const(complete), n - complete)
    complete = m // isa.DIM * isa.DIM
    if complete:
        fb.for_loop(0, complete, isa.DIM,
            lambda row: row_tile(row, isa.DIM), retain_loop=True)
    if complete < m:
        row_tile(fb.const(complete), m - complete)
    ctx.fence()
    return dict(selection='on', route='disjoint_acc_batched_loads',
        batch_columns_full_rows=min(n, isa.ACC_ROWS // min(m, isa.DIM)),
        maximum_live_accumulator_rows=max(
            min(n, isa.ACC_ROWS // rows) * rows
            for rows in ({isa.DIM} if m >= isa.DIM and m % isa.DIM == 0
                         else ({isa.DIM, m % isa.DIM} if m >= isa.DIM else {m}))),
        raw_row_stride_bytes=n * 4, output_row_stride_bytes=n,
        load_drain='one full drain after all loads in each batch',
        store_drain='one full drain after every original scaled store',
        ownership='private disjoint ACC intervals; full drain before reuse',
        admission=admission)


def _palette(n, attributes):
    """Partition original columns by their exact complete store command."""
    ep = attributes.get('epilogue', ())
    if any('pool' in str(stage).lower() for stage in ep):
        raise ValueError('palette readout requires separate spatial pooling')
    if attributes.get('requant_shift') is not None and 'requant' in ep:
        raise ValueError('half-up requantization has no exact Gemmini readout contract')
    scales = attributes.get('acc_scale_by_column')
    if scales is None:
        scales = [attributes.get('acc_scale', 1.0)] * n
    if len(scales) != n:
        raise ValueError('palette source scales differ from output channels')
    activation = int(bool(attributes.get('relu', False) or 'relu' in ep))
    groups = {}
    for column, scale in enumerate(scales):
        command = isa.config_st(n, scale=scale, activation=activation)
        groups.setdefault(command, []).append(column)
    groups = tuple((command, tuple(columns)) for command, columns in groups.items())
    if sorted(column for _, columns in groups for column in columns) != list(range(n)):
        raise ValueError('palette must partition every original column exactly once')
    for command, columns in groups:
        if any(isa.config_st(n, scale=scales[c], activation=activation) != command for c in columns):
            raise ValueError('palette configuration differs from an original column')
    return groups


def _column_at(fb, position, columns):
    """Immutable address selection without configuration or tensor loads."""
    if len(columns) == 1:
        return fb.const(columns[0])
    step = columns[1] - columns[0]
    if all(value == columns[0] + i * step for i, value in enumerate(columns)):
        return fb.add_i(fb.const(columns[0]), fb.mul_i(position, fb.const(step)))
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import IntegerAttr, i64
    from xdsl.ir import Block
    merge = Block(arg_types=[i64])
    def choose(lo, hi):
        if hi - lo == 1:
            fb.add(llvm.BrOp(merge, fb.const(columns[lo])))
            return
        mid = (lo + hi) // 2
        left, right = Block(), Block()
        fb.region.add_block(left)
        fb.region.add_block(right)
        condition = fb.add(llvm.ICmpOp(position, fb.const(mid),
            IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))).results[0]
        fb.add(llvm.CondBrOp(condition, left, [], right, []))
        fb.blk = left
        choose(lo, mid)
        fb.blk = right
        choose(mid, hi)
    choose(0, len(columns))
    fb.region.add_block(merge)
    fb.blk = merge
    return merge.args[0]


def _emit_palette_readout(ctx, raw, dst, m, n, attributes):
    """Issue stores with one unchanged configuration until a real group drain.

    Column writes commute because their original output elements and ACC row
    intervals are disjoint. The compile-time partition preserves every original
    complete CONFIG_ST word. No younger, different configuration is issued
    until the prior group's stores fully complete. Finite store backpressure
    and selected same-lane in-order completion serialize the unchanged-config
    stores; neither varying-config snapshots nor cross-queue hazards are used.
    """
    groups = _palette(n, attributes)
    # Bound static control expansion independently of N. Unusually large
    # palettes keep the existing synchronous varying-scale store consumer.
    if len(groups) > 32:
        result = emit_batched_column_readout(ctx, raw, dst, m, n,
            dict(attributes, column_readout_batching='on'))
        return dict(result, requested_selection='palette',
                    palette_declined='more than 32 distinct complete CONFIG_ST words')
    admission = _admission()
    admission['store_configuration_snapshot'] = 'not assumed; identical configuration through group drain'
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n * 4, channel=2, shrink=False))

    def row_tile(row, rows):
        capacity = isa.ACC_ROWS // rows
        for command, columns in groups:
            def batch(first, count):
                def load(slot):
                    column = _column_at(fb, fb.add_i(first, slot), columns)
                    offset = fb.add_i(fb.mul_i(row, fb.const(n)), column)
                    local_row = fb.mul_i(slot, fb.const(rows))
                    ctx.emit(*isa.mvin(ctx.ptr(raw, offset), isa.acc_addr(local_row, fb=fb),
                                      rows, 1, channel=2, fb=fb))
                fb.for_loop(0, count, 1, load, retain_loop=True)
                ctx.fence()
                ctx.emit(*command)
                def store(slot):
                    column = _column_at(fb, fb.add_i(first, slot), columns)
                    offset = fb.add_i(fb.mul_i(row, fb.const(n)), column)
                    local_row = fb.mul_i(slot, fb.const(rows))
                    ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(local_row, fb=fb),
                                       rows, 1, fb=fb))
                fb.for_loop(0, count, 1, store, retain_loop=True)
                ctx.fence()  # Before another configuration or any ACC reuse.
            complete = len(columns) // capacity * capacity
            if complete:
                fb.for_loop(0, complete, capacity,
                    lambda first: batch(first, capacity), retain_loop=True)
            if complete < len(columns):
                batch(fb.const(complete), len(columns) - complete)
    complete = m // isa.DIM * isa.DIM
    if complete:
        fb.for_loop(0, complete, isa.DIM,
            lambda row: row_tile(row, isa.DIM), retain_loop=True)
    if complete < m:
        row_tile(fb.const(complete), m - complete)
    ctx.fence()
    row_extents = ([isa.DIM] if complete else []) + ([m - complete] if complete < m else [])
    return dict(selection='palette', route='identical_config_disjoint_acc_groups',
        palette_groups=len(groups), palette_column_counts=[len(columns) for _, columns in groups],
        palette_commands=[list(command) for command, _ in groups],
        maximum_live_accumulator_rows=max(min(len(columns), isa.ACC_ROWS // rows) * rows
            for _, columns in groups for rows in row_extents),
        raw_row_stride_bytes=n * 4, output_row_stride_bytes=n,
        load_drain='one full drain before every fixed-configuration store group',
        store_drain='one full drain before configuration change or ACC reuse',
        partition='every original column exactly once; original complete store word preserved',
        admission=admission)
