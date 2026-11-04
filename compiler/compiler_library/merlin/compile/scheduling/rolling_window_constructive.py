"""Bounded constructive check of canonical counted rolling-row transport.

The IR match includes every operation, operand/use, type, property, attribute,
region and successor. Independent interval interpretation proves all actual
integer intermediates. Source algebra below proves coverage and the ring
induction for arbitrary loop extents; it does not execute or sample rows.
Noncanonical editable programs remain the original SMT checker's domain.
"""
from __future__ import annotations

import hashlib
from xdsl.dialects import arith, builtin, func, scf


def _require(condition, reason):
    if not condition:
        raise ValueError('constructive rolling-window check: ' + reason)


def _same_ir(actual, expected):
    if not actual.is_structurally_equivalent(expected):
        return False
    # xDSL's structural comparison checks attributes/properties and SSA uses,
    # but does not itself compare operation-result types.
    aa, bb = tuple(actual.walk()), tuple(expected.walk())
    return len(aa) == len(bb) and all(
        type(a) is type(b)
        and tuple(v.type for v in a.results) == tuple(v.type for v in b.results)
        and tuple(tuple(v.type for v in block.args) for r in a.regions for block in r.blocks)
            == tuple(tuple(v.type for v in block.args) for r in b.regions for block in r.blocks)
        for a, b in zip(aa, bb, strict=True))


def _check_intervals(module):
    """Interpret each actual SSA op once over closed integer loop intervals."""
    checked = 0
    def block_body(block, inherited):
        nonlocal checked
        env = dict(inherited)
        for op in block.ops:
            if type(op) is arith.ConstantOp:
                v = op.value.value.data
                result = (v, v)
            elif type(op) in (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.RemUIOp):
                a, b = env[op.lhs], env[op.rhs]
                if type(op) is arith.AddiOp:
                    result = (a[0] + b[0], a[1] + b[1])
                elif type(op) is arith.SubiOp:
                    result = (a[0] - b[1], a[1] - b[0])
                elif type(op) is arith.MuliOp:
                    products = tuple(x * y for x in a for y in b)
                    result = (min(products), max(products))
                else:
                    _require(a[0] >= 0 and b[0] == b[1] and b[0] > 0,
                             'unsigned ring remainder lacks nonnegative numerator/positive constant divisor')
                    result = (0, min(a[1], b[0] - 1))
            elif type(op) is arith.CmpiOp:
                _require(op.predicate.value.data == 6, 'padding guard is not unsigned less-than')
                result = (0, 1)
            elif type(op) is scf.ForOp:
                low, high, step = env[op.lb], env[op.ub], env[op.step]
                _require(low[0] == low[1] and high[0] == high[1]
                         and step == (1, 1) and low[0] < high[0], 'loop is not a nonempty constant counted domain')
                block_body(op.body.block, {**env, op.body.block.args[0]: (low[0], high[0] - 1)})
                continue
            else:
                continue
            checked += 1
            if op.results[0].type == builtin.i64:
                _require(-(1 << 63) <= result[0] <= result[1] < 1 << 63,
                         'actual SSA address intermediate may overflow signed i64')
            env[op.results[0]] = result
    block_body(tuple(module.body.block.ops)[0].body.block, {})
    return checked


# Exact affine normalization for independently checking the actual lookup
# builder. Remainders are opaque, structurally bound terms of positive modulus.
def _affine(items=()):
    terms = {}
    for atom, coefficient in items:
        terms[atom] = terms.get(atom, 0) + coefficient
    return tuple(sorted(((a, c) for a, c in terms.items() if c), key=repr))


def _const(value):
    return _affine(((('constant',), value),))


def _add(a, b, sign=1):
    return _affine((*a, *((atom, sign * coefficient) for atom, coefficient in b)))


def _scale(a, factor):
    return _affine((atom, coefficient * factor) for atom, coefficient in a)


class _LookupBuilder:
    def op(self, name, a, b, *unused):
        if type(a) is int: a = _const(a)
        if type(b) is int: b = _const(b)
        if name == 'and': return a | b
        if name == 'add': return _add(a, b)
        if name == 'sub': return _add(a, b, -1)
        if name == 'mul':
            for constant, other in ((a, b), (b, a)):
                if not constant: return ()
                if len(constant) == 1 and constant[0][0] == ('constant',):
                    return _scale(other, constant[0][1])
            raise ValueError('lookup introduced non-affine multiplication')
        if name == 'urem':
            _require(len(b) == 1 and b[0][0] == ('constant',) and b[0][1] > 0,
                     'lookup introduced unproved remainder')
            return _affine(((('remainder', a, b[0][1]), 1),))
        raise ValueError('lookup introduced unknown arithmetic')

    def cmp(self, name, a, b):
        if type(a) is int: a = _const(a)
        if type(b) is int: b = _const(b)
        return frozenset(((name, a, b),))


def _check_lookup(plan):
    from .rolling_window import emit_rolling_window_lookup
    builder = _LookupBuilder()
    names = ('y', 'x', 'ky', 'kx', 'channel', 'image')
    y, x, ky, kx, channel, image = (_affine((((name,), 1),)) for name in names)
    w = plan.window
    unbiased = _add(_scale(y, w.stride[0]), _scale(ky, w.dilation[0]))
    iy = _add(unbiased, _const(w.padding[0]), -1)
    ix = _add(_add(_scale(x, w.stride[1]), _const(w.padding[1]), -1), _scale(kx, w.dilation[1]))
    slot = _affine(((('remainder', unbiased, plan.image_slots), 1),))
    if plan.image_windows > 1:
        slot = _add(slot, _affine(((('remainder', image, plan.image_windows), plan.image_slots),)))
    address = _add(_add(_const(plan.cache.start), _scale(slot, plan.row_stride_bytes)),
                   _add(_scale(_add(ix, _const(plan.source_columns[0]), -1), w.input_shape[-1]), channel))
    guards = ((_add(y, _const(plan.output_y_first), -1), plan.output_y_count),
              (_add(x, _const(plan.output_x_first), -1), plan.output_x_count),
              (ky, w.kernel[0]), (kx, w.kernel[1]), (channel, w.input_shape[-1]),
              (iy, w.input_shape[1]), (ix, w.input_shape[2]))
    if plan.image_windows > 1: guards = (*guards, (image, w.input_shape[0]))
    predicate = frozenset(('ult', expression, _const(bound)) for expression, bound in guards)
    actual = emit_rolling_window_lookup(plan, builder, y, x, ky, kx, channel, image=image)
    _require(actual == dict(predicate=predicate, cache_byte=address),
             'actual guarded retained-row lookup differs from original source/tap/ring equations')


def _check_effect_equations(module, plan, output_rows):
    """Normalize actual stage SSA, independently from the constructor's order."""
    from merlin.xdsl_dialects.rolling_window import BeginOp, ConsumeOp, ConsumeGroupOp, StageRowOp
    builder = _LookupBuilder()
    values, stages, serial = {}, [], 0
    # Locate by operation kind rather than assuming where constants were placed.
    image_loop = next(op for op in tuple(module.body.block.ops)[0].body.block.ops
                      if type(op) is scf.ForOp)
    def walk(block):
        nonlocal serial
        for op in block.ops:
            if type(op) is arith.ConstantOp:
                values[op.result] = _const(op.value.value.data)
            elif type(op) in (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.RemUIOp):
                name = {arith.AddiOp:'add', arith.SubiOp:'sub', arith.MuliOp:'mul', arith.RemUIOp:'urem'}[type(op)]
                values[op.result] = builder.op(name, values[op.lhs], values[op.rhs])
            elif type(op) is arith.CmpiOp:
                values[op.result] = builder.cmp('ult', values[op.lhs], values[op.rhs])
            elif type(op) is scf.ForOp:
                serial += 1
                values[op.body.block.args[0]] = _affine((((('iv', serial),), 1),))
                walk(op.body.block)
            elif type(op) is StageRowOp:
                stages.append(op)
    walk(tuple(module.body.block.ops)[0].body.block)
    image = values[image_loop.body.block.args[0]]
    w = plan.window
    for stage in stages:
        begin = stage.transfer.owner
        _require(type(begin) is BeginOp, 'stage transfer is not an original epoch')
        epoch_block = begin.parent
        operations = tuple(epoch_block.ops)
        begin_index = operations.index(begin)
        consume = next(op for op in operations[begin_index + 1:]
                       if type(op) in (ConsumeOp, ConsumeGroupOp))
        initial = epoch_block is image_loop.body.block and not any(
            type(op) in (ConsumeOp, ConsumeGroupOp) for op in operations[:begin_index])
        first = 0 if initial else plan.span_rows - (
            plan.frontier_rows if output_rows is None else w.stride[0])
        row_loop = stage.parent.parent_op()
        row = values[row_loop.body.block.args[0]]
        y = values[consume.output_y]
        unbiased = _add(_add(_scale(y, w.stride[0]), row), _const(first))
        source_y = _add(unbiased, _const(w.padding[0]), -1)
        source = _add(_scale(_add(_scale(image, w.input_shape[1]), source_y),
                             w.input_shape[2] * w.input_shape[3]),
                      _const(plan.source_columns[0] * w.input_shape[3]))
        slot = builder.op('urem', unbiased, plan.image_slots)
        if plan.image_windows > 1:
            slot = _add(slot, _scale(builder.op('urem', image, plan.image_windows), plan.image_slots))
        cache = _add(_const(plan.cache.start), _scale(slot, plan.row_stride_bytes))
        expected = (image, source_y, source, cache, builder.cmp('ult', source_y, w.input_shape[1]))
        actual = tuple(values[value] for value in
            (stage.image, stage.input_row, stage.source_byte, stage.cache_byte, stage.valid))
        _require(actual == expected and values[consume.image] == image,
                 'actual stage/consumer SSA equations differ from original row transport')


def _check_induction(plan, output_rows):
    """Check affine interval endpoints establishing complete arbitrary-Y induction."""
    w = plan.window
    span, advance = plan.span_rows, w.stride[0]
    y, retained = (_affine(((((name,),), 1),)) for name in ('y', 'retained'))
    current = _add(_scale(y, advance), retained)
    if output_rows is None:
        frontier = plan.frontier_rows
        if advance < span:
            old = _add(_scale(_add(y, _const(1), -1), advance),
                       _add(_const(advance), retained))
            previous_end = _add(_scale(_add(y, _const(1), -1), advance), _const(span))
            frontier_start = _add(_scale(y, advance), _const(span - frontier))
            _require(old == current and previous_end == frontier_start,
                     'retained identities or preceding/frontier boundary changed')
        else:
            _require(frontier == span, 'disjoint window must stage its complete span')
        return
    full, tail = divmod(plan.output_y_count, output_rows)
    for consumers in {output_rows, tail} - {0}:
        union = span + (consumers - 1) * advance
        retained_count = span - advance
        _require(retained_count >= 0 and retained_count + consumers * advance == union,
                 'retained prefix and actual frontier do not partition the consumer union')
        _require((consumers - 1) * advance + (w.kernel[0] - 1) * w.dilation[0] == union - 1,
                 'group tap endpoints do not cover the original union')
    old = _add(_scale(_add(y, _const(output_rows), -1), advance),
               _add(_const(output_rows * advance), retained))
    previous_end = _add(_scale(_add(y, _const(output_rows), -1), advance),
                        _const(span + (output_rows - 1) * advance))
    frontier_start = _add(_scale(y, advance), _const(span - advance))
    _require(old == current and previous_end == frontier_start,
             'previous group does not retain precisely the next source prefix')
    _require(full * output_rows + tail == plan.output_y_count,
             'full groups and tail do not partition the original output rows')


def verify_canonical_rolling_window(module, plan, output_rows):
    """Return a proof report for exact canonical IR, or None for SMT fallback."""
    from .rolling_window import (RollingWindowPlan, SCHEMA, GROUP_SCHEMA,
        _construct_rolling_window, _group_horizon, rolling_window_text)
    _require(type(plan) is RollingWindowPlan, 'original typed source proposal required')
    plan.__post_init__()
    module.verify()
    canonical = _construct_rolling_window(plan, output_rows)
    if not _same_ir(module, canonical):
        return None
    checked = _check_intervals(module)
    _check_effect_equations(module, plan, output_rows)
    _check_induction(plan, output_rows)
    w = plan.window
    b, h, width, ci = w.input_shape
    span, advance, slots = plan.span_rows, w.stride[0], plan.image_slots
    horizon = span if output_rows is None else _group_horizon(plan, output_rows)
    _require(0 < horizon <= slots and plan.row_bytes <= plan.row_stride_bytes,
             'complete live union is not injectively assigned to physical rows')
    # Injectivity: two live offsets differ by less than slots. If residues
    # coincide their difference is divisible by slots, hence it is zero.
    # Every residue row is inside one image partition; the complete partition
    # ends at required_bytes. Adjacent image residues differ when windows>1.
    _require(plan.required_bytes == plan.image_windows * slots * plan.row_stride_bytes
             and plan.required_bytes <= plan.cache.size,
             'image partitions exceed caller-owned capacity')
    first, last = plan.source_columns
    xlow = plan.output_x_first * w.stride[1] - w.padding[1]
    xhigh = ((plan.output_x_first + plan.output_x_count - 1) * w.stride[1]
             - w.padding[1] + (w.kernel[1] - 1) * w.dilation[1] + 1)
    _require((first, last) == (max(0, xlow), min(width, xhigh)) and 0 <= first < last <= width,
             'cropped source interval misses an original horizontal tap')
    _require(span == (w.kernel[0] - 1) * w.dilation[0] + 1,
             'source span omits an original dilated row tap')
    if output_rows is not None:
        full, tail = divmod(plan.output_y_count, output_rows)
        _require(advance <= span and full * output_rows + tail == plan.output_y_count
                 and 0 <= tail < output_rows
                 and horizon == span + (min(output_rows, plan.output_y_count) - 1) * advance,
                 'group/full-tail union coverage changed')
    # Initial epoch writes all rows of its consumer union. Each next epoch
    # starts at current_y*advance+span-frontier. The preceding union ends one
    # byte-row earlier than the new frontier starts, and contains exactly the
    # retained prefix. Its source integer and modulo slot are unchanged.
    # For grouped epochs the frontier has consumers*advance rows and retained
    # prefix span-advance; full/tail groups partition the original Y interval.
    # Canonical SSA equality proves those precise bounds and token sequence:
    # begin -> all stages -> await -> original consumer -> complete, before
    # any subsequent epoch can overwrite a row. No payload/target law follows.
    _check_lookup(plan)
    staged = b * (span + (plan.output_y_count - 1) * plan.frontier_rows)
    return dict(schema=SCHEMA if output_rows is None else GROUP_SCHEMA,
        plan_sha256=plan.identity,
        actual_ir_sha256=hashlib.sha256(rolling_window_text(module).encode()).hexdigest(),
        symbolic_queries=0, proof_method='canonical SSA equality plus constructive interval/coverage/ring induction',
        checked_arithmetic_operations=checked,
        dynamic_stage_rows=staged, dynamic_stage_bytes=staged * plan.row_bytes,
        consumers=b * plan.output_y_count,
        unstaged_reference_rows=b * plan.output_y_count * span,
        retained_rows_per_advance=span - plan.frontier_rows,
        cache_bytes=plan.required_bytes,
        actual_counted_ir_operations=sum(1 for _ in module.walk()),
        conditional_on=['original source identity and immutable input', 'disjoint caller-owned cache',
            'exact guarded row copy/zero completed at await',
            'original consumer reads only checked window view and completes at complete'],
        target_qualification=False,
        **({} if output_rows is None else dict(output_rows_per_group=output_rows,
            consumer_groups=b * ((plan.output_y_count + output_rows - 1) // output_rows),
            source_row_union=horizon, max_staged_rows_per_epoch=horizon,
            completion='await all actual group transports before consumer; full consumer completion before any reuse')))
