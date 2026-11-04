"""Bounded shared command-loop expansion before the common target encoder.

This is an optional representation choice, not a scheduling or completion policy.
The shallow mode expands original innermost loops of at most four iterations;
deep16 iterates the same pass over exposed loops of at most sixteen trips. The
shared pass budgets the entire command-bearing SCF module, including expression
operations and declarations, to at most 16384 operations. Consequently static
command call sites are also bounded by 16384; dynamic commands are unchanged.
Auto consumes that detached candidate only when it has no counted loops and
at most 2048 SCF operations. This is a compiler representation budget; rejected
candidates use the original checked emitter without target callbacks in planning.
"""
from __future__ import annotations

import hashlib
import io
from contextlib import contextmanager

from xdsl.dialects import arith, builtin, func, scf, llvm
from xdsl.dialects.builtin import i64
from xdsl.printer import Printer

from merlin.compile.scheduling.scf_commands import (
    materialize_scf_commands, unroll_small_command_loops,
    verify_scf_commands, emit_scf_commands,
)
from merlin.compile.scheduling.llvm_index_bounds import FnBuilderBoundedControlAdapter

MAX_EXTENT = 4
MAX_OPERATIONS = 16384
AUTO_MAX_OPERATIONS = 2048


@contextmanager
def _pack_bounds(builder, bridge, enabled):
    """Publish only ranges derived by the actual counted-control bridge.

    Target address additions are ordinary LLVM operations outside that bridge.
    Derive their ranges recursively, admitting only total signed-i64 arithmetic
    with no overflow flags; unknown producers and overflow discard the proof.
    """
    previous = getattr(builder, '_gemmini_pack_bounds', None)
    previous_stats = getattr(builder, '_gemmini_pack_bounds_stats', None)
    stats = {'proven_masks_removed': 0, 'unknown_or_outside_masks_retained': 0,
             'basis': 'actual counted i64 intervals; signed-overflow-free arithmetic'}
    cache = {}
    def bounds(value):
        known = bridge.bounds(value)
        if known is not None:
            return known
        if value in cache:
            return cache[value]
        cache[value] = None
        if getattr(value, 'type', None) != i64:
            return None
        owner = value.owner
        answer = None
        if getattr(owner, 'name', None) == 'llvm.mlir.constant':
            attr = owner.properties.get('value')
            if hasattr(attr, 'value'):
                number = attr.value.data
                if -(1 << 63) <= number < 1 << 63:
                    answer = (number, number)
        elif type(owner) is llvm.AndOp:
            # AND with any proven nonnegative operand has its sign bit clear
            # and is unsigned <= that operand, even if the other is unknown.
            # Preserve the AND itself (e.g. clearing accumulator selectors).
            ceilings = []
            for operand in owner.operands:
                interval = bounds(operand)
                if interval is not None and 0 <= interval[0] <= interval[1] < (1 << 63):
                    ceilings.append(interval[1])
            if ceilings:
                answer = (0, min(ceilings))
        elif type(owner) in (llvm.AddOp, llvm.SubOp, llvm.MulOp, llvm.SelectOp, llvm.OrOp):
            flags = owner.properties.get('overflowFlags')
            if flags is not None and getattr(getattr(flags, 'value', None), 'data', None) != 0:
                return None
            operands = owner.operands[-2:]
            left, right = (bounds(v) for v in operands)
            if left is not None and right is not None:
                a, b = left; c, d = right
                if type(owner) is llvm.AddOp: answer = (a+c, b+d)
                elif type(owner) is llvm.SubOp: answer = (a-d, b-c)
                elif type(owner) is llvm.MulOp:
                    products = (a*c, a*d, b*c, b*d)
                    answer = (min(products), max(products))
                elif type(owner) is llvm.OrOp:
                    # Nonnegative i64 values cannot set a bit above either
                    # operand's highest possible bit. This admits flagged
                    # accumulator rows without assuming flag/row disjointness.
                    if not (0 <= a <= b < (1 << 63) and 0 <= c <= d < (1 << 63)):
                        return None
                    answer = (0, (1 << max(b,d).bit_length()) - 1)
                else: answer = (min(a,c), max(b,d))
                if not -(1 << 63) <= answer[0] <= answer[1] < 1 << 63:
                    answer = None
        cache[value] = answer
        return answer
    if enabled:
        builder._gemmini_pack_bounds = bounds
        builder._gemmini_pack_bounds_stats = stats
    try:
        yield stats if enabled else None
    finally:
        if previous is None:
            builder.__dict__.pop('_gemmini_pack_bounds', None)
        else:
            builder._gemmini_pack_bounds = previous
        if previous_stats is None:
            builder.__dict__.pop('_gemmini_pack_bounds_stats', None)
        else:
            builder._gemmini_pack_bounds_stats = previous_stats


def _bounds_enabled():
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    return knob('command_pack_bounds', default='off', values=('off','on'),
                root=Path(__file__).resolve().parent) == 'on'


def emit_bounded_compact(result, builder, lower_call):
    """Incumbent source index program with independently optional pack proofs."""
    from merlin.compile.scheduling.llvm_program import emit_index_program
    result.revalidate()
    if result.index_bits != 64:
        raise ValueError('bounded command indices require checked i64 indices')
    bridge = FnBuilderBoundedControlAdapter(builder)
    with _pack_bounds(builder, bridge, _bounds_enabled()) as pack_stats:
        report = emit_index_program(result.kernel, bridge,
            lambda instruction, arguments: lower_call(instruction,
                {key: bridge.value(value) for key, value in arguments.items()}))
    result.revalidate()
    return {**report, 'pack_bounds': pack_stats}


def emit_bounded_effects(result, builder, lower_effect, *, complete_epoch):
    """Publish actual counted ranges through a checked asynchronous owner.

    The owner's emitter retains its original command operands and epoch-drain
    callback. The same registered control bridge introduces only loop/constant
    facts; the existing encoder consumes them for exact unsigned field packing.
    No schedule, target effect or completion boundary is transformed here.
    """
    bridge = FnBuilderBoundedControlAdapter(builder)
    with _pack_bounds(builder, bridge, _bounds_enabled()) as pack_stats:
        report = result.emit_index_program(
            bridge, lower_effect, complete_epoch=complete_epoch)
    return {**report, 'pack_bounds': pack_stats}


def _summary(module):
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    operations = tuple(module.walk())
    return {
        'sha256': hashlib.sha256(stream.getvalue().encode()).hexdigest(),
        'operations': len(operations),
        'counted_loops': sum(type(op) is scf.ForOp for op in operations),
        'static_command_calls': sum(type(op) is func.CallOp for op in operations),
    }


def _expanded_operation_lower_bound(module, limit):
    """Count operations preserved by complete SCF expansion, with saturation.

    The shared unroller clones every body operation except for/yield, replaces
    block arguments and inserts IV constants. It performs no folding or DCE.
    Thus these preserved operations alone bound a loop-free result from below.
    Unknown control or bounds decline the proof; this is not a new unroller.
    """
    def count(operation):
        if type(operation) is scf.YieldOp:
            return 0
        if type(operation) is scf.ForOp:
            bounds = []
            for value in (operation.lb, operation.ub, operation.step):
                owner = value.owner
                if type(owner) is not arith.ConstantOp or not isinstance(owner.value, builtin.IntegerAttr):
                    return None
                bounds.append(owner.value.value.data)
            low, high, step = bounds
            if step <= 0:
                return None
            trips = max(0, (high - low + step - 1) // step)
            if trips == 0:
                return 0
            body = total(operation.body.block.ops)
            return None if body is None else min(limit + 1, trips * body)
        if operation.regions and type(operation) not in (builtin.ModuleOp, func.FuncOp):
            return None
        children = total(child for region in operation.regions
                         for block in region.blocks for child in block.ops)
        return None if children is None else min(limit + 1, 1 + children)
    def total(operations):
        result = 0
        for operation in operations:
            amount = count(operation)
            if amount is None:
                return None
            result = min(limit + 1, result + amount)
        return result
    return count(module)


def plan_compact(result, *, mode='on'):
    """Return a verified consumed SCF alternative and source-bound receipt.

    Refusals happen before any target callback. The original checked schedule
    remains untouched. Its revalidation is required both before and after the
    shared transform; no schedule analysis is reused for the transformed SCF.
    """
    result.revalidate()
    if result.index_bits != 64:
        raise ValueError('command-loop expansion requires checked i64 indices')
    before = materialize_scf_commands(result.kernel)
    original = _summary(before)
    if mode not in ('on', 'deep16', 'auto'):
        raise ValueError('unknown command-loop expansion mode')
    def depth(operation):
        parent = operation.parent_op()
        return int(type(operation) is scf.ForOp) + (depth(parent) if parent is not None else 0)
    original_depth = max((depth(op) for op in before.walk()), default=0)
    after = before
    current = original
    steps = []
    refusal = None
    # Every pass selects innermost loops only. Unit-loop removal can expose a
    # previously nested K loop; expanded paths lose one nesting level per pass.
    # Two phases of at most original depth therefore bound pass invocations.
    deep = mode in ('deep16', 'auto')
    lower_bound = (_expanded_operation_lower_bound(before, AUTO_MAX_OPERATIONS)
                   if mode == 'auto' else None)
    budget_proven = lower_bound is not None and lower_bound > AUTO_MAX_OPERATIONS
    phases = ((1, original_depth), (16, original_depth)) if deep else ((MAX_EXTENT, 1),)
    if budget_proven:
        phases = ()
    for extent, repeats in phases:
        for _ in range(repeats):
            old = current
            try:
                candidate = unroll_small_command_loops(
                    after, max_extent=extent, max_operations=MAX_OPERATIONS)
            except ValueError as error:
                if 'budget' not in str(error):
                    raise
                refusal = str(error)
                break
            new = _summary(candidate)
            if new['sha256'] == old['sha256']:
                break
            after = candidate
            current = new
            steps.append({'max_extent':extent, 'before':old, 'after':new})
        if refusal is not None:
            break
    verify_scf_commands(after, max_operations=MAX_OPERATIONS)
    realized = current
    result.revalidate()
    selected = mode != 'auto' or (not budget_proven and
        realized['counted_loops'] == 0 and realized['operations'] <= AUTO_MAX_OPERATIONS)
    return after, {
        'mechanism': 'unroll_small_command_loops',
        'mode': mode, 'max_extent': 16 if deep else MAX_EXTENT,
        'original_loop_depth': original_depth,
        'max_pass_invocations': 2 * original_depth if deep else 1,
        'steps': steps, 'budget_fallback': refusal,
        'max_static_scf_operations': MAX_OPERATIONS,
        'before': original, 'after': realized,
        'applied': selected and original['sha256'] != realized['sha256'],
        'selected': selected,
        'auto_decision': ({
            'selected': selected,
            'max_realized_scf_operations': AUTO_MAX_OPERATIONS,
            'requires_loop_free': True,
            **({'expanded_operation_lower_bound': lower_bound}
               if budget_proven else {}),
            'reason': ('proven expanded operation lower bound exceeds compiler code budget'
                       if budget_proven else
                       'loop-free candidate within compiler code budget' if selected else
                       'candidate retains counted loops' if realized['counted_loops'] else
                       'candidate exceeds compiler code budget'),
        } if mode == 'auto' else None),
        'scope': 'command representation; original dynamic effect/completion order',
    }


def emit_compact(result, builder, lower_call, *, mode='on'):
    """Emit the verified alternative through the incumbent target callback.

    The caller must discard a partially populated builder if emission fails.
    This bridge does not suppress or move any terminal fence or DMA operation.
    """
    try:
        module, receipt = plan_compact(result,mode=mode)
    except ValueError as error:
        if 'budget' not in str(error):
            raise
        # Planning ran before any target callback; an oversized source program
        # can safely retain its incumbent counted representation in full.
        result.emit_llvm(builder,lower_call)
        return {'mechanism':'unroll_small_command_loops','mode':mode,
                'applied':False,'budget_fallback':str(error),
                'selected':False,
                'auto_decision': ({'selected':False,
                    'max_realized_scf_operations':AUTO_MAX_OPERATIONS,
                    'requires_loop_free':True,
                    'reason':'shared planning operation budget refusal'} if mode=='auto' else None),
                'max_static_scf_operations':MAX_OPERATIONS}
    if not receipt['selected']:
        # The candidate was detached and no target callback has run. Retaining
        # the original emitter also retains its expression/control spelling.
        if _bounds_enabled():
            receipt['pack_bounds'] = emit_bounded_compact(result, builder, lower_call)['pack_bounds']
        else:
            result.emit_llvm(builder, lower_call)
        return receipt
    bridge = FnBuilderBoundedControlAdapter(builder)
    with _pack_bounds(builder, bridge, _bounds_enabled()) as pack_stats:
        emit_scf_commands(module, bridge,
            lambda instruction, arguments, metadata: lower_call(
                instruction, {key: bridge.value(value) for key, value in arguments.items()}))
    result.revalidate()
    if pack_stats is not None:
        receipt['pack_bounds'] = pack_stats
    return receipt
