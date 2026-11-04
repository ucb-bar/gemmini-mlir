"""Pure host-index LICM before the common dense SCF/ISA terminal.

This is address arithmetic motion, not asynchronous DMA motion. Only total
integer arithmetic moves; original device commands, operands, counted loops,
configuration, and completion remain in their original order and scope.
"""
from __future__ import annotations

from xdsl.dialects import arith, builtin, func, scf
from merlin.compile.scheduling.scf_commands import (
    materialize_scf_commands, verify_scf_commands, emit_scf_commands,
)
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter

_PURE = (arith.ConstantOp, arith.AddiOp, arith.MuliOp, arith.CmpiOp, arith.SelectOp)


def _inventory(module):
    """Preserve exact SSA topology/flags and every nonarithmetic parent/order."""
    topology = {op: (tuple(op.operands), tuple(op.results), dict(op.attributes),
                     dict(op.properties), tuple(op.regions)) for op in module.walk()}
    effects = tuple((op, op.parent_block()) for op in module.walk() if type(op) not in _PURE)
    return topology, effects


def _work(module):
    total = 0
    def walk(block, multiplier):
        nonlocal total
        for op in block.ops:
            if type(op) in _PURE:
                total += multiplier
            elif type(op) is scf.ForOp:
                # Shared verification establishes static positive step bounds.
                low, high, step = (x.owner.value.value.data for x in (op.lb, op.ub, op.step))
                walk(op.body.block, multiplier * ((high-low+step-1)//step))
    for op in module.body.block.ops:
        if type(op) is func.FuncOp and op.body.blocks:
            walk(op.body.block, 1)
    return total


def hoist_dense_address_invariants(module):
    """Return (private transformed SCF module, receipt), or raise ValueError.

    Input must be the actual shared command-bearing SCF, with closed i64/i1
    arithmetic. The shared verifier rejects unknown memory/control operations,
    events without their contracts, and unsupported types before mutation.
    Empty/zero-trip loops are lawful: speculated operations are total integers.
    No operand/flag rewrite, CSE, loop removal, command motion or wait motion.
    """
    source = verify_scf_commands(module)
    for op in module.walk():
        if type(op) not in _PURE:
            continue
        expected = ({'value'} if type(op) is arith.ConstantOp else
                    {'overflowFlags'} if type(op) in (arith.AddiOp, arith.MuliOp) else
                    {'predicate'} if type(op) is arith.CmpiOp else set())
        if op.attributes or set(op.properties) != expected:
            raise ValueError('dense index hoisting refuses unknown arithmetic attributes or flags')
        if type(op) in (arith.AddiOp, arith.MuliOp) and op.overflow_flags.data:
            raise ValueError('dense index hoisting requires total wraparound integer arithmetic')
    actual = module.clone()
    original = _inventory(actual)
    before_work = _work(actual)
    moves = 0
    moved = set()

    def visit(block):
        nonlocal moves
        for loop in tuple(block.ops):
            if type(loop) is not scf.ForOp:
                continue
            visit(loop.body.block)
            # All definitions nested in this loop, including its induction
            # variable and event handles, cannot dominate its preheader.
            local = set(loop.body.block.args)
            for nested in loop.body.walk():
                local.update(nested.results)
                for region in nested.regions:
                    for child in region.blocks:
                        local.update(child.args)
            for op in tuple(loop.body.block.ops):
                if type(op) not in _PURE or any(value in local for value in op.operands):
                    continue
                # The whitelist is total and memory-free; retain exact flags,
                # bit widths and SSA operands, including wraparound semantics.
                loop.body.block.detach_op(op)
                block.insert_op_before(op, loop)
                local.difference_update(op.results)
                moves += 1
                moved.add(op)

    for entry in tuple(actual.body.block.ops):
        if type(entry) is func.FuncOp and entry.body.blocks:
            visit(entry.body.block)
    if _inventory(actual) != original:
        raise ValueError('dense index hoisting changed SSA topology, flags, command order or scope')
    checked = verify_scf_commands(actual)
    if verify_scf_commands(module) != source:
        raise ValueError('dense index hoisting mutated its source')
    report = dict(schema='dense_address_invariants.v1', applied=bool(moves),
        source_schedule_sha256=source['source_schedule_sha256'],
        source_command_ir_sha256=source['command_ir_sha256'],
        command_ir_sha256=checked['command_ir_sha256'],
        hoisted_static_operations=len(moved), loop_boundaries_crossed=moves,
        original_index_operation_executions=before_work,
        transformed_index_operation_executions=_work(actual),
        effects='identical commands, SSA operands, flags, loop scope/order and completion',
        arithmetic='only total i64/i1 constant/add/mul/eq/select; dominant SSA operands',
        timing='UNKNOWN')
    return actual, report


def emit_dense_address_hoisted(schedule, builder, lower_call):
    """Consume one actual transformed SCF artifact through the same terminal.

    Call after schedule/store-epoch selection. The original complete dense
    product owner is revalidated before and after emission. Materialization
    happens exactly once; the transformed artifact reaches emit_scf_commands.
    The target callback still emits every real configuration and full drain.
    """
    schedule.revalidate()
    kernel_identity = schedule.kernel.digest()
    source = materialize_scf_commands(schedule.kernel)
    actual, report = hoist_dense_address_invariants(source)
    emission = emit_scf_commands(actual, FnBuilderIndexAdapter(builder), lower_call)
    schedule.revalidate()
    if schedule.kernel.digest() != kernel_identity:
        raise ValueError('dense source kernel changed during address emission')
    if emission['command_ir_sha256'] != report['command_ir_sha256']:
        raise ValueError('dense emission did not consume its transformed address artifact')
    return dict(report, emission=emission, kernel_sha256=kernel_identity)

# Common-terminal hook spelling; both names consume the same artifact once.
emit_hoisted_dense = emit_dense_address_hoisted
