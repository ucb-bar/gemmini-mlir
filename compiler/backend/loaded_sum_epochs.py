"""Checked loaded-sum epochs, consumed by the ordinary ISA terminal encoder.

The RHS of a bias add is a logical broadcast view: (r,c) maps to bias[c].
That view, its zero DMA stride and full-width scalar arithmetic are byte-bound
in the original proof, rather than claiming a materialized M*N bias buffer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.compact_loaded_sum import (
    LoadedSumLayout, LoadedSumContract, LoadedSumEffect,
)
from merlin.compile.scheduling.loaded_sum_epochs import (
    LoadedSumEpochLaw, plan_loaded_sum_epochs, emit_loaded_sum_epochs,
)
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
from . import isa
from .column_batches import _admission


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def selected(attributes):
    value = attributes.get('elementwise_sum_epochs', knob('ELEMENTWISE_SUM_EPOCHS',
        default='off', values=('off', 'on'), root=Path(__file__).resolve().parents[1]))
    if value not in ('off', 'on'):
        raise ValueError('ELEMENTWISE_SUM_EPOCHS must be off or on')
    return value == 'on'


class _Indices(FnBuilderIndexAdapter):
    def op(self, opcode, lhs, rhs):
        if opcode in ('udiv', 'urem'):
            from xdsl.dialects import llvm
            cls = llvm.UDivOp if opcode == 'udiv' else llvm.URemOp
            return self.builder.add(cls(self.value(lhs), self.value(rhs))).results[0]
        return super().op(opcode, lhs, rhs)


def emit_sum(ctx, lhs, rhs, dst, rows, columns, *, dtype, scalar_proof,
             commands, broadcast_rhs=False):
    """Check full logical cover, then emit the checked bounded epoch artifact.

    No target configuration changes inside the plan. Initial and final drains
    delimit exclusive ACC ownership; every phase boundary is a real fence.
    """
    # One tile has no independent work to group; retain the incumbent encoder.
    if ((rows + 15) // 16) * ((columns + 15) // 16) <= 1:
        return None
    if any(ctx.buffer(a) is ctx.buffer(b) for a, b in ((lhs, rhs), (lhs, dst), (rhs, dst))):
        raise ValueError('loaded-sum epochs require disjoint external buffers')
    admission = _admission()  # Pins selected public dispatch/capacity/finite-queue facts.
    if isa.DIM != 16 or isa.ACC_ROWS < isa.DIM:
        raise ValueError('loaded-sum epochs lack selected accumulator tile geometry')
    proof = dict(scalar_proof, source_view=dict(rows=rows, columns=columns,
        lhs='row*columns+column', rhs='column' if broadcast_rhs else 'row*columns+column',
        rhs_broadcast_rows=broadcast_rhs), exact_configuration_words=commands,
        input_dtype=dtype, output_dtype=dtype)
    identity = _digest(proof)
    parameters = tuple(tuple((f'word{i}', int(word)) for i, word in enumerate(command))
                       for command in commands)
    hardware = 'selected-gemmini:' + admission['base_facts_sha256']
    contract = LoadedSumContract(identity, parameters, hardware, True, True)
    layout = LoadedSumLayout(rows, columns, 16, 16, 0, 64, isa.ACC_ROWS * 64,
                             input_dtype=dtype, output_dtype=dtype)
    mt, nt = (rows + 15) // 16, (columns + 15) // 16
    args = tuple(TensorArg(name, (rows, columns), dtype, access)
                 for name, access in (('lhs', 'read'), ('rhs', 'read'), ('output', 'write')))
    kinds = ('initialize', 'accumulate', 'store')
    row, col = Var('m') * 16, Var('n') * 16
    height = select('m', mt - 1, (rows - 1) % 16 + 1, 16)
    width = select('n', nt - 1, (columns - 1) % 16 + 1, 16)
    calls = tuple(Call(kind, (('row', row), ('column', col), ('rows', height),
        ('columns', width), ('local_byte', Const(0)),
        *tuple((key, Const(value)) for key, value in parameters[i]))) for i, kind in enumerate(kinds))
    source = Kernel('loaded_sum', args, (Loop('m', mt, (Loop('n', nt, calls),)),))

    def project(call):
        index = kinds.index(call.instr)
        # Decode the current actual instruction's coordinate and parameter operands.
        return LoadedSumEffect(call.instr, ('lhs', 'rhs', 'output')[index],
            call.arg('row'), call.arg('column'), call.arg('rows'), call.arg('columns'),
            call.arg('local_byte'), 64, identity,
            tuple((key, call.arg(key).value) for key, _ in parameters[index]), 'synchronous')

    # Stable configuration is completed before any DMA, so capture across a
    # changing CONFIG word is not assumed. Pending same-phase DMA rectangles
    # are disjoint; selected load release-at-issue and finite queues admit them.
    law = LoadedSumEpochLaw(hardware, identity,
        'uniform-config-full-drain:' + admission['ordering_sha256'],
        disjoint_loads_pending=True, disjoint_stores_pending=True,
        ordered_configuration_capture=True, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
    plan = plan_loaded_sum_epochs(source, layout, contract, project,
        scalar_proof=proof, target_law=law, max_live_tiles=isa.ACC_ROWS // 16,
        flatten_rows=True)
    plan.revalidate(source, layout, contract, project, scalar_proof=proof)
    fb, bridge = ctx.fb, _Indices(ctx.fb)
    ctx.fence()
    for command in commands:
        ctx.emit(*command)
    ctx.fence()  # Establish every uniform load/store configuration before data.

    def lower(kind, effect):
        if kind == 'complete_phase':
            ctx.fence()
            return
        index = kinds.index(kind)
        if effect['parameters'] != parameters[index]:
            raise ValueError('loaded-sum emitter changed original scalar parameters')
        offset = bridge.op('add', bridge.op('mul', effect['row'], columns), effect['column'])
        local = bridge.op('udiv', effect['local_byte'], 64)
        if kind == 'store':
            command = isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(local, full=dtype == 'i32', fb=fb),
                rows=effect['rows'], cols=effect['columns'], fb=fb)
        else:
            accumulate = kind == 'accumulate'
            operand = rhs if accumulate else lhs
            address = effect['column'] if accumulate and broadcast_rhs else offset
            command = isa.mvin(ctx.ptr(operand, address), isa.acc_addr(local, accumulate=accumulate, fb=fb),
                rows=effect['rows'], cols=effect['columns'], channel=int(accumulate), fb=fb)
        ctx.emit(*command)

    report = emit_loaded_sum_epochs(plan, bridge, lower)
    plan.revalidate(source, layout, contract, project, scalar_proof=proof)
    report.update(admission=admission, scalar_proof=proof,
        configuration='uniform words completed before data; no changing-state capture assumed',
        pass_order=['original scalar proof', 'compact_loaded_sum', 'loaded_sum_epochs', 'terminal ISA encoder'])
    ctx.metadata.setdefault('loaded_sum_epochs', []).append(report)
    return report
