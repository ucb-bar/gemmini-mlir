"""Optional counted array-weight reuse with a separate target admission.

The shared pass proves the exact fresh/remaining-M rewrite and preserves the
original source, storage and completion checks. It does not derive array
persistence from a weight-stationary label or from opcode names. Callers must
first supply the independently selected fresh/STAY/output-descriptor law.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.compiler.schedule.blocks import Compute, Preload
from merlin.compiler.schedule.dense_array_retention import (
    DenseArrayRetentionAdmission, propose_dense_array_retention,
    verify_dense_array_retention,
)
from merlin.sched.ir import Call, as_expr

from . import isa

PRELOAD_RETAINED = 'gemmini_preload_retained'
COMPUTE_RETAINED = 'gemmini_compute_accumulated'
MISSING_ARRAY_LAW = (
    'selected target lacks an independently qualified fresh-capture/STAY/'
    'output-descriptor array-persistence law; WS mode and FLIP/STAY opcode '
    'names alone do not establish it'
)


def retention_adapter(adapter):
    """Extend the actual operand codec, preserving every ordinary source call."""
    def encode(operation):
        if type(operation) is Preload and operation.weight_row is None:
            if operation.weight_block is not None or operation.weight_rows is None:
                raise ValueError('retained descriptor requires no RHS identity and an explicit depth')
            names = ('acc', 'accumulate', 'rows', 'cols', 'depth')
            values = (operation.accumulator_row, operation.accumulate, operation.rows,
                      operation.cols, operation.weight_rows)
            return Call(PRELOAD_RETAINED, tuple(zip(names, map(as_expr, values))))
        if type(operation) is Compute and operation.fresh_weights is False:
            if operation.input_gather is not None:
                raise ValueError('dense array retention does not qualify gathered inputs')
            names = ('lhs', 'block0', 'block1', 'rows', 'depth')
            values = (operation.input_row, *operation.input_block, operation.rows, operation.cols)
            return Call(COMPUTE_RETAINED, tuple(zip(names, map(as_expr, values))))
        return adapter.encode(operation)

    def project(call):
        values = dict(call.args)
        if call.instr == PRELOAD_RETAINED:
            if set(values) != {'acc', 'accumulate', 'rows', 'cols', 'depth'}:
                raise ValueError('retained descriptor has unprojected operands')
            return Preload(None, None, values['acc'], values['accumulate'],
                           values['rows'], values['cols'], values['depth'])
        if call.instr == COMPUTE_RETAINED:
            if set(values) != {'lhs', 'block0', 'block1', 'rows', 'depth'}:
                raise ValueError('retained compute has unprojected operands')
            return Compute(values['lhs'], (values['block0'], values['block1']),
                           values['rows'], values['depth'], False)
        return adapter.project(call)

    return replace(adapter, encode=encode, project=project)


@dataclass(frozen=True)
class DenseWeightRetentionResult:
    original: object
    proposal: object
    adapter: object
    realization: dict

    @property
    def kernel(self):
        return self.proposal.kernel

    @property
    def source_identity(self):
        return self.original.source_identity

    def revalidate(self):
        self.original.revalidate()
        if self.original.kernel != self.proposal.checking_kernel:
            raise ValueError('array retention original checked program changed')
        self.proposal.revalidate(self.adapter, self.proposal.admission)

    def materialize_scf(self):
        from merlin.compiler.schedule.scf_commands import materialize_scf_commands
        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        self.revalidate()
        return module

    def emit_llvm(self, builder, lower_call):
        from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        bridge = FnBuilderIndexAdapter(builder)
        result = emit_index_program(self.kernel, bridge, lambda instruction, arguments:
            lower_call(instruction, {key: bridge.value(value) for key, value in arguments.items()}))
        self.revalidate()
        return result


def select_weight_retention(schedule, *, problem, block, resources, storage,
                            adapter, array_admission=None, static_budget=128,
                            timeout_ms=5000):
    """Return a checked transformed program and receipt, or the exact incumbent.

    Invoke after original retained/segmented source verification and before
    transformations which obscure its pure K,N,M compute nests. The supplied
    problem/block/resources/storage are the original planning objects, not a
    second independently reconstructed schedule. No completion is removed.
    """
    schedule.revalidate()
    if type(array_admission) is not DenseArrayRetentionAdmission:
        return schedule, dict(applied=False, refusal=MISSING_ARRAY_LAW)
    selected = retention_adapter(adapter)
    try:
        proposal = propose_dense_array_retention(schedule.kernel, selected, array_admission)
        report = verify_dense_array_retention(
            proposal, problem, block, resources, storage, selected, schedule.admission,
            static_budget=static_budget, timeout_ms=timeout_ms,
            output_readout=schedule.output_readout, bias_source=schedule.bias_source)
    except (ValueError, AttributeError) as exc:
        return schedule, dict(applied=False, refusal=str(exc))
    receipt = dict(applied=True, **report)
    result = DenseWeightRetentionResult(schedule, proposal, selected,
        dict(schedule.realization, kernel_sha256=proposal.actual_sha256,
             array_weight_retention=receipt))
    result.revalidate()
    return result, receipt


def emit_retained_call(ctx, instruction, arguments, *, descriptor_columns):
    """Terminal callback extension; use only for a separately admitted result.

    ``descriptor_columns`` is the same one-element descriptor state used by
    ordinary dense preload/compute emission. Return False for an ordinary call
    so the existing terminal encoder owns all unchanged commands.
    """
    if instruction not in (PRELOAD_RETAINED, COMPUTE_RETAINED):
        return False
    fb = ctx.fb
    if instruction == PRELOAD_RETAINED:
        descriptor_columns[0] = arguments['cols']
        ctx.emit(*isa.preload(isa.GARBAGE,
            isa.acc_addr(arguments['acc'], accumulate=arguments['accumulate'], fb=fb),
            weight_rows=arguments['depth'], weight_cols=arguments['cols'],
            c_rows=arguments['rows'], c_cols=arguments['cols'], fb=fb))
    else:
        if descriptor_columns[0] is None:
            raise ValueError('retained compute requires its preceding output descriptor')
        ctx.emit(*isa.compute_accumulated(arguments['lhs'], isa.GARBAGE,
            a_rows=arguments['rows'], a_cols=arguments['depth'],
            b_rows=arguments['depth'], b_cols=descriptor_columns[0], fb=fb))
    return True
