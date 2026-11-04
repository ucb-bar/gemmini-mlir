"""Remove empty completion boundaries in the actual checked dense Kernel.

This adapter follows the store-epoch transform. BEGIN already performs the
full producer readiness drain before its original uniform CONFIG_ST; END is
its closing full store drain. An original panel drain immediately next to
these completions orders no additional device operation. No RAW/WAR or posted
configuration premise replaces a required wait, and no data call is renamed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.sched.ir import Call, Loop, check_structure

DRAIN = 'merlin_dense_panel_drain'
BEGIN = 'gemmini_dense_store_epoch_begin'
END = 'gemmini_dense_store_epoch_end'
_COMPLETES = frozenset((DRAIN, END))


def _first(body):
    for statement in body:
        if type(statement) is Loop:
            event = _first(statement.body)
            if event is not None:
                return event
        else:
            return statement.instr
    return None


def _last(body):
    for statement in reversed(body):
        if type(statement) is Loop:
            event = _last(statement.body)
            if event is not None:
                return event
        else:
            return statement.instr
    return None


def _count(body, name=DRAIN):
    return sum(statement.extent*_count(statement.body,name) if type(statement) is Loop
               else int(statement.instr == name) for statement in body)


def _projection(body):
    return tuple(replace(s,body=_projection(s.body)) if type(s) is Loop else s
                 for s in body if not (type(s) is Call and s.instr == DRAIN))


def _transform(kernel):
    events = []
    def audit(body):
        for statement in body:
            if type(statement) is Loop:
                audit(statement.body)
            elif statement.instr in (DRAIN,BEGIN,END) and statement != Call(statement.instr,()):
                raise ValueError('completion calls must have no additional token or scheduling effects')
    audit(kernel.body)
    def backward(body, following, multiplicity):
        result = []
        next_event = following
        for statement in reversed(body):
            if type(statement) is Loop:
                if statement.iter_args or statement.yields or statement.results:
                    raise ValueError('completion adapter cannot rewrite token-carrying loops')
                # An external successor follows only the LAST iteration. Never
                # apply that successor to every iteration of a repeated loop.
                child_next = next_event if statement.extent == 1 else None
                child = backward(statement.body,child_next,multiplicity*statement.extent)
                statement = replace(statement,body=child)
            elif statement.instr == DRAIN and next_event in (DRAIN,BEGIN,END):
                events.append(dict(reason='next original full completion before any device operation',
                                   successor=next_event,dynamic_drains=multiplicity))
                continue
            result.append(statement)
            first = _first((statement,))
            if first is not None:
                next_event = first
        return tuple(reversed(result))

    def forward(body, complete, multiplicity):
        result = []
        for statement in body:
            if type(statement) is Loop:
                # The first iteration and every backedge must both arrive
                # complete before removing the body's leading completion.
                incoming = complete and (statement.extent == 1 or _last(statement.body) in _COMPLETES)
                child, outgoing = forward(statement.body,incoming,multiplicity*statement.extent)
                statement = replace(statement,body=child)
                complete = outgoing
            elif statement.instr == DRAIN and complete:
                events.append(dict(reason='previous original full completion; no intervening device operation',
                                   dynamic_drains=multiplicity))
                continue
            else:
                # BEGIN contains CONFIG_ST after its readiness completion.
                # Configuration is a real queued device operation here.
                complete = statement.instr in _COMPLETES
            result.append(statement)
        return tuple(result),complete

    body = backward(kernel.body,None,1)
    body,_ = forward(body,False,1)
    actual = replace(kernel,body=body)
    if check_structure(actual) or _projection(actual.body) != _projection(kernel.body):
        raise ValueError('completion adapter changed data/configuration operations or scopes')
    before,after = _count(kernel.body),_count(actual.body)
    if sum(row['dynamic_drains'] for row in events) != before-after:
        raise ValueError('completion adapter drain accounting changed')
    return actual,dict(input_dynamic_panel_drains=before,dynamic_panel_drains=after,
        removed_dynamic_drains=before-after,proofs=events)


@dataclass(frozen=True)
class DenseCompletionEpochResult:
    original: object
    kernel: object
    realization: dict
    source_identity: str

    def __getattr__(self,key):
        # Preserve original store-epoch CONFIG_ST and construction admission.
        return getattr(self.original,key)

    def revalidate(self):
        self.original.revalidate()
        actual,report = _transform(self.original.kernel)
        if (self.original.kernel.digest()!=self.source_identity or actual!=self.kernel
                or report!=self.realization['completion_epochs']['completion_proof']):
            raise ValueError('completion adapter original owner or exact boundary proof changed')

    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate()
        bridge=FnBuilderIndexAdapter(builder)
        result=emit_index_program(self.kernel,bridge,lambda instruction,arguments:
            lower_call(instruction,{key:bridge.value(value) for key,value in arguments.items()}))
        self.revalidate()
        return result


def select_completion_epochs(schedule,attributes=None):
    """Consume after select_store_epochs, before terminal emission/load cleanup.

    The original source/storage admission remains authoritative. This complete
    transform does not change its issued commands or pending queue budgets.
    """
    attributes=attributes or {}
    selection=attributes.get('dense_completion_epochs',knob('DENSE_COMPLETION_EPOCHS',
        default='off',values=('off','empty_boundaries'),root=Path(__file__).resolve().parents[1]))
    if selection not in ('off','empty_boundaries'):
        raise ValueError('DENSE_COMPLETION_EPOCHS must be off or empty_boundaries')
    if selection=='off':
        return schedule,None
    schedule.revalidate()
    if _count(schedule.kernel.body,BEGIN) or _count(schedule.kernel.body,END):
        from .dense_store_epochs import DenseStoreEpochResult
        owner=schedule
        for _ in range(8):
            if type(owner) is DenseStoreEpochResult:
                break
            owner=getattr(owner,'original',None)
            if owner is None:
                break
        if type(owner) is not DenseStoreEpochResult:
            return schedule,dict(selection=selection,applied=False,
                refusal='store completion semantics require the actual checked store-epoch owner')
    try:
        actual,proof=_transform(schedule.kernel)
    except ValueError as exc:
        return schedule,dict(selection=selection,applied=False,refusal=str(exc))
    if not proof['removed_dynamic_drains']:
        return schedule,dict(selection=selection,applied=False,refusal='no empty full-completion boundaries')
    report=dict(schema='backend.dense_completion_epochs.v1',selection=selection,applied=True,
        source_kernel_sha256=schedule.kernel.digest(),kernel_sha256=actual.digest(),
        completion_proof=proof,data_projection='all original non-panel-drain calls and lexical scopes identical',
        completion_contract={'panel_drain':'real full ctx.fence()',
            'store_epoch_begin':'real full ctx.fence(), then original uniform CONFIG_ST',
            'store_epoch_end':'real full ctx.fence()'},
        preserved='operand/ACC ownership drains with intervening work; all store-region begin/end; final whole-call completion',
        ordering='idempotence of consecutive complete device drains; no cross-queue ordering assumed')
    result=DenseCompletionEpochResult(schedule,actual,
        dict(schedule.realization,kernel_sha256=actual.digest(),completion_epochs=report),schedule.kernel.digest())
    result.revalidate()
    return result,report
