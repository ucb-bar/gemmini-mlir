"""Bounded ordered command-loop specialization over checked schedule IR.

This changes host submission control only: the dynamic instruction sequence,
operands, configuration, staging and completion calls keep their source order.
The original schedule remains the numerical/resource authority. No payload or
workload-domain enumeration occurs during ordinary compilation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.sched.ir import Call, Loop, check_structure

STATIC_OPERATION_BUDGET = 512
MAX_TRANSFORMS = 16


def _size(body):
    return sum(1 + (_size(x.body) if type(x) is Loop else 0) for x in body)


def _has_work_loop(body):
    return any(type(x) is Loop and (x.extent > 1 or _has_work_loop(x.body)) for x in body)


def _check_effects(body):
    for node in body:
        if type(node) is Loop:
            if node.iter_args or node.yields or node.results:
                raise ValueError('completion-token loops retain their original control')
            _check_effects(node.body)
        elif type(node) is not Call or node.produces or node.awaits:
            raise ValueError('asynchronous token ownership retains its original control')


def _paths(body, factor, path=()):
    for node in body:
        if type(node) is Loop:
            here = (*path, node.var)
            if node.iter_args or node.yields or node.results:
                raise ValueError('completion-token loops retain their original control')
            if node.extent >= factor and not _has_work_loop(node.body):
                yield here, node.extent
            else:
                yield from _paths(node.body, factor, here)
        elif type(node) is not Call or node.produces or node.awaits:
            raise ValueError('asynchronous token ownership retains its original control')


def transform_kernel(kernel, factor, *, complete=False):
    """Use only the library's order-preserving unroll, with a counted tail.

    Independent deepest work loops are chosen from the actual current Kernel.
    At most16 transforms and512 static IR statements bound code growth without
    depending on a workload trip count. A refused proposal keeps that loop.
    """
    if factor not in (1, 2, 4):
        raise ValueError('COMMAND_UNROLL must be 1, 2 or 4')
    if check_structure(kernel):
        raise ValueError('command specialization needs a checked kernel')
    if factor == 1:
        return kernel, ()
    _check_effects(kernel.body)
    paths = tuple(_paths(kernel.body, 2 if complete else factor))
    if complete and (len(paths) > MAX_TRANSFORMS or any(extent > factor for _,extent in paths)):
        raise ValueError('small-complete work loops must fit the full-unroll bound')
    actual = kernel
    requests = []
    from merlin.sched.primitives import NotApplicable
    for path,extent in paths[:MAX_TRANSFORMS]:
        request = LoopTransform('unroll', path, factor=extent if complete else factor)
        try:
            candidate = propose_loop_transforms(actual, (request,), index_bits=64,
                max_static_operations=STATIC_OPERATION_BUDGET)
        except (NotApplicable, ValueError):
            if complete:
                raise ValueError('small-complete specialization exceeds the checked static budget')
            continue
        actual = candidate
        requests.append(request)
    return actual, tuple(requests)


def _physical_backoff(k, n, dim):
    """Retain incumbent control in two regions contradicted by matched L3.

    This is a conservative restriction of the original small-complete class,
    not a cycle predictor or a claim about unmeasured geometries. Keep the
    reduction-tail region separate: ragged K with single-tile N has positive
    measured support and does not inherit the multi-column regression.
    """
    if k > 2*dim and n > 2*dim:
        return 'physical backoff: both K and N exceed two primitive tiles'
    if k % dim and n > dim:
        return 'physical backoff: reduction tail with multiple output-column tiles'
    return None


def _small_complete_policy(schedule, m, k, n):
    """Execute the typed policy and return its actual transformed Kernel."""
    from . import isa
    from merlin.transforms.pipeline_policy import (PipelinePolicy, PipelineVariant,
        PolicyFacts, RegisteredPass, ResourcePredicate, ShapePredicate, execute_policy)
    from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
    source = schedule.kernel
    if (len(source.args) < 2 or source.args[0].shape != (m,k)
            or source.args[1].shape != (k,n)):
        raise ValueError('command policy dimensions must match the original typed dense operands')
    def verify(kernel):
        if check_structure(kernel):
            raise ValueError('command policy requires structurally valid schedule IR')
    from hashlib import sha256
    context = PipelineContext(resources=sha256(repr((isa.DIM,STATIC_OPERATION_BUDGET,MAX_TRANSFORMS,'physical-backoff-v1')).encode()).hexdigest(),
        execution='ordered-submission-control')
    session = PipelineSession(source, context=context, ir_level='schedule_command_control',
        fingerprint=lambda kernel: kernel.digest(), verify_ir=verify)
    try:
        candidate, requests = transform_kernel(source, 4, complete=True)
        refusal = None
    except ValueError as exc:
        candidate, requests, refusal = source, (), str(exc)
    backoff = _physical_backoff(k,n,isa.DIM)
    facts = PolicyFacts(session.stamp(), shapes=((m,k),(k,n)),
        resources=(('full_unroll_legal', int(bool(requests))),
                   ('physical_backoff_clear', int(backoff is None))))
    predicates = (ShapePredicate(0,0,1,isa.DIM),
        ShapePredicate(0,1,1,4*isa.DIM), ShapePredicate(1,1,1,4*isa.DIM),
        ResourcePredicate('full_unroll_legal',1), ResourcePredicate('physical_backoff_clear',1))
    policy = PipelinePolicy((PipelineVariant('small_complete',('command_full_unroll',),predicates),
        PipelineVariant('incumbent',('command_incumbent',))), 'incumbent')
    output = 'checked_command_control_kernel'
    def specialize(state):
        if state.module != source:
            raise ValueError('command policy cannot reuse a different source artifact')
        state.module = candidate
        return {output:(candidate,requests)}
    def incumbent(state):
        return {output:(state.module,())}
    registry = {
        'command_full_unroll': RegisteredPass(PassSpec('command_full_unroll',0,
            produces=(output,),ir_level='schedule_command_control'),specialize),
        'command_incumbent': RegisteredPass(PassSpec('command_incumbent',0,
            produces=(output,),mutates_ir=False,ir_level='schedule_command_control'),incumbent)}
    selected = execute_policy(session,policy,registry,facts,output=output)
    actual, requests = selected.artifact
    if actual != selected.module:
        raise ValueError('command policy artifact differs from its executed Kernel')
    receipt = {key:value for key,value in selected.receipt.items() if key != 'host_pass_profile'}
    return actual, requests, receipt, backoff or refusal


@dataclass(frozen=True)
class CommandControlResult:
    original: object
    kernel: object
    realization: dict
    transforms: tuple
    original_identity: str

    @property
    def command(self):
        # Dense store epochs still own their exact CONFIG_ST command.
        return self.original.command

    def revalidate(self):
        self.original.revalidate()
        if self.original.kernel.digest() != self.original_identity:
            raise ValueError('command-control source changed; replan before emission')
        expected = propose_loop_transforms(self.original.kernel, self.transforms,
            index_bits=64, max_static_operations=STATIC_OPERATION_BUDGET)
        if expected != self.kernel:
            raise ValueError('command-control ordered specialization changed')

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        bridge = FnBuilderIndexAdapter(builder)
        result = emit_index_program(self.kernel, bridge, lambda instruction, arguments:
            lower_call(instruction, {key: bridge.value(value) for key, value in arguments.items()}))
        self.revalidate()
        return result


def select_control(schedule, m, k, n, attributes=None):
    """Return the checked executable alternative, or the unchanged incumbent."""
    attrs = attributes or {}
    selection = str(attrs.get('command_unroll', knob('COMMAND_UNROLL', default='1',
        values=('1', '2', '4', 'small_complete'), root=Path(__file__).resolve().parents[1])))
    if selection not in ('1', '2', '4', 'small_complete'):
        raise ValueError('COMMAND_UNROLL must be 1, 2, 4 or small_complete')
    if selection == '1':
        return schedule, None
    complete = selection == 'small_complete'
    schedule.revalidate()
    policy = None
    try:
        if complete:
            kernel, requests, policy, refusal = _small_complete_policy(schedule,m,k,n)
        else:
            kernel, requests = transform_kernel(schedule.kernel, int(selection))
            refusal = None
    except ValueError as exc:
        return schedule, dict(selection=selection, applied=False, refusal=str(exc))
    if not requests:
        return schedule, dict(selection=selection, applied=False, policy=policy,
            refusal=refusal or 'no bounded work loop admits the selected shape/control class')
    report = dict(selection=selection, applied=True, source_shape=[m,k,n], policy=policy,
        source_kernel_sha256=schedule.kernel.digest(), kernel_sha256=kernel.digest(),
        static_operations_before=_size(schedule.kernel.body), static_operations_after=_size(kernel.body),
        transforms=[dict(action=t.action,path=list(t.path),factor=t.factor) for t in requests],
        ordering=('library ordered complete unroll; every original call/effect retained; no work-loop remainder' if complete
            else 'library ordered partial unroll; every original call/effect retained; counted remainder'),
        numerical_authority='unchanged original checked schedule; no source arithmetic changes',
        emission='actual transformed Kernel through original terminal instruction callback')
    result = CommandControlResult(schedule,kernel,
        dict(schedule.realization,command_control=report),requests,schedule.kernel.digest())
    result.revalidate()
    return result, report
