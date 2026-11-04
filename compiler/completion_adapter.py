"""Compose completion placement with Fresh6's actual dense result routes.

The small public entry is also the production entry.  Its context contains the
original problem/block/resources/storage/target adapter, an explicit issued
budget and a plain premise dictionary.  Missing laws never get defaulted true.
Data construction stays with the inherited scheduling dialect services.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from collections.abc import Mapping

# The public safety entry is importable independently of compiler.py.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'compiler_library'))

from merlin.sched.ir import Kernel, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, _digest
from merlin.compile.scheduling.dense_completion import (
    CompletionIssueBudget, CompletionRefusal, MODES, PASS_SPECS, REQUIRED_PREMISES,
    check_completion_proposal, make_completion_issue_budget, module_identity,
    propose_dense_completion,
)
from merlin.compile.scheduling.dense_completion_plan import make_completion_plan, read_completion_plan
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN, DensePanelOverlapAdmission, DenseRetainedReductionAdmission,
    verify_dense_panel_pipeline, verify_dense_retained_reduction,
)
from merlin.compile.scheduling.dense_segmented_pipeline import (
    DenseSegmentedAdmission, verify_dense_segmented_pipeline,
)


def _get(context, key, default=None):
    return context.get(key, default) if isinstance(context, Mapping) else getattr(context, key, default)


def _owners(original):
    # RetainedLhsResult carries the checked transformed stream in .checked;
    # resident results carry their actual kernel directly and keep boundaries
    # on their original. No result is relabelled as another scheduling route.
    result, seen = [], set()
    pending = [original]
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current)); result.append(current)
        pending.extend(o for name in ('checked', 'original')
                       if (o := getattr(current, name, None)) is not None)
    return result


def _component(original, field, default=None):
    return next((getattr(owner, field) for owner in _owners(original) if hasattr(owner, field)), default)


def _route(original):
    report = original.realization
    if 'rhs_residency' in report:
        return 'resident_rhs'
    plan = report.get('schedule_ir')
    return (plan.get('pipeline_kind', type(original).__name__)
            if isinstance(plan, dict) else type(original).__name__)


def _validate_source(original, context):
    original.revalidate()
    kernel = original.kernel
    if type(kernel) is not Kernel or check_structure(kernel):
        raise CompletionRefusal('completion: malformed original schedule')
    p, block, resources, storage, adapter = (_get(context, field) for field in
        ('problem', 'block', 'resources', 'storage', 'adapter'))
    if any(value is None for value in (p, block, resources, storage, adapter)):
        raise CompletionRefusal('completion: complete original scheduling context required')
    if type(storage) is not DenseLoopStorage or storage.accumulator is None:
        raise CompletionRefusal('completion: original private storage assignment is unknown')
    # Use canonical backing BYTE coordinates, never view names as nonaliasing.
    views = (storage.lhs, storage.rhs, storage.accumulator)
    for i, left in enumerate(views):
        for right in views[i+1:]:
            if (left.backing == right.backing
                    and left.base*left.unit_bytes < (right.base+right.capacity)*right.unit_bytes
                    and right.base*right.unit_bytes < (left.base+left.capacity)*left.unit_bytes):
                raise CompletionRefusal('completion: overlapping live operand/accumulator ownership windows')
    if not storage.admits(p, block, resources):
        raise CompletionRefusal('completion: original block exceeds assigned storage/capacity')
    if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation:
        raise CompletionRefusal('completion: original full-drain or exact integer continuation is unknown')
    admission = _component(original, 'admission')
    boundary = dict(output_readout=_component(original, 'output_readout'),
                    bias_source=_component(original, 'bias_source'))
    kwargs = dict(static_budget=_get(context, 'static_budget', 128),
                  timeout_ms=_get(context, 'timeout_ms', 30000), **boundary)
    if type(admission) is DensePanelOverlapAdmission:
        checker = verify_dense_panel_pipeline
    elif type(admission) is DenseRetainedReductionAdmission:
        checker = verify_dense_retained_reduction
    elif type(admission) is DenseSegmentedAdmission:
        checker = verify_dense_segmented_pipeline
    else:
        raise CompletionRefusal('completion: unsupported original execution admission/result composition')
    # Recheck actual original addresses, tails, source coverage, paired
    # descriptors and every old epoch, not just a stale report or result label.
    source = checker(kernel, p, block, resources, storage, adapter, admission, **kwargs)
    original.revalidate()
    return source, admission


def _adapter_source_identity():
    # Bind the production ownership/full-drain mapping and selected law source,
    # rather than treating a retained Boolean dictionary as fresh target facts.
    root = Path(__file__).resolve().parent
    names = ('completion_adapter.py', 'dense.py', 'target.py',
             'COMPLETION_PREMISES.json', 'COMPLETION_ORDERING.json', 'rtl_facts/facts.json')
    return _digest({name:sha256((root/name).read_bytes()).hexdigest() for name in names})


@dataclass(frozen=True)
class CompletionResult:
    original: object
    kernel: object
    realization: dict
    admission: object
    context: dict
    completion_plan: object
    checked_plan_sha256: str
    checked_kernel_sha256: str
    checked_context_sha256: str
    checked_core_sha256: str
    checked_adapter_sha256: str

    def revalidate(self):
        self.original.revalidate()
        adapter = self.context['adapter']
        context_pin = _context_identity(self.context)
        mode, decision = read_completion_plan(self.completion_plan, self.original.kernel, context_pin)
        if (mode != self.context['mode'] or decision['sha256'] != self.checked_plan_sha256
                or decision != self.realization['completion_schedule_ir']):
            raise ValueError('completion: checked executable scheduling policy changed')
        if (self.kernel.digest() != self.checked_kernel_sha256
                or context_pin != self.checked_context_sha256
                or module_identity() != self.checked_core_sha256
                or _adapter_source_identity() != self.checked_adapter_sha256):
            raise ValueError('completion: checked schedule, original context or implementation changed')
        checked = check_completion_proposal(self.original.kernel, self.kernel, adapter, self.admission,
            self.context['premises'], mode=self.context['mode'],
            original_admission=_component(self.original, 'admission'), storage=self.context['storage'], block=self.context['block'])
        if checked != self.realization['completion']:
            raise ValueError('completion: actual proposal or conditional completion proof changed')

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        self.revalidate()
        receipt = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt

    def emit_resident_index_program(self, control, lower_effect, *, complete_epoch, resident_live=False):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        self.revalidate()
        if not hasattr(self.original, 'emit_resident_index_program'):
            raise ValueError('completion: no original resident emission contract')
        if self.kernel.body[0] != self.original.kernel.body[0]:
            raise ValueError('completion: resident preparation boundary changed')
        kernel = replace(self.kernel, body=self.kernel.body[1:]) if resident_live else self.kernel
        receipt = emit_index_effect_program(kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        module.verify()
        self.revalidate()
        return module

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import emit_index_program, FnBuilderIndexAdapter
        self.revalidate()
        result = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return result


def _callable_identity(callback):
    import inspect
    import marshal
    target = getattr(callback, '__func__', callback)
    if not inspect.isfunction(target) and not inspect.isbuiltin(target):
        target = getattr(target, '__call__', target)
    try:
        source = inspect.getsourcefile(target)
    except TypeError:
        source = None
    path = Path(source) if source else None
    code = getattr(target, '__code__', None)
    return dict(module=getattr(target, '__module__', None),
        name=getattr(target, '__qualname__', type(target).__qualname__),
        source_sha256=sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
        code_sha256=sha256(marshal.dumps(code)).hexdigest() if code is not None else None)


def _context_identity(context):
    adapter = context['adapter']
    return _digest(dict(problem=asdict(context['problem']), block=asdict(context['block']),
        resources=asdict(context['resources']), storage=asdict(context['storage']),
        admission=asdict(context['admission']), premises=context['premises'], mode=context['mode'],
        adapter=(adapter.configuration_identity, adapter.numerical_policy_identity,
                 adapter.synchronous_completion, adapter.exact_accumulator_continuation, adapter.index_bits),
        actual_encoder=_callable_identity(adapter.encode), actual_projector=_callable_identity(adapter.project)))


def apply_completion(original, context):
    """Return the actual checked result used by production emission or refuse.

    No implicit law/budget extension is performed at this entry. A caller can
    supply the ORIGINAL admission, or a CompletionIssueBudget bound to exact
    counted scopes by make_completion_issue_budget. Insufficient bounds refuse.
    """
    if (not callable(getattr(original, 'revalidate', None))
            or type(getattr(original, 'kernel', None)) is not Kernel):
        raise CompletionRefusal('completion: an original checked result is required')
    mode = _get(context, 'mode', 'output_block')
    if mode not in MODES:
        raise CompletionRefusal('completion: unknown requested mode')
    if mode == 'original':
        original.revalidate()
        return original
    premises = _get(context, 'premises')
    if type(premises) is not dict or any(premises.get(key) is not True for key in REQUIRED_PREMISES):
        missing = [key for key in REQUIRED_PREMISES if type(premises) is not dict or premises.get(key) is not True]
        raise CompletionRefusal('completion: missing or unknown ordering/ownership premise: ' + ', '.join(missing))
    source, original_admission = _validate_source(original, context)
    if type(_get(context, 'admission')) not in (CompletionIssueBudget, DensePanelOverlapAdmission,
            DenseRetainedReductionAdmission, DenseSegmentedAdmission):
        raise CompletionRefusal('completion: explicit original or program-scoped issued admission required')
    frozen_context = {field:_get(context, field) for field in
        ('problem','block','resources','storage','adapter','admission')}
    frozen_context.update(premises=dict(premises), mode=mode)
    context_pin = _context_identity(frozen_context)
    plan = _get(context, 'completion_plan')
    if plan is None:
        plan = make_completion_plan(original.kernel, context_pin, mode=mode)
    consumed_mode, decision = read_completion_plan(plan, original.kernel, context_pin)
    if consumed_mode != mode:
        raise CompletionRefusal('completion: schedule policy differs from scoped mode/budget')
    # The consumed schedule.apply_policy selects the actual transformation.
    mode = consumed_mode
    adapter = _get(context, 'adapter')
    actual, _ = propose_dense_completion(original.kernel, adapter, mode=mode)
    proof = check_completion_proposal(original.kernel, actual, adapter, _get(context, 'admission'),
        premises, mode=mode, original_admission=original_admission, storage=_get(context,'storage'), block=_get(context,'block'),
        timeout_ms=_get(context,'timeout_ms',30000))
    if proof['removed_dynamic_drains'] == 0:
        return original
    # Run the declared phase order on the actual scheduling dialect Kernel.
    session = PipelineSession(original.kernel, context=PipelineContext(
        hardware=adapter.configuration_identity, numerical_policy=adapter.numerical_policy_identity),
        ir_level='schedule', fingerprint=lambda k:k.digest(),
        verify_ir=lambda k: _verify_structure(k, adapter.index_bits))
    session.run(PASS_SPECS[0], lambda: {'source_schedule':source})
    def transform():
        session.analysis('source_schedule')
        session.module = actual
        return {'completion_proof':proof}
    session.run(PASS_SPECS[1], transform)
    session.run(PASS_SPECS[2], lambda: {'emission_input':session.analysis('completion_proof')})
    # A composed asynchronous program is not a newly certified original
    # per-K/pure-reduction result. Keep the old epoch/source evidence under its
    # original input owner, and expose only the actual new completion proof.
    report = dict(schema='merlin.dense_completion_composition.v1',
        kernel_sha256=actual.digest(), dynamic_epoch_drains=proof['dynamic_drains_after'],
        source=original.realization['source'],
        schedule_ir=original.realization.get('schedule_ir'),
        admission=asdict(frozen_context['admission']), completion=proof, completion_schedule_ir=decision,
        completion_composition=dict(original_route=_route(original),
            original_result_type=type(original).__name__, source_kernel_sha256=original.kernel.digest(),
            original_source_identity=_component(original, 'source_identity'),
            source_and_original_epochs=source, pass_order=[r.name for r in session.records],
            data_construction='unchanged original schedule'),
        input_schedule=original.realization, target_execution='not_established')
    result = CompletionResult(original, actual, report, frozen_context['admission'], frozen_context,
        plan, decision['sha256'], actual.digest(), context_pin, module_identity(),
        _adapter_source_identity())
    result.revalidate()
    return result


def _verify_structure(kernel, bits):
    if check_structure(kernel):
        raise ValueError('completion: malformed transformed scheduling dialect')
    check_integer_domain(kernel, bits)


def selected_premises(adapter):
    """Selected public target facts plus this emitter's explicit program proofs."""
    root = Path(__file__).resolve().parent
    card = json.loads((root/'COMPLETION_PREMISES.json').read_text())
    if (sha256((root/'rtl_facts/facts.json').read_bytes()).hexdigest() != card['inputs']['facts']['sha256']
            or sha256((root/'COMPLETION_ORDERING.json').read_bytes()).hexdigest() != card['inputs']['ordering']['sha256']):
        raise CompletionRefusal('completion: selected ordering/facts do not match public premise bindings')
    premises = {key:value.get('value') if value.get('status') == 'fact' else None
                for key,value in card['premises'].items()}
    # Program proof: backend owns private local storage during this invocation,
    # fenced on entry/exit; all data fields/configuration are fixed; storage
    # nonaliasing and source lifetimes are independently rechecked at the entry.
    # Callback is Builder.fence, the same full drain the original result uses.
    premises['exclusive_private_state'] = True
    premises['complete_device_drain'] = adapter.synchronous_completion is True
    return premises


def complete_production_result(original, problem, block, resources, storage, adapter):
    from merlin.compile.tuning_knobs import knob
    mode = knob('dense_completion', default='original', values=MODES, root=Path(__file__).parent)
    if mode == 'original':
        return apply_completion(original, dict(mode=mode))
    admission = _component(original, 'admission')
    # This count proof covers exactly one output K-stream (or a two-epoch
    # period), not an arbitrary machine queue extension. Pending capacities and
    # all original epoch admissions remain byte-for-byte the original values.
    try:
        budget = make_completion_issue_budget(original.kernel, adapter, admission, mode=mode)
        context = dict(problem=problem, block=block, resources=resources, storage=storage,
            adapter=adapter, admission=budget, premises=selected_premises(adapter), mode=mode)
        return apply_completion(original, context)
    except CompletionRefusal as error:
        # Keep the checked original program; record a refusal without a claim
        # of optimized safety or a fabricated performance result.
        report = dict(original.realization)
        report['completion_refusal'] = str(error)
        return replace(original, realization=report)
