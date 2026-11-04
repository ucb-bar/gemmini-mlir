"""Compose checked completion placement on unchanged dense route results.

The route owns source coverage, arithmetic and storage. This consumer owns only
completion deletion, using the original panel epoch core's structural proposal
and symbolic load-reader proof. An optional, separately checked issued scope
counts the actual K-stream commands; it never enlarges physical pending queues.
No readout, output-block or final completion is removed.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import time

from merlin.sched.ir import Call, Const, Loop, Scaled, Select, Sum, Var, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from .block_schedule import Compute, Load, Preload
from .compact_dense import _digest
from .dense_panel_epochs import (
    COMPLETION, STREAM_FLAGS, DensePanelEpochLaw, _plan, _prove_stream,
    _record, _drains, _count, _sites,
)
from .dense_panel_pipeline import (
    DRAIN, DensePanelPipelineResult, DensePanelOverlapAdmission,
    DenseRetainedReductionAdmission, _calls, _serial_projection, _windows,
    verify_dense_panel_pipeline, verify_dense_retained_reduction,
)
from .dense_segmented_pipeline import (
    DenseSegmentedPipelineResult, DenseSegmentedAdmission,
    verify_dense_segmented_pipeline, _assignment,
)


def _owner():
    return (sha256(Path(__file__).read_bytes()).hexdigest(),
            sha256(Path(__file__).with_name('dense_panel_epochs.py').read_bytes()).hexdigest())


def checked_source(original):
    """A composition hook exposes the existing checked owner, without relabeling it."""
    original.revalidate()
    if type(original) is DenseCompletionInput:
        return original.source
    if type(original) in (DensePanelPipelineResult, DenseSegmentedPipelineResult):
        return original
    hook = getattr(original, 'completion_source', None)
    if hook is None:
        raise ValueError('completion composition: no checked source ownership interface')
    source = hook()
    if type(source) not in (DensePanelPipelineResult, DenseSegmentedPipelineResult):
        raise ValueError('completion composition: unsupported checked source owner')
    source.revalidate()
    if source.kernel != original.kernel:
        raise ValueError('completion composition: owner and actual emitted kernel differ')
    return source


def effective_admission(admission):
    """Keep a segmented law's two independent issued limits, using their minimum."""
    admission.__post_init__()
    if type(admission) is DenseSegmentedAdmission:
        return replace(admission.overlap,
            max_issued_compute_pairs=min(admission.overlap.max_issued_compute_pairs,
                                         admission.reduction.max_issued_compute_pairs),
            max_issued_loads=min(admission.overlap.max_issued_loads, admission.reduction.max_issued_loads),
            max_issued_readouts=min(admission.overlap.max_issued_readouts, admission.reduction.max_issued_readouts))
    if type(admission) not in (DensePanelOverlapAdmission, DenseRetainedReductionAdmission):
        raise ValueError('completion composition: unknown execution admission')
    return admission


def normalize_completion_context(original,context):
    """Accept the real route admission as well as its conservative effective view.

    A combined segmented admission is never replaced by one component alone:
    its original conjunction stays with the checked source and both issued
    limits are intersected for completion placement.
    """
    from collections.abc import Mapping
    if isinstance(context,Mapping):
        context=dict(context)
    else:
        original_context=context
        try:
            context={name:getattr(original_context,name) for name in
                ('problem','block','resources','storage','adapter','admission')}
        except AttributeError as error:
            raise ValueError('completion context lacks a required problem/storage/admission field') from error
        for name in ('premises','ordering_identity','completion_plan'):
            if hasattr(original_context,name): context[name]=getattr(original_context,name)
    if type(context.get('premises',{})) is not dict:
        raise ValueError('completion premises must be a plain dictionary')
    given = context['admission']
    if type(given) is DenseSegmentedAdmission:
        source = checked_source(original)
        if given != source.admission:
            raise ValueError('completion context changes the original combined segmented law')
        return dict(context,admission=effective_admission(given))
    return dict(context)


def _pin(context):
    return _digest((*(asdict(context[k]) for k in ('problem','block','resources','storage','admission')),
        repr(context['adapter']),
        {flag:context.get('premises',{}).get(flag) is True for flag in STREAM_FLAGS},
        context.get('ordering_identity')))


def _source_pin(source):
    return _digest((source.source_identity,source.realization.get('schedule_ir'),asdict(source.admission)))


def _validate_original(original, context, *, timeout_ms, static_budget):
    source = checked_source(original)
    p,b,r,s,a = (context[k] for k in ('problem','block','resources','storage','adapter'))
    expected = original.admission if type(original) is DenseCompletionInput else effective_admission(source.admission)
    if context['admission'] != expected:
        raise ValueError('completion composition: context changes the selected issued or pending budgets')
    kind = source.realization.get('schedule_ir',{}).get('pipeline_kind')
    if kind not in ('two_window','retained_full_k','retained_lhs_segmented_k'):
        raise ValueError('completion composition: unknown original pipeline kind')
    from .schedule_ir import parse_dense_schedule_plan,read_dense_schedule_plan
    text = source.realization.get('schedule_ir',{}).get('text')
    if type(text) is not str:
        raise ValueError('completion composition: original executable schedule plan is unavailable')
    decision = read_dense_schedule_plan(parse_dense_schedule_plan(text),p,r,s,a,
        source_identity=source.source_identity,
        dtypes=(source.kernel.args[0].dtype,source.kernel.args[1].dtype,'i'+str(p.accumulator_bytes*8)),
        static_budget=static_budget,output_readout=source.output_readout,bias_source=source.bias_source)
    if decision.block != b or decision.pipeline_kind != kind:
        raise ValueError('completion context differs from original actual block or route')
    # Replay the original row-base decision inside the caller's assigned view.
    # This selects the already-emitted storage, without moving an allocation.
    s = decision.storage
    if kind == 'two_window':
        _windows(p,b,s,r,decision.pipeline_window_offsets)
    elif kind == 'retained_lhs_segmented_k':
        _assignment(p,b,r,s,decision.pipeline_window_offsets)
    # Run the original route's full verifier on the original program and current
    # context. This checks storage ownership, exact source/tails and every
    # original closing completion, before any boundary may disappear.
    if type(source.admission) is DenseSegmentedAdmission:
        verifier = verify_dense_segmented_pipeline
    elif type(source.admission) is DenseRetainedReductionAdmission:
        verifier = verify_dense_retained_reduction
    else:
        verifier = verify_dense_panel_pipeline
        if kind == 'retained_lhs_segmented_k':
            # The inherited K=1 segmented builder has a disjoint-output panel
            # proof. Also check its true retained-LHS/two-RHS allocation.
            _assignment(p,b,r,s)
    verifier(source.kernel,p,b,r,s,a,source.admission,static_budget=static_budget,
        timeout_ms=timeout_ms,output_readout=source.output_readout,bias_source=source.bias_source)
    return source,s


def _streams(kernel, adapter, law, admission):
    # A retained full-LHS initializer can be adjacent to the first RHS stage
    # when an output dimension has only one group (no enclosing counted N
    # loop). Keep that initializer's completion: its immediate successor need
    # not read LHS yet. This classification uses actual effects, not names or
    # dimensions, and changes no data construction or operand allocation.
    protected = []
    def visit(body):
        for statement in body:
            if type(statement) is not Loop:
                continue
            if statement.body and statement.body[-1] == COMPLETION:
                operations = [adapter.project(call) for call in _calls(statement.body[:-1])]
                if operations and all(type(op) is Load and op.role == 'lhs' for op in operations):
                    protected.append(statement.var)
            else:
                visit(statement.body)
    visit(kernel.body)
    return _plan(kernel,adapter,law,admission,merge_readout=False,protected_epochs=tuple(protected))


def _prove_output_block(stream,adapter,block,deadline):
    """All actual compute instances of a merged stream belong to one output block.

    Outer coordinates remain symbolic; local K/M/N and epoch loops range over
    their original exact extents. This proves scope instead of inferring it
    from a loop name or merely from the total number of issued commands.
    """
    import z3
    def expression(value,env):
        if type(value) is int: return z3.IntVal(value)
        if type(value) is Const: return z3.IntVal(value.value)
        if type(value) is Var: return env[value.name]
        if type(value) is Sum: return z3.Sum(*(expression(v,env) for v in value.terms))
        if type(value) is Scaled: return expression(value.expr,env)*value.factor
        if type(value) is Select:
            return z3.If(env[value.var]==value.equals,expression(value.then,env),expression(value.other,env))
        raise ValueError('completion scope has an unknown source coordinate expression')
    pairs=[]
    pending=None
    for call,local in _sites(stream['elements']):
        op=adapter.project(call)
        if type(op) is Preload:
            if pending is not None: raise ValueError('completion scope overwrites a live descriptor')
            pending=(op,local)
        elif type(op) is Compute:
            if pending is None or pending[1]!=local:
                raise ValueError('completion scope splits a source compute pair')
            pairs.append((op.input_block[0],pending[0].weight_block[1],local))
            pending=None
        elif pending is not None:
            raise ValueError('completion scope interposes an effect inside a compute pair')
    if pending is not None or not pairs:
        raise ValueError('completion scope needs complete original compute pairs')
    outer={name:z3.Int('output-block:'+name) for name,_ in stream['ancestry']}
    constraints=[z3.And(outer[name]>=0,outer[name]<extent) for name,extent in stream['ancestry']]
    first_m,first_n,first_local=pairs[0]
    anchor={**outer,**{name:z3.IntVal(0) for name,_ in first_local}}
    anchor_m=expression(first_m,anchor)/block.m
    anchor_n=expression(first_n,anchor)/block.n
    for index,(m,n,local) in enumerate(pairs):
        env={**outer,**{name:z3.Int(f'output-site:{index}:'+name) for name,_ in local}}
        domain=[z3.And(env[name]>=0,env[name]<extent) for name,extent in local]
        remaining=int((deadline-time.monotonic())*1000)
        if remaining<=0: raise ValueError('completion output-block scope proof budget exhausted (UNKNOWN)')
        solver=z3.Solver();solver.set(timeout=remaining)
        solver.add(*constraints,*domain,z3.Or(expression(m,env)/block.m!=anchor_m,
                                              expression(n,env)/block.n!=anchor_n))
        verdict=solver.check()
        if verdict!=z3.unsat:
            raise ValueError('completion stream crosses an original output-block boundary or its scope is UNKNOWN')
    return {'compute_sites':len(pairs),'block_partition':[block.m,block.n],
            'proof':'all actual output tile coordinates share one original block, with outer coordinates symbolic'}


def _scope_bounds(kernel, problem, block, adapter, law, admission, *, timeout_ms=5000):
    # This provisional bound is a finite counted-program upper bound, used only
    # for discovery. The selected bound below is the maximum actual K stream.
    total_pairs = sum(_count(s,adapter,{Compute}) for s in kernel.body)
    total_loads = sum(_count(s,adapter,{Load}) for s in kernel.body)
    search = replace(admission,max_issued_compute_pairs=max(1,total_pairs),
        max_issued_loads=max(1,total_loads))
    _, streams = _streams(kernel,adapter,law,search)
    if any(s['compute_pairs'] > block.m*block.n*problem.tiles[1] for s in streams):
        raise ValueError('completion issued scope escapes one original output block')
    deadline=time.monotonic()+timeout_ms/1000
    records=[dict(_record(s),output_block_proof=_prove_output_block(s,adapter,block,deadline)) for s in streams]
    bound = replace(admission,
        max_issued_compute_pairs=max((admission.max_issued_compute_pairs,*(s['compute_pairs'] for s in streams))),
        max_issued_loads=max((admission.max_issued_loads,*(s['loads'] for s in streams))))
    return bound, records


class _Metadata:
    @property
    def source_identity(self): return self.source.source_identity
    @property
    def output_readout(self): return self.source.output_readout
    @property
    def bias_source(self): return self.source.bias_source
    @property
    def readout_qualification_identity(self): return self.source.readout_qualification_identity

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from .llvm_program import emit_index_effect_program
        self.revalidate()
        receipt = emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate()
        return receipt

    def emit_llvm(self,builder,lower_call):
        from .llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate()
        receipt = emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call)
        self.revalidate()
        return receipt

    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands
        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        self.revalidate()
        return module


@dataclass(frozen=True)
class DenseCompletionInput(_Metadata):
    """Unchanged original program plus an exact, separately checked issued scope."""
    original: object
    source: object
    kernel: object
    realization: dict
    admission: object
    context: dict
    law: DensePanelEpochLaw
    checked_identity: str
    owner_sha256: tuple

    def revalidate(self):
        self.original.revalidate()
        self.source.revalidate()
        base = effective_admission(self.source.admission)
        expected, records = _scope_bounds(self.source.kernel,self.context['problem'],
            self.context['block'],self.context['adapter'],self.law,base)
        if (self.source is not checked_source(self.original)
                or self.kernel != self.original.kernel or self.kernel != self.source.kernel
                or self.admission != expected or self.owner_sha256 != _owner()
                or self.checked_identity != _digest((self.kernel.digest(),_pin(self.context),
                    _source_pin(self.source),asdict(self.law),asdict(expected),records))):
            raise ValueError('completion issued-scope proof or original checked program changed')


def bind_dense_completion_scope(original, context, law, *, static_budget=128,timeout_ms=15000):
    """Prove finite issued counts of original output-block streams; pending limits stay fixed.

    This explicit constructor is separate from completion application: a consumer
    must never silently increase an input admission that is insufficient.
    """
    if type(law) is not DensePanelEpochLaw:
        raise ValueError('completion composition requires a typed selected epoch law')
    context = normalize_completion_context(original,context)
    source,actual_storage = _validate_original(original,context,timeout_ms=timeout_ms,static_budget=static_budget)
    base = effective_admission(source.admission)
    law.validate_for(context['adapter'],base)
    if law.unknown(STREAM_FLAGS):
        return original
    bound, records = _scope_bounds(source.kernel,context['problem'],context['block'],
        context['adapter'],law,base)
    report = dict(original.realization)
    report['completion_issued_scope'] = {
        'source_kernel_sha256':source.kernel.digest(),'original_admission':asdict(base),
        'scoped_admission':asdict(bound),'streams':records,
        'proof':'exact counted data calls in a store-free original output-block K stream',
        'pending_capacity':'unchanged; selected finite ready/valid backpressure',
    }
    frozen_context = dict(context,admission=bound,premises=dict(context.get('premises',{})))
    result = DenseCompletionInput(original,source,source.kernel,report,bound,frozen_context,law,
        _digest((source.kernel.digest(),_pin(frozen_context),_source_pin(source),asdict(law),asdict(bound),records)),_owner())
    result.revalidate()
    return result


@dataclass(frozen=True)
class DenseCompletionResult(_Metadata):
    original: object
    source: object
    kernel: object
    realization: dict
    admission: object
    context: dict
    law: DensePanelEpochLaw
    checked_identity: str
    owner_sha256: tuple

    def revalidate(self):
        self.original.revalidate()
        expected,_ = _streams(self.original.kernel,self.context['adapter'],self.law,self.admission)
        if (self.source is not checked_source(self.original)
                or expected != self.kernel or check_structure(self.kernel)
                or self.realization['kernel_sha256'] != self.kernel.digest()
                or self.realization['dynamic_epoch_drains'] != _drains(self.kernel.body)
                or self.admission != self.context['admission'] or self.owner_sha256 != _owner()
                or self.checked_identity != _digest((self.original.kernel.digest(),self.kernel.digest(),
                    _pin(self.context),_source_pin(self.source),asdict(self.law)))):
            raise ValueError('completion composition: original, context, law or emitted effects changed')


def schedule_dense_completion(original,context,law,*,static_budget=128,timeout_ms=15000,max_read_instances=4096):
    """Consume real checked two-window, retained or segmented effects, preserving readout barriers."""
    if type(law) is not DensePanelEpochLaw:
        raise ValueError('completion composition requires a typed selected epoch law')
    context = normalize_completion_context(original,context)
    source,actual_storage = _validate_original(original,context,timeout_ms=timeout_ms,static_budget=static_budget)
    if type(max_read_instances) is not int or max_read_instances<=0:
        raise ValueError('completion composition needs an explicit positive read-instance proof budget')
    p,b,r,_,a,admission = (context[k] for k in ('problem','block','resources','storage','adapter','admission'))
    s = actual_storage
    law.validate_for(a,admission)
    actual, streams = _streams(original.kernel,a,law,admission)
    if not streams:
        return original
    if check_structure(actual) or _serial_projection(actual) != _serial_projection(original.kernel):
        raise ValueError('completion composition changed a data call, address, scope or issue order')
    check_integer_domain(actual,a.index_bits)
    if sum(1 for _ in _calls(actual.body)) > static_budget:
        raise ValueError('completion composition exceeds static command budget')
    if streams and s.accumulator.backing in (s.lhs.backing,s.rhs.backing):
        raise ValueError('completion composition needs separate operand and accumulator stores')
    deadline = time.monotonic()+timeout_ms/1000
    output_scope_proofs=[_prove_output_block(stream,a,b,deadline) for stream in streams]
    proofs = sum(_prove_stream(stream,a,s,deadline,max_read_instances) for stream in streams)
    before,after = _drains(original.kernel.body),_drains(actual.body)
    report = dict(original.realization)
    report.update(schema='merlin.dense_completion.v1',kernel_sha256=actual.digest(),
        input_schedule=original.realization,dynamic_epoch_drains=after,
        completion_composition={
            'input_dynamic_drains':before,'dynamic_drains':after,'removed_dynamic_drains':before-after,
            'streams':[_record(stream) for stream in streams],'consumption_proofs':proofs,
            'output_block_proofs':output_scope_proofs,
            'original_pipeline_kind':source.realization['schedule_ir']['pipeline_kind'],
            'unknown_stream_premises':list(law.unknown(STREAM_FLAGS)),
            'law':asdict(law),'admission':asdict(admission),
            'preserved':['every original readout boundary','every output-block boundary',
                         'closing completion','ordered data calls and storage'],
            'qualification':'conditional selected premises; no physical timing or qualification'})
    result = DenseCompletionResult(original,source,actual,report,admission,
        dict(context,premises=dict(context.get('premises',{}))),law,
        _digest((original.kernel.digest(),actual.digest(),_pin(context),_source_pin(source),asdict(law))),_owner())
    result.revalidate()
    return result
