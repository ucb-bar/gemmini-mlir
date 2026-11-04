"""Checked completion-only composition over existing dense construction routes.

No constructor is selected or called here. The actual checked input is consumed,
including its original storage and issue order. Selected hardware laws replace
only internal compute/prefetch completions. Initialization, readout and closing
completions stay. Issued scope growth is a separate source-bound count proof;
physical pending budgets and every other admission field stay unchanged.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import time

from merlin.sched.ir import Loop, check_structure
from .block_schedule import Compute, Load, Preload
from .compact_dense import _digest, DenseLoopStorage, DenseLoopAdapter
from merlin.transforms.dense_planning import DenseProblem, DenseResources, DenseBlock
from .dense_panel_pipeline import (
    DensePanelOverlapAdmission, DensePanelPipelineResult,
    DenseRetainedReductionAdmission, _calls, _serial_projection,
    verify_dense_panel_pipeline, verify_dense_retained_reduction, _windows,
)
from .dense_segmented_pipeline import (
    DenseSegmentedPipelineResult, DenseSegmentedAdmission,
    verify_dense_segmented_pipeline, _assignment,
)
from .dense_panel_epochs import (
    DensePanelEpochLaw, STREAM_FLAGS, _is_epoch, _is_chain, _kinds,
    _count, _epochs, _strip, _drains, _record, _prove_stream, _OPERANDS, _sites,
)


def _owner():
    # The reused law, structural epoch helpers and dependence proof are part
    # of this result's owner even though the legacy roster predates them.
    from . import dense_panel_epochs
    return _digest((sha256(Path(__file__).read_bytes()).hexdigest(),
                    sha256(Path(dense_panel_epochs.__file__).read_bytes()).hexdigest()))


def _base(original):
    """The original wrapper remains a revalidated owner, never a route relabel."""
    candidate = getattr(original, 'checked', original)
    if type(candidate) not in (DensePanelPipelineResult, DenseSegmentedPipelineResult):
        raise ValueError('completion composition needs a supported checked dense owner')
    if original.kernel != candidate.kernel:
        raise ValueError('completion wrapper does not own the actual checked kernel')
    return candidate


def effective_completion_admission(a):
    """Intersect segmented component budgets; retain all physical queue facts."""
    if type(a) is DenseSegmentedAdmission:
        a.__post_init__()
        return replace(a.overlap,
            max_issued_compute_pairs=min(a.overlap.max_issued_compute_pairs, a.reduction.max_issued_compute_pairs),
            max_issued_loads=min(a.overlap.max_issued_loads, a.reduction.max_issued_loads),
            max_issued_readouts=min(a.overlap.max_issued_readouts, a.reduction.max_issued_readouts))
    if type(a) not in (DensePanelOverlapAdmission, DenseRetainedReductionAdmission):
        raise ValueError('completion composition has no selected original admission')
    return a


def _admission(original):
    return effective_completion_admission(_base(original).admission)


def _route(original):
    base = _base(original)
    kind = base.realization.get('schedule_ir', {}).get('pipeline_kind')
    if type(base.admission) is DenseRetainedReductionAdmission:
        return 'retained'
    if kind not in ('two_window', 'retained_lhs_segmented_k'):
        raise ValueError('completion composition has no proved construction route')
    return kind


def _plan(kernel, adapter, law, admission=None):
    """Find K chains ending at their ORIGINAL final compute completion.

    A stream contains at least two original compute epochs and may begin with
    one operand-load epoch proved consumed by its first compute. Its interior
    may include serialized loads. Other invocation/retained preparation stays.
    Stores/initializations are barriers. Counted loops with those effects are
    searched recursively, never treated as one cross-output stream.
    """
    streams = []
    enabled = not law.unknown(STREAM_FLAGS)

    def operand_chain(statement):
        if not _is_chain(statement):
            return False
        kinds = _kinds(statement, adapter)
        return bool(kinds) and kinds <= _OPERANDS

    def starts_compute(statement):
        return operand_chain(statement) and Compute in _kinds(statement, adapter)

    def compute_epochs(statement):
        if _is_epoch(statement):
            return int(Compute in _kinds(statement, adapter))
        return statement.extent * sum(compute_epochs(child) for child in statement.body)

    def starts_stream(body, index):
        statement = body[index]
        return starts_compute(statement) or (
            _is_epoch(statement) and _kinds(statement, adapter) == {Load}
            and index + 1 < len(body) and starts_compute(body[index + 1]))

    def walk(body, ancestry):
        out, index = [], 0
        while index < len(body):
            end = index
            if starts_stream(body, index):
                while end < len(body) and operand_chain(body[end]):
                    end += 1
            last = end - 1
            while last >= index and not (_is_epoch(body[last]) and Compute in _kinds(body[last], adapter)):
                last -= 1
            elements = tuple(body[index:last + 1])
            pairs = sum(_count(e, adapter, {Compute}) for e in elements)
            loads = sum(_count(e, adapter, {Load}) for e in elements)
            fits = admission is None or (pairs <= admission.max_issued_compute_pairs
                                        and loads <= admission.max_issued_loads)
            if enabled and elements and sum(compute_epochs(e) for e in elements) > 1 and fits:
                out.extend((*(_strip(e) for e in elements[:-1]), elements[-1]))
                streams.append(dict(ancestry=ancestry, elements=elements, readout=None,
                    compute_pairs=pairs, loads=loads, readouts=0,
                    removed_drains=_drains(elements)-1))
                index = last + 1
                continue
            statement = body[index]
            if type(statement) is Loop and not _is_epoch(statement):
                statement = replace(statement, body=walk(statement.body,
                    (*ancestry, (statement.var, statement.extent))))
            out.append(statement)
            index += 1
        return tuple(out)
    return replace(kernel, body=walk(kernel.body, ())), streams


def _prove_output_scope(stream, adapter, block, timeout_ms=5000):
    """Every paired update in a removed-boundary stream is one output block.

    Use actual projected source tile coordinates, not loop names or a count
    heuristic. All chain/local indices vary; enclosing output indices stay
    symbolic. Full and ragged output groups are covered by the same proof.
    """
    import z3
    from merlin.sched.ir import Const, Var, Sum, Scaled, Select

    def expr(value, env):
        if type(value) is int:
            return z3.IntVal(value)
        if type(value) is Const:
            return z3.IntVal(value.value)
        if type(value) is Var:
            return env[value.name]
        if type(value) is Sum:
            return z3.Sum(*(expr(term, env) for term in value.terms))
        if type(value) is Scaled:
            return expr(value.expr, env) * value.factor
        if type(value) is Select:
            return z3.If(env[value.var] == value.equals, expr(value.then, env), expr(value.other, env))
        raise ValueError('completion output scope has an unknown index expression')

    outer = {name:z3.Int('scope-outer:'+name) for name, _ in stream['ancestry']}
    outer_constraints = [z3.And(outer[name]>=0, outer[name]<extent) for name,extent in stream['ancestry']]
    reference, paired = None, []
    pending = None
    for element in stream['elements']:
        for call, local in _sites((element,)):
            op = adapter.project(call)
            if type(op) is Preload:
                if pending is not None:
                    raise ValueError('completion output scope encountered an unpaired descriptor')
                pending = (op, local)
            elif type(op) is Compute:
                if pending is None or pending[1] != local:
                    raise ValueError('completion output scope requires an adjacent same-domain compute pair')
                descriptor, _ = pending
                paired.append((op.input_block[0],descriptor.weight_block[1],local))
                pending = None
            elif pending is not None:
                raise ValueError('completion output scope split the original descriptor/compute pair')
    if pending is not None or not paired:
        raise ValueError('completion output scope lacks complete paired updates')
    m,n,local = paired[0]
    fixed = {**outer, **{name:z3.IntVal(0) for name,_ in local}}
    reference = (expr(m,fixed)/block.m, expr(n,fixed)/block.n)
    for m,n,local in paired:
        env = {**outer, **{name:z3.Int('scope-local:'+name) for name,_ in local}}
        solver=z3.Solver();solver.set(timeout=timeout_ms)
        solver.add(*outer_constraints,
            *(z3.And(env[name]>=0,env[name]<extent) for name,extent in local),
            z3.Or(expr(m,env)/block.m != reference[0], expr(n,env)/block.n != reference[1]))
        if solver.check() != z3.unsat:
            raise ValueError('completion stream crosses an original logical output-block boundary')
    return len(paired)


def _decision_geometry(original, p, b):
    """Bind this completion context to the ORIGINAL consumed schedule decision.

    Read its typed geometry directly. Layout has already been consumed and may
    narrow/relocate the original view, so do not reinterpret an actual layout as
    an unconsumed planner input or regenerate data commands from it.
    """
    from .schedule_ir import parse_dense_schedule_plan
    from xdsl.dialects.builtin import DenseArrayBase
    from xdsl.utils.exceptions import ParseError, VerifyException
    from merlin.transforms.dense_ir import DenseScheduleOp
    from merlin.xdsl_dialects.schedule import BindOp

    base = _base(original)
    text = base.realization.get('schedule_ir', {}).get('text')
    if type(text) is not str or not text.strip():
        raise ValueError('completion composition lacks the original consumed schedule decision')
    try:
        module = parse_dense_schedule_plan(text)
        module.verify()
    except (ParseError, VerifyException) as exc:
        raise ValueError('completion composition cannot validate the original consumed decision: '+str(exc)) from exc
    declarations = [op for op in module.walk() if isinstance(op, DenseScheduleOp)]
    bindings = [op for op in module.walk() if isinstance(op, BindOp)]
    if len(declarations) != 1 or len(bindings) != 1:
        raise ValueError('completion composition needs one original bound contraction decision')
    op = declarations[0]
    fields = {
        'problem': (p.m, p.k, p.n),
        'primitive_tile': (p.tile_m, p.tile_k, p.tile_n),
        'block_tiles': (b.m, b.k, b.n),
        'element_bytes': (p.lhs_bytes, p.rhs_bytes, p.accumulator_bytes),
    }
    for name, expected in fields.items():
        attribute = op.attributes.get(name)
        if not isinstance(attribute, DenseArrayBase) or tuple(attribute.iter_values()) != expected:
            raise ValueError('completion context differs from the original consumed '+name+' decision')
    if not bindings[0].target.data.startswith('dense:'+base.source_identity+':'):
        raise ValueError('completion decision selects another original source identity')


def _storage(original, p, b, r, s):
    if (type(p) is not DenseProblem or type(b) is not DenseBlock
        or type(r) is not DenseResources or type(s) is not DenseLoopStorage):
        raise ValueError('completion composition needs typed original problem/block/resources/storage')
    _decision_geometry(original, p, b)
    route = _route(original)
    if route == 'two_window':
        _windows(p, b, s, r)
    elif route == 'retained_lhs_segmented_k':
        _assignment(p, b, r, s)
    elif s.accumulator is None or not s.admits(p, b, r):
        raise ValueError('retained completion needs original private storage admission')
    # Whole role views must have disjoint live byte ownership as well as slots.
    views = (s.lhs, s.rhs, s.accumulator)
    for i, left in enumerate(views):
        for right in views[i+1:]:
            if (left.backing == right.backing
                and left.base * left.unit_bytes < (right.base + right.capacity) * right.unit_bytes
                and right.base * right.unit_bytes < (left.base + left.capacity) * left.unit_bytes):
                raise ValueError('completion composition: overlapping live operand/accumulator views')


def _verify_original(original, p, b, r, s, adapter, static_budget, timeout_ms):
    original.revalidate()
    base = _base(original)
    _storage(original, p, b, r, s)
    if type(base.admission) is DenseSegmentedAdmission:
        verify = verify_dense_segmented_pipeline
    elif type(base.admission) is DenseRetainedReductionAdmission:
        verify = verify_dense_retained_reduction
    else:
        verify = verify_dense_panel_pipeline
    # Rebind problem, resources, storage and decoder to the ACTUAL original.
    # This also rejects uncovered data calls and missing closing completions.
    verify(original.kernel, p, b, r, s, adapter, base.admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=base.output_readout, bias_source=base.bias_source)


@dataclass(frozen=True)
class DenseCompletionScope:
    """Exact issued-count envelope of source-bound output-block K streams.

    The progress premise admits bounded issue under physical queue backpressure;
    counts are program multiplicities, not claimed pending queue capacities.
    """
    kernel_sha256: str
    problem_sha256: str
    original_admission_sha256: str
    admission_sha256: str
    counts: tuple
    law_sha256: str
    storage_sha256: str
    owner_sha256: str


def prove_dense_completion_scope(original, problem, block, resources, storage, adapter, law):
    """Propose a finite, exact issued envelope; no completion is removed here.

    This grant is usable only with this original, geometry/storage, laws and
    every exact per-stream count. schedule_dense_completion_epochs subsequently
    checks original source/epochs and the load-reader chain before consuming it.
    """
    original.revalidate()
    base = _admission(original)
    law.validate_for(adapter, base)
    if law.unknown(STREAM_FLAGS):
        raise ValueError('issued-scope proof lacks ordering, ownership or finite-progress premises')
    _storage(original, problem, block, resources, storage)
    _, streams = _plan(original.kernel, adapter, law)
    # Independent source-work ceiling per output block. No output row or
    # column is fused with another block, and no data command is introduced.
    pair_limit = block.m * block.n * problem.tiles[1]
    load_limit = (block.m + block.n) * problem.tiles[1]
    for stream in streams:
        _prove_output_scope(stream, adapter, block)
        if stream['compute_pairs'] > pair_limit or stream['loads'] > load_limit:
            raise ValueError('completion stream crosses the original output-block source-work bound')
    counts = tuple((s['ancestry'], s['compute_pairs'], s['loads'], _epochs_count(s)) for s in streams)
    bound = replace(base,
        max_issued_compute_pairs=max((s['compute_pairs'] for s in streams), default=base.max_issued_compute_pairs),
        max_issued_loads=max(base.max_issued_loads, max((s['loads'] for s in streams), default=0)))
    bound = replace(bound, max_issued_compute_pairs=max(base.max_issued_compute_pairs, bound.max_issued_compute_pairs))
    scope = DenseCompletionScope(original.kernel.digest(),
        _digest((asdict(problem), asdict(block), asdict(resources))),
        _digest(asdict(base)), _digest(asdict(bound)), counts,
        _digest((law.hardware_identity, law.numerical_policy_identity, law.qualification_identity,
            tuple((flag, getattr(law, flag)) for flag in STREAM_FLAGS))),
        _digest(asdict(storage)), _owner())
    return bound, scope


def _epochs_count(stream):
    return sum(_epochs(e) for e in stream['elements'])


def _identity(original, p, b, r, s, adapter, admission, law, scope):
    return _digest((original.kernel.digest(), _base(original).source_identity,
        _base(original).realization.get('schedule_ir', {}).get('text'), _route(original),
        asdict(p), asdict(b), asdict(r), asdict(s),
        repr(adapter), asdict(admission), asdict(law), None if scope is None else asdict(scope)))


@dataclass(frozen=True)
class DenseCompletionResult:
    original: object
    kernel: object
    realization: dict
    admission: object
    law: DensePanelEpochLaw
    scope: object
    problem: object
    block: object
    resources: object
    storage: object
    adapter: object
    checked_identity: str
    owner_sha256: str

    def __getattr__(self, name):
        return getattr(_base(self.original), name)

    def revalidate(self):
        self.original.revalidate()
        _storage(self.original, self.problem, self.block, self.resources, self.storage)
        actual, _ = _plan(self.original.kernel, self.adapter, self.law, self.admission)
        if (actual != self.kernel or self.kernel.digest() != self.realization['kernel_sha256']
            or self.checked_identity != _identity(self.original, self.problem, self.block,
                self.resources, self.storage, self.adapter, self.admission, self.law, self.scope)
            or self.owner_sha256 != _owner()):
            raise ValueError('completion composition: checked program, premises, scope or owner changed')

    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands
        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        module.verify()
        self.revalidate()
        return module

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from .llvm_program import emit_index_effect_program
        from .dense_panel_pipeline import DRAIN
        self.revalidate()
        receipt = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt

    def emit_llvm(self, builder, lower_call):
        from .llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        receipt = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return receipt


def schedule_dense_completion_epochs(original, problem, block, resources, storage,
    adapter, admission, law, *, scope=None, static_budget=128, timeout_ms=30000,
    max_read_instances=4096):
    """Compose checked routes; preserve initialization/readout/block completions."""
    if type(law) is not DensePanelEpochLaw or type(adapter) is not DenseLoopAdapter:
        raise ValueError('completion composition needs an explicit typed law and adapter')
    admission = effective_completion_admission(admission)
    _verify_original(original, problem, block, resources, storage, adapter, static_budget, timeout_ms)
    base = _admission(original)
    if type(admission) is not type(base):
        raise ValueError('completion composition cannot change the original admission type')
    admission.__post_init__()
    law.validate_for(adapter, admission)
    if admission != base:
        expected, proof = prove_dense_completion_scope(original, problem, block, resources, storage, adapter, law)
        if type(scope) is not DenseCompletionScope or scope != proof or admission != expected:
            raise ValueError('completion composition: issued-budget increase lacks its exact scoped proof')
    elif scope is not None:
        expected, proof = prove_dense_completion_scope(original, problem, block, resources, storage, adapter, law)
        if type(scope) is not DenseCompletionScope or scope != proof or admission != expected:
            raise ValueError('completion composition: supplied issued budgets do not cover the proved scope')
    actual, streams = _plan(original.kernel, adapter, law, admission)
    if check_structure(actual) or _serial_projection(actual) != _serial_projection(original.kernel):
        raise ValueError('completion composition changed ordered data operations or loop scopes')
    if sum(1 for _ in _calls(actual.body)) > static_budget:
        raise ValueError('completion composition exceeds original static command budget')
    if type(timeout_ms) is not int or timeout_ms <= 0 or type(max_read_instances) is not int or max_read_instances <= 0:
        raise ValueError('completion composition requires positive proof budgets')
    deadline = time.monotonic() + timeout_ms / 1000
    scope_proofs = sum(_prove_output_scope(s, adapter, block, timeout_ms) for s in streams)
    proofs = sum(_prove_stream(s, adapter, storage, deadline, max_read_instances) for s in streams)
    before, after = _drains(original.kernel.body), _drains(actual.body)
    report = dict(original.realization)
    report.update(schema='merlin.dense_completion_composition.v1', kernel_sha256=actual.digest(),
        dynamic_epoch_drains=after, input_schedule=original.realization,
        completion_extension=dict(status='optimized' if before > after else 'unchanged',
            route=_route(original), reason=None if before > after else 'no redundant internal K boundary inside the proved issued scope; initialization and readout barriers retained',
            original_kernel_sha256=original.kernel.digest(),
            actual_kernel_sha256=actual.digest(), original_dynamic_drains=before,
            removed_dynamic_drains=before-after, dynamic_drains=after,
            streams=[_record(s) for s in streams], consumption_proofs=proofs, output_scope_proofs=scope_proofs,
            law=asdict(law), unknown_premises=list(law.unknown(STREAM_FLAGS)),
            issued_scope=None if scope is None else asdict(scope),
            original_admission=asdict(base), actual_admission=asdict(admission),
            preserved=['all ordered data calls and loop scopes', 'all storage and addresses',
                'preparation outside the proved K stream', 'final compute before readout',
                'every original readout completion', 'output-block and final completion'],
            target_execution='conditional on selected laws; no physical timing qualification'))
    result = DenseCompletionResult(original, actual, report, admission, law, scope,
        problem, block, resources, storage, adapter,
        _identity(original, problem, block, resources, storage, adapter, admission, law, scope), _owner())
    result.revalidate()
    return result
