"""Checked completion-only consumer for private counted dense constructions.

No construction family is renamed and no issued budget is changed here. The
original source owner proves arithmetic/ABI; this pass proves exact call-tree
preservation, closure, declared budgets, private windows and a write/read/write
chain for every physical operand row reused without a full drain. The target
must independently provide all selected ordering laws and finite backpressure.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import prod
from pathlib import Path
import json

from merlin.sched.ir import Call, Const, Kernel, Loop, evaluate, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from .block_schedule import Load, Preload, Compute, Store
from .compact_dense import _digest
from .dense_panel_pipeline import DRAIN, DensePanelPipelineResult, _windows, _serial_projection
from .dense_segmented_pipeline import DenseSegmentedPipelineResult, _assignment
from .compact_verification import verify_compact_dense
from .schedule_ir import read_dense_schedule_plan, parse_dense_schedule_plan


class CompletionNotApplicable(ValueError):
    """A missing obligation, distinct from a numerical or infrastructure error."""


REQUIRED_LAWS = (
    'local_read_after_write', 'local_write_after_read',
    'ordered_accumulator_writes', 'ordered_configuration_capture',
    'finite_queue_progress', 'accumulator_result_visibility',
    'exclusive_private_state', 'complete_device_drain',
)


def _truth(premises, name):
    value = premises.get(name)
    if type(value) is dict:
        value = value.get('value')
    return value is True


def _context_digest(context):
    return _digest((*(asdict(context[k]) for k in ('problem', 'block', 'resources', 'storage', 'admission')),
                    context['adapter'].configuration_identity,
                    context['adapter'].numerical_policy_identity,
                    context['adapter'].synchronous_completion,
                    context['adapter'].exact_accumulator_continuation,
                    context['adapter'].index_bits, context['premises']))


def _data_tree(body):
    return tuple(replace(s, body=_data_tree(s.body)) if type(s) is Loop else s
                 for s in body if s != Call(DRAIN, ()))


def _calls(body):
    for s in body:
        if type(s) is Loop:
            yield from _calls(s.body)
        else:
            yield s


def _owner(result):
    # Backend wrapper composition is deliberately explicit: the adapter unwraps
    # its RetainedLhsResult. No duck-typed result is accepted as a checked owner.
    if type(result) not in (DensePanelPipelineResult, DenseSegmentedPipelineResult):
        raise CompletionNotApplicable('unsupported checked construction owner')
    return result


def _bound(admission):
    # Segmented admission preserves both original grants. Use their intersection
    # rather than silently picking the larger budget from either component.
    if hasattr(admission, 'overlap') and hasattr(admission, 'reduction'):
        return {k: min(getattr(admission.overlap, k), getattr(admission.reduction, k))
                for k in ('max_issued_compute_pairs', 'max_issued_loads', 'max_issued_readouts',
                          'pending_compute_pairs', 'pending_loads', 'pending_readouts')}
    return {k: getattr(admission, k) for k in
            ('max_issued_compute_pairs', 'max_issued_loads', 'max_issued_readouts',
             'pending_compute_pairs', 'pending_loads', 'pending_readouts')}


def _ownership(result, context):
    p, b, r, s, a, admission = (context[k] for k in
        ('problem', 'block', 'resources', 'storage', 'adapter', 'admission'))
    if admission != result.admission:
        raise CompletionNotApplicable('context admission differs from checked construction')
    laws = (admission.overlap, admission.reduction) if hasattr(admission, 'overlap') else (admission,)
    for law in laws:
        law.__post_init__()
        if (law.hardware_identity != a.configuration_identity or
                law.numerical_policy_identity != a.numerical_policy_identity):
            raise CompletionNotApplicable('selected target/numerical identity differs')
    if (p.bias or (p.lhs_bytes, p.rhs_bytes, p.accumulator_bytes) != (1, 1, 4)
            or getattr(result, 'output_readout', None) is not None
            or getattr(result, 'bias_source', None) is not None
            or not a.synchronous_completion or not a.exact_accumulator_continuation):
        raise CompletionNotApplicable('completion route requires raw signed-i8 to exact-i32 without epilogue')
    views = (s.lhs, s.rhs, s.accumulator)
    if any(v is None for v in views):
        raise CompletionNotApplicable('unknown private storage ownership')
    if (sum(v.capacity*v.unit_bytes for v in views[:2]) > r.operand_bytes or
            views[2].capacity*views[2].unit_bytes > r.accumulator_bytes):
        raise CompletionNotApplicable('assigned storage exceeds original resource budget')
    for i, left in enumerate(views):
        for right in views[i+1:]:
            if (left.backing == right.backing and
                    left.base*left.unit_bytes < (right.base+right.capacity)*right.unit_bytes and
                    right.base*right.unit_bytes < (left.base+left.capacity)*left.unit_bytes):
                raise CompletionNotApplicable('simultaneously live storage views overlap')
    schedule = result.realization.get('schedule_ir', {})
    if not schedule.get('text'):
        raise CompletionNotApplicable('missing consumed source-bound construction schedule')
    decision = read_dense_schedule_plan(parse_dense_schedule_plan(schedule['text']), p, r, s, a,
        source_identity=result.source_identity, dtypes=('i8','i8','i32'), static_budget=128)
    if decision.block != b or decision.storage != s:
        raise CompletionNotApplicable('context block/storage differs from consumed construction')
    kind = decision.pipeline_kind
    if kind == 'two_window':
        _windows(p, b, s, r, decision.pipeline_window_offsets)
    elif kind == 'retained_lhs_segmented_k':
        _assignment(p, b, r, s, decision.pipeline_window_offsets)
    elif kind == 'retained_full_k':
        if not s.admits(p, b, r):
            raise CompletionNotApplicable('retained full-K assignment is not admitted')
    else:
        raise CompletionNotApplicable('unproved construction/window family')
    # Recheck actual decoded operands against THIS context, not a report copied
    # from another storage assignment. Serial projection is only the source
    # arithmetic/lifetime proof; asynchronous legality is checked below.
    source = verify_compact_dense(_serial_projection(result.kernel), p, r, s, a,
                                  static_budget=128, timeout_ms=15000)
    return kind, source


def _proposal(kernel, adapter):
    """Remove only intermediate K-stream drains, keep all loop/data structure.

    A drain immediately before any readout, after a readout, at the end of a
    block/invocation or after retained-LHS initialization is never removed.
    Repeated-loop frontiers are unioned so the same static decision is safe on
    every dynamic iteration, including the last one.
    """
    def original_closure(body, inside=False):
        for s in body:
            if type(s) is Loop:
                has_drain = any(type(c) is Call and c.instr == DRAIN for c in s.body)
                if has_drain:
                    if inside or s.extent != 1 or s.body[-1] != Call(DRAIN, ()):
                        raise CompletionNotApplicable('malformed or missing original closing completion')
                    original_closure(s.body[:-1], True)
                else:
                    original_closure(s.body, inside)
            elif type(s) is not Call or not inside or s.instr == DRAIN:
                raise CompletionNotApplicable('original effect lacks its closing full completion')
    original_closure(kernel.body)
    epochs, removed = {}, []
    def types(body):
        return tuple(type(adapter.project(c)) for c in _calls(body) if c.instr != DRAIN)
    def first(body):
        for s in body:
            if type(s) is Loop:
                found = first(s.body)
                if found:
                    return found
            elif s.instr != DRAIN:
                return frozenset((type(adapter.project(s)),))
        return frozenset()
    def walk(body, following, outer=1, path=()):
        result = []
        next_types = following
        for index in reversed(range(len(body))):
            s = body[index]
            at = (*path, index)
            if type(s) is not Loop:
                if s.instr == DRAIN:
                    raise CompletionNotApplicable('completion is outside an original singleton closing epoch')
                result.append(s)
                next_types = frozenset((type(adapter.project(s)),))
                continue
            drains = [c for c in s.body if type(c) is Call and c.instr == DRAIN]
            if drains:
                if (s.extent != 1 or len(drains) != 1 or s.body[-1] != Call(DRAIN, ())
                        or any(c.instr == DRAIN for c in _calls(s.body[:-1]))):
                    raise CompletionNotApplicable('missing or malformed original closing completion')
                kinds = types(s.body[:-1])
                if not kinds:
                    raise CompletionNotApplicable('empty/unknown original completion epoch')
                ops = [adapter.project(c) for c in _calls(s.body[:-1])]
                pure_lhs = all(type(op) is Load and op.role == 'lhs' for op in ops)
                eligible = (set(kinds) <= {Load, Preload, Compute} and not pure_lhs and
                            bool(next_types) and next_types <= {Load, Preload, Compute})
                epochs[at] = dict(removed=eligible, dynamic=outer, types=tuple(t.__name__ for t in kinds))
                if eligible:
                    s = replace(s, body=s.body[:-1])
                    removed.append(at)
                result.append(s)
                next_types = first(s.body)
            else:
                successors = next_types | (first(s.body) if s.extent > 1 else frozenset())
                result.append(replace(s, body=walk(s.body, successors, outer*s.extent, at)))
                next_types = first(s.body) or next_types
        return tuple(reversed(result))
    candidate = replace(kernel, body=walk(kernel.body, frozenset()))
    if _data_tree(kernel.body) != _data_tree(candidate.body):
        raise CompletionNotApplicable('completion proposal changed ordered data calls/control/storage')
    if check_structure(candidate):
        raise CompletionNotApplicable('completion proposal has malformed counted IR')
    return candidate, epochs


def _trace_proof(original, candidate, context, *, operation_limit):
    """Bounded exact physical-row proof; emission remains compact counted IR.

    A row's previous load must be followed by an overlapping consuming execute
    read before another load writes it without a kept drain. This establishes
    WAW through RAW then WAR even when load completions themselves are OOO.
    Existing source checking establishes which exact tensor tile every read
    consumes. No issue ticket, delay or queue-capacity enlargement is invented.
    """
    a, s = context['adapter'], context['storage']
    bound = _bound(context['admission'])
    if any(type(v) is not int or v <= 0 for v in bound.values()):
        raise CompletionNotApplicable('missing finite issued/pending queue budgets')
    total = 0
    def count(body, outer=1):
        return sum(count(x.body, outer*x.extent) if type(x) is Loop else outer for x in body)
    dynamic = count(original.body)
    if dynamic > operation_limit:
        raise CompletionNotApplicable(f'exact row-chain checking work {dynamic} exceeds budget {operation_limit}')
    # Store only physical rows, bounded by the caller's assigned stores.
    rows = {}
    epoch = 0
    pending = None
    counters = [0,0,0]
    maximum = [0,0,0]
    drains = 0
    waw_bridges = 0
    epoch_classes = {}
    last_was_drain = False
    def value(v, env):
        return v if type(v) in (int,bool) else evaluate(v, env)
    def span(view, base, length):
        if base < view.base or length <= 0 or base+length > view.base+view.capacity:
            raise CompletionNotApplicable('actual physical effect escapes owned storage')
        return range(base, base+length)
    def read(view, base, length):
        for row in span(view, base, length):
            key = (view.backing, view.unit_bytes, row)
            if key not in rows:
                raise CompletionNotApplicable('operand read has no preceding owned load')
            rows[key][1] = True
    def emit(body, env):
        nonlocal total, epoch, pending, drains, waw_bridges, last_was_drain
        for c in body:
            if type(c) is Loop:
                for i in range(c.extent):
                    yield from emit(c.body, {**env, c.var:i})
            else:
                yield c, env
    for c, env in emit(candidate.body, {}):
        total += 1
        last_was_drain = c == Call(DRAIN, ())
        if last_was_drain:
            if pending is not None:
                raise CompletionNotApplicable('completion splits paired preload/compute')
            epoch_classes[tuple(counters)] = epoch_classes.get(tuple(counters),0) + 1
            for i, (actual, field) in enumerate(zip(counters,
                    ('max_issued_compute_pairs','max_issued_loads','max_issued_readouts'))):
                if actual > bound[field]:
                    raise CompletionNotApplicable(f'insufficient {field}: issued {actual}, budget {bound[field]}')
                maximum[i] = max(maximum[i], actual)
                counters[i] = 0
            epoch += 1
            drains += 1
            continue
        if c.unit or c.produces or c.awaits or c.stages or c.sets or c.assumes:
            raise CompletionNotApplicable('unknown asynchronous/configuration effect')
        op = a.project(c)
        if counters[2] and type(op) is not Store:
            raise CompletionNotApplicable('required output-block completion is missing')
        if type(op) is Load:
            if pending is not None:
                raise CompletionNotApplicable('load splits paired preload/compute')
            if op.role not in ('lhs','weight'):
                raise CompletionNotApplicable('unknown operand ownership')
            view = s.lhs if op.role == 'lhs' else s.rhs
            base, length, cols = (value(x,env) for x in (op.row,op.rows,op.cols))
            if cols <= 0 or cols*context['problem'].lhs_bytes > view.unit_bytes:
                raise CompletionNotApplicable('wide/unproved physical operand mapping')
            for row in span(view, base, length):
                key = (view.backing, view.unit_bytes, row)
                prior = rows.get(key)
                if prior is not None and prior[0] == epoch:
                    if not prior[1]:
                        raise CompletionNotApplicable('unordered operand WAW lacks an intervening consuming read')
                    waw_bridges += 1
                rows[key] = [epoch,False]
            counters[1] += 1
        elif type(op) is Preload:
            if pending is not None:
                raise CompletionNotApplicable('unconsumed preload descriptor overwritten')
            # Weight read occurs at preload, not after its paired compute.
            read(s.rhs, value(op.weight_row,env), value(op.weight_rows,env))
            pending = op, dict(env)
        elif type(op) is Compute:
            if pending is None:
                raise CompletionNotApplicable('unpaired compute')
            pre, pre_env = pending
            read(s.lhs, value(op.input_row,env), value(op.rows,env))
            span(s.accumulator, value(pre.accumulator_row,pre_env), value(pre.rows,pre_env))
            counters[0] += 1
            pending = None
        elif type(op) is Store:
            if pending is not None or counters[0] or counters[1]:
                raise CompletionNotApplicable('required readout boundary has been removed')
            span(s.accumulator, value(op.accumulator_row,env), value(op.rows,env))
            counters[2] += 1
        else:
            raise CompletionNotApplicable('unsupported effect in raw dense completion route')
    if pending is not None or not last_was_drain or any(counters):
        raise CompletionNotApplicable('missing closing full-device completion')
    return dict(dynamic_checked_calls=total, original_dynamic_calls=dynamic,
                maximum_issued=dict(zip(('compute_pairs','loads','readouts'),maximum)),
                issued_and_pending_bounds=bound, kept_dynamic_drains=drains,
                merged_epoch_classes=[dict(compute_pairs=k[0],loads=k[1],readouts=k[2],dynamic_epochs=v)
                                      for k,v in epoch_classes.items()],
                operand_row_waw_bridges=waw_bridges,
                operand_lifetime_proof='original exact source/lifetime proof plus actual row RAW/WAR chains',
                load_waw_completion_order='not assumed; every reused row has execute read bridge or kept drain')


@dataclass(frozen=True)
class DenseKCompletionResult:
    original: object
    kernel: Kernel
    realization: dict
    context: dict
    checked_context_sha256: str
    checked_kernel_sha256: str
    checked_report_sha256: str
    checked_owner_sha256: str
    @property
    def admission(self):
        return self.original.admission
    @property
    def source_identity(self):
        return self.original.source_identity
    def revalidate(self):
        self.original.revalidate()
        if (self.kernel.digest() != self.checked_kernel_sha256 or
                _context_digest(self.context) != self.checked_context_sha256 or
                _digest(self.realization) != self.checked_report_sha256 or
                sha256(Path(__file__).read_bytes()).hexdigest() != self.checked_owner_sha256):
            raise ValueError('completion source, context, actual emitted program or proof changed')
    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from .llvm_program import emit_index_effect_program
        self.revalidate()
        result = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return result
    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands
        self.revalidate()
        return materialize_scf_commands(self.kernel)


def complete_dense_k_stream(original, context, *, operation_limit=2_000_000):
    """Return an actual checked completion result or a precise proof refusal.

    Context's issued admission is immutable. Unknown target law, unknown owner,
    aliases, oversized checking work or insufficient bounds never grant a safe
    optimized result. A backend may catch CompletionNotApplicable and return its
    valid original result. Numerical execution is a separate obligation.
    """
    required = ('problem','block','resources','storage','adapter','admission','premises')
    if type(context) is not dict or any(k not in context for k in required):
        raise CompletionNotApplicable('incomplete source-bound completion context')
    if type(context['premises']) is not dict:
        raise CompletionNotApplicable('ordering premises must be a plain dictionary')
    missing = [name for name in REQUIRED_LAWS if not _truth(context['premises'],name)]
    if missing:
        raise CompletionNotApplicable('unknown required premise: '+', '.join(missing))
    original = _owner(original)
    original.revalidate()
    kind, source = _ownership(original,context)
    kernel, epochs = _proposal(original.kernel,context['adapter'])
    removed = sum(v['dynamic'] for v in epochs.values() if v['removed'])
    if not removed:
        raise CompletionNotApplicable('no intermediate K-stream completion is removable')
    check_integer_domain(kernel,context['adapter'].index_bits)
    proof = _trace_proof(original.kernel,kernel,context,operation_limit=operation_limit)
    report = dict(original.realization)
    report['completion'] = dict(status='optimized', construction_kind=kind,
        original_kernel_sha256=original.kernel.digest(), actual_kernel_sha256=kernel.digest(),
        removed_dynamic_drains=removed, kept_boundaries=[list(k) for k,v in epochs.items() if not v['removed']],
        data_tree_sha256=sha256(repr(_data_tree(kernel.body)).encode()).hexdigest(),
        exact_ordered_data_tree_preserved=True, storage_preserved=True,
        admission_preserved=True, source_rechecked=source['kernel_sha256'] if 'kernel_sha256' in source else True,
        premises=context['premises'], **proof)
    report['construction_epochs'] = original.realization.get('epochs')
    report['epochs'] = proof['merged_epoch_classes']
    report['kernel_sha256'] = kernel.digest()
    report['dynamic_epoch_drains'] = proof['kept_dynamic_drains']
    result = DenseKCompletionResult(original,kernel,report,dict(context),_context_digest(context),
        kernel.digest(),_digest(report),sha256(Path(__file__).read_bytes()).hexdigest())
    result.revalidate()
    return result
