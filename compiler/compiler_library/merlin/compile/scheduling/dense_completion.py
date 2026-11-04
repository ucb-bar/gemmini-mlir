"""Completion-only composition on actual counted dense schedules.

No data operation is built, moved, renamed, reprojected for emission or edited.
A proposal deletes only full drains inside an output's K stream.  The last
original drain of that stream, every readout/initialization drain and all outer
boundaries remain.  Source equivalence is inherited through exact symbolic
identity; target dependency order is a separate, explicit conditional premise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import prod
from pathlib import Path

from merlin.sched.ir import Call, Kernel, Loop, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import NotApplicable
from merlin.transforms.pipeline_state import PassSpec, validate_pass_order

from .block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from .compact_dense import _digest
from .dense_contracts import TiledIntegerStore
from .dense_panel_pipeline import DRAIN

REQUIRED_PREMISES = (
    'local_read_after_write', 'local_write_after_read',
    'ordered_accumulator_writes', 'ordered_configuration_capture',
    'accumulator_result_visibility', 'finite_queue_progress',
    'exclusive_private_state', 'complete_device_drain',
)
MODES = ('original', 'period', 'output_block')
PASS_SPECS = (
    PassSpec('completion_source', 3, produces=('source_schedule',), mutates_ir=False, ir_level='schedule'),
    PassSpec('dense_k_completion', 4, requires=('source_schedule',), produces=('completion_proof',), ir_level='schedule'),
    PassSpec('completion_emission', 5, requires=('completion_proof',), produces=('emission_input',), mutates_ir=False, ir_level='schedule'),
)
validate_pass_order(PASS_SPECS)


class CompletionRefusal(NotApplicable):
    """A precise, safe refusal; never a numerical or infrastructure verdict."""


def _body_data(body):
    # Keep the COMPLETE counted tree, including original singleton wrappers.
    # Thus equality proves exact dynamic order, operands, metadata and domains.
    return tuple(replace(op, body=_body_data(op.body)) if type(op) is Loop else op
                 for op in body if op != Call(DRAIN, ()))


def _count(body):
    counts = dict(loads=0, compute_pairs=0, readouts=0, drains=0)
    for op in body:
        if type(op) is Loop:
            for key, value in _count(op.body).items():
                counts[key] += op.extent * value
        elif op == Call(DRAIN, ()):
            counts['drains'] += 1
    return counts


def dynamic_drains(kernel):
    return _count(kernel.body)['drains']


def _classify(body, adapter):
    """Decode actual calls; opcode names are not a memory/effect proof."""
    kinds = set()
    for op in body:
        if type(op) is Loop:
            if op.iter_args or op.yields or op.results:
                raise CompletionRefusal('completion: event-carried loops need a separate composition contract')
            kinds.update(_classify(op.body, adapter))
        elif type(op) is Call:
            if op.unit or op.produces or op.awaits or op.stages or op.sets or op.assumes:
                raise CompletionRefusal('completion: annotated effects need a complete independent projector')
            if op.instr == DRAIN:
                if op != Call(DRAIN, ()):
                    raise CompletionRefusal('completion: unknown full-drain operands or effects')
                kinds.add('drain')
            else:
                try:
                    effect = adapter.project(op)
                except (ValueError, KeyError, TypeError) as error:
                    raise CompletionRefusal('completion: unknown actual effect: ' + str(error)) from error
                if type(effect) not in (Load, Preload, Compute, Store, TiledIntegerStore, InitializeAccumulator):
                    raise CompletionRefusal('completion: unsupported projected effect')
                kinds.add(type(effect))
        else:
            raise CompletionRefusal('completion: unknown schedule statement')
    return kinds


def _counts(body, adapter):
    values = dict(loads=0, compute_pairs=0, readouts=0, drains=0)
    for op in body:
        if type(op) is Loop:
            for key, value in _counts(op.body, adapter).items():
                values[key] += op.extent * value
        elif op == Call(DRAIN, ()):
            values['drains'] += 1
        else:
            effect = adapter.project(op)
            if type(effect) is Load:
                values['loads'] += 1
            elif type(effect) is Compute:
                values['compute_pairs'] += 1
            elif type(effect) in (Store, TiledIntegerStore):
                values['readouts'] += 1
    return values


def _last_drain(body, prefix=(), multiplicity=1):
    last = None
    for i, op in enumerate(body):
        path = (*prefix, i)
        if op == Call(DRAIN, ()):
            last = path, multiplicity
        elif type(op) is Loop:
            child = _last_drain(op.body, path, multiplicity * op.extent)
            if child is not None:
                last = child
    return last


def _strip(body, keep, prefix=()):
    result = []
    for i, op in enumerate(body):
        path = (*prefix, i)
        if op == Call(DRAIN, ()) and path != keep:
            continue
        result.append(replace(op, body=_strip(op.body, keep, path)) if type(op) is Loop else op)
    return tuple(result)


def propose_dense_completion(kernel, adapter, *, mode='output_block'):
    """Return the actual deletion-only proposal and exact counted issue scopes.

    Runs stop at stores, initialization and unknown effects. A closing drain is
    an ORIGINAL singleton instance, never a newly invented completion. Repeated
    loops are not expanded and no loop or operand expression changes. The
    'period' alternative only merges adjacent singleton compute epochs.
    """
    if type(kernel) is not Kernel or check_structure(kernel) or mode not in MODES:
        raise CompletionRefusal('completion: invalid schedule or requested completion mode')
    check_integer_domain(kernel, adapter.index_bits)
    _classify(kernel.body, adapter)
    groups = []
    stream_types = {Load, Preload, Compute, 'drain'}

    def eligible(op):
        kinds = _classify((op,), adapter)
        if not kinds or not kinds <= stream_types or kinds == {'drain'}:
            return False
        if mode == 'period':
            return (type(op) is Loop and op.extent == 1 and op.body
                    and op.body[-1] == Call(DRAIN, ()) and Compute in kinds)
        return True

    def walk(body, path=(), outer=1):
        output = []
        index = 0
        while index < len(body):
            end = index
            while end < len(body) and eligible(body[end]):
                end += 1
            run = body[index:end]
            counts = _counts(run, adapter)
            closing = _last_drain(run)
            # A last static site inside a repeated loop is not one final event.
            # Recurse instead, retaining that loop's original closing boundary.
            if (mode != 'original' and counts['drains'] > 1 and counts['compute_pairs']
                    and closing is not None and closing[1] == 1
                    and _ends_in_drain(run) and Compute in _classify((run[-1],), adapter)):
                actual = _strip(run, closing[0])
                if _body_data(run) != _body_data(actual):
                    raise ValueError('completion proposal altered ordered data or counted storage operands')
                groups.append(dict(path=(*path, index), siblings=end-index,
                    dynamic_instances=outer, issued=counts, retained_closing_site=closing[0],
                    removed_drains_per_instance=counts['drains']-1))
                output.extend(actual)
                index = end
                continue
            # Recurse one statement at a time. Output blocks contain stores,
            # so no outer block loop can be merged into its neighbour.
            op = body[index]
            output.append(replace(op, body=walk(op.body, (*path,index), outer*op.extent))
                          if type(op) is Loop else op)
            index += 1
        return tuple(output)

    actual = replace(kernel, body=walk(kernel.body))
    if (check_structure(actual) or _body_data(kernel.body) != _body_data(actual.body)
            or kernel.args != actual.args or kernel.attrs != actual.attrs):
        raise ValueError('completion did not preserve the complete symbolic data schedule')
    check_integer_domain(actual, adapter.index_bits)
    return actual, tuple(groups)


def _ends_in_drain(body):
    if not body:
        return False
    last = body[-1]
    return last == Call(DRAIN, ()) or type(last) is Loop and _ends_in_drain(last.body)


def _base(admission):
    if hasattr(admission, 'overlap') and hasattr(admission, 'reduction'):
        overlap, reduction = admission.overlap, admission.reduction
        return replace(overlap, **{field:min(getattr(overlap,field),getattr(reduction,field))
            for field in ('max_issued_compute_pairs','max_issued_loads','max_issued_readouts')})
    return admission


@dataclass(frozen=True)
class CompletionIssueBudget:
    """Program-scoped compiler ISSUE bounds, never new physical queue entries.

    Counts are exact symbolic products over the selected K runs of this one
    kernel. The base execution admission/pending capacities remain unchanged.
    A selected finite-queue progress/backpressure law is separately required.
    A consumer cannot silently enlarge a supplied bound to fit its proposal.
    """
    original_admission: object
    kernel_sha256: str
    mode: str
    counted_scopes_sha256: str
    max_issued_compute_pairs: int
    max_issued_loads: int
    max_issued_readouts: int


def make_completion_issue_budget(original_kernel, adapter, admission, *, mode):
    """Construct and bind a finite issue bound from actual counted commands."""
    _, groups = propose_dense_completion(original_kernel, adapter, mode=mode)
    base = _base(admission)
    return CompletionIssueBudget(admission, original_kernel.digest(), mode, _digest(groups),
        max(base.max_issued_compute_pairs, max((g['issued']['compute_pairs'] for g in groups), default=0)),
        max(base.max_issued_loads, max((g['issued']['loads'] for g in groups), default=0)),
        base.max_issued_readouts)


def check_completion_proposal(original, actual, adapter, admission, premises, *, mode,
                              original_admission=None, storage=None, block=None, timeout_ms=30000):
    """Rebuild a deletion-only proposal, check budgets and every required law.

    The ordinary route verifier must first validate the original source and all
    physical windows. Exact data identity carries that source/storage proof to
    this program. The new ordering conjunction discharges the additional RAW,
    WAR and overlapping ACC-WAW edges exposed by deleting drains. It does not
    turn source order into a target ordering law, nor claim physical execution.
    """
    if type(premises) is not dict:
        raise CompletionRefusal('completion: plain explicit premises dictionary required')
    missing = [key for key in REQUIRED_PREMISES if premises.get(key) is not True]
    if missing:
        raise CompletionRefusal('completion: missing or unknown ordering/ownership premise: ' + ', '.join(missing))
    expected, groups = propose_dense_completion(original, adapter, mode=mode)
    if actual != expected:
        raise ValueError('completion: actual program differs from checked deletion-only proposal')
    if type(admission) is CompletionIssueBudget:
        if (admission.original_admission != original_admission
                or admission.kernel_sha256 != original.digest() or admission.mode != mode
                or admission.counted_scopes_sha256 != _digest(groups)):
            raise CompletionRefusal('completion: stale or differently scoped issue-budget proof')
        budget = admission
    else:
        if admission != original_admission:
            raise CompletionRefusal('completion: changed original admission lacks a program-scoped issue proof')
        budget = _base(admission)
    for field in ('max_issued_compute_pairs', 'max_issued_loads', 'max_issued_readouts'):
        if type(getattr(budget, field, None)) is not int or getattr(budget, field) <= 0:
            raise CompletionRefusal('completion: invalid issued budget: ' + field)
    base = _base(original_admission)
    for field in ('max_issued_compute_pairs', 'max_issued_loads', 'max_issued_readouts'):
        if getattr(budget, field) < getattr(base, field):
            raise CompletionRefusal('completion: insufficient issued budget for original epochs: ' + field)
    for group in groups:
        for field, count in (('max_issued_compute_pairs','compute_pairs'), ('max_issued_loads','loads'),
                             ('max_issued_readouts','readouts')):
            if group['issued'][count] > getattr(budget, field):
                raise CompletionRefusal('completion: insufficient issued budget: ' + field)
    if storage is None or block is None:
        raise CompletionRefusal('completion: exact original storage/block required for operand reuse ordering')
    from .dense_completion_order import verify_load_reuse_order
    reuse_order = verify_load_reuse_order(original, groups, storage, adapter, block=block, timeout_ms=timeout_ms)
    before, after = dynamic_drains(original), dynamic_drains(actual)
    removed = sum(g['dynamic_instances']*g['removed_drains_per_instance'] for g in groups)
    if before-after != removed or _body_data(original.body) != _body_data(actual.body):
        raise ValueError('completion: counted deletion or exact ordered data conservation failed')
    return dict(schema='merlin.dense_completion.v1', mode=mode, status='optimized' if removed else 'unchanged',
        source_kernel_sha256=original.digest(), actual_kernel_sha256=actual.digest(),
        data_schedule_sha256=replace(actual, body=_body_data(actual.body)).digest(), groups=groups, load_reuse_order=reuse_order,
        dynamic_drains_before=before, dynamic_drains_after=after, removed_dynamic_drains=removed,
        issue_budget=asdict(admission) if type(admission) is CompletionIssueBudget else asdict(_base(admission)),
        premises={key:premises[key] for key in REQUIRED_PREMISES},
        preserved='all exact ordered data calls, counted domains, storage, readout/initialization and closing drains',
        qualification='conditional selected target ordering and original source/ownership proofs',
        target_execution='not_established')


def module_identity():
    # Bind both the completion rewrite and its independent load-reuse checker.
    return _digest({name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('dense_completion.py', 'dense_completion_order.py', 'dense_completion_plan.py')})
