"""Shared, verified scalar reuse and invariant motion on counted commands.

This is a representation bridge, not a schedule or completion policy. Production
consumption uses the common pipeline's host verification gate. Explicit
``plan_only=True`` is reserved for isolated development artifacts and supplies no
formal or target execution qualification.
"""
from __future__ import annotations

from collections import Counter

from merlin.compile.scheduling.scf_commands import (
    materialize_scf_commands, verify_scf_commands, emit_scf_commands,
)
from merlin.compile.scheduling.llvm_index_bounds import FnBuilderBoundedControlAdapter
from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline


def _summary(module):
    operations = tuple(module.walk())
    return {'sha256': ir_digest(module), 'operations': len(operations),
            'operation_counts': dict(sorted(Counter(op.name for op in operations).items()))}


def plan_recurrence(result, *, verifier=None, plan_only=False, changes_sink=None):
    """Return detached command IR and an account of the actual shared passes.

    The checked source schedule is revalidated before and after planning. Every
    transformed module is checked for retained command metadata, SSA completion
    edges and private-row lifetimes. Common scalar rewrites retain their exact
    before/after changes and replay witnesses; production additionally requires
    the host's accepted receipt. No target callback runs during planning.
    """
    if type(plan_only) is not bool:
        raise ValueError('plan_only must be an explicit Boolean')
    result.revalidate()
    if result.index_bits != 64:
        raise ValueError('command scalar motion requires checked i64 indices')
    original = materialize_scf_commands(result.kernel)
    source = verify_scf_commands(original)
    before = _summary(original)
    current = original
    changes = []
    passes = []
    # Shared LICM walks inner loops first, so one traversal moves chained
    # invariants through the complete existing nesting. The final CSE merges
    # expressions brought into a common dominating scope. No fixpoint policy.
    for hoist in (True, False):
        transformed = apply_common_pipeline(current, PipelineOptions(
            index_bits=64, normalize_shapes=False, simplify_integers=False,
            reuse_buffers=False, eliminate_redundancy=True,
            hoist_invariants=hoist), verifier=verifier, plan_only=plan_only)
        current = transformed.module
        verify_scf_commands(current)
        changes.extend(transformed.changes)
        passes.extend(record.name for record in transformed.passes)
    result.revalidate()
    if current.attributes != original.attributes:
        raise ValueError('scalar motion changed source schedule custody')
    if changes_sink is not None:
        changes_sink(tuple(changes))
    return current, {
        'mechanism': 'shared_scalar_redundancy_and_loop_invariant_motion',
        'source_schedule_sha256': source['source_schedule_sha256'],
        'before': before, 'after': _summary(current),
        'passes': passes, 'plan_only': plan_only,
        'host_verification': 'NOT_ESTABLISHED' if plan_only else 'accepted_by_common_pipeline',
        'changes': [{'kind': change.kind,
                     'before_sha256': ir_digest(change.before_ir),
                     'after_sha256': ir_digest(change.after_ir),
                     'witnesses': len(change.certificate.witnesses)} for change in changes],
        'scope': 'total scalar integer SSA expressions; unchanged commands and effect order',
    }


def emit_recurrence(result, builder, lower_call, *, verifier=None,
                    plan_only=False, changes_sink=None):
    """Consume the detached transformed commands through the ordinary terminal.

    Callers discard the builder on any emission failure. Async command schedules
    requiring target event callbacks are intentionally refused by the shared
    emitter; this bridge does not invent completion semantics.
    """
    module, receipt = plan_recurrence(result, verifier=verifier,
        plan_only=plan_only, changes_sink=changes_sink)
    bridge = FnBuilderBoundedControlAdapter(builder)
    receipt['emission'] = emit_scf_commands(module, bridge,
        lambda instruction, arguments, metadata: lower_call(instruction,
            {key: bridge.value(value) for key, value in arguments.items()}))
    result.revalidate()
    return receipt
