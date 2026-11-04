"""Optional checked full-K scratchpad RHS reuse across complete M panels.

This uses the library's single-product M-panel constructor, not array-weight
retention. The original source/assigned windows/ordered reduction law and final
integer source/lifetime checks remain authoritative. Complete M families share
immutable staged RHS rows; separate M tails retain their own checked loads.
"""
from dataclasses import replace
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.dense_panel_pipeline import (
    DenseRetainedReductionAdmission, dense_retained_choices, schedule_dense_m_panels,
)
from .dense_verification import verify_dense_with_budget


class _ConstructionRefused(Exception):
    def __init__(self, feedback):
        self.feedback = feedback


def enabled():
    return knob('DENSE_RHS_RESIDENCY', default='off', values=('off', 'on'),
                root=Path(__file__).resolve().parents[1]) == 'on'


def select(schedule, *, problem, resources, storage, adapter, admission,
           block, source_identity, bias_source=None):
    """Return the actual checked result/block; preserve the incumbent on refusal.

    The selected alternative is bounded by the caller's original geometric and
    physical domains. Choice counts include full/tail families, not cycles.
    Issued-operation bounds describe the selected finite epoch; pending queue
    capacities and every selected target premise remain unchanged.
    """
    schedule.revalidate()
    original_source = schedule.realization['source']
    original_loads = original_source['rhs_loads']
    original_operand_bytes = (original_source['requested_lhs_bytes']
                              + original_source['requested_rhs_bytes'])
    choices = dense_retained_choices(problem, resources, storage, roles=('weight',))
    if (type(admission) is not DenseRetainedReductionAdmission or not choices
            or choices[0].rhs_loads >= original_loads):
        schedule.realization['rhs_residency'] = dict(applied=False,
            refusal='no admitted full-K RHS reuse across complete M panels')
        return schedule, block
    choice = choices[0]
    # One-dimensional output panels can exchange a small immutable RHS reload
    # for many additional ownership epochs. Keep their existing batching; this
    # alternative is for reuse across two-dimensional output panels.
    if choice.block.m < 2 or choice.block.n < 2:
        schedule.realization['rhs_residency'] = dict(applied=False,
            refusal='RHS residency requires a two-dimensional output panel',
            proposed_block_tiles=[choice.block.m,choice.block.k,choice.block.n])
        return schedule, block
    # Residency exchanges LHS reuse for RHS reuse.  Require a strict reduction
    # in the whole counted source's operand bytes, including remainder families;
    # fewer RHS loads alone is insufficient.  This profitability filter grants
    # no legality: the selected product still passes the original full proof.
    if choice.requested_operand_bytes >= original_operand_bytes:
        schedule.realization['rhs_residency'] = dict(applied=False,
            refusal='no strict total LHS+RHS operand-byte reduction',
            original_operand_bytes=original_operand_bytes,
            proposed_operand_bytes=choice.requested_operand_bytes)
        return schedule, block
    proposed = choice.block
    # The existing source epoch has no larger physical queue assumptions.
    selected_admission = replace(admission,
        max_issued_compute_pairs=max(1, proposed.m*proposed.k*proposed.n),
        max_issued_loads=max(1, (proposed.m+proposed.n)*proposed.k))
    original_kernel = schedule.kernel.digest()

    def build(timeout):
        attempt = schedule_dense_m_panels(problem, resources, storage, adapter,
            selected_admission, n_tiles=proposed.n, m_tiles=proposed.m,
            name='dense_rhs_resident', dtypes=('i8','i8','i32'),
            source_identity=source_identity, external_buffers_disjoint=True,
            static_budget=128, timeout_ms=timeout, bias_source=bias_source)
        if attempt.result is None:
            raise _ConstructionRefused(attempt.feedback)
        actual_source = attempt.result.realization['source']
        actual_operand_bytes = (actual_source['requested_lhs_bytes']
                                + actual_source['requested_rhs_bytes'])
        if actual_operand_bytes != choice.requested_operand_bytes:
            raise ValueError('RHS residency candidate/actual operand-byte mismatch')
        # The common bounded verifier expects the actual checked artifact.
        attempt.result.realization['rhs_residency'] = dict(applied=True,
            mechanism='library schedule_dense_m_panels', storage='original assigned scratchpad windows',
            residency='full-K RHS N-panel across complete M groups; tail family reloads',
            array_weight_retention=False, original_kernel_sha256=original_kernel,
            source_kernel_sha256=attempt.result.kernel.digest(),
            original_rhs_loads=original_loads, actual_rhs_loads=attempt.result.realization['source']['rhs_loads'],
            actual_lhs_loads=attempt.result.realization['source']['lhs_loads'],
            original_operand_bytes=original_operand_bytes,
            actual_operand_bytes=actual_operand_bytes,
            profitability='strict total original-source LHS+RHS operand-byte reduction',
            selected_block_tiles=[proposed.m,proposed.k,proposed.n],
            source_order='original increasing K per output; independent output groups N then M',
            target_premise=selected_admission.execution,
            qualification_identity=selected_admission.qualification_identity,
            pending_compute_pairs=selected_admission.pending_compute_pairs,
            pending_loads=selected_admission.pending_loads, construction=attempt.feedback)
        return attempt.result

    try:
        result = verify_dense_with_budget(build, initial_timeout_ms=5000)
    except _ConstructionRefused as refusal:
        schedule.realization['rhs_residency'] = dict(applied=False, refusal=refusal.feedback)
        return schedule, block
    result.revalidate()
    return result, proposed
