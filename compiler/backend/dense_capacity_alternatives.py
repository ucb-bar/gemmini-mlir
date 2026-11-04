"""Opt-in checked segmented/full-K tradeoff, without runtime measurement.

Both variants are artifacts of the existing planner and preserve its source,
assigned storage, completion admission and terminal encoder. Capacity is only
legality; it makes no performance claim.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PipelineContext


def execute_segmented_alternative(m, k, n, bias, column_scales, retained, refusal,
                                  *, ready_readout=False, storage_split='half'):
    from . import isa
    from .dense_pipeline import retained_schedule
    from .dense_pipeline_policy import (
        DensePipelineSource, DensePipelineCandidates, pipeline_policy_session,
        pipeline_policy_facts, pipeline_policy_registry, OUTPUT,
    )
    # Never turn an incumbent refusal or another source ABI into admission.
    if retained is None or column_scales:
        return retained, refusal, {'selected': 'retained', 'reason': 'incumbent unavailable or column scales'}
    retained.revalidate()
    geometry = retained.realization['panel_geometry']['selection']
    block = retained.realization['panel_geometry']['selected_block_tiles']
    if block[1] != (k + isa.DIM - 1) // isa.DIM:
        return retained, refusal, {'selected': 'retained', 'reason': 'incumbent is not full K'}
    # This is the exact same original segmented planner, with current selected
    # issue, placement, transfer and readout consumers; no hand-authored stream.
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    try:
        segmented, segmented_refusal = retained_schedule(
            m, k, n, bias, column_scales, True,
            geometry=geometry, ready_readout=ready_readout)
    except DenseVerificationUnavailable as exc:
        # An unavailable original source proof is not a legal candidate. Keep
        # its exact refusal and let the existing registered fallback execute.
        segmented, segmented_refusal = None, type(exc).__name__ + ': ' + str(exc)
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split)
    candidates = DensePipelineCandidates(source, retained, refusal, segmented,
        segmented_refusal, tuple((name, None if value is None else value.kernel.digest())
        for name, value in (('retained', retained), ('segmented', segmented))))
    candidates.revalidate()
    ordering = Path(__file__).with_name('dense_ordering.json')
    context = PipelineContext(resources=sha256(json.dumps({
        'dim': isa.DIM, 'spad_rows': isa.SPAD_ROWS, 'acc_rows': isa.ACC_ROWS,
        'geometry': geometry, 'retained_block': block,
        'retained_kernel': retained.kernel.digest(),
        'segmented_kernel': None if segmented is None else segmented.kernel.digest(),
    }, sort_keys=True).encode()).hexdigest(), execution=sha256(ordering.read_bytes()).hexdigest())
    session = pipeline_policy_session(source, context=context)
    original = pipeline_policy_facts(session, candidates)
    resources = dict(original.resources)
    resources.update(rhs_panel_expansion=int(segmented is not None and
        segmented.realization['panel_geometry']['selected_block_tiles'][2] > block[2]),
        distinct_schedule=int(segmented is not None and segmented.kernel.digest() != retained.kernel.digest()))
    facts = PolicyFacts(session.stamp(), shapes=original.shapes,
        element_types=original.element_types, layouts=original.layouts,
        operations=original.operations, resources=tuple(sorted(resources.items())))
    policy = PipelinePolicy((
        PipelineVariant('segmented', ('dense_pipeline_segmented',), (
            ResourcePredicate('retained_legal', 1),
            ResourcePredicate('segmented_legal', 1),
            ResourcePredicate('rhs_panel_expansion', 1),
            ResourcePredicate('distinct_schedule', 1))),
        PipelineVariant('retained', ('dense_pipeline_retained',))), fallback='retained')
    selected = execute_policy(session, policy, pipeline_policy_registry(candidates), facts, output=OUTPUT)
    schedule, selected_refusal = selected.artifact
    name = 'segmented' if schedule is segmented else 'retained'
    receipt = {'selected': name, 'family': 'segmented_if_legal',
        'resources': resources, 'candidate_kernels': dict(candidates.kernel_identities),
        'candidate_refusals': {'retained': refusal, 'segmented': segmented_refusal},
        'policy': selected.receipt}
    if schedule is not None:
        schedule.realization['capacity_family_alternative'] = receipt
    return schedule, selected_refusal, receipt
