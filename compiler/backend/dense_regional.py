"""Provisional regional choice over the existing checked dense artifacts.

This optional compiler policy is not an assessed_shape_policy qualification.
It neither manufactures observation bindings nor replaces either planner's
source, storage, ordering or numerical checks. The caller retains terminal
emission and the ordinary handling of a retained capacity refusal.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

from merlin.transforms.pipeline_policy import (
    DimensionRef, FeaturePredicate, PipelinePolicy, PipelineVariant,
    ResourcePredicate, ResourceRef, TileFeature, execute_policy,
)
from merlin.transforms.pipeline_state import PipelineContext

from . import isa
from .dense_pipeline_policy import (
    OUTPUT, DensePipelineSource, pipeline_policy_candidates,
    pipeline_policy_facts, pipeline_policy_registry, pipeline_policy_session,
)


def regional_pipeline_policy():
    """Describe the evidenced capacity region with existing typed predicates."""
    return PipelinePolicy((
        PipelineVariant('segmented', ('dense_pipeline_segmented',), (
            FeaturePredicate(TileFeature(DimensionRef(0, 0), ResourceRef('dim')),
                             at_least=(1, 1), at_most=(1, 1)),
            ResourcePredicate('uniform_column_readout', 1),
            ResourcePredicate('retained_legal', 1),
            ResourcePredicate('segmented_legal', 1),
            ResourcePredicate('rhs_capacity_starved', 1),
            ResourcePredicate('n_tiles', 8),
        )),
        PipelineVariant('retained', ('dense_pipeline_retained',)),
    ), fallback='retained')


def _selected_resource_context(source):
    # An identity of the current authored resource view, not an assertion of
    # hardware/execution authority. Actual legality remains with both planners.
    selected = dict(dim=isa.DIM, spad_rows=isa.SPAD_ROWS, acc_rows=isa.ACC_ROWS,
                    geometry=source.geometry, storage_split=source.storage_split)
    identity = sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    return PipelineContext(resources='authored_dense_resource_view:'+identity)


def execute_dense_regional(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', storage_split='half', context=None, retained_planner=None):
    """Return ``(checked_schedule_or_None, original_refusal, receipt)``.

    Geometry and storage selection pass unchanged to the existing planners.
    Unknown caller resource context selects retained. If that planner refuses,
    its original refusal reaches the caller; this helper does not retry using
    streaming, compact or a different storage partition.
    """
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split)
    if context is None:
        context = _selected_resource_context(source)
    session = pipeline_policy_session(source, context=context)
    candidates = pipeline_policy_candidates(source, retained_planner=retained_planner)
    # Existing facts validate full-LHS/two-disjoint-RHS epoch bounds, actual
    # retained/segmented block widths and current kernel source signatures.
    facts = pipeline_policy_facts(session, candidates)
    # This is source-selected readout geometry, not a hardware capability.
    facts = replace(facts, resources=tuple(sorted((*facts.resources,
        ('uniform_column_readout', int(not source.column_scales))))))
    selected = execute_policy(session, regional_pipeline_policy(),
        pipeline_policy_registry(candidates), facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    receipt = dict(
        schema='authored_dense_regional.v1',
        qualification='provisional regional option; not assessed_shape_policy qualification',
        policy=selected.receipt,
        candidate_kernels=dict(candidates.kernel_identities),
        candidate_refusals=dict(retained=candidates.retained_refusal,
                                segmented=candidates.segmented_refusal),
        refusal=refusal,
        realization=None if schedule is None else dict(schedule.realization),
    )
    return schedule, refusal, receipt
