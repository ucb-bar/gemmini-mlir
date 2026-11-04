"""Optional crossover using actual checked fallback and segmented schedules.

This is a frozen performance hypothesis, not a storage or ordering grant.
The existing expansion rule runs first; this alternative only handles its
retained fallback. Neither projected load counts nor matching guards predict
cycles. Complete paired timing remains necessary before default promotion.
"""
from dataclasses import replace
from pathlib import Path
import os

from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, ResourcePredicate, execute_policy,
)

CHOICES = ('expansion', 'equal_loads')


def resolve_crossover_policy(selection=None):
    from merlin.compile.tuning_knobs import knob
    configured = knob('DENSE_CROSSOVER_POLICY', default='expansion', values=CHOICES,
        root=Path(__file__).resolve().parent.parent)
    selection = selection if selection is not None else os.environ.get('DENSE_CROSSOVER_POLICY', configured)
    if selection not in CHOICES:
        raise ValueError('DENSE_CROSSOVER_POLICY must be expansion or equal_loads')
    return selection


def execute_equal_loads_crossover(session, candidates, incumbent):
    from .dense_pipeline_policy import (
        OUTPUT, pipeline_crossover_facts, pipeline_policy_registry, pipeline_policy_session,
    )
    if session.module != candidates.source:
        raise ValueError('crossover requires the current source session')
    # The first policy session has already executed its expansion/fallback
    # pass. Start a fresh current-source session for this second selection;
    # no stale analysis or second execute on a consumed session is permitted.
    session = pipeline_policy_session(candidates.source, context=session.context)
    # These facts revalidate the supplied final incumbent, including its
    # original ABI, geometry, admission and kernel identity. Capacity-selected
    # whole-RHS retention must never be compared as a retained-LHS surrogate.
    facts = pipeline_crossover_facts(session, candidates, incumbent)
    resources = dict(facts.resources)
    resources['incumbent_not_whole_rhs_reuse'] = int(
        resources['retained_legal'] and not resources['incumbent_whole_rhs_reuse'])
    facts = replace(facts, resources=tuple(sorted(resources.items())))
    result, refusal = incumbent
    identities = dict(candidates.kernel_identities)
    identities['retained'] = None if result is None else result.kernel.digest()
    actual = replace(candidates, retained=result, retained_refusal=refusal,
        kernel_identities=tuple(identities.items()))
    actual.revalidate()
    guards = (ResourcePredicate('segmented_legal', 1),
        ResourcePredicate('segmented_operand_loads_not_increased', 1),
        ResourcePredicate('incumbent_not_whole_rhs_reuse', 1),
        ResourcePredicate('k_tiles', 4), ResourcePredicate('n_tiles', 4))
    policy = PipelinePolicy((
        PipelineVariant('segmented', ('dense_pipeline_segmented',), guards),
        PipelineVariant('retained', ('dense_pipeline_retained',))), 'retained')
    selected = execute_policy(session, policy, pipeline_policy_registry(actual), facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    receipt = dict(schema='authored_dense_equal_loads_crossover.v1',
        policy=selected.receipt, candidate_resources=resources,
        candidate_kernels=dict(actual.kernel_identities),
        candidate_refusals=dict(retained=actual.retained_refusal, segmented=actual.segmented_refusal),
        predicates=[dict(resource=p.resource, at_least=p.at_least) for p in guards],
        hypothesis='componentwise no additional operand loads, nontrivial K/N granularity, no whole-RHS incumbent reuse',
        timing_authority=False, default_promotion=False)
    return schedule, refusal, receipt
