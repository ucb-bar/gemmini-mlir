"""Opt-in single-M-tile dense panel expansion over the resource incumbent.

A full-K panel can be physically legal yet serialize narrow N panels. The
experimental class admits a shared two-slot segmented schedule only when its
actual checked N panel is wider. Capacity is legality; complete-call offline
paired cycles, recorded separately, are performance evidence. No profiling or
capsule identity participates in compilation.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, ResourcePredicate, execute_policy,
)
from . import isa
from .dense_pipeline_policy import (
    DensePipelineSource, DensePipelineCandidates, OUTPUT,
    execute_dense_capacity_policy, pipeline_policy_session,
    pipeline_policy_registry, pipeline_policy_facts,
)


def execute_dense_decode_policy(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', retained_operand='traffic', storage_split='half',
        issue_order='interleaved', retained_planner=None, narrow_long_k='off'):
    """Return actual checked schedule, original refusal, and detached receipt.

    The incumbent remains the existing resource policy, including traffic
    retention and capacity refusal/streaming behavior. Only a single primitive
    M tile with a legal full-K incumbent spanning multiple N panels can enter
    this additional class. A wider *checked* segmented N panel is mandatory;
    neither source geometry alone nor the performance hypothesis grants it.
    """
    incumbent, refusal, base = execute_dense_capacity_policy(
        m, k, n, bias, column_scales, geometry=geometry,
        retained_operand=retained_operand, storage_split=storage_split,
        issue_order=issue_order, retained_planner=retained_planner)
    resources = dict(base['policy']['facts']['resources'])
    preliminary = (not column_scales and resources['m_tiles'] == 1
        and resources['retained_legal'] == 1
        and resources['retained_n_blocks'] >= 2
        and base['selection'] == 'retained'
        and resources['n_tiles'] > resources['retained_n_blocks'])
    receipt = dict(schema='authored_dense_decode_policy.v1',
        hypothesis='one M tile and strictly wider segmented N panels than the legal full-K incumbent',
        incumbent=base, preliminary_region=preliminary,
        segmented_construction='not_constructed', selection='incumbent',
        emitted_kernel=None if incumbent is None else incumbent.kernel.digest(),
        performance='opt-in offline hypothesis; no compilation timing or profiling',
        numerical_and_execution_authority='original selected schedule source, capacity and epoch checks')
    if not preliminary:
        from .dense_narrow_policy import select_dense_narrow_policy
        schedule, selected_refusal, narrow = select_dense_narrow_policy(
            incumbent, refusal, m=m,k=k,n=n,bias=bias,column_scales=column_scales,
            geometry=geometry,mode=narrow_long_k,retained_planner=retained_planner)
        if narrow is not None:
            receipt['narrow_long_k'] = narrow
            if narrow['applied']:
                receipt.update(selection='segmented', emitted_kernel=schedule.kernel.digest())
        return schedule, selected_refusal, receipt
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner = _retained_schedule
    segmented, sr = retained_planner(m, k, n, bias, column_scales,
        True, geometry, storage_split, issue_order, 'lhs')
    receipt['segmented_construction'] = 'constructed'
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split)
    identities = tuple((name, None if result is None else result.kernel.digest())
        for name, result in (('retained', incumbent), ('segmented', segmented)))
    candidates = DensePipelineCandidates(source, incumbent, refusal, segmented, sr, identities)
    candidates.revalidate()
    session = pipeline_policy_session(source)
    facts = pipeline_policy_facts(session, candidates)
    current = dict(facts.resources)
    expansion = (segmented is not None
        and current['segmented_n_blocks'] > resources['retained_n_blocks']
        and resources['n_tiles'] > current['segmented_n_blocks'])
    facts = replace(facts, resources=tuple(sorted((*facts.resources,
        ('single_m_panel_expansion', int(expansion))))))
    policy = PipelinePolicy((
        PipelineVariant('segmented', ('dense_pipeline_segmented',),
            (ResourcePredicate('single_m_panel_expansion', 1),
             ResourcePredicate('segmented_legal', 1))),
        PipelineVariant('retained', ('dense_pipeline_retained',))), 'retained')
    selected = execute_policy(session, policy, pipeline_policy_registry(candidates), facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    receipt.update(policy=selected.receipt, candidate_kernels=dict(identities),
        candidate_refusals={'retained': candidates.retained_refusal, 'segmented': sr},
        panel_expansion=expansion, selection=selected.receipt['selected_variant'],
        emitted_kernel=None if schedule is None else schedule.kernel.digest())
    return schedule, refusal, receipt


def decode_pipeline_schedule(m, k, n, bias=False, column_scales=False, *, geometry=None):
    """Two-value dense dispatch ABI for the opt-in decode_policy alternative."""
    from .dense_pipeline import PANEL_GEOMETRIES
    root = Path(__file__).resolve().parents[1]
    def setting(name, default, values):
        return os.environ.get(name, knob(name, default=default, values=values, root=root))
    geometry = geometry or setting('DENSE_PANEL_GEOMETRY', 'balanced', tuple(PANEL_GEOMETRIES))
    from .dense_verification_fallback import optional_verified_dense_plan
    schedule, refusal, receipt = optional_verified_dense_plan(
        execute_dense_decode_policy, m, k, n, bias, column_scales,
        geometry=geometry,
        retained_operand=setting('DENSE_RETAINED_OPERAND', 'lhs', ('lhs', 'weight', 'traffic')),
        storage_split=setting('DENSE_STORAGE_SPLIT', 'half', ('half', 'adaptive')),
        issue_order=setting('DENSE_SEGMENTED_ISSUE_ORDER', 'grouped', ('grouped', 'interleaved')),
        narrow_long_k=setting('DENSE_NARROW_LONG_K', 'off', ('off', 'resource_fit')))
    if schedule is not None:
        schedule.realization['dense_decode_selection'] = receipt
        # Preserve the existing dense dispatcher receipt ABI. The regional
        # "incumbent" decision still names its actual retained/segmented route.
        physical = receipt['incumbent']['selection'] if receipt['selection'] == 'incumbent' else receipt['selection']
        schedule.realization['dense_pipeline_selection'] = dict(receipt, selection=physical)
    return schedule, refusal
