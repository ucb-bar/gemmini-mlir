"""Opt-in long-reduction, single-output-tile shared segmented alternative.

The frozen performance hypothesis is that a full-K operand consuming more than
half its assigned half-SPAD window benefits from bounded RHS issue epochs.
Capacity is independent of source/epoch legality; both actual alternatives must
retain their original complete checks. No capsule identity or profiling enters
selection, and no terminal encoder is duplicated here.
"""
from __future__ import annotations

from dataclasses import replace
from merlin.transforms.pipeline_policy import PipelinePolicy, PipelineVariant, ResourcePredicate, execute_policy
from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
from . import isa
from .dense_pipeline_policy import (
    DensePipelineSource, DensePipelineCandidates, OUTPUT, pipeline_policy_session,
    pipeline_policy_registry, pipeline_policy_facts,
)


def select_dense_narrow_policy(incumbent, refusal, *, m, k, n, bias=False,
        column_scales=False, geometry='wide', mode='off', retained_planner=None):
    """Keep the checked incumbent unless the frozen opt-in class is admitted."""
    if mode not in ('off', 'resource_fit'):
        raise ValueError('unknown narrow long-K dense policy')
    if mode == 'off':
        return incumbent, refusal, None
    from .dense_pipeline import half_retained_capacity, _retained_schedule
    _, _, _, capacity = half_retained_capacity(m,k,n,bias,column_scales,geometry)
    mt,kt,nt = ((value+isa.DIM-1)//isa.DIM for value in (m,k,n))
    # Resource-relative cutoff frozen before checking neighbors: each complete
    # operand occupies >1/2 of its assigned half-SPAD window, but still fits.
    threshold = (isa.SPAD_ROWS//2)//(2*isa.DIM)
    rb = None
    if incumbent is not None:
        incumbent.revalidate()
        rb = incumbent.realization['panel_geometry']['selected_block_tiles']
    preliminary = bool(not column_scales and mt == nt == 1 and kt > threshold
        and capacity is not None and (capacity.m,capacity.k,capacity.n)==(1,kt,1)
        and rb == [1,kt,1]
        and incumbent.realization.get('storage_partition',{}).get('selection') == 'half')
    receipt = dict(schema='authored_dense_narrow_policy.v1', mode=mode,
        hypothesis='single M/N tile; full-K operand exceeds half its assigned half-SPAD window',
        source_axes=dict(m=m,k=k,n=n,bias=bias,column_scales=column_scales),
        k_tiles_threshold_exclusive=threshold,
        capacity_block=None if capacity is None else [capacity.m,capacity.k,capacity.n],
        preliminary_region=preliminary, candidate_construction='not_constructed',
        incumbent_kernel=None if incumbent is None else incumbent.kernel.digest(),
        candidate_kernel=None, applied=False,
        numerical_and_execution_authority='both actual original source and epoch admissions',
        performance='opt-in frozen hypothesis; complete paired evidence remains separate')
    if not preliminary:
        return incumbent, refusal, receipt
    planner = retained_planner or _retained_schedule
    from .dense_narrow_geometry import narrow_geometry_schedule, narrow_k_tile_setting
    selected_k_tiles = narrow_k_tile_setting()
    try:
        segmented, sr, selected_k_tiles = narrow_geometry_schedule(
            m,k,n,bias,column_scales,geometry,planner=planner,k_tiles=selected_k_tiles)
    except DenseVerificationUnavailable as error:
        segmented, sr = None, 'segmented candidate proof unavailable: '+str(error)
    receipt.update(candidate_construction='constructed',candidate_refusal=sr,
        requested_segment_k_tiles=selected_k_tiles)
    source = DensePipelineSource(m,k,n,bias,column_scales,geometry,'half')
    identities=tuple((name,None if result is None else result.kernel.digest())
        for name,result in (('retained',incumbent),('segmented',segmented)))
    candidates=DensePipelineCandidates(source,incumbent,refusal,segmented,sr,identities)
    candidates.revalidate()
    session=pipeline_policy_session(source)
    facts=pipeline_policy_facts(session,candidates)
    # pipeline_policy_facts verifies the actual full-LHS/two-RHS epoch bounds.
    eligible = bool(segmented is not None and
        segmented.realization.get('segmented_issue_order') == 'interleaved' and
        segmented.realization['panel_geometry']['selected_block_tiles'][::2] == [1,1])
    facts=replace(facts,resources=tuple(sorted((*facts.resources,('narrow_long_k_fit',int(eligible))))))
    policy=PipelinePolicy((
        PipelineVariant('segmented',('dense_pipeline_segmented',),
            (ResourcePredicate('narrow_long_k_fit',1),ResourcePredicate('segmented_legal',1))),
        PipelineVariant('retained',('dense_pipeline_retained',))), 'retained')
    selected=execute_policy(session,policy,pipeline_policy_registry(candidates),facts,output=OUTPUT)
    schedule, selected_refusal=selected.artifact
    receipt.update(policy=selected.receipt,candidate_kernel=dict(identities)['segmented'],
        applied=selected.receipt['selected_variant']=='segmented',
        emitted_kernel=None if schedule is None else schedule.kernel.digest(),
        realization=None if segmented is None else dict(segmented.realization))
    return schedule, selected_refusal, receipt
