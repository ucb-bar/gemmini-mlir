"""Optional wide-output storage experiment over the checked dense incumbent.

The frozen structural class is experimental, not a capacity or numerical law.
The existing physical partition planner and full source/epoch checks establish
legality. Ordinary compilation neither profiles nor simulates any candidate.
"""
from __future__ import annotations

import os
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass, ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec

from . import isa
from .dense_pipeline_policy import DensePipelineSource, LEVEL, pipeline_policy_session


OUTPUT = 'dense_storage_policy_checked_artifact'
BYTE_FIELDS = ('requested_lhs_bytes', 'requested_rhs_bytes',
               'requested_store_bytes', 'output_tiles', 'bias_initializations')


def _description(schedule):
    schedule.revalidate()
    realization = schedule.realization
    return dict(kernel_sha256=schedule.kernel.digest(),
        block=realization.get('panel_geometry', {}).get('selected_block_tiles'),
        geometry=realization.get('panel_geometry', {}).get('selection'),
        bounds=realization.get('segmented_epoch_bounds'),
        storage=realization.get('storage_partition'),
        issue=realization.get('frontier_issue', {}).get('issue_order'),
        source_check={key: realization.get('source', {}).get(key) for key in
            ('kernel_sha256', 'configuration_identity', 'numerical_policy_identity')},
        admission=realization.get('admission'),
        qualification=realization.get('qualification'),
        schedule_ir=realization.get('schedule_ir'),
        traffic={field: realization.get('source', {}).get(field) for field in BYTE_FIELDS})


def select_dense_storage_policy(schedule, attributes=None, *, m, k, n,
        bias=False, column_scales=False):
    """Return an actual checked artifact and receipt before readout transforms.

    Only original half-store segmented schedules with full-LHS/two-RHS epochs
    enter the experiment. Exact N groups are a frozen scope restriction, not
    an assertion that tails are illegal. The incumbent remains authoritative
    if the adaptive plan refuses or changes blocking or logical traffic.
    """
    attributes = attributes or {}
    root = Path(__file__).resolve().parents[1]
    selection = attributes.get('dense_storage_policy', os.environ.get(
        'DENSE_STORAGE_POLICY', knob('DENSE_STORAGE_POLICY', default='off',
        values=('off', 'wide_output'), root=root)))
    if selection not in ('off', 'wide_output'):
        raise ValueError('unknown dense storage policy')
    if selection == 'off':
        return schedule, None
    original = _description(schedule)
    mt, kt, nt = ((value+isa.DIM-1)//isa.DIM for value in (m, k, n))
    block, bounds = original['block'], original['bounds']
    preliminary = bool(not column_scales and mt == 1 and nt > kt and block
        and block[1] < kt and nt > block[2] and nt % block[2] == 0
        and bounds and bounds['rhs_slots'] == 2
        and bounds['retained_lhs_rows'] == block[0]*kt*isa.DIM
        and original['storage'] and original['storage']['selection'] == 'half'
        and original['issue'] == 'prefetch_rhs_between_k_tiles')
    report = dict(schema='authored_dense_storage_policy.v1', selection=selection,
        hypothesis='wide exact output groups preserve N blocking and bytes while reducing K ownership epochs',
        preliminary_region=preliminary, candidate_construction='not_constructed',
        source=original, candidate=None, applied=False,
        source_axes=dict(m=m, k=k, n=n, bias=bias, column_scales=column_scales),
        structural_scope='one primitive M tile; Ntiles>Ktiles; complete incumbent N groups',
        legality='original actual source, physical partition and combined epoch admission',
        performance='UNKNOWN until matched offline complete-call timing; no compile-time profiling')
    if not preliminary:
        report['refusal'] = 'outside frozen structural experiment scope; incumbent preserved'
        return schedule, report
    from .dense_pipeline import _retained_schedule
    from .dense_verification_fallback import optional_verified_dense_plan

    def construct():
        result, refusal = _retained_schedule(m, k, n, bias, column_scales,
            True, original['geometry'], 'adaptive', 'interleaved', 'lhs')
        return result, refusal, None

    candidate, refusal, _ = optional_verified_dense_plan(construct)
    report['candidate_construction'] = 'constructed'
    report['refusal'] = refusal
    if candidate is None:
        return schedule, report
    alternate = _description(candidate)
    report['candidate'] = alternate
    cb = alternate['block']
    old_segments = (kt+block[1]-1)//block[1]
    new_segments = (kt+cb[1]-1)//cb[1]
    same_traffic = (all(value is not None for value in original['traffic'].values())
        and alternate['traffic'] == original['traffic'])
    admitted = bool(cb[0] == block[0] and cb[2] == block[2]
        and cb[1] > block[1] and new_segments < old_segments and same_traffic
        and candidate.kernel.args == schedule.kernel.args
        and alternate['issue'] == original['issue'])
    report['comparison'] = dict(same_logical_traffic=same_traffic,
        original_k_segments=old_segments, candidate_k_segments=new_segments,
        same_m_n_block=cb[::2] == block[::2], strict_epoch_reduction=admitted)
    source = DensePipelineSource(m, k, n, bias, column_scales, original['geometry'], 'half')
    session = pipeline_policy_session(source)
    facts = PolicyFacts(session.stamp(), shapes=((m, k), (k, n)),
        element_types=('i8', 'i8'), layouts=('row_major', 'row_major'),
        operations=frozenset({'dense'}),
        resources=(('wide_exact_same_traffic_epoch_reduction', int(admitted)),))
    artifacts = dict(incumbent=schedule, adaptive=candidate)

    def registered(name):
        def action(current):
            if current.module != source:
                raise ValueError('storage policy cannot reuse another source artifact')
            artifacts[name].revalidate()
            return {OUTPUT: artifacts[name]}
        key = 'dense_storage_'+name
        return key, RegisteredPass(PassSpec(key, 0, produces=(OUTPUT,),
            mutates_ir=False, ir_level=LEVEL), action)

    policy = PipelinePolicy((PipelineVariant('adaptive', ('dense_storage_adaptive',),
        (ResourcePredicate('wide_exact_same_traffic_epoch_reduction', 1),)),
        PipelineVariant('incumbent', ('dense_storage_incumbent',))), 'incumbent')
    selected = execute_policy(session, policy,
        dict(registered(name) for name in artifacts), facts, output=OUTPUT)
    result = selected.artifact
    result.revalidate()
    report.update(applied=selected.receipt['selected_variant'] == 'adaptive',
        policy=selected.receipt, emitted_kernel_sha256=result.kernel.digest())
    result.realization['dense_storage_selection'] = report
    return result, report
