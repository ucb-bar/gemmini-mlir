"""Opt-in bounded-output panel policy; production defaults are unchanged.

This is a frozen resource hypothesis, not a calibrated performance cutoff. The
only timing observation motivating it covers one geometry. Every successful
selection returns the existing library's checked panel-epochs artifact for the
ordinary dense terminal; no instructions, source copies or arithmetic live here.
"""
from dataclasses import asdict
from pathlib import Path

from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy)
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
from gemmini_dense_coalesced_policy import _raw_source_supported, _digest
from gemmini_dense_readout import ReadoutUnavailable


_LIMITS = (
    ('original_raw_source_supported', 1), ('source_bound_adapter', 1),
    ('single_m_tile', 1), ('output_image_fits', 1), ('n_panels', 2),
    ('reduction_tiles_cover_output_tiles', 1),
    ('retained_lhs_half_store_fits', 1),
    ('full_k_rhs_double_window_overflows', 1), ('two_window_block_fits', 1))


def predicates():
    return tuple(ResourcePredicate(name, at_least=minimum) for name, minimum in _LIMITS)


def design():
    body = dict(schema='gemmini.dense.short_panel_policy.v1',
        status='opt-in hypothesis; one matched provisional modeled observation, regional cutoff unproven',
        source_boundary='original contiguous signed i8*i8 -> raw i32; zero seed, no bias, scales, activation, shift or epilogue; original m,n order',
        predicates=[asdict(p) for p in predicates()],
        resources=dict(single_m_tile='source M tiles == 1',
            output_image_fits='source M tiles * source N tiles * selected edge <= accumulator rows',
            n_panels='ceil(source N tiles / original checked N block) >= 2',
            reduction_tiles_cover_output_tiles='source K tiles >= source N tiles',
            retained_lhs_half_store_fits='2 * original M block * source K tiles * edge <= operand rows',
            full_k_rhs_double_window_overflows='2 * original N block * source K tiles * edge > operand rows - retained LHS rows',
            two_window_block_fits='original K block == 1 and two original M/N blocks fit their disjoint operand halves'),
        pass_order=['original checked compact dense source/block',
            'selected ordering premises and physical issue capacities',
            'make_dense_schedule_plan(pipeline_prefetch=True)',
            'schedule_dense_pipeline(batch_readouts=True, epoch_law=DensePanelEpochLaw)',
            'revalidate', 'unchanged ordinary dense terminal'],
        logical_static_bounds=4096,
        physical_issue_capacity='derived by existing ordering_premises; logical bounds are not queue slots',
        fallback='caller supplied ordinary automatic planner, with all original arguments and options',
        integration='attempt only in automatic mode, checked override absent and explicit GEMV override off; before long-K selection; explicit schedule/N16 overrides remain caller-owned',
        performance='UNKNOWN outside the single observed geometry; no source identity or feedback consulted while compiling')
    return {**body, 'design_sha256': _digest(body)}


def features(incumbent, adapter, problem, geometry, *, bias=False, initial=False,
        attrs=None, output_dtype='i32', outer_order=('m', 'n')):
    edge, operand, operand_rows, accumulator, accumulator_rows = geometry
    block = incumbent.choice.block
    lhs_rows = block.m * problem.tiles[1] * edge
    split = operand_rows // 2
    return dict(original_raw_source_supported=int(_raw_source_supported(problem,
            attrs=dict(attrs or {}), bias=bias, initial=initial,
            output_dtype=output_dtype, outer_order=outer_order)),
        source_bound_adapter=int(adapter.numerical_policy_identity.startswith('source-ordered-i8-i8-i32:')),
        single_m_tile=int(problem.tiles[0] == 1),
        output_image_fits=int(problem.tiles[0] * problem.tiles[2] * edge <= accumulator_rows),
        n_panels=(problem.tiles[2] + block.n - 1) // block.n,
        reduction_tiles_cover_output_tiles=int(problem.tiles[1] >= problem.tiles[2]),
        retained_lhs_half_store_fits=int(2 * lhs_rows <= operand_rows),
        full_k_rhs_double_window_overflows=int(2 * block.n * problem.tiles[1] * edge > operand_rows - lhs_rows),
        two_window_block_fits=int(block.k == 1 and 2 * block.m * edge <= split
            and 2 * block.n * edge <= operand_rows - split))


def plan_short_panel(incumbent, adapter, problem, geometry, facts_path, *,
        bias=False, initial=False, attrs=None, output_dtype='i32', outer_order=('m', 'n')):
    """Return (checked artifact, 'panel_epochs', receipt), or safe refusal.

    The caller retains the original checked seed for fallback. Apply before the
    existing automatic long-K branch so that an already selected retained plan
    does not accidentally become the source seed of this different alternative.
    """
    from gemmini_dense_pipeline import plan_pipeline
    incumbent.revalidate()
    resources = features(incumbent, adapter, problem, geometry, bias=bias,
        initial=initial, attrs=attrs, output_dtype=output_dtype, outer_order=outer_order)
    failed = [name for name, minimum in _LIMITS if resources[name] < minimum]
    if failed:
        raise ReadoutUnavailable('short panel region refused: ' + ', '.join(failed))
    try:
        result, receipt = plan_pipeline(incumbent, adapter, problem, geometry,
            facts_path, mode='panel_epochs', bias=bias, initial=initial, outer_order=outer_order)
        result.revalidate()
    except ValueError as error:
        raise ReadoutUnavailable('short panel owner refused: ' + str(error)) from error
    receipt = {**receipt, 'short_panel_policy': dict(design=design(), resources=resources,
        source_problem=asdict(problem), original_block=asdict(incumbent.choice.block),
        original_schedule_sha256=incumbent.kernel.digest(),
        transformed_schedule_sha256=result.kernel.digest(),
        selected_facts_sha256=_digest(Path(facts_path).read_bytes()),
        qualification_identity=None, timing='UNKNOWN; construction is not cycle evidence')}
    return result, 'panel_epochs', receipt


def short_auto_schedule(ordinary_scheduler, incumbent, adapter, problem, geometry,
        facts_path, *, bias=False, initial=False, attrs=None, output_dtype='i32',
        outer_order=('m', 'n'), **ordinary_options):
    """Select real checked panel epochs or call the ordinary scheduler unchanged.

    ordinary_scheduler accepts the same source arguments and returns a checked
    artifact, ordinary mode and receipt. It can enclose the entire incumbent
    automatic selection sequence. This wrapper never catches fallback failures.
    """
    arguments = dict(bias=bias, initial=initial, attrs=attrs,
        output_dtype=output_dtype, outer_order=outer_order)
    candidate = None
    refusal = None
    try:
        candidate = plan_short_panel(incumbent, adapter, problem, geometry, facts_path, **arguments)
    except ReadoutUnavailable as error:
        refusal = str(error)
    resources = features(incumbent, adapter, problem, geometry, **arguments)
    resources['owner_admitted'] = int(candidate is not None)
    context = PipelineContext(hardware=_digest(Path(facts_path).read_bytes()),
        numerical_policy=adapter.numerical_policy_identity, resources=_digest(resources))
    session = PipelineSession(dict(problem=asdict(problem), source=incumbent.kernel.digest()),
        context=context, ir_level='checked_dense', fingerprint=_digest, verify_ir=lambda _: None)
    facts = PolicyFacts(session.stamp(), shapes=((problem.m, problem.k), (problem.k, problem.n)),
        operations=frozenset({'dense'}), resources=tuple(resources.items()))
    def panel(_):
        candidate[0].revalidate()
        return {'selection': candidate}
    def fallback(_):
        return {'selection': ordinary_scheduler(incumbent, adapter, problem, geometry,
            facts_path, **arguments, **ordinary_options)}
    registry = {name: RegisteredPass(PassSpec(name, 0, produces=('selection',),
        mutates_ir=False, ir_level='checked_dense'), action)
        for name, action in (('panel_epochs', panel), ('incumbent', fallback))}
    policy = PipelinePolicy((PipelineVariant('panel_epochs', ('panel_epochs',),
        predicates() + (ResourcePredicate('owner_admitted', at_least=1),)),
        PipelineVariant('incumbent', ('incumbent',))), 'incumbent')
    selected = execute_policy(session, policy, registry, facts, output='selection')
    result, mode, receipt = selected.artifact
    return result, mode, {**(receipt or {}), 'short_panel_selection': dict(
        design_sha256=design()['design_sha256'], resources=resources,
        selected_policy=selected.receipt, candidate_refusal=refusal)}
