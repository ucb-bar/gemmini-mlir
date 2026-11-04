"""Optional development policy for short, wide multirow dense products.

No default hook or emission lives here. Geometry thresholds are a frozen
performance hypothesis; the existing checked coalesced owner proves legality.
No timing or source/capsule identity participates in selection.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy)
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
from gemmini_dense_readout import ReadoutUnavailable
from gemmini_dense_coalesced_policy import _raw_source_supported, _digest


def predicates():
    return tuple(ResourcePredicate(name, at_least=minimum) for name, minimum in (
        ('original_raw_source_supported', 1), ('short_multirow', 1),
        ('wide_output', 1), ('long_k_window', 1),
        ('retained_lhs_half_store_fits', 1), ('k_segments', 2),
        ('wide_tiles_per_transfer', 2), ('current_source_owner_admitted', 1)))


def design():
    body = dict(schema='gemmini.dense.small_wide_coalesced_design.v1',
        predicates=[asdict(p) for p in predicates()],
        empirical_region=dict(short_multirow='1 < M <= 4 * selected DIM',
            wide_output='N >= 2 * K',
            long_k_window='16 * selected DIM <= K <= 128 * selected DIM'),
        structural_guards=dict(retained_lhs_half_store_fits=
            '2 * incumbent M tiles * source K tiles * DIM <= selected operand rows',
            current_source_owner_admitted='existing original-source/epoch checked segmented transfer owner',
            k_segments='actual segmented owner has at least two K segments',
            wide_tiles_per_transfer='selected DMA/decoded MVIN2 width permits at least two tiles'),
        source_boundary='original contiguous signed i8*i8 -> raw i32; no bias, initializer, scale or activation',
        selected_path='plan_transfers -> unchanged ordinary dense terminal, segmented_ready',
        fallback='unchanged m1_coalesced_auto_schedule',
        status='development hypothesis; reserved validation pending; not promoted')
    return {**body, 'design_sha256': _digest(body)}


def small_wide_coalesced_auto_schedule(incumbent, adapter, problem, geometry, facts_path, *,
        bias=False, initial=False, attrs=None, output_dtype='i32', outer_order=('m', 'n')):
    """Return checked artifact, ordinary mode and receipt, preserving fallback.

    Candidate construction is attempted only inside the frozen geometric and
    storage region. Source/capacity/ordering refusal retains the incumbent owner.
    The caller must consume the returned artifact with the common dense encoder.
    """
    from gemmini_dense_coalesced_policy import m1_coalesced_auto_schedule
    from gemmini_dense_transfers import plan_transfers
    incumbent.revalidate()
    attrs = dict(attrs or {})
    edge, operand, operand_rows, accumulator, accumulator_rows = geometry
    lhs_rows = incumbent.choice.block.m * problem.tiles[1] * edge
    resources = dict(
        original_raw_source_supported=int(_raw_source_supported(problem, attrs=attrs,
            bias=bias, initial=initial, output_dtype=output_dtype, outer_order=outer_order)),
        short_multirow=int(1 < problem.m <= 4 * edge),
        wide_output=int(problem.n >= 2 * problem.k),
        long_k_window=int(16 * edge <= problem.k <= 128 * edge),
        retained_lhs_half_store_fits=int(2 * lhs_rows <= operand_rows),
        k_segments=0, wide_tiles_per_transfer=0, current_source_owner_admitted=0)
    candidate = None
    candidate_receipt = None
    refusal = None
    if all(resources[name] for name in ('original_raw_source_supported',
            'short_multirow', 'wide_output', 'long_k_window', 'retained_lhs_half_store_fits')):
        try:
            candidate, candidate_receipt = plan_transfers(incumbent, adapter, problem,
                geometry, facts_path, bias=bias, initial=initial, outer_order=outer_order)
            candidate.revalidate()
            resources.update(k_segments=candidate_receipt['k_segments'],
                wide_tiles_per_transfer=candidate_receipt['max_tiles'], current_source_owner_admitted=1)
        except ReadoutUnavailable as error:
            refusal = str(error)
    selected_facts = Path(facts_path).read_bytes()
    context = PipelineContext(hardware=_digest(selected_facts),
        numerical_policy=adapter.numerical_policy_identity,
        resources=_digest(dict(features=resources, edge=edge, operand_rows=operand_rows,
            accumulator_rows=accumulator_rows)))
    session = PipelineSession(dict(problem=asdict(problem),
        transformed_schedule=None if candidate is None else candidate.kernel.digest()),
        context=context, ir_level='checked_dense', fingerprint=_digest, verify_ir=lambda _: None)
    facts = PolicyFacts(session.stamp(), shapes=((problem.m, problem.k), (problem.k, problem.n)),
        operations=frozenset({'dense'}), resources=tuple(resources.items()))
    def coalesced(_):
        candidate.revalidate()
        return {'selection': (candidate, 'segmented_ready', candidate_receipt)}
    def fallback(_):
        return {'selection': m1_coalesced_auto_schedule(incumbent, adapter, problem,
            geometry, facts_path, bias=bias, initial=initial, attrs=attrs,
            output_dtype=output_dtype, outer_order=outer_order)}
    registry = {name: RegisteredPass(PassSpec(name, 0, produces=('selection',),
        mutates_ir=False, ir_level='checked_dense'), action)
        for name, action in (('small_wide_coalesced', coalesced), ('auto_ready', fallback))}
    policy = PipelinePolicy((PipelineVariant('small_wide_coalesced',
        ('small_wide_coalesced',), predicates()), PipelineVariant('auto_ready', ('auto_ready',))), 'auto_ready')
    selection = execute_policy(session, policy, registry, facts, output='selection')
    result, mode, receipt = selection.artifact
    return result, mode, {**(receipt or {}), 'small_wide_coalesced_policy': dict(
        design=design(), resources=resources, selected_policy=selection.receipt,
        candidate_refusal=refusal, selected_facts_sha256=_digest(selected_facts),
        retained_lhs_rows=lhs_rows, qualification_identity=None)}
