"""Structural policy for complete-K RHS reuse with ready readout over M groups.

No emission lives here. Resource guards express full-panel
reuse applicability; the existing checked combined owner proves legality.
No timing or source/capsule identity participates in selection.
"""
from dataclasses import asdict
from pathlib import Path
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy)
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
from gemmini_dense_readout import ReadoutUnavailable
from gemmini_dense_coalesced_policy import _raw_source_supported, _digest


def predicates():
    return tuple(ResourcePredicate(name, at_least=minimum) for name, minimum in (
        ('original_raw_source_supported', 1),
        ('full_m_groups', 2), ('complete_panels_fit', 1),
        ('current_source_owner_admitted', 1)))


def design():
    body = dict(schema='gemmini.dense.rhs_ready_design.v1',
        predicates=[asdict(p) for p in predicates()],
        empirical_region='structural full-panel reuse region; paired training and reserved tall/wide validation passed with unchanged integer numerics',
        structural_guards=dict(full_m_groups='floor(source M tiles / incumbent M block) >= 2',
            complete_panels_fit='complete source K times incumbent M and N blocks fit their original disjoint operand halves',
            current_source_owner_admitted='existing plan_rhs_ready_readout checks source, actual invariant RHS hoists, original storage, epochs and separate ready completion laws'),
        source_boundary='original contiguous signed i8*i8 accumulation in i32; raw i32 by default, or opt-in existing scalar i8 readout with common scale/activation semantics; no bias, initializer, vector scales, shift or explicit epilogue stages',
        selected_path='checked full-K retained RHS -> ready readout -> optional admitted wide RHS transfers -> unchanged ordinary dense terminal, panel_epochs',
        fallback='unchanged m1_coalesced_auto_schedule',
        status='integrated structural scalar region; complete matched provisional Spike estimates improve all six admitted training/heldout shapes; reserved ragged fallback unchanged; timing qualification and required RTL tiers remain separate',
        validation=dict(training='sweep_2_1793180349708713685_97c4abc38519fb332ac5eb00527fe827',
            heldout='sweep_2_1793180645811035930_549dc9ef216f7c6b75b3869c367c25c4',
            evidence='artifacts/dense-work/heldout-comparison-evidence.json',
            neighbor='M255/256 source integer projection pass; M257 owner refusal retains checked incumbent'))
    return {**body, 'design_sha256': _digest(body)}


def rhs_ready_auto_schedule(incumbent, adapter, problem, geometry, facts_path, *,
        bias=False, initial=False, attrs=None, output_dtype='i32', outer_order=('m', 'n'),
        aggregate_transfers=False, allow_scalar_readout=False):
    """Return checked artifact, ordinary mode and receipt, preserving fallback.

    Candidate construction is attempted only inside the frozen geometric and
    storage region. Source/capacity/ordering refusal retains the incumbent owner.
    The caller must consume the returned artifact with the common dense encoder.
    """
    from gemmini_dense_coalesced_policy import m1_coalesced_auto_schedule
    from gemmini_dense_rhs_overlap import plan_rhs_ready_readout
    from gemmini_dense_rhs_experiment import scalar_terminal_source_supported
    if aggregate_transfers:
        from gemmini_dense_rhs_experiment import plan_rhs_ready_transfers as plan_rhs_ready_readout
    incumbent.revalidate()
    attrs = dict(attrs or {})
    edge, operand, operand_rows, accumulator, accumulator_rows = geometry
    block = incumbent.choice.block
    lhs_rows = block.m * problem.tiles[1] * edge
    rhs_rows = block.n * problem.tiles[1] * edge
    split = operand_rows // 2
    resources = dict(
        original_raw_source_supported=int((scalar_terminal_source_supported if allow_scalar_readout
            else _raw_source_supported)(problem, attrs=attrs,
            bias=bias, initial=initial, output_dtype=output_dtype, outer_order=outer_order)),
        full_m_groups=problem.tiles[0] // block.m,
        complete_panels_fit=int(lhs_rows <= split and rhs_rows <= operand_rows-split),
        current_source_owner_admitted=0)
    candidate = None
    candidate_receipt = None
    refusal = None
    if (resources['original_raw_source_supported']
            and resources['full_m_groups'] >= 2 and resources['complete_panels_fit']):
        try:
            candidate, candidate_receipt = plan_rhs_ready_readout(incumbent, adapter, problem,
                geometry, facts_path, bias=bias, initial=initial, outer_order=outer_order)
            candidate.revalidate()
            resources['current_source_owner_admitted'] = 1
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
    def retained(_):
        candidate.revalidate()
        return {'selection': (candidate, 'panel_epochs', candidate_receipt)}
    def fallback(_):
        return {'selection': m1_coalesced_auto_schedule(incumbent, adapter, problem,
            geometry, facts_path, bias=bias, initial=initial, attrs=attrs,
            output_dtype=output_dtype, outer_order=outer_order)}
    registry = {name: RegisteredPass(PassSpec(name, 0, produces=('selection',),
        mutates_ir=False, ir_level='checked_dense'), action)
        for name, action in (('rhs_ready', retained), ('auto_ready', fallback))}
    policy = PipelinePolicy((PipelineVariant('rhs_ready',
        ('rhs_ready',), predicates()), PipelineVariant('auto_ready', ('auto_ready',))), 'auto_ready')
    selection = execute_policy(session, policy, registry, facts, output='selection')
    result, mode, receipt = selection.artifact
    return result, mode, {**(receipt or {}), 'rhs_ready_policy': dict(
        design=design(), resources=resources, selected_policy=selection.receipt,
        experimental_options=dict(aggregate_transfers=aggregate_transfers,
            allow_scalar_readout=allow_scalar_readout),
        candidate_refusal=refusal, selected_facts_sha256=_digest(selected_facts),
        retained_lhs_rows=lhs_rows, retained_rhs_rows=rhs_rows, qualification_identity=None)}
