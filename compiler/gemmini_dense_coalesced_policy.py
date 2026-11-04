"""Capacity-relative M=1 transfer policy over the ordinary dense adapter.

This module emits nothing. The returned checked artifact must reach the existing
dense terminal exactly as auto_ready artifacts do. All other sources retain the
existing ready_auto_schedule. The ordinary auto_ready caller consumes this path;
its fixed design record is retained verbatim for experiment custody. Selection
uses no case identity or runtime timing.
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


def _digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def predicates():
    return tuple(ResourcePredicate(name, at_least=minimum) for name, minimum in (
        ('original_raw_source_supported', 1), ('m_equals_one', 1),
        ('retained_lhs_half_store_fits', 1), ('n_panels', 2),
        ('reduction_tiles_cover_output_tiles', 1),
        ('full_k_rhs_double_window_overflows', 1),
        ('k_segments', 2), ('wide_tiles_per_transfer', 2),
        ('current_source_owner_admitted', 1)))


def design():
    """Stable capacity-relative hypothesis; numerical and timing proof are separate."""
    body = dict(schema='gemmini.dense.m1_coalesced_design.v1',
        predicates=[asdict(p) for p in predicates()],
        source_boundary='original contiguous row-major signed i8 * i8 -> raw i32; no initializer, bias, scaling or activation',
        resources=dict(m_equals_one='source M == 1',
            retained_lhs_half_store_fits='2 * incumbent M tiles * source K tiles * selected array edge <= operand capacity rows',
            n_panels='ceil(source N tiles / incumbent N block tiles)',
            reduction_tiles_cover_output_tiles='source K tiles >= source N tiles; conservative aspect-ratio region after wide-N regression',
            full_k_rhs_double_window_overflows='two full-K incumbent RHS panels exceed operand rows remaining after retained LHS',
            k_segments='ceil(source K tiles / actual checked segmented K block tiles)',
            wide_tiles_per_transfer='actual checked aggregation bound derived from decoded MVIN2 columns and selected DMA request bytes',
            current_source_owner_admitted='actual coalesced commands pass original source/address/epoch laws'),
        selected_path='existing plan_transfers -> ordinary dense terminal; mode segmented_ready',
        fallback='unchanged ready_auto_schedule',
        status='integrated conservative aspect-ratio region; provisional modeled evidence only, fresh boundary validation pending')
    return {**body, 'design_sha256': _digest(body)}


def _raw_source_supported(problem, *, attrs, bias, initial, output_dtype, outer_order):
    from gemmini_readout import source_scales, _relu
    if (bias or initial or problem.bias or output_dtype != 'i32'
            or (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4)
            or tuple(outer_order) != ('m', 'n')):
        return False
    if (attrs.get('lhs_stride', problem.k) != problem.k
            or attrs.get('rhs_stride', problem.n) != problem.n
            or attrs.get('dst_stride', problem.n) != problem.n):
        return False
    try:
        scale = source_scales(attrs)
        return (not isinstance(scale, (tuple, list)) and float(scale) == 1.0
            and not _relu(attrs) and int(attrs.get('requant_shift', attrs.get('shift', 0))) == 0
            and not attrs.get('epilogue', []))
    except (TypeError, ValueError):
        return False


def m1_coalesced_auto_schedule(incumbent, adapter, problem, geometry, facts_path, *,
        bias=False, initial=False, attrs=None, output_dtype='i32', outer_order=('m', 'n')):
    """Return (checked result, ordinary mode, receipt); never emit or change defaults.

    The caller supplies the actual source-bound DenseLoopAdapter. Successful
    selection consumes the returned transformed result through the existing
    asynchronous dense encoder. Refusal executes the current auto_ready owner.
    """
    from dense_policy import ready_auto_schedule
    from gemmini_dense_transfers import plan_transfers
    incumbent.revalidate()
    attrs = dict(attrs or {})
    edge, operand, operand_rows, accumulator, accumulator_rows = geometry
    block = incumbent.choice.block
    lhs_rows = block.m * problem.tiles[1] * edge
    resources = dict(
        original_raw_source_supported=int(_raw_source_supported(problem, attrs=attrs,
            bias=bias, initial=initial, output_dtype=output_dtype, outer_order=outer_order)),
        m_equals_one=int(problem.m == 1),
        retained_lhs_half_store_fits=int(2 * lhs_rows <= operand_rows),
        n_panels=(problem.tiles[2] + block.n - 1) // block.n,
        reduction_tiles_cover_output_tiles=int(problem.tiles[1] >= problem.tiles[2]),
        full_k_rhs_double_window_overflows=int(
            2 * block.n * problem.tiles[1] * edge > operand_rows - lhs_rows),
        k_segments=0, wide_tiles_per_transfer=0, current_source_owner_admitted=0)
    candidate = None
    candidate_receipt = None
    refusal = None
    if all(resources[name] >= minimum for name, minimum in (
            ('original_raw_source_supported', 1), ('m_equals_one', 1),
            ('retained_lhs_half_store_fits', 1), ('n_panels', 2),
            ('reduction_tiles_cover_output_tiles', 1),
            ('full_k_rhs_double_window_overflows', 1))):
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
        return {'selection': ready_auto_schedule(incumbent, adapter, problem, geometry,
            facts_path, bias=bias, initial=initial)}
    registry = {name: RegisteredPass(PassSpec(name, 0, produces=('selection',),
        mutates_ir=False, ir_level='checked_dense'), action)
        for name, action in (('m1_coalesced', coalesced), ('auto_ready', fallback))}
    policy = PipelinePolicy((PipelineVariant('m1_coalesced', ('m1_coalesced',), predicates()),
        PipelineVariant('auto_ready', ('auto_ready',))), 'auto_ready')
    selection = execute_policy(session, policy, registry, facts, output='selection')
    result, mode, receipt = selection.artifact
    return result, mode, {**(receipt or {}), 'm1_coalesced_policy': dict(
        design=design(), resources=resources, selected_policy=selection.receipt,
        candidate_refusal=refusal, selected_facts_sha256=_digest(selected_facts),
        retained_lhs_rows=lhs_rows, qualification_identity=None)}
