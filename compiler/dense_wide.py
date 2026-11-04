"""Opt-in full-LHS retention for wide-output dense contractions.

This binds the existing segmented/ready-readout owner to a broader, bounded
capacity domain. The ordinary compact incumbent reloads an LHS tile for every
output group. The library instead stages complete K once per M group, reuses
it across N, and alternates two physical RHS segments. Larger N groups can
also amortize transfer/setup commands. These are latency hypotheses, not cycle
claims; full transfer width and complete invocation timing need paired checks.

shared_rhs_epochs does not apply to a single source product, and requires its
complete W to fit. copy_pipeline needs token-copy StorageAdapter effects and a
separately selected concurrent/crossing-copy admission; neither exists on the
current dense boundary. No such laws are invented here. This path consumes the
same selected RAW/WAR/ACC/configuration ordering facts and checked indexed
schedule used by the existing segmented ready route. One terminal encoder owns
every data effect and every library completion.

Integration: ``result, info = build_wide_schedule(p, resources, storage,
adapter, options, bias_source=...)``. Emit with
``result.emit_index_program(control, lower_effect, complete_epoch=full_fence)``
and revalidate before/after emission. ``info['block']`` is the actual chosen
block. Caller declines transpose/per-column staging and invalidates resident
RHS state before this route writes its private windows. A solver-budget refusal retries the identical full verification once with a
20s budget after the ordinary 5s attempt; concrete invariant failures never retry.
Failure keeps the existing exact incumbent. No backend or settings changes live in this module.
"""
from dataclasses import replace
from math import gcd
from pathlib import Path
import json

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks

KNOB_DECLARATION={'dense_wide_schedule':{
    'values':['off','retained_lhs_wide','supported_geometry'],
    'mechanism':'checked full-LHS retention, capacity-bounded two-RHS K segments and causal ready readout',
    'note':'Opt-in wider planning domain for N>2048, K>edge. Up to 64 N tiles, four M tiles and capacity-bound K; explicit axis caps win. Same selected execution laws and encoder. Complete cycles remain unqualified; off is incumbent.'}}


def setting():
    return knob('dense_wide_schedule',default='off',values=('off','retained_lhs_wide','supported_geometry','matrix_coalesced_geometry'),
        root=Path(__file__).resolve().parent)


def _is_retryable_proof_timeout(reason):
    reason=reason.lower()
    return ('verification budget exhausted' in reason or
        'proof unavailable' in reason and any(word in reason for word in ('timeout','canceled','cancelled','resource limit')))


def build_wide_schedule(problem,resources,storage,adapter,options=None,*,bias_source=None,
                        rhs_transposed=False):
    """Return the actual source-checked retained/segmented owner and evidence.

    Storage legality is derived by dense_segmented_blocks, including retained
    padded K, both disjoint RHS slots, accumulator rows, and original budgets.
    No empirical shape selector is introduced; this function is an explicit
    experiment with an N>2048 coverage premise rather than a speed crossover.
    """
    if rhs_transposed or problem.tile_n!=problem.tile_k or problem.tile_m!=problem.tile_k:
        raise ValueError('wide retention requires ordinary square primitive operand/readout views')
    if problem.n<=2048 or problem.tiles[1]<2:
        raise ValueError('wide retention experiment requires N>2048 and multiple K tiles')
    options=dict(options or {})
    overrides=options.get('axis_overrides',{})
    maxima={
        'm':min(4,problem.tiles[0]),
        'k':problem.tiles[1]-1,
        # Avoid a second partial output-group proof domain in this bounded
        # experiment. Primitive column tails remain exact library operands.
        'n':min(gcd(64,problem.tiles[2]),storage.accumulator.capacity//problem.tile_m),
    }
    for axis in ('m','k','n'):
        if overrides.get(axis,False):
            maxima[axis]=int(options.get('max_'+axis+'_blocks',getattr(resources,'max_'+axis+'_blocks')))
    if any(value<=0 for value in maxima.values()):
        raise ValueError('wide retention needs positive bounded axis maxima')
    limits=replace(resources,**{'max_'+axis+'_blocks':value for axis,value in maxima.items()})
    choices=tuple(dense_segmented_blocks(problem,limits,storage))
    if not choices:
        raise ValueError('full LHS and both RHS segments do not fit original private stores')
    # Existing owner chooses the block and applies/validates the actual plan;
    # this adapter supplies only a bounded domain, never an unchecked schedule.
    from dense_epochs import build_readout_schedule
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    attempts=[]
    for timeout_ms in (5000,20000):
        try:
            result,info=build_readout_schedule(problem,limits,storage,adapter,
                dict(options,readout_schedule='segmented_ready'),bias_source=bias_source,
                timeout_ms=timeout_ms)
            result.revalidate()
        except DenseVerificationUnavailable as error:
            # UNKNOWN caused by an exhausted solver budget proves neither the
            # original invariant nor its negation. Retry the SAME full plan and
            # all original checks; a concrete failed invariant is a ValueError
            # and deliberately never enters this handler.
            reason=str(error)
            retryable=_is_retryable_proof_timeout(reason)
            attempts.append({'timeout_ms':timeout_ms,'status':'unproved',
                'refusal':reason,'retryable_solver_timeout':retryable})
            if not retryable:
                raise
            if timeout_ms==20000:
                raise DenseVerificationUnavailable(
                    'wide retained-LHS full verification remained unavailable after bounded retries: '
                    +json.dumps(attempts,sort_keys=True)) from error
        else:
            attempts.append({'timeout_ms':timeout_ms,'status':'fully_verified'})
            break
    info=dict(info)
    info.update(selected_path='retained_lhs_wide',
        planning_maxima=maxima,
        candidate_count=len(choices),
        verification_attempts=attempts,
        verification_policy='same complete plan and full original invariants; 5s then one 20s retry only for solver timeouts; concrete failures never retried',
        output_group_domain='power-of-two divisor of primitive N tile count; primitive tails preserved',
        source_shape=[problem.m,problem.k,problem.n],
        mechanism='existing retained_lhs_segmented_k plus causal ready readout',
        hypothesis='retain complete LHS across output groups; amortize wider RHS movement/setup',
        performance_status='experimental-unmeasured',
        coverage_premise={'n_greater_than':2048,'meaning':'explicit experiment coverage, not profitable crossover'},
        terminal_encoding='unchanged dense effect callback and explicit full epoch completion')
    from dense_wide_loads import maybe_coalesce
    return maybe_coalesce(result,problem,limits,storage,adapter,info)
