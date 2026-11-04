"""Bounded cold-proof recovery without changing the checked dense schedule.

Each attempt runs the original complete source and epoch verifier. A cold
solver may exhaust a short interactive budget on quantified continuation or
ragged readout obligations; only that unavailability permits one retry. Failed
invariants and unavailable contracts propagate immediately. Successful shared
source proofs retain the library's original cache and identity checks.
"""
from __future__ import annotations

from time import monotonic
from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable


def verify_dense_with_budget(build, *, initial_timeout_ms, total_timeout_ms=60000):
    """Build one identical candidate, with at most one bounded proof retry.

    ``build(timeout_ms)`` must close over the same typed source, schedule plan,
    storage and admission on both calls. The total wall-clock allowance covers
    both attempts, so the retry cannot multiply the endpoint's compile budget.
    No successful proof or solver UNKNOWN is replaced by an assumed verdict.
    """
    if (not callable(build) or type(initial_timeout_ms) is not int
            or type(total_timeout_ms) is not int
            or not 0 < initial_timeout_ms <= total_timeout_ms <= 60000):
        raise ValueError('dense proof requires finite positive budgets bounded by 60 seconds')
    start = monotonic()
    try:
        result = build(initial_timeout_ms)
    except ValueError as exc:
        # Older panel checkers report their explicit exhausted deadline as
        # ValueError. Enumerate those availability failures; invariant errors
        # (including an unclassified solver result) remain immediate refusals.
        availability_errors = {
            'retained staging proof budget exhausted',
            'retained source proof budget exhausted',
            'causal readout epoch checking budget exhausted',
            'panel assigned-access checking budget exhausted',
            'panel source checking budget exhausted'}
        if not isinstance(exc, DenseVerificationUnavailable) and str(exc) not in availability_errors:
            raise
        reason = str(exc).lower()
        if not any(word in reason for word in ('budget exhausted', 'timeout', 'canceled')):
            raise
        remaining = int(total_timeout_ms - (monotonic() - start) * 1000)
        if remaining < 1000:
            raise
        result = build(remaining)
    result.realization['verification_budget'] = {
        'initial_timeout_ms': initial_timeout_ms,
        'total_timeout_ms': total_timeout_ms,
        'retry_policy': 'at most one retry within shared wall-clock allowance',
        'contract': 'same complete source and epoch checks on unchanged schedule'}
    return result
