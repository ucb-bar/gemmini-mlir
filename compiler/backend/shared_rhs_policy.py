"""Coarse opt-in shared-RHS profitability policy, separate from legality.

This pinned hypothesis uses only source dimensions and the proved group count.
It does not inspect names, payloads, scale values, timings or source identities.
"""
from __future__ import annotations


def select_columns(members):
    """Return the internal N bundle size, or None for incumbent source tasks.

    The graph transform owns canonical raw-i32 semantics, common immutable RHS,
    independent effects and storage alias proofs. Exactly eight consumers and
    at least1024 RHS elements are required for this bounded policy experiment.
    """
    if not isinstance(members, (list, tuple)) or len(members) != 8:
        return None
    first = members[0]
    try:
        m, k, n = first['m'], first['k'], first['n']
        if (any(type(d) is not int for d in (m, k, n))
                or not 1 <= m <= 16 or k < 16 or n < 16 or k * n < 1024
                or any(any(type(member[key]) is not int for key in ('m', 'k', 'n'))
                       or (member['m'], member['k'], member['n']) != (m, k, n)
                       for member in members)):
            return None
    except (KeyError, TypeError):
        return None
    return 4 if n >= 32 else 1


def combine_rows(members):
    """Bounded tiny-M refinement; eight rows share one physical array tile.

    Whole-call provisional support covers the selected decode region; fresh
    numerical geometry checks do not establish a fresh timing crossover.
    """
    return select_columns(members) is not None and members[0]["m"] == 1
