"""Pinned dimension-only GEMV grouping/residency performance hypothesis.

Resource, layout and numerical admission remain in the actual lowerers. This
predicate contains no kernel names, source identities or observations at runtime.
"""


def select_grouped_gemv(m: int, k: int, n: int) -> bool:
    return (all(type(d) is int and d > 0 for d in (m, k, n))
            and m == 1 and k >= 128 and n >= 128)
