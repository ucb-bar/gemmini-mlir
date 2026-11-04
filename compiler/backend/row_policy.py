"""Optional dimension-based row residency policy, independent of kernel names."""
from merlin.compile.tuning_knobs import knob


def select_row_tiles(m, k, maximum, *, root):
    """Cap row residency using actual contraction dimensions.

    ``fixed`` preserves the selected maximum. ``reuse_large`` applies the
    provisional model-selected maximum16 only beyond128 output rows and with
    at least128 reduction elements; other dimensions retain maximum8.
    ``unroll_reuse`` uses that same residency limit and separately permits
    static issue of full row groups containing at most16 tiles.
    """
    policy = knob("row_policy", default="fixed",
                  values=("fixed", "reuse_large", "unroll_reuse"), root=root)
    if policy == "fixed":
        return maximum
    return min(maximum, 16 if int(m) > 128 and int(k) >= 128 else 8)


def unroll_full_rows(*, root):
    """Select static issue of bounded full row groups without changing residency."""
    return knob("row_policy", default="fixed",
                values=("fixed", "reuse_large", "unroll_reuse"),
                root=root) == "unroll_reuse"
