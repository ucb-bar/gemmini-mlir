"""Numerical boundary policy for the whole-model mesh reference executor.

The policy is target-neutral: the selected datapath binding supplies the operand
and accumulator formats and whether subnormal operands are flushed.
"""

from __future__ import annotations

import math

import numpy as np


def boundary_scale(x, fmt: str) -> float:
    """Power-of-two scale placing the largest finite operand inside ``fmt``."""
    from . import fp8_formats as FF

    mx = float(np.abs(x).max()) if x.size else 0.0
    if not (mx > 0.0) or not np.isfinite(mx):
        return 1.0
    _min_normal, max_finite = FF.normal_range(fmt)
    return float(2.0 ** math.ceil(math.log2(mx / max_finite)))


def _representability(x, fmt: str, *, flush: bool) -> dict:
    """Count subnormal, flushed and saturating values at the selected boundary."""
    from . import fp8_formats as FF

    min_normal, max_finite = FF.normal_range(fmt)
    ax = np.abs(np.asarray(x, dtype=np.float64))
    subnormal = int(np.count_nonzero((ax > 0.0) & (ax < min_normal)))
    return {
        "n_values": int(ax.size),
        "n_subnormal": subnormal,
        "n_flushed_to_zero": subnormal if flush else 0,
        "n_saturating": int(np.count_nonzero(ax > max_finite)),
    }


def float_boundary_operands(a, b, binding) -> tuple[list, list, float, dict]:
    """Scale float operands and report what their selected format cannot carry.

    Power-of-two scaling preserves mantissas. If the resulting reduction could
    overflow the selected accumulator, back off the scales in whole powers of
    two and retain the remaining loss in the representability record.
    """
    from . import fp8_formats as FF

    fmt = binding.operand_dtype
    af, bf = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    sa, sb = boundary_scale(af, fmt), boundary_scale(bf, fmt)

    ma = float(np.abs(af).max()) if af.size else 0.0
    mb = float(np.abs(bf).max()) if bf.size else 0.0
    try:
        _min_acc, max_acc = FF.normal_range(binding.accum_dtype)
    except KeyError:  # integer or unresolvable accumulator: no float cap
        max_acc = None
    if max_acc is not None and ma > 0.0 and mb > 0.0:
        k = max(1, int(af.shape[1])) if af.ndim == 2 else 1
        headroom = max_acc / 2.0
        worst = (ma / sa) * (mb / sb) * k
        if np.isfinite(worst) and worst > headroom:
            back = math.ceil(math.log2(worst / headroom))
            sa *= float(2.0 ** ((back + 1) // 2))
            sb *= float(2.0 ** (back // 2))

    qa, qb = af / sa, bf / sb
    flush = bool(getattr(binding, "subnormal_operand_flush", False))
    rec = {
        "operand_dtype": fmt,
        "scale_a": sa,
        "scale_b": sb,
        "a": _representability(qa, fmt, flush=flush),
        "b": _representability(qb, fmt, flush=flush),
    }
    return qa.tolist(), qb.tolist(), sa * sb, rec
