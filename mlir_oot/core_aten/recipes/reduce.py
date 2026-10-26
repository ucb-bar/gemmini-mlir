"""Sum reductions with one readout scale (mean.dim: s_in / (L s_out)).

* ``mean_last``: the reduced dims are the contiguous trailing block (``ew_sum_lastdim`` /
  ``ep_gm_mean_last``): per 16-row tile, X . ones-column accumulated over the L/16 K tiles
  in the int32 accumulator (one ACC row per output), then a 1-column readout.
* ``mean_middle``: the reduced dims are a contiguous middle block of x viewed as
  [O, L, I]: per output row o, ones-row . X_o with X_o tiles as the stationary operand,
  accumulated over L in the accumulator; one-row readouts.
Sums stay exact in int32; only the readout rounds (gate 3: 0 LSB).
"""

from __future__ import annotations

import numpy as np

from ..stream import ACC, DIM, NO_ACT, SP, Stream

ONES = 0          # spad rows 0..15: ones column (col 0) / row 16: ones row
ONEROW = 16
SLOT = 32


def _ones(s: Stream):
    if "ones16" not in s._index:
        s.const("ones16", np.ones(DIM, np.int8))


def mean_last(s: Stream, x: str, out: str, R: int, L: int, scale: float, *, out_i32: bool = False,
              multiplier_bits: int | None = None) -> None:
    """``multiplier_bits`` programs a proved narrow f32 readout instead of ``scale``."""
    _ones(s)
    s.config_ex()
    s.config_ld(1, 1)
    s.mvin(s.at("ones16"), SP(ONES), 1, DIM, 1)
    s.config_ld(0, L)
    t = 0
    for c, r0 in enumerate(range(0, R, 512)):
        rr = min(512, R - r0)
        acc = (c & 1) * 512
        for i0 in range(0, rr, DIM):
            mr = min(DIM, rr - i0)
            for k0 in range(0, L, DIM):
                kr = min(DIM, L - k0)
                slot = SLOT + (t & 3) * DIM
                t += 1
                s.mvin(s.at(x, (r0 + i0) * L + k0), SP(slot), kr, mr, 0)
                s.preload(SP(ONES), ACC(acc + i0, accumulate=k0 > 0), 1, kr, 1, mr)
                s.compute(SP(slot), kr, mr)
        if multiplier_bits is None:
            s.config_st(4 if out_i32 else 1, NO_ACT, scale)
        else:
            s.config_st(1, NO_ACT, multiplier_bits=multiplier_bits)
        for i in range(0, rr, DIM):
            s.mvout(s.at(out, (r0 + i)*(4 if out_i32 else 1)), ACC(acc + i, full=out_i32), 1, min(DIM, rr - i))


def mean_middle(s: Stream, x: str, out: str, O: int, L: int, I: int, scale: float, *, out_i32: bool = False) -> None:
    _ones(s)
    s.config_ex()
    s.config_ld(0, 0)
    s.mvin(s.at("ones16"), SP(ONEROW), DIM, 1, 0)
    s.config_ld(1, I)
    s.config_st(I*(4 if out_i32 else 1), NO_ACT, scale)
    t = u = 0
    for o in range(O):
        for j0 in range(0, I, DIM):
            nc = min(DIM, I - j0)
            acc = u % 512
            u += 1
            for k0 in range(0, L, DIM):
                kr = min(DIM, L - k0)
                slot = SLOT + (t & 3) * DIM
                t += 1
                s.mvin(s.at(x, (o * L + k0) * I + j0), SP(slot), nc, kr, 1)
                s.preload(SP(slot), ACC(acc, accumulate=k0 > 0), nc, kr, nc, 1)
                s.compute(SP(ONEROW), kr, 1)
            s.mvout(s.at(out, (o * I + j0)*(4 if out_i32 else 1)), ACC(acc, full=out_i32), nc, 1)
