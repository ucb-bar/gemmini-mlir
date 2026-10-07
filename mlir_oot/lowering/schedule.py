"""Tiling / residency scheduling for the gemmini mesh.

This is a HEURISTIC over derived capabilities, not a table of shapes: it is handed the tile counts
of the contraction it is about to emit and returns the loop-nest blocking, which it derives purely
from the RTL facts (mesh DIM, scratchpad rows, accumulator rows, DMA block length). No capsule
extent is special-cased anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..target import facts


@dataclass(frozen=True)
class Blocking:
    """One legal blocking of an (Mt x Kt x Nt)-tile contraction, in DIM-sized tiles."""

    ti: int
    tj: int
    tk: int
    #: scratchpad row where the A (moving) operand's tile array starts
    a_base: int
    #: scratchpad row where the B (stationary) operand's tile array starts
    b_base: int

    def a_row(self, il: int, kl: int) -> int:
        return self.a_base + (il * self.tk + kl) * facts.DIM

    def b_row(self, kl: int, jl: int) -> int:
        return self.b_base + (kl * self.tj + jl) * facts.DIM

    def acc_row(self, il: int, jl: int, jb: int) -> int:
        return (il * jb + jl) * facts.DIM


class Scheduler:
    """Chooses the whole-region schedule from the derived target capabilities."""

    def __init__(
        self,
        sp_rows: int = facts.SP_ROWS,
        acc_rows: int = facts.ACC_ROWS,
        dim: int = facts.DIM,
    ) -> None:
        self.sp_rows = sp_rows
        self.acc_rows = acc_rows
        self.dim = dim

    def transfers(self, mt: int, kt: int, nt: int, ti: int, tj: int) -> int:
        """DMA bursts an (ti, tj) blocking costs, as a function of the extents alone.

        A moving-operand tile is re-fetched once per N block and a stationary tile once per M block,
        so the two terms trade against each other under one accumulator budget; the output is
        written once either way. This is the cost the schedule is chosen to minimise.
        """
        n_blocks = -(-nt // tj)
        m_blocks = -(-mt // ti)
        return n_blocks * mt * kt + m_blocks * kt * nt + mt * nt

    def choose(self, mt: int, kt: int, nt: int, *, b_transposed: bool = False) -> Blocking:
        """Return the legal blocking with the fewest DMA bursts.

        The accumulator bounds ``ti*tj`` (one DIM-row tile per live output tile) and the operand
        store bounds ``(ti + tj) * tk``. Within those two derived budgets the choice is a search,
        not a preference: fixing either extreme (all-N, or all-M) re-fetches the other operand once
        per block of the first, and which one that costs more depends entirely on the shape.
        """
        acc_tiles = max(1, self.acc_rows // self.dim)
        sp_tiles = max(1, self.sp_rows // self.dim)

        best: tuple[int, int, int] | None = None
        for ti in range(1, min(mt, acc_tiles) + 1):
            tj = max(1, min(nt, acc_tiles // ti))
            if ti * tj > acc_tiles or ti + tj > sp_tiles:
                continue
            cost = self.transfers(mt, kt, nt, ti, tj)
            if best is None or cost < best[0]:
                best = (cost, ti, tj)
        if best is None:
            raise ValueError("no blocking of this contraction fits the derived on-chip budgets")
        _, ti, tj = best

        budget = sp_tiles // max(1, ti + tj)
        tk = max(1, min(kt, budget))
        b_base = self.sp_rows - tk * tj * self.dim
        a_base = 0
        if a_base + ti * tk * self.dim > b_base:
            raise ValueError("scratchpad budget exceeded by the chosen blocking")
        return Blocking(ti=ti, tj=tj, tk=tk, a_base=a_base, b_base=b_base)
