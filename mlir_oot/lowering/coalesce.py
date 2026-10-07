"""Coalescing consecutive DMA tile loads into one block transfer.

This target's load path can fill more than one on-chip tile per transfer. The capability is a
DERIVED fact, not a choice: ``gemmini_params.h`` sets ``MAX_BYTES`` to 64 and defines
``MAX_BLOCK_LEN = MAX_BYTES / (DIM * sizeof(elem_t))``, so one transfer carries up to
:data:`~..target.facts.MAX_BLOCK_LEN` DIM-wide column blocks; ``LoadController.scala`` places
block ``b`` of such a transfer at ``spad + block_stride * b``, where ``block_stride`` is the
CONFIG_LD field this package always declares as ``DIM`` (matching the header's own
``gemmini_extended3_config_ld(..., DIM, id)``).

Two facts make this a pure rewrite of the transfer schedule rather than a change of meaning:

* the destination of block ``b`` is exactly the address the un-merged transfer ``b`` would have
  used, because the scheduler lays consecutive operand tiles out ``DIM`` rows apart
  (``Blocking.a_row`` steps ``DIM`` per k-tile, ``Blocking.b_row`` steps ``DIM`` per n-tile); and
* the source is one contiguous DRAM run, because the merged tiles' byte offsets form an arithmetic
  progression of exactly one tile's width.

Both are CHECKED here per candidate run rather than assumed, so a load stream that does not satisfy
them (a gathered im2col row-run, a transposed operand, an edge tile in the middle of a run) is left
exactly as it was. The accumulator is never merged into: its element is four bytes wide, so the same
derivation gives ``MAX_BLOCK_LEN_ACC == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..target import facts


@dataclass(frozen=True)
class Burst:
    """One DMA transfer: a DRAM run landing at an on-chip address."""

    spad: int
    tensor: str
    byte_offset: int
    cols: int
    rows: int
    note: str


def _extends(run: list[Burst], nxt: Burst, elem_bytes: int) -> bool:
    """True when ``nxt`` is the next column block of the transfer ``run`` already describes."""
    prev = run[-1]
    return (
        nxt.tensor == prev.tensor
        # every block of one transfer reads the same number of rows
        and nxt.rows == prev.rows
        # ... lands one block further on chip (the declared CONFIG_LD block stride)
        and nxt.spad == prev.spad + facts.DIM
        # ... and continues the same contiguous DRAM run
        and nxt.byte_offset == prev.byte_offset + facts.DIM * elem_bytes
        # only the FINAL block of a transfer may be a partial (edge) tile
        and prev.cols == facts.DIM
        and not (nxt.spad & facts.BIT_IS_ACC)
    )


def coalesce(loads: list[Burst], *, elem_bytes: int,
             max_blocks: int = facts.MAX_BLOCK_LEN) -> list[Burst]:
    """Merge maximal runs of ``loads`` into block transfers, preserving order and meaning.

    Returns a list of transfers whose union of (source byte, destination row) pairs is identical to
    the input's. ``max_blocks`` bounds a transfer at the derived DMA payload.
    """
    if max_blocks <= 1 or not loads:
        return list(loads)

    out: list[Burst] = []
    run: list[Burst] = []

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            out.append(
                replace(
                    run[0],
                    cols=(len(run) - 1) * facts.DIM + run[-1].cols,
                    note=f"{run[0].note}x{len(run)}",
                )
            )
        run.clear()

    for ld in loads:
        if ld.spad & facts.BIT_IS_ACC:
            # an accumulator destination carries a 4-byte element: one block per transfer
            flush()
            out.append(ld)
            continue
        if run and len(run) < max_blocks and _extends(run, ld, elem_bytes):
            run.append(ld)
            continue
        flush()
        run.append(ld)
    flush()
    return out


def footprint(burst: Burst) -> tuple[int, int]:
    """``(blocks, rows)`` -- the on-chip rows one transfer writes, as block count and rows each."""
    blocks = (burst.cols + facts.DIM - 1) // facts.DIM
    return max(1, blocks), burst.rows
