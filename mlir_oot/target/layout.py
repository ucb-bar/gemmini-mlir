"""How a tensor the harness allocates is laid out in DRAM.

``mlir_oot_backend_contract.yaml`` states the pointee layout once: "row-major, edge tiles
zero-padded to a multiple of 16 (DIM); weight/out as packed per the i8/i32 readout". So a tensor is
the row-major matrix ``[prod(shape[:-1]), shape[-1]]`` whose ROW PITCH is the trailing extent rounded
up to a multiple of DIM -- an edge tile is a whole tile whose surplus lanes are zero.

The distinction is invisible for a tensor whose trailing extent is already a multiple of DIM, which
is why it is isolated here behind ``row_pitch``: every address this backend forms goes through it.
"""

from __future__ import annotations

from . import facts

#: The contract's `pointee_layout` clause, applied. Kept as one named switch because it is the single
#: assumption every DRAM address in the package rests on.
PAD_TRAILING_EXTENT_TO_DIM = True


def row_pitch(shape) -> int:
    """Elements between the starts of two consecutive rows of ``shape``'s row-major matrix view."""
    if not shape:
        return 1
    trailing = int(shape[-1])
    if not PAD_TRAILING_EXTENT_TO_DIM:
        return trailing
    return -(-trailing // facts.DIM) * facts.DIM


def row_count(shape) -> int:
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    return n
