"""RTL-derived facts for the gemmini target.

Every number here was DERIVED, not chosen:

* ``DIM``, ``SP_BYTES``, ``ACC_BYTES`` come from ``rtl.facts.load_facts("gemmini")["facts"]``
  (``arrays[0].rows/cols`` = 16x16 mesh, ``memories.scratchpad.bytes`` = 262144,
  ``memories.accumulator.bytes`` = 65536) and are corroborated by the shipped
  ``gemmini_params.h`` (``DIM 16``, ``BANK_NUM*BANK_ROWS*DIM`` bytes of operand store).
* ``LEGAL_FUNCTS`` is the decoder's own icmp fan-out (26 values) reported by
  ``rtl_backend.target_profile("gemmini")["legal_opcodes"]``.
* The on-chip address layout is read off ``rtl/gemmini/LocalAddr.scala``: the bundle declares
  ``is_acc_addr``, ``accumulate``, ``read_full_acc_row``, ``norm_cmd`` from the MSB down, over a
  32-bit local address whose low ``log2(sp_banks*sp_bank_entries)`` bits are the row index.

See ``docs/public_facts_used.md`` for the full provenance table.
"""

from __future__ import annotations

# --- mesh -------------------------------------------------------------------
DIM = 16

# --- element widths (datapaths[] in the RTL facts: input i8, accumulator i32) ---
ELEM_BYTES = 1
ACC_BYTES_PER_ELEM = 4

# --- on-chip capacity -------------------------------------------------------
SP_BYTES = 262144
ACC_BYTES = 65536
#: scratchpad rows: one row holds DIM operand elements
SP_ROWS = SP_BYTES // (DIM * ELEM_BYTES)          # 16384
#: accumulator rows: one row holds DIM accumulator elements
ACC_ROWS = ACC_BYTES // (DIM * ACC_BYTES_PER_ELEM)  # 1024
#: the operand store is BANK_NUM singly-ported banks of BANK_ROWS rows each
#: (``gemmini_params.h``: ``BANK_NUM*BANK_ROWS*DIM*ELEM_BYTES == SP_BYTES``, 4*4096*16 == 262144,
#: and ``load_facts("gemmini")`` reports the same store as "depth 4096"). A read and a write that
#: land in the SAME bank in one cycle serialise on its single port, so two operand homes meant to
#: be used CONCURRENTLY -- one being read by the mesh while the other is written by a prefetch --
#: must not share a bank.
SP_BANK_ROWS = 4096
SP_BANKS = SP_ROWS // SP_BANK_ROWS                  # 4

#: the largest number of DIM-wide column blocks one DMA transaction may carry
MAX_BYTES = 64
MAX_BLOCK_LEN = MAX_BYTES // (DIM * ELEM_BYTES)        # 4
MAX_BLOCK_LEN_ACC = MAX_BYTES // (DIM * ACC_BYTES_PER_ELEM)  # 1

# --- local (on-chip) address layout, from LocalAddr.scala -------------------
ADDR_LEN = 32
BIT_IS_ACC = 1 << (ADDR_LEN - 1)        # 0x80000000
BIT_ACCUMULATE = 1 << (ADDR_LEN - 2)    # 0x40000000
BIT_READ_FULL_ACC = 1 << (ADDR_LEN - 3)  # 0x20000000
GARBAGE_ADDR = 0xFFFFFFFF

# --- decoder gate -----------------------------------------------------------
LEGAL_FUNCTS = frozenset(
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 126)
)

#: RoCC envelope for this accelerator: custom-3 opcode, xd=0/xs1=1/xs2=1 -> func3 = 0b011.
ROCC_OPCODE = 0x7B
ROCC_FUNC3 = 0x3


def acc_addr(row: int, accumulate: bool, read_full: bool) -> int:
    """An accumulator local address for ``row``, with the two readout-mode bits set."""
    a = BIT_IS_ACC | (row & 0x3FFF)
    if accumulate:
        a |= BIT_ACCUMULATE
    if read_full:
        a |= BIT_READ_FULL_ACC
    return a
