"""Gemmini RoCC instruction encoding, derived from the shipped ISA definition.

Field layout (from the RTL decoder / `xcustom.h` ROCC_INSTRUCTION_RS1_RS2):

    funct[31:25] rs2[24:20] rs1[19:15] xd[14] xs1[13] xs2[12] rd[11:7] opcode[6:0]

which for a two-source, no-destination accelerator command is exactly

    .insn r <CUSTOM_OPCODE>, <FUNCT3>, <funct>, x0, $0, $1

with FUNCT3 = 0b011 (xd=0, xs1=1, xs2=1).  The rs1/rs2 bit packings below are transcribed
one-for-one from the `gemmini_*` macros in `gemmini.h`; every shift is quoted in
`docs/public_facts_used.md`.
"""
from __future__ import annotations

import struct

from . import rtl_facts as F

DIM = F.DIM
ADDR_LEN = 32                      # gemmini_params.h

# ---- funct7 values (gemmini.h `#define k_*`) ------------------------------------------------
K_CONFIG = 0
K_MVIN2 = 1
K_MVIN = 2
K_MVOUT = 3
K_COMPUTE_PRELOADED = 4
K_COMPUTE_ACCUMULATE = 5
K_PRELOAD = 6
K_FLUSH = 7
K_MVIN3 = 14

#: CONFIG subtypes, carried in rs1[1:0].
CONFIG_EX = 0
CONFIG_LD = 1
CONFIG_ST = 2

#: dataflow bit (rs1[2] of a CONFIG_EX)
OUTPUT_STATIONARY = 0
WEIGHT_STATIONARY = 1

#: activation selector (rs1[4:3] of CONFIG_EX / rs1[3:2] of CONFIG_ST)
NO_ACTIVATION = 0
RELU = 1

GARBAGE_ADDR = 0xFFFFFFFF

#: local-address metadata bits (LocalAddr.scala bundle order, MSB first)
ACC_ADDR_BIT = 1 << (ADDR_LEN - 1)          # bit 31 -- address names the accumulator
ACC_ACCUMULATE_BIT = 1 << (ADDR_LEN - 2)    # bit 30 -- add into the accumulator row
ACC_FULL_ROW_BIT = 1 << (ADDR_LEN - 3)      # bit 29 -- full i32 row for DMA readout

MASK64 = (1 << 64) - 1


def f32_bits(value: float) -> int:
    """IEEE-754 single-precision bit pattern (`acc_scale_t_to_acc_scale_t_bits`)."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _u64(value: int) -> int:
    return _field(value, 64, "register word")


def _field(value, width: int, name: str) -> int:
    """``value`` as an unsigned ``width``-bit field; anything out of range refuses.

    A masked encoding would silently issue a different, legal command than the
    one the stream describes (``cols=0x80000010`` becoming 16 columns).
    """
    if isinstance(value, bool):
        value = int(value)
    if type(value) is not int or not 0 <= value < 1 << width:
        raise ValueError(f"{name}={value!r} does not fit its {width}-bit field")
    return value


def _flag(value, name: str) -> int:
    return _field(value, 1, name)


def assert_legal(funct: int) -> int:
    if funct not in F.LEGAL_FUNCTS:
        raise ValueError(f"funct {funct} is not in the decoder's legal set")
    name = F.FUNCT_NAMES.get(funct, "")
    if name.startswith("LOOP_"):
        raise ValueError(f"funct {funct} ({name}) uses a hardware loop FSM")
    return funct


def asm_string(funct: int) -> str:
    """The `.insn` directive text for one RoCC command."""
    assert_legal(funct)
    return f".insn r {hex(F.CUSTOM_OPCODE)}, {hex(F.FUNCT3)}, {hex(funct)}, x0, $0, $1"


# ---- rs1/rs2 packers -------------------------------------------------------------------------
# Each returns (funct, rs1, rs2).  A packer whose rs1 carries a DRAM address returns `None` in
# that slot: the caller substitutes the runtime pointer value.

def flush(skip: int = 0):
    return K_FLUSH, _u64(skip), 0


def config_ex(*, dataflow: int = WEIGHT_STATIONARY, sys_act: int = NO_ACTIVATION,
              sys_shift: int = 0, acc_scale: float = 1.0, c_stride: int = 1,
              a_stride: int = 1, a_transpose: bool = False, b_transpose: bool = False,
              set_only_strides: bool = False):
    """`gemmini_extended3_config_ex`.

    rs1: [63:32] acc_scale | [31:16] a_stride | [9] b_transpose | [8] a_transpose |
         [7] set_only_strides | [5] uselut | [4:3] activation | [2] dataflow | [1:0] cmd_type
    rs2: [63:48] c_stride | [31:0] in_shift
    """
    rs1 = ((f32_bits(acc_scale) << 32)
           | (_field(a_stride, 16, "a_stride") << 16)
           | (_flag(b_transpose, "b_transpose") << 9)
           | (_flag(a_transpose, "a_transpose") << 8)
           | (_flag(set_only_strides, "set_only_strides") << 7)
           | (_field(sys_act, 2, "activation") << 3)
           | (_field(dataflow, 1, "dataflow") << 2)
           | CONFIG_EX)
    rs2 = (_field(c_stride, 16, "c_stride") << 48) | _field(sys_shift, 32, "in_shift")
    return K_CONFIG, _u64(rs1), _u64(rs2)


def config_ld(*, stride: int, scale: float = 1.0, shrunk: bool = False,
              block_stride: int = DIM, pixel_repeats: int = 1, load_id: int = 0):
    """`gemmini_extended5_config_ld`.

    rs1: [63:32] mvin scale | [31:16] block_mvin_stride | [15:8] pixel_repeats |
         [4:3] load id | [2] shrunk | [1:0] cmd_type
    rs2: DRAM row stride in bytes
    """
    rs1 = ((f32_bits(scale) << 32)
           | (_field(block_stride, 16, "block_stride") << 16)
           | (_field(pixel_repeats, 8, "pixel_repeats") << 8)
           | (_field(load_id, 2, "load_id") << 3)
           | (_flag(shrunk, "shrunk") << 2)
           | CONFIG_LD)
    return K_CONFIG, _u64(rs1), _u64(stride)


def config_st(*, stride: int, acc_act: int = NO_ACTIVATION, acc_scale: float = 1.0,
              pool_stride: int = 0, pool_size: int = 0, pool_out_dim: int = 0,
              porows: int = 0, pocols: int = 0, orows: int = 0, ocols: int = 0,
              upad: int = 0, lpad: int = 0):
    """`gemmini_extended2_config_st`.

    rs1: [63:56] ocols | [55:48] orows | [47:40] pocols | [39:32] porows |
         [31:24] pool_out_dim | [15:10] lpad | [9:8] upad | [7:6] pool_size |
         [5:4] pool_stride | [3:2] acc_act | [1:0] cmd_type
    rs2: [63:32] acc_scale | [31:0] DRAM row stride in bytes
    """
    rs1 = ((_field(ocols, 8, "ocols") << 56)
           | (_field(orows, 8, "orows") << 48)
           | (_field(pocols, 8, "pocols") << 40)
           | (_field(porows, 8, "porows") << 32)
           | (_field(pool_out_dim, 8, "pool_out_dim") << 24)
           | (_field(lpad, 6, "lpad") << 10)
           | (_field(upad, 2, "upad") << 8)
           | (_field(pool_size, 2, "pool_size") << 6)
           | (_field(pool_stride, 2, "pool_stride") << 4)
           | (_field(acc_act, 2, "activation") << 2)
           | CONFIG_ST)
    rs2 = (f32_bits(acc_scale) << 32) | _field(stride, 32, "stride")
    return K_CONFIG, _u64(rs1), _u64(rs2)


def _mem_rs2(local_addr: int, cols: int, rows: int) -> int:
    return _u64((_field(rows, 16, "rows") << (ADDR_LEN + 16))
                | (_field(cols, 16, "cols") << ADDR_LEN)
                | _field(local_addr, ADDR_LEN, "local_addr"))


def mvin(*, local_addr: int, cols: int, rows: int, load_id: int = 0):
    """`gemmini_extended_mvin`; rs1 is the DRAM address (runtime pointer)."""
    funct = {0: K_MVIN, 1: K_MVIN2, 2: K_MVIN3}.get(load_id)
    if funct is None or type(load_id) is not int:
        raise ValueError(f"load_id={load_id!r} names no load state")
    return funct, None, _mem_rs2(local_addr, cols, rows)


def mvout(*, local_addr: int, cols: int, rows: int):
    """`gemmini_extended_mvout`; rs1 is the DRAM address (runtime pointer)."""
    return K_MVOUT, None, _mem_rs2(local_addr, cols, rows)


def preload(*, bd_addr: int, c_addr: int, bd_cols: int, bd_rows: int,
            c_cols: int, c_rows: int):
    """`gemmini_extended_preload`."""
    return K_PRELOAD, _mem_rs2(bd_addr, bd_cols, bd_rows), _mem_rs2(c_addr, c_cols, c_rows)


def compute(*, a_addr: int, bd_addr: int, a_cols: int, a_rows: int,
            bd_cols: int = DIM, bd_rows: int = DIM, accumulate: bool = False):
    """`gemmini_extended_compute_{preloaded,accumulated}`."""
    funct = K_COMPUTE_ACCUMULATE if _flag(accumulate, "accumulate") else K_COMPUTE_PRELOADED
    return funct, _mem_rs2(a_addr, a_cols, a_rows), _mem_rs2(bd_addr, bd_cols, bd_rows)


def acc_addr(row: int, *, accumulate: bool = False, full_row: bool = False) -> int:
    a = ACC_ADDR_BIT | (row & 0x3FFF)
    if accumulate:
        a |= ACC_ACCUMULATE_BIT
    if full_row:
        a |= ACC_FULL_ROW_BIT
    return a
