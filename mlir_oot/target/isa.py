"""RoCC instruction encoding for gemmini.

ONE packer per instruction class, each a transcription of the ``gemmini_*`` macro named in its
docstring (shipped ``isa_include/gemmini.h``). Nothing here is invented: the funct values come from
``tables/funct_table.py`` (generated from the same header) and the rs1/rs2 bit layouts come from the
macros. The 32-bit word itself is the RoCC R-type this target's decoder reads --
``funct[31:25] rs2[24:20] rs1[19:15] xd[14] xs1[13] xs2[12] rd[11:7] opcode[6:0]`` -- which is what
``.insn r 0x7b, 0x3, <funct>, x0, $0, $1`` assembles to on stock clang/LLVM.

An operand is either an immediate (``Imm``) or a DRAM address formed from a kernel POINTER ARGUMENT
(``ArgAddr``): no DRAM address is ever a literal, because the harness allocates the buffers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..tables.funct_table import CONFIG_SUBTYPE, FUNCT
from . import facts

MASK64 = (1 << 64) - 1


class EncodingError(Exception):
    """A field did not fit, or a funct this target's decoder does not accept was requested."""


# --------------------------------------------------------------------------- operands
@dataclass(frozen=True)
class Imm:
    """A plain 64-bit immediate; lowered to ``llvm.mlir.constant(<v> : i64) : i64``."""

    value: int

    def normalised(self) -> "Imm":
        return Imm(self.value & MASK64)


@dataclass(frozen=True)
class ArgAddr:
    """``ptrtoint(<kernel arg for tensor>) + byte_offset``. Never a baked DRAM address."""

    tensor: str
    byte_offset: int = 0


Operand = Imm | ArgAddr


# --------------------------------------------------------------------------- instruction
@dataclass(frozen=True)
class Instr:
    """One issued RoCC command: an instruction class plus its two source operands."""

    cls: str
    rs1: Operand
    rs2: Operand
    #: free-form provenance so the emitted MLIR can be reconciled against the command buffer
    note: str = ""

    @property
    def funct(self) -> int:
        try:
            f = FUNCT[self.cls]
        except KeyError as exc:  # pragma: no cover - guarded by the builders below
            raise EncodingError(f"no funct for instruction class {self.cls!r}") from exc
        if f not in facts.LEGAL_FUNCTS:
            raise EncodingError(f"funct {f} ({self.cls}) is not decoded by this target's RTL")
        return f

    @property
    def asm(self) -> str:
        return f".insn r {facts.ROCC_OPCODE:#x}, {facts.ROCC_FUNC3:#x}, {self.funct:#x}, x0, $0, $1"


def _fit(value: int, width: int, what: str) -> int:
    v = int(value)
    if v < 0:
        v &= (1 << width) - 1
    if v >= (1 << width):
        raise EncodingError(f"{what}={value} does not fit in {width} bits")
    return v


def _f32_bits(x: float) -> int:
    """The IEEE-754 single-precision pattern of ``x`` (``acc_scale_t_to_acc_scale_t_bits``)."""
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _rows_cols_addr(rows: int, cols: int, addr: int) -> int:
    """``(rows << (ADDR_LEN+16)) | (cols << ADDR_LEN) | addr`` -- the shared mvin/mvout/compute rs."""
    return (
        (_fit(rows, 16, "rows") << (facts.ADDR_LEN + 16))
        | (_fit(cols, 16, "cols") << facts.ADDR_LEN)
        | (addr & 0xFFFFFFFF)
    )


# --------------------------------------------------------------------------- builders
def flush(skip: int = 0) -> Instr:
    """``gemmini_flush(skip)``."""
    return Instr("FLUSH", Imm(skip), Imm(0), "flush")


def config_ex(
    dataflow: int,
    sys_act: int,
    sys_shift: int = 0,
    sys_acc_scale: float = 1.0,
    c_stride: int = 1,
    a_stride: int = 1,
    a_transpose: bool = False,
    b_transpose: bool = False,
) -> Instr:
    """``gemmini_extended3_config_ex`` (RS1 bit map is quoted verbatim above the macro)."""
    rs1 = (
        (_f32_bits(sys_acc_scale) << 32)
        | (_fit(a_stride, 16, "a_stride") << 16)
        | ((1 if b_transpose else 0) << 9)
        | ((1 if a_transpose else 0) << 8)
        | (_fit(sys_act, 2, "sys_act") << 3)
        | (_fit(dataflow, 1, "dataflow") << 2)
        | CONFIG_SUBTYPE["EX"]
    )
    rs2 = (_fit(c_stride, 16, "c_stride") << 48) | (int(sys_shift) & 0xFFFFFFFF)
    return Instr("CONFIG", Imm(rs1), Imm(rs2), "config_ex")


def config_ld(stride_bytes: int, scale: float = 1.0, shrunk: bool = False, ld_id: int = 0) -> Instr:
    """``gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride=DIM, pixel_repeats=1, id)``."""
    rs1 = (
        (_f32_bits(scale) << 32)
        | (_fit(facts.DIM, 16, "block_mvin_stride") << 16)
        | (1 << 8)
        | (_fit(ld_id, 2, "ld_id") << 3)
        | ((1 if shrunk else 0) << 2)
        | CONFIG_SUBTYPE["LD"]
    )
    return Instr("CONFIG", Imm(rs1), Imm(int(stride_bytes) & MASK64), f"config_ld id={ld_id}")


#: Width, in bits, of each field ``gemmini_extended2_config_st`` packs into rs1 (the layout is
#: read off the shipped ISA header). :func:`config_st` packs through this table, so a planner that
#: checks a value against it is checking against what the encoder will actually emit.
CONFIG_ST_FIELD_BITS = {
    "ocols": 8,
    "orows": 8,
    "pocols": 8,
    "porows": 8,
    "pool_out_dim": 8,
    "lpad": 2,
    "upad": 2,
    "pool_size": 2,
    "pool_stride": 2,
    "acc_act": 2,
}


def config_st_field_max(field: str) -> int:
    """The largest value this store-path field can carry."""
    return (1 << CONFIG_ST_FIELD_BITS[field]) - 1


def config_st(
    stride_bytes: int,
    acc_act: int = 0,
    acc_scale: float = 1.0,
    pool_stride: int = 0,
    pool_size: int = 0,
    pool_out_dim: int = 0,
    porows: int = 0,
    pocols: int = 0,
    orows: int = 0,
    ocols: int = 0,
    upad: int = 0,
    lpad: int = 0,
) -> Instr:
    """``gemmini_extended2_config_st`` -- the readout path: activation, acc scale, fused pooling."""
    _b = CONFIG_ST_FIELD_BITS
    rs1 = (
        (_fit(ocols, _b["ocols"], "ocols") << 56)
        | (_fit(orows, _b["orows"], "orows") << 48)
        | (_fit(pocols, _b["pocols"], "pocols") << 40)
        | (_fit(porows, _b["porows"], "porows") << 32)
        | (_fit(pool_out_dim, _b["pool_out_dim"], "pool_out_dim") << 24)
        | (_fit(lpad, _b["lpad"], "lpad") << 10)
        | (_fit(upad, _b["upad"], "upad") << 8)
        | (_fit(pool_size, _b["pool_size"], "pool_size") << 6)
        | (_fit(pool_stride, _b["pool_stride"], "pool_stride") << 4)
        | (_fit(acc_act, _b["acc_act"], "acc_act") << 2)
        | CONFIG_SUBTYPE["ST"]
    )
    rs2 = (_f32_bits(acc_scale) << 32) | (int(stride_bytes) & 0xFFFFFFFF)
    return Instr("CONFIG", Imm(rs1), Imm(rs2), "config_st")


def _mvin(cls: str, dram: Operand, spad_addr: int, cols: int, rows: int, note: str) -> Instr:
    return Instr(cls, dram, Imm(_rows_cols_addr(rows, cols, spad_addr)), note)


def mvin(dram: Operand, spad_addr: int, cols: int, rows: int, note: str = "mvin") -> Instr:
    """``gemmini_extended_mvin`` -- load unit 0 (the matmul's A operand)."""
    return _mvin("MVIN", dram, spad_addr, cols, rows, note)


def mvin2(dram: Operand, spad_addr: int, cols: int, rows: int, note: str = "mvin2") -> Instr:
    """``gemmini_extended_mvin2`` -- load unit 1 (the stationary B operand)."""
    return _mvin("MVIN2", dram, spad_addr, cols, rows, note)


def mvin3(dram: Operand, spad_addr: int, cols: int, rows: int, note: str = "mvin3") -> Instr:
    """``gemmini_extended_mvin3`` -- load unit 2 (the D/bias operand, straight into the accumulator)."""
    return _mvin("MVIN3", dram, spad_addr, cols, rows, note)


def mvout(dram: Operand, local_addr: int, cols: int, rows: int, note: str = "mvout") -> Instr:
    """``gemmini_extended_mvout``."""
    return Instr("MVOUT", dram, Imm(_rows_cols_addr(rows, cols, local_addr)), note)


def preload(bd_addr: int, c_addr: int, bd_cols: int, bd_rows: int, c_cols: int, c_rows: int) -> Instr:
    """``gemmini_extended_preload`` -- stage B into the mesh and name the accumulator destination."""
    return Instr(
        "PRELOAD",
        Imm(_rows_cols_addr(bd_rows, bd_cols, bd_addr)),
        Imm(_rows_cols_addr(c_rows, c_cols, c_addr)),
        "preload",
    )


def compute(
    a_addr: int, a_cols: int, a_rows: int, preloaded: bool, bd_addr: int | None = None
) -> Instr:
    """``gemmini_extended_compute_preloaded`` / ``..._accumulated``.

    ``preloaded`` picks COMPUTE_PRELOADED (funct 4, the mesh flips to the weights the matching
    PRELOAD staged) over COMPUTE_ACCUMULATE (funct 5, the mesh keeps the weights it already holds).
    """
    bd = facts.GARBAGE_ADDR if bd_addr is None else bd_addr
    return Instr(
        "COMPUTE_PRELOADED" if preloaded else "COMPUTE_ACCUMULATE",
        Imm(_rows_cols_addr(a_rows, a_cols, a_addr)),
        Imm(_rows_cols_addr(facts.DIM, facts.DIM, bd)),
        "compute",
    )
