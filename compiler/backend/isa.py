"""Terminal Gemmini custom-3 encoding from selected public RTL bundle facts.

Local addresses are caller-supplied encoded 32-bit addresses. This module does
not allocate memory or choose the accumulator/address metadata flags.
"""
from __future__ import annotations

import struct
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i64
from xdsl.ir import SSAValue

CUSTOM_OPCODE = 0x7B
FUNCT3 = 3
DIM = 16
GARBAGE_ADDR = 0xFFFFFFFF


def _value(builder, value: int | SSAValue) -> SSAValue:
    if isinstance(value, int):
        # llvm constants are signed textual i64, while packed fields are bit patterns.
        value &= (1 << 64) - 1
        return builder.const(value if value < (1 << 63) else value - (1 << 64))
    if value.type != i64:
        raise TypeError(f"RoCC operand must be i64, got {value.type}")
    return value


def ptr_int(builder, pointer: SSAValue) -> SSAValue:
    """Convert a logical kernel pointer to the RV64 RoCC address register."""
    if pointer.type == i64:
        return pointer
    if not isinstance(pointer.type, llvm.LLVMPointerType):
        raise TypeError("expected an LLVM pointer or i64 address")
    return builder.add(llvm.PtrToIntOp(pointer, i64)).results[0]


def emit(builder, funct: int, rs1: int | SSAValue, rs2: int | SSAValue):
    """Emit one side-effecting basic instruction; hardware loop opcodes are excluded."""
    if funct not in (0, 1, 2, 3, 4, 5, 6, 7, 14, 23):
        raise ValueError(f"unsupported basic Gemmini funct {funct}")
    return builder.add(llvm.InlineAsmOp(
        f".insn r {CUSTOM_OPCODE}, {FUNCT3}, {funct}, x0, $0, $1",
        "r,r,~{memory}", [_value(builder, rs1), _value(builder, rs2)],
        has_side_effects=True))


def fence(builder):
    """Emit the public reference's completion fence and a compiler memory barrier."""
    return builder.add(llvm.InlineAsmOp("fence", "~{memory}", [], has_side_effects=True))


def _field(value: int, width: int, name: str) -> int:
    if not isinstance(value, int) or not 0 <= value < (1 << width):
        raise ValueError(f"{name} must fit unsigned {width} bits")
    return value


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _pack(builder, fields):
    """Pack disjoint (value, offset, width) fields, allowing dynamic i64 fields."""
    fixed = 0
    dynamic = []
    for value, offset, width in fields:
        if isinstance(value, int):
            fixed |= _field(value, width, "packed field") << offset
        else:
            dynamic.append((value, offset, width))
    if not dynamic:
        return fixed
    result = _value(builder, fixed)
    for value, offset, width in dynamic:
        part = builder.and_i(_value(builder, value), _value(builder, (1 << width) - 1))
        if offset:
            part = builder.shl_i(part, offset)
        result = builder.or_i(result, part)
    return result


def matrix(builder, local: int | SSAValue, cols: int | SSAValue, rows: int | SSAValue):
    """Pack MvinRs2/MvoutRs2/PreloadRs/ComputeRs common address/shape slots."""
    return _pack(builder, [(local, 0, 32), (cols, 32, 16), (rows, 48, 16)])


def config_ex(builder, *, dataflow=1, activation=0, shift=0, a_stride=1,
              c_stride=1, a_transpose=False, b_transpose=False, scale=1.0,
              set_only_strides=False, relu6_shift=0):
    rs1 = _pack(builder, [(0, 0, 2), (dataflow, 2, 1), (activation, 3, 2),
        (int(set_only_strides), 7, 1), (int(a_transpose), 8, 1),
        (int(b_transpose), 9, 1), (a_stride, 16, 16), (f32_bits(scale), 32, 32)])
    rs2 = _pack(builder, [(shift, 0, 32), (relu6_shift, 32, 16), (c_stride, 48, 16)])
    return emit(builder, 0, rs1, rs2)


def config_ld(builder, stride: int | SSAValue, *, state=0, scale=1.0,
              shrink=False, block_stride=16, pixel_repeats=1):
    _field(state, 2, "load state")
    if state > 2:
        raise ValueError("load state must be 0, 1, or 2")
    rs1 = _pack(builder, [(1, 0, 2), (int(shrink), 2, 1), (state, 3, 2),
        (pixel_repeats, 8, 8), (block_stride, 16, 16), (f32_bits(scale), 32, 32)])
    return emit(builder, 0, rs1, stride)


def config_st(builder, stride: int | SSAValue, *, scale=1.0, activation=0,
              scale_bits: int | SSAValue | None = None):
    """Configure readout; scale_bits overrides scale with its exact f32 payload.

    Dynamic payloads use the standard i64 register representation; only their
    low 32 bits are packed into ConfigMvoutRs2.acc_scale. No float conversion occurs.
    """
    rs1 = _pack(builder, [(2, 0, 2), (activation, 2, 2)])
    payload = f32_bits(scale) if scale_bits is None else scale_bits
    rs2 = _pack(builder, [(stride, 0, 32), (payload, 32, 32)])
    return emit(builder, 0, rs1, rs2)


def mvin(builder, dram: int | SSAValue, local, cols, rows, *, state=0):
    if state not in (0, 1, 2):
        raise ValueError("load state must be 0, 1, or 2")
    address = ptr_int(builder, dram) if isinstance(dram, SSAValue) else dram
    return emit(builder, (2, 1, 14)[state], address, matrix(builder, local, cols, rows))


def preload(builder, bd, cd, bd_cols=16, bd_rows=16, cd_cols=16, cd_rows=16):
    return emit(builder, 6, matrix(builder, bd, bd_cols, bd_rows),
                matrix(builder, cd, cd_cols, cd_rows))


def compute(builder, a, bd, a_cols=16, a_rows=16, bd_cols=16, bd_rows=16, *, flip=True):
    return emit(builder, 4 if flip else 5, matrix(builder, a, a_cols, a_rows),
                matrix(builder, bd, bd_cols, bd_rows))


def mvout(builder, dram: int | SSAValue, local, cols, rows):
    address = ptr_int(builder, dram) if isinstance(dram, SSAValue) else dram
    return emit(builder, 3, address, matrix(builder, local, cols, rows))
