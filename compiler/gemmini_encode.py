"""Terminal Gemmini RoCC encoder built from the public register bundle layouts.

This module encodes instructions only: callers own storage allocation, dependencies,
completion placement and numerical legality. Local addresses are raw 32-bit ISA
addresses, including accumulator, accumulation and readout selector flags.
"""
from __future__ import annotations

import struct
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerType, i8, i64
from xdsl.ir import SSAValue

DIM = 16
GARBAGE = 0xFFFFFFFF
ACC = 0x80000000
ACCUMULATE = 1 << 30
FULL = 1 << 29
ACC_ADD = ACC | ACCUMULATE
ACC_FULL = ACC | FULL


def _i64(fn, value):
    if isinstance(value, int):
        # IntegerAttr accepts the signed spelling of the complete 64-bit word.
        value &= (1 << 64) - 1
        return fn.const(value if value < (1 << 63) else value - (1 << 64))
    value = SSAValue.get(value)
    if isinstance(value.type, llvm.LLVMPointerType):
        return fn.add(llvm.PtrToIntOp(value, i64)).results[0]
    if value.type == i64:
        return value
    if isinstance(value.type, IntegerType) and value.type.width.data < 64:
        return fn.add(llvm.ZExtOp(value, i64)).results[0]
    raise TypeError(f"instruction operand requires an integer or pointer, got {value.type}")


def _pack(fn, *fields):
    """Pack (value, offset, slot width), masking SSA operands to their slots."""
    constant = 0
    dynamic = []
    for value, shift, width in fields:
        mask = (1 << width) - 1
        if isinstance(value, int):
            if not 0 <= value <= mask:
                raise ValueError(f"field value {value} exceeds unsigned {width}-bit slot")
            constant |= value << shift
        else:
            operand = _i64(fn, value)
            lookup = getattr(fn, '_gemmini_pack_bounds', None)
            interval = lookup(operand) if callable(lookup) else None
            # Actual counted-loop ranges and overflow-free integer expressions
            # establish unsigned slot containment; unknown fields keep masking.
            if interval is not None and 0 <= interval[0] <= interval[1] <= mask:
                part = operand
                stats = getattr(fn, '_gemmini_pack_bounds_stats', None)
                if stats is not None: stats['proven_masks_removed'] += 1
            else:
                part = fn.and_i(operand, _i64(fn, mask))
                stats = getattr(fn, '_gemmini_pack_bounds_stats', None)
                if stats is not None: stats['unknown_or_outside_masks_retained'] += 1
            dynamic.append(fn.shl_i(part, shift) if shift else part)
    result = _i64(fn, constant)
    for part in dynamic:
        result = fn.or_i(result, part)
    return result


def _f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def insn(fn, funct: int, rs1, rs2):
    """Emit custom3 (opcode 0x7b), xs1=xs2=1, xd=0, no destination."""
    if not isinstance(funct, int) or not 0 <= funct < 128:
        raise ValueError('RoCC funct must be an unsigned seven-bit integer')
    return fn.add(llvm.InlineAsmOp(
        f'.insn r 0x7b, 3, {funct}, x0, $0, $1',
        'r,r,~{memory}', [_i64(fn, rs1), _i64(fn, rs2)],
        has_side_effects=True))


def fence(fn):
    """Complete prior accelerator memory effects before subsequent host access.

    The canonical no-operand form assembles to 0x0ff0000f, exactly the same
    full iorw/iorw fence, and is recognized by the public instruction trace.
    """
    return fn.add(fence_op())


def fence_op():
    """Construct the common completion instruction for explicit block placement."""
    return llvm.InlineAsmOp('fence', '~{memory}', [], has_side_effects=True)


def flush_op():
    """Invalidate DMA address translations, retaining scratchpad contents.

    The selected funct table assigns FLUSH_CMD to 7. Zero rs1 retries rather
    than skips a faulting request; rs2 is unused. This bypasses command queues,
    so callers must complete preceding DMA before invalidating translations.
    """
    return llvm.InlineAsmOp('.insn r 0x7b, 3, 7, x0, x0, x0',
                            '~{memory}', [], has_side_effects=True)


def flush(fn):
    """Refresh the DMA TLB without discarding a pending memory request."""
    return fn.add(flush_op())


def pointer_offset(fn, ptr, byte_offset):
    """Return a pointer advanced by a byte offset (integer or SSA)."""
    if isinstance(ptr, int):
        return fn.add_i(_i64(fn, ptr), _i64(fn, byte_offset))
    ptr = SSAValue.get(ptr)
    if isinstance(ptr.type, llvm.LLVMPointerType):
        return fn.gep(ptr, _i64(fn, byte_offset), i8)
    return fn.add_i(_i64(fn, ptr), _i64(fn, byte_offset))


def config_ex(fn, *, dataflow='ws', activation=0, shift=0, a_stride=1, c_stride=1):
    """Configure execute. Strides are local rows, shift is integer input shift."""
    if dataflow not in ('ws', 'os', 0, 1):
        raise ValueError('dataflow must be ws, os, 1 or 0')
    flow = int(dataflow in ('ws', 1))
    rs1 = _pack(fn, (0, 0, 2), (flow, 2, 1), (activation, 3, 2),
                (a_stride, 16, 16), (_f32_bits(1.0), 32, 32))
    rs2 = _pack(fn, (shift, 0, 32), (c_stride, 48, 16))
    return insn(fn, 0, rs1, rs2)


def config_load(fn, stride_bytes, *, stream=0, scale=1.0, shrunk=False,
                block_stride=16, repeats=1):
    """Configure mvin stream 0/1/2; block_stride is in local rows."""
    if stream not in (0, 1, 2):
        raise ValueError('load stream must be 0, 1 or 2')
    rs1 = _pack(fn, (1, 0, 2), (int(shrunk), 2, 1), (stream, 3, 2),
                (repeats, 8, 8), (block_stride, 16, 16), (_f32_bits(scale), 32, 32))
    return insn(fn, 0, rs1, _i64(fn, stride_bytes))


def config_store(fn, stride_bytes, *, scale=1.0, activation=0):
    rs1 = _pack(fn, (2, 0, 2), (activation, 2, 2))
    rs2 = _pack(fn, (stride_bytes, 0, 32), (_f32_bits(scale), 32, 32))
    return insn(fn, 0, rs1, rs2)


def _matrix(fn, local, rows, cols):
    # MvinRs2/MvoutRs2/PreloadRs/ComputeRs share these 32/16/16 slots.
    return _pack(fn, (local, 0, 32), (cols, 32, 16), (rows, 48, 16))


def mvin(fn, ptr, local, rows, cols, stream=0):
    if stream not in (0, 1, 2):
        raise ValueError('load stream must be 0, 1 or 2')
    return insn(fn, (2, 1, 14)[stream], _i64(fn, ptr), _matrix(fn, local, rows, cols))


def mvout(fn, ptr, local, rows, cols):
    return insn(fn, 3, _i64(fn, ptr), _matrix(fn, local, rows, cols))


def preload(fn, bd, cs, br, bc, cr, cc):
    return insn(fn, 6, _matrix(fn, bd, br, bc), _matrix(fn, cs, cr, cc))


def compute(fn, a, bd, ar, ac, br, bc, accumulate=False):
    """Flip/preload weights, or retain weights with accumulate=True (stay).

    Accumulator addition is selected separately by bit30 of preload's cs address.
    """
    return insn(fn, 5 if accumulate else 4,
                _matrix(fn, a, ar, ac), _matrix(fn, bd, br, bc))
