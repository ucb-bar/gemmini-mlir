"""Constant tensor initialization through authored Gemmini DMA commands.

The NULL source is LoadController's all_zeros command (FIRRTL 229229);
stride zero repeats a constant source row (229264-229270). Each full-width
readout remains one 16-column block, per StoreController.scala:225.
"""
from __future__ import annotations

import math
import re
import struct


_HEADER = r'''
#ifndef GM_DEVICE_FILL_DEFINED
#define GM_DEVICE_FILL_DEFINED
#define DF_INLINE static __attribute__((always_inline)) inline
typedef unsigned long df_u64;
typedef unsigned int df_u32;
#define DF_INSN(F,A,B) __asm__ volatile (".insn r 0x7b, 3, " #F ", x0, %0, %1" :: "r"((df_u64)(A)), "r"((df_u64)(B)) : "memory")
DF_INLINE df_u64 df_shape(df_u32 address,int rows,int cols) {
  return (df_u64)address | ((df_u64)(unsigned)cols<<32) | ((df_u64)(unsigned)rows<<48);
}

#endif
'''


def emit_fill(dtype, value, count, dst, tag):
    """Return (global declarations, body statements), or None if unsupported.

    ``value`` must be the already parsed scalar constant. Out-of-range integer
    literals retain the declared dtype's low bits, matching MLIR integers.
    Finite f32 constants are rounded to IEEE binary32 and copied as raw i32
    bits through the full-width accumulator path, including signed zero.
    Nonempty f32 fills below 256 elements retain their existing lowering.
    """
    if dtype == 'f32':
        if not isinstance(value, (int, float)):
            return None
        try:
            scalar = float(value)
            if not math.isfinite(scalar):
                return None
            raw = struct.pack('<f', scalar)
        except (OverflowError, struct.error):
            return None
        value = int.from_bytes(raw, 'little')
    elif dtype not in ('i8', 'i32') or not isinstance(value, int):
        return None
    if type(count) is not int or count < 0:
        raise ValueError('device fill requires a nonnegative static element count')
    if count == 0:
        return [], []
    if dtype == 'f32' and count < 256:
        return None
    if not isinstance(tag, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', tag):
        raise ValueError('device fill tag must be a C identifier')
    width = 1 if dtype == 'i8' else 4
    bits = width * 8
    signed = value & ((1 << bits) - 1)
    if signed >= 1 << (bits - 1):
        signed -= 1 << bits
    globals_ = [_HEADER]
    if signed == 0:
        row = '(const void*)0'
    else:
        row = tag + '_fill_row'
        ctype = 'signed char' if width == 1 else 'int'
        # Avoid the unsigned decimal type of C's 2147483648 token.
        literal = '(-2147483647-1)' if signed == -(1 << 31) else str(signed)
        globals_.append(f'static const {ctype} {row}[16] __attribute__((aligned(64))) = {{{",".join([literal] * 16)}}};')
    # Resolve all geometry here so emitted control has literal positive steps
    # and bounds. In particular, no dynamic min/modulo or conditional tails
    # remain for the host control-flow audit to prove after LLVM inlining.
    fence = '__asm__ volatile("fence" ::: "memory");'
    load_config = (0x3f800000 << 32) | (16 << 16) | ((width == 1) << 2) | 1
    store_config = (0x3f800000 << 32) | (16 * width)
    body = [fence, f'DF_INSN(0,{load_config}UL,0);',
            f'DF_INSN(0,2,{store_config}UL);']
    read_base = '0xa0000000u' if width == 4 else '0x80000000u'

    def transfer(offset, rows, cols):
        return [f'DF_INSN(2,{row},df_shape(0x80000000u,{rows},{cols}));',
                f'DF_INSN(3,(unsigned char*)({dst})+({offset}),'
                f'df_shape({read_base},{rows},{cols}));']

    blocks, remaining = divmod(count, 256)
    if blocks:
        induction = tag + '_fill_block'
        body.append(f'for(int {induction}=0;{induction}<{blocks};{induction}+=1) {{')
        body.extend(transfer(f'(df_u64){induction}*{256 * width}', 16, 16))
        body.append('}')
    rows, tail = divmod(remaining, 16)
    if rows:
        body.extend(transfer(blocks * 256 * width, rows, 16))
    if tail:
        body.extend(transfer((blocks * 256 + rows * 16) * width, 1, tail))
    body.append(fence)
    return globals_, body
