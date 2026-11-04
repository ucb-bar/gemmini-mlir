"""Optional contiguous multi-block RHS DMA for adjacent matmul columns.

The public tinyllama_raw/exo_llm.c mm_down example (lines193-205) configures
load-state1 block stride704, loads64 columns, then preloads the four blocks at
8192+704*column. Its state0 configuration uses block stride16 with64-column
loads. This helper applies that same disclosed block layout with stride16.
The64-byte reader request limit is an upper bound, not a guaranteed burst size.
"""
from __future__ import annotations

from pathlib import Path

from merlin.compile.tuning_knobs import knob
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i8, i64
from xdsl.ir import SSAValue

from . import isa

ROOT = Path(__file__).resolve().parents[1]
MAX_COLUMNS = 64
BASE_ROW = 16


def _index(value):
    return type(value) is int or isinstance(value, SSAValue) and value.type == i64


def try_load_columns(f, b, n: int, kk: int | SSAValue, nj: int | SSAValue,
                     widths, tk: int | SSAValue) -> bool:
    """Emit one state1 MVIN into row16+16*column, or False without IR effects.

    Caller supplies original row-major i8 B[K,N] and guarantees source bounds
    for dynamic kk,nj,tk, with1<=tk<=16. State1 must already be configured for
    DRAM stride N, block_stride16, identity scale, shrink=False and one pixel
    repeat. The destination blocks must be reserved until consumers complete.
    This is the adjacent-column scheduler's existing configuration/layout;
    transposed, packed or differently configured inputs must not use this API.

    Widths are static adjacent blocks, each16 columns except an optional last
    partial block. Dynamic dimensions are ordinary i64 CPU address registers.
    No CPU tensor values are read and no hardware-loop command is emitted.
    """
    if knob('column_dma', default='tile', values=('tile', 'wide'), root=ROOT) != 'wide':
        return False
    if (type(n) is not int or n <= 0
            or not isinstance(b, SSAValue) or not isinstance(b.type, llvm.LLVMPointerType)
            or not all(_index(v) for v in (kk, nj, tk))
            or not isinstance(widths, (tuple, list)) or not 1 <= len(widths) <= 4
            or any(type(w) is not int or not 1 <= w <= isa.DIM for w in widths)
            or any(w != isa.DIM for w in widths[:-1])):
        return False
    total = sum(widths)
    if total > min(MAX_COLUMNS, n):
        return False
    if (type(kk) is int and kk < 0
            or type(nj) is int and (nj < 0 or nj + total > n)
            or type(tk) is int and not 1 <= tk <= isa.DIM):
        return False
    value = lambda x: f.const(x) if type(x) is int else x
    offset = f.add_i(f.mul_i(value(kk), f.const(n)), value(nj))
    isa.mvin(f, f.gep(b, offset, i8), BASE_ROW, total, tk, state=1)
    return True
