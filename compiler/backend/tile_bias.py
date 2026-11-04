"""Typed signed-i32 accumulator initialization for shared tiled matmul.

Admission is pure. Emission uses only the common state-2 DMA encoder: no host
payload loads, arithmetic or copying. The contraction schedule owns every local
accumulator row and must initialize it once before its first accumulating MAC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from xdsl.dialects import llvm
from xdsl.dialects.builtin import i32
from xdsl.ir import SSAValue

from . import isa


def has_bias(attrs: Mapping) -> bool:
    value = attrs.get('bias')
    return value not in (None, False, '', 'none')


@dataclass(frozen=True)
class BiasPreparation:
    """An i32 source under the caller's row-major bias operand contract."""

    source: str | SSAValue
    n: int
    broadcast: bool
    stride_bytes: int
    element_type = i32
    compute_flags = 0xC0000000

    def matches(self, attrs: Mapping, n: int) -> bool:
        value = attrs.get('bias')
        same = value == self.source if isinstance(self.source, str) else value is self.source
        return bool(same and n == self.n
                    and bool(attrs.get('bias_broadcast', True)) == self.broadcast)

    def configure(self, f):
        return isa.config_ld(f, self.stride_bytes, state=2, shrink=False)

    def preload(self, f, pointer, start, column, local, cols, rows):
        """Initialize exactly the active lanes/rows of one scheduled accumulator.

        `pointer` may already point at a convolution row-block's bias slice.
        `start` is relative to that pointer, never the whole source tensor.
        """
        index = column if self.broadcast else f.add_i(f.mul_i(start, f.const(self.n)), column)
        return isa.mvin(f, f.gep(pointer, index, i32),
                        f.or_i(f.const(0x80000000), local), cols, rows, state=2)


def select_bias(attrs: Mapping, m: int, n: int,
                tensors: Mapping) -> BiasPreparation | None:
    """Admit declared signed-i32 bias or a caller-typed LLVM pointer.

    Named operands must declare a broadcast vector or a row-major matrix with
    enough rows for this invocation. An opaque pointer is the existing common
    matmul API's typed i32 bias pointer, possibly already offset by its caller;
    it retains that caller's element-type/extent proof rather than inventing one.
    """
    if (not has_bias(attrs) or any(type(d) is not int or d <= 0 for d in (m, n))
            or n * 4 >= 1 << 32):
        return None
    value = attrs['bias']
    broadcast = bool(attrs.get('bias_broadcast', True))
    if isinstance(value, str):
        spec = tensors.get(value)
        if not isinstance(spec, Mapping) or spec.get('dtype') != 'i32':
            return None
        shape = spec.get('shape', [])
        if not isinstance(shape, (tuple, list)) or any(type(d) is not int or d <= 0 for d in shape):
            return None
        if broadcast:
            if list(shape) not in ([n], [1, n]):
                return None
        elif len(shape) != 2 or shape[0] < m or shape[1] != n:
            return None
    elif not isinstance(value, SSAValue) or not isinstance(value.type, llvm.LLVMPointerType):
        return None
    return BiasPreparation(value, n, broadcast, 0 if broadcast else n * 4)
