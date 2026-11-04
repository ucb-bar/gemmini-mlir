"""Exact nonzero summaries of explicitly encoded signed-byte panels.

Zero means the stored integer byte is zero. No source floating dtype or radix
precision implies this fact. Producers own initialization and complete joins;
consumers require the encoded storage and summary to remain unchanged together.
Block extents are supplied by the consumer, never inferred hardware facts.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodedI8Nonzero:
    planes: int
    rows: int
    reduction_length: int
    block_rows: int
    nonzero: tuple[bool, ...]

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.planes, self.rows, self.reduction_length, self.block_rows)):
            raise ValueError("positive explicit encoded extents required")
        blocks = (self.rows + self.block_rows - 1) // self.block_rows
        if (
            type(self.nonzero) is not tuple
            or len(self.nonzero) != self.planes * blocks
            or any(type(v) is not bool for v in self.nonzero)
        ):
            raise ValueError("complete immutable boolean panel metadata required")

    def active(self, plane: int, block: int) -> bool:
        blocks = (self.rows + self.block_rows - 1) // self.block_rows
        if not 0 <= plane < self.planes or not 0 <= block < blocks:
            raise ValueError("summary index outside encoded storage")
        return self.nonzero[plane * blocks + block]


def summarize_encoded_i8(
    values: Sequence[int], *, planes: int, rows: int, reduction_length: int, block_rows: int, transposed: bool = False
) -> EncodedI8Nonzero:
    """Reference summary for [plane,row,K] or [plane,K,row] storage."""
    if any(type(v) is not int or v <= 0 for v in (planes, rows, reduction_length, block_rows)):
        raise ValueError("positive explicit encoded extents required")
    if len(values) != planes * rows * reduction_length:
        raise ValueError("complete encoded storage required")
    blocks = (rows + block_rows - 1) // block_rows
    flags = [False] * (planes * blocks)
    for p in range(planes):
        for r in range(rows):
            for k in range(reduction_length):
                at = p * rows * reduction_length + (k * rows + r if transposed else r * reduction_length + k)
                value = values[at]
                if not isinstance(value, int) or not -128 <= value <= 127:
                    raise ValueError("explicit signed-i8 encoded domain required")
                flags[p * blocks + r // block_rows] |= value != 0
    return EncodedI8Nonzero(planes, rows, reduction_length, block_rows, tuple(flags))


def c_header() -> str:
    """Producer joins and independent validation; no target primitive emission.

    A row's OR is over uint8 casts of every stored signed byte. Summary zero
    proves every byte in all joined rows is zero; values such as -128 remain
    nonzero. Replacing signed bytes by their nonoverflowing absolute values
    preserves these flags. Consumers need nonaliasing immutable input storage.
    """
    return r"""#ifndef MERLIN_ENCODED_I8_ZEROS_H
#define MERLIN_ENCODED_I8_ZEROS_H
#include <stddef.h>
#include <stdint.h>
static inline void merlin_encoded_i8_nonzero_begin(uint8_t *flags,size_t planes,
                                                 size_t rows,size_t block_rows) {
  const size_t blocks=(rows+block_rows-1)/block_rows;
  for(size_t t=0;t<planes*blocks;t++)flags[t]=0;
}
static inline void merlin_encoded_i8_nonzero_join_row(uint8_t *flags,size_t plane,
    size_t row,size_t rows,size_t block_rows,uint8_t stored_byte_or) {
  const size_t blocks=(rows+block_rows-1)/block_rows;
  flags[plane*blocks+row/block_rows]|=stored_byte_or!=0;
}
/* Independent diagnostic recheck, outside performance intervals unless stated.
 * Inputs must have the declared complete plane-major span. */
static inline int merlin_encoded_i8_nonzero_verify(const int8_t *values,
    const uint8_t *flags,size_t planes,size_t rows,size_t k,size_t block_rows,
    int transposed) {
  if(!planes||!rows||!k||!block_rows)return 0;
  const size_t blocks=(rows+block_rows-1)/block_rows;
  for(size_t p=0;p<planes;p++)for(size_t b=0;b<blocks;b++) {
    uint8_t nonzero=0;
    for(size_t r=b*block_rows;r<rows&&r<(b+1)*block_rows;r++)
      for(size_t z=0;z<k;z++)
        nonzero|=values[p*rows*k+(transposed?z*rows+r:r*k+z)]!=0;
    if(flags[p*blocks+b]!=nonzero)return 0;
  }
  return 1;
}
#endif
"""
