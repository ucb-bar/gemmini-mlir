"""Pair-preserving native i64 permutations using the existing logical planner."""
from dataclasses import dataclass, replace
from math import prod
from .layout_transfer import LayoutTransfer

@dataclass(frozen=True)
class SourceWordLayoutTransfer:
    source_shape: tuple[int, ...]
    permutation: tuple[int, ...]
    dtype: str = 'i64'
    tile: int = 8

    def __post_init__(self):
        if self.dtype != 'i64' or self.tile != 8 or prod(self.source_shape)*8 >= 1 << 31:
            raise ValueError('bounded native i64 word pairs required')
        self.movement.__post_init__()

    @property
    def movement(self):
        return LayoutTransfer(self.source_shape,self.permutation,'i32',self.tile)

    @property
    def output_shape(self):
        return self.movement.output_shape

    @property
    def itemsize(self):
        return 8

    def emit(self,builder,consume):
        self.__post_init__()
        # Both words of each original element retain increasing address order.
        # Doubling the complete logical bijection preserves pair adjacency and
        # disjoint coverage, including each ragged final run.
        def pair(run):
            consume(replace(run,
                source_element=builder.op('mul',run.source_element,2),
                output_element=builder.op('mul',run.output_element,2),
                columns=run.columns*2,source_stride=run.source_stride*2,
                output_stride=run.output_stride*2))
        self.movement.emit(builder,pair)
