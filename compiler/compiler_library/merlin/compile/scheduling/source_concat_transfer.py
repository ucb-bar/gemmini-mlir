"""Exact original concat coordinates as bounded raw LayoutRun transfers.

Every input owns a disjoint interval on the result's concat axis. Adjacent
suffix axes retain their original C order. This module supplies address loops,
not payload arithmetic, target instructions, alias permission or timing.
"""
from dataclasses import dataclass
from math import prod
from .layout_transfer import LayoutRun


@dataclass(frozen=True)
class SourceConcatTransfer:
    input_shapes: tuple[tuple[int, ...], ...]
    axis: int
    result_shape: tuple[int, ...]
    dtype: str
    tile: int

    def __post_init__(self):
        if (type(self.input_shapes) is not tuple or not self.input_shapes
            or type(self.result_shape) is not tuple or not self.result_shape
            or any(type(s) is not tuple for s in self.input_shapes)
            or type(self.axis) is not int or not 0 <= self.axis < len(self.result_shape)
            or self.dtype not in ('i8', 'i32', 'f32') or type(self.tile) is not int
            or not 0 < self.tile <= 16
            or any(type(d) is not int or not 0 <= d < 1 << 31
                   for s in (*self.input_shapes, self.result_shape) for d in s)
            or any(len(s) != len(self.result_shape) for s in self.input_shapes)
            or any(prod(s)*self.itemsize >= 1 << 31
                   for s in (*self.input_shapes, self.result_shape))):
            raise ValueError('bounded static same-type ranked concat required')
        if any(s[d] != self.result_shape[d] for s in self.input_shapes
               for d in range(len(s)) if d != self.axis):
            raise ValueError('non-concat axes differ from complete original result')
        if sum(s[self.axis] for s in self.input_shapes) != self.result_shape[self.axis]:
            raise ValueError('concat intervals do not exactly partition original result')

    @property
    def itemsize(self):
        return 1 if self.dtype == 'i8' else 4

    @property
    def outer_extent(self):
        return prod(self.result_shape[:self.axis])

    @property
    def inner_extent(self):
        return prod(self.result_shape[self.axis+1:])

    def emit(self, builder, consume):
        """Call consume(input_index, LayoutRun) for every original input span.

        Runs have one row and at most tile contiguous columns, so native
        transfer strides stay bounded even when a full original pitch is large.
        Zero-sized inputs or results have no payload transfers.
        """
        self.__post_init__()
        if not prod(self.result_shape):
            return
        output_extent = self.result_shape[self.axis]*self.inner_extent
        prefix = 0
        for index, shape in enumerate(self.input_shapes):
            extent = shape[self.axis]*self.inner_extent
            if extent:
                def outer(group):
                    source_base = builder.op('mul', group, extent)
                    output_base = builder.op('add', builder.op('mul', group, output_extent), prefix)
                    def run(first, count):
                        consume(index, LayoutRun(builder.op('add', source_base, first),
                            builder.op('add', output_base, first), 1, count, count, count,
                            reads='complete_original_concat_input'))
                    full = extent//self.tile*self.tile
                    if full:
                        builder.loop(0, full, self.tile, lambda first: run(first, self.tile))
                    if full < extent:
                        run(full, extent-full)
                builder.loop(0, self.outer_extent, 1, outer)
            prefix += extent
