"""Exact scalar broadcast over the existing complete static layout mapping.

The target supplies one immutable materialized source element, distinct output,
and synchronous raw transfers. No payload arithmetic or scalar value is inferred.
"""
from dataclasses import dataclass, replace
from math import prod
from .layout_transfer import LayoutTransfer

@dataclass(frozen=True)
class SourceBroadcastTransfer:
    shape: tuple[int, ...]
    dtype: str
    tile: int = 16

    def __post_init__(self):
        if (type(self.shape) is not tuple or self.dtype not in ('i8','i32','f32')
            or any(type(v) is not int or not 0 < v < 1 << 31 for v in self.shape)
            or prod(self.shape)*self.itemsize >= 1 << 31):
            raise ValueError('complete bounded static same-type scalar broadcast required')
        self.movement.__post_init__()

    @property
    def itemsize(self):
        return 1 if self.dtype=='i8' else 4

    @property
    def movement(self):
        shape=self.shape or (1,)
        return LayoutTransfer(shape,tuple(range(len(shape))),self.dtype,self.tile)

    def emit(self,builder,consume):
        self.__post_init__()
        # Identity movement may return a whole 2D contiguous panel. Split its
        # original rows, then turn each row's columns into scalar-repeat rows.
        # Preserve every original output base and pitch; do not cross a panel.
        def run(original):
            if (not 0 < original.rows <= self.tile
                or not 0 < original.columns <= self.tile
                or original.output_stride != original.columns):
                raise ValueError('broadcast requires bounded contiguous identity panels')
            def row(index):
                output=builder.op('add',original.output_element,
                    builder.op('mul',index,original.output_stride))
                consume(replace(original,source_element=0,output_element=output,
                    rows=original.columns,columns=1,source_stride=0,output_stride=1,
                    reads='original_materialized_scalar',
                    writes='complete_distinct_original_result'))
            builder.loop(0,original.rows,1,row)
        self.movement.emit(builder,run)
