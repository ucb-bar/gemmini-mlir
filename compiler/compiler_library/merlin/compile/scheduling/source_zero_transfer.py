"""Complete exact zero source effects over a checked same-type movement plan.

A zero source is not a host memory read. The target owner must bind the NULL
source transfer semantics, raw zero-preserving readout and completion boundaries.
"""
from dataclasses import dataclass, replace
from math import prod
from .layout_transfer import LayoutTransfer


@dataclass(frozen=True)
class SourceZeroTransfer:
    shape: tuple[int, ...]
    dtype: str
    tile: int

    def __post_init__(self):
        if (type(self.shape) is not tuple or self.dtype not in ('i8','i32')
            or type(self.tile) is not int or not 0 < self.tile <= 16
            or any(type(v) is not int or not 0 < v < 1 << 31 for v in self.shape)
            or prod(self.shape)*self.itemsize >= 1 << 31):
            raise ValueError('complete bounded static integer zero tensor required')
        self.movement.__post_init__()

    @property
    def itemsize(self):
        return 1 if self.dtype == 'i8' else 4

    @property
    def movement(self):
        shape = self.shape or (1,)
        return LayoutTransfer(shape,tuple(range(len(shape))),self.dtype,self.tile)

    def emit(self,builder,consume):
        """Retain original complete output mapping and zero-source effects."""
        self.__post_init__()
        self.movement.emit(builder,lambda run:consume(replace(run,source_element=None,
            source_stride=0,reads='none_selected_null_source_zero',
            writes='complete_distinct_original_result')))
