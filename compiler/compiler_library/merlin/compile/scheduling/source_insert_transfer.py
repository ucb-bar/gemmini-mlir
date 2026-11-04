"""Exact static insertion: complete immutable seed copy, then raw patch stores.

Reuse the extraction coordinate map in reverse, retaining its static bounds and
rank-reduction proof. No tensor payload arithmetic or target encoding is added.
"""
from dataclasses import dataclass
from .source_slice_transfer import SourceSliceTransfer
from .layout_transfer import LayoutRun

@dataclass(frozen=True)
class SourceInsertTransfer:
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    tile: int

    def __post_init__(self):
        self.patch.__post_init__()
        self.seed.__post_init__()

    @property
    def itemsize(self):
        return 1 if self.dtype == 'i8' else 4

    @property
    def patch(self):
        return SourceSliceTransfer(self.destination_shape,self.offsets,self.sizes,
            self.strides,self.source_shape,self.dtype,self.tile)

    @property
    def seed(self):
        shape=self.destination_shape
        return SourceSliceTransfer(shape,(0,)*len(shape),shape,(1,)*len(shape),shape,self.dtype,self.tile)

    def emit(self,builder,consume_seed,consume_patch):
        self.__post_init__()
        self.seed.emit(builder,consume_seed)
        def reverse(run):
            consume_patch(LayoutRun(run.output_element,run.source_element,run.rows,
                run.columns,run.output_stride,run.source_stride,
                reads='complete_original_insert_source',writes='exact_fresh_result_patch'))
        self.patch.emit(builder,reverse)
