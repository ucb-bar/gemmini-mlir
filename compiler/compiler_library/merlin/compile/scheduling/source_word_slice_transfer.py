"""Native i64 slices as adjacent raw32 pairs of existing logical slice runs."""
from dataclasses import dataclass,replace
from math import prod
from .source_slice_transfer import SourceSliceTransfer


@dataclass(frozen=True)
class SourceWordSliceTransfer:
    source_shape: tuple[int,...]
    offsets: tuple[int,...]
    sizes: tuple[int,...]
    strides: tuple[int,...]
    result_shape: tuple[int,...]
    dtype: str = 'i64'
    tile: int = 8

    def __post_init__(self):
        if self.dtype!='i64' or type(self.tile)is not int or self.tile!=8:
            raise ValueError('native i64 requires adjacent pairs and eight-element logical tiles')
        self.movement.__post_init__()
        if prod(self.source_shape)*8>=1<<31:
            raise ValueError('native i64 slice source exceeds bounded raw byte addressing')

    @property
    def movement(self):
        return SourceSliceTransfer(self.source_shape,self.offsets,self.sizes,
            self.strides,self.result_shape,'i32',self.tile)

    def emit(self,builder,consume):
        self.__post_init__()
        def pair(run):
            consume(replace(run,source_element=builder.op('mul',run.source_element,2),
                output_element=builder.op('mul',run.output_element,2),
                columns=run.columns*2,source_stride=run.source_stride*2,
                output_stride=run.output_stride*2,
                reads='both adjacent raw32 words of original logical slice element',
                writes='both adjacent raw32 words of complete disjoint i64 output'))
        self.movement.emit(builder,pair)
