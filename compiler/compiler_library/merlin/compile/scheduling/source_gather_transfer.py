"""Complete indexed rows composed from the existing exact source-slice runs.

Runtime indices retain tensor.extract's in-bounds precondition. This schedule
introduces no wrapping, clamping, masking or behavior for undefined source cases.
The adapter lowers the original index producer and grants raw transfer effects.
"""
from dataclasses import dataclass
from math import prod
from .source_slice_transfer import SourceSliceTransfer
from .layout_transfer import LayoutRun


@dataclass(frozen=True)
class SourceRowGather:
    source_shape: tuple[int,int]
    index_shape: tuple[int,...]
    dtype: str = 'f32'
    tile: int = 16

    def __post_init__(self):
        if (self.dtype!='f32' or self.tile!=16 or type(self.source_shape)is not tuple
                or len(self.source_shape)!=2 or type(self.index_shape)is not tuple
                or not self.index_shape or any(type(x)is not int or not 0<x<1<<31
                    for x in (*self.source_shape,*self.index_shape))
                or max(prod(self.source_shape)*4,prod(self.index_shape)*8,
                       prod(self.output_shape)*4)>=1<<31):
            raise ValueError('bounded static complete f32 rows and signed i64 index source required')
        self.row_slice.__post_init__()

    @property
    def output_shape(self):return (*self.index_shape,self.source_shape[1])

    @property
    def row_slice(self):
        width=self.source_shape[1]
        return SourceSliceTransfer((width,),(0,),(width,),(1,),(width,),self.dtype,self.tile)

    def emit(self,builder,load_original_index,consume):
        """One original signed row index per complete output row; no payload CPU op."""
        self.__post_init__()
        width=self.source_shape[1]
        def row(output_row):
            source_row=load_original_index(output_row)
            source_base=builder.op('mul',source_row,width)
            output_base=builder.op('mul',output_row,width)
            def run(part):
                consume(LayoutRun(builder.op('add',source_base,part.source_element),
                    builder.op('add',output_base,part.output_element),
                    part.rows,part.columns,part.source_stride,part.output_stride,
                    reads='original tensor.extract in-bounds row and exact column'))
            self.row_slice.emit(builder,run)
        builder.loop(0,prod(self.index_shape),1,row)
