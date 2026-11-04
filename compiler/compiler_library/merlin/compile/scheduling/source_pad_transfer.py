"""Complete static padding as disjoint raw border fills and an interior copy.

Border slabs are partitioned by the first axis outside the source interior.
The coordinate map reuses the checked source slice map in reverse; no source
payload is evaluated, zero-filled, or overwritten by another output region.
"""
from dataclasses import dataclass
from .source_slice_transfer import SourceSliceTransfer
from .layout_transfer import LayoutRun

@dataclass(frozen=True)
class SourcePadTransfer:
    source_shape: tuple[int, ...]
    low: tuple[int, ...]
    high: tuple[int, ...]
    dtype: str
    tile: int = 16

    def __post_init__(self):
        if (any(type(x) is not tuple for x in (self.source_shape,self.low,self.high))
            or len(self.low)!=len(self.source_shape) or len(self.high)!=len(self.source_shape)
            or any(type(x) is not int or x<0 for x in (*self.low,*self.high))):
            raise ValueError('static nonnegative padding required')
        self.interior.__post_init__()
        for slab in self.borders:
            slab.__post_init__()

    @property
    def itemsize(self):
        return 1 if self.dtype=='i8' else 4

    @property
    def output_shape(self):
        return tuple(s+l+h for s,l,h in zip(self.source_shape,self.low,self.high))

    def _slice(self,offsets,sizes):
        return SourceSliceTransfer(self.output_shape or (1,),offsets or (0,),
            sizes or (1,),(1,)*max(1,len(sizes)),sizes or (1,),self.dtype,self.tile)

    @property
    def interior(self):
        return self._slice(self.low,self.source_shape)

    @property
    def borders(self):
        slabs=[]
        for axis,(lo,hi) in enumerate(zip(self.low,self.high)):
            for size,start in ((lo,0),(hi,self.low[axis]+self.source_shape[axis])):
                if not size:
                    continue
                sizes=self.source_shape[:axis]+(size,)+self.output_shape[axis+1:]
                offsets=self.low[:axis]+(start,)+(0,)*(len(self.low)-axis-1)
                slabs.append(self._slice(offsets,sizes))
        return tuple(slabs)

    def emit(self,builder,consume_fill,consume_source):
        self.__post_init__()
        def fill(run):
            # Convert a contiguous source-slice row to scalar-stride-zero rows.
            # Slice runs are either one contiguous row or a strided column.
            contiguous=run.rows==1
            consume_fill(LayoutRun(0,run.source_element,
                run.columns if contiguous else run.rows,1,0,
                1 if contiguous else run.source_stride,
                reads='original_padding_scalar',writes='disjoint_complete_padding_border'))
        for slab in self.borders:
            slab.emit(builder,fill)
        def copy(run):
            consume_source(LayoutRun(run.output_element,run.source_element,
                run.rows,run.columns,run.output_stride,run.source_stride,
                reads='complete_original_padding_source',writes='disjoint_padding_interior'))
        self.interior.emit(builder,copy)
