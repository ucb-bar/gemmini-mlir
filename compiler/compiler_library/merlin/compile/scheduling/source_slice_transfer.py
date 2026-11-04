"""Source-coordinate exact slice copies with explicit raw transfer effects.

Rank reduction drops only unit size axes and leaves C-order element order
unchanged. The execution owner grants disjoint buffers and raw same-type device
transfers; this module adds no payload arithmetic or target instruction.
"""
from dataclasses import dataclass
from math import prod
from .layout_transfer import LayoutRun


@dataclass(frozen=True)
class SourceSliceTransfer:
    source_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    result_shape: tuple[int, ...]
    dtype: str
    tile: int

    def __post_init__(self):
        if (self.dtype not in ('i8','i32','f32') or type(self.tile) is not int or not 0 < self.tile <= 16
            or any(type(v) is not tuple for v in (self.source_shape,self.offsets,self.sizes,self.strides,self.result_shape))
            or not self.source_shape or any(type(v) is not int for row in (self.source_shape,self.offsets,self.sizes,self.strides,self.result_shape) for v in row)
            or any(not 0 < v < 1 << 31 for row in (self.source_shape,self.sizes,self.strides,self.result_shape) for v in row)
            or any(not 0 <= v < 1 << 31 for v in self.offsets)
            or any(len(row) != len(self.source_shape) for row in (self.offsets,self.sizes,self.strides))
            or prod(self.source_shape)*self.itemsize >= 1 << 31):
            raise ValueError('exact bounded positive static same-type slice required')
        if any(o+(n-1)*s >= d for o,n,s,d in zip(self.offsets,self.sizes,self.strides,self.source_shape)):
            raise ValueError('slice exceeds complete original source bounds')
        kept = 0
        for size in self.sizes:
            if kept < len(self.result_shape) and size == self.result_shape[kept]:
                kept += 1
            elif size != 1:
                raise ValueError('rank reduction may remove only singleton slice axes')
        if kept != len(self.result_shape):
            raise ValueError('rank-reduced result geometry differs from source slice')

    @property
    def itemsize(self):
        return 1 if self.dtype == 'i8' else 4

    @property
    def source_pitches(self):
        return tuple(prod(self.source_shape[d+1:]) for d in range(len(self.source_shape)))

    @property
    def first_element(self):
        return sum(o*p for o,p in zip(self.offsets,self.source_pitches))

    @property
    def contiguous_suffix(self):
        count = 1
        for n,s,p in reversed(tuple(zip(self.sizes,self.strides,self.source_pitches))):
            if n > 1 and s*p != count:
                break
            count *= n
        return count

    @property
    def run_geometry(self):
        contiguous = self.contiguous_suffix
        if contiguous > 1:
            return contiguous, 1, True
        axis = next((d for d in reversed(range(len(self.sizes))) if self.sizes[d] > 1), None)
        if axis is None:
            return 1, 1, True
        return self.sizes[axis], self.source_pitches[axis]*self.strides[axis], False

    @property
    def is_contiguous(self):
        return self.contiguous_suffix == prod(self.sizes)

    def emit(self, builder, consume):
        """Emit only counted address loops; consumers retain each run's effects."""
        self.__post_init__()
        extent, step, contiguous = self.run_geometry
        pitches = tuple(prod(self.sizes[d+1:]) for d in range(len(self.sizes)))
        def outer(group):
            first = builder.op('mul', group, extent)
            source = self.first_element
            for n,pitch,source_pitch,stride in zip(self.sizes,pitches,self.source_pitches,self.strides):
                coordinate = builder.op('urem',builder.op('udiv',first,pitch),n)
                source = builder.op('add',source,builder.op('mul',coordinate,source_pitch*stride))
            def run(begin,count):
                consume(LayoutRun(builder.op('add',source,builder.op('mul',begin,step)),
                    builder.op('add',first,begin), 1 if contiguous else count,
                    count if contiguous else 1, count if contiguous else step,
                    count if contiguous else 1, reads='exact_original_source_slice'))
            full = extent//self.tile*self.tile
            if full:
                builder.loop(0,full,self.tile,lambda begin:run(begin,self.tile))
            if full < extent:
                run(full,extent-full)
        builder.loop(0,prod(self.sizes)//extent,1,outer)
