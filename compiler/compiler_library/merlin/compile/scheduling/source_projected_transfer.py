"""Complete static projected tensor copies as existing raw LayoutRun effects.

Source axes select one original output coordinate or constant zero. The target
owns raw storage, disjoint full buffers, field widths and synchronous completion.
No payload arithmetic or target instruction is introduced here.
"""
from dataclasses import dataclass
from math import prod
from .layout_transfer import LayoutRun


@dataclass(frozen=True)
class SourceProjectedTransfer:
    source_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    source_axes: tuple[int | None, ...]
    dtype: str
    tile: int = 16

    def __post_init__(self):
        if (any(type(row) is not tuple for row in (self.source_shape,self.output_shape,self.source_axes))
            or self.dtype not in ('i8','i32','f32') or type(self.tile) is not int or not 0 < self.tile <= 16
            or any(type(n) is not int or not 0 < n < 1 << 31 for row in (self.source_shape,self.output_shape) for n in row)
            or any(prod(row)*self.itemsize >= 1 << 31 for row in (self.source_shape,self.output_shape))
            or len(self.source_axes) != len(self.source_shape)):
            raise ValueError('complete bounded static same-type projected copy required')
        selected=[]
        for size,axis in zip(self.source_shape,self.source_axes):
            if axis is None:
                continue
            if (type(axis) is not int or not 0 <= axis < len(self.output_shape)
                or self.output_shape[axis] > size or axis in selected):
                raise ValueError('source projection requires distinct in-bounds original output axes')
            selected.append(axis)

    @property
    def itemsize(self):
        return 1 if self.dtype == 'i8' else 4

    @property
    def source_steps(self):
        steps=[0]*len(self.output_shape)
        for dim,axis in enumerate(self.source_axes):
            if axis is not None:
                steps[axis]=prod(self.source_shape[dim+1:])
        return tuple(steps)

    @property
    def contiguous_suffix(self):
        count=1
        for size,step in reversed(tuple(zip(self.output_shape,self.source_steps))):
            if size > 1 and step != count:
                break
            count*=size
        return count

    @property
    def run_geometry(self):
        count=self.contiguous_suffix
        if count > 1 or not self.output_shape:
            return count,1,True
        return self.output_shape[-1],self.source_steps[-1],False

    @property
    def maximum_stride_elements(self):
        extent,step,contiguous=self.run_geometry
        return self.tile if contiguous else max(step,1)

    def emit(self,builder,consume):
        """Counted address loops partition the complete output exactly once."""
        self.__post_init__()
        extent,step,contiguous=self.run_geometry
        pitches=tuple(prod(self.output_shape[d+1:])for d in range(len(self.output_shape)))
        def outer(group):
            first=builder.op('mul',group,extent)
            source=0
            for size,pitch,source_step in zip(self.output_shape,pitches,self.source_steps):
                if source_step:
                    coordinate=builder.op('urem',builder.op('udiv',first,pitch),size)
                    source=builder.op('add',source,builder.op('mul',coordinate,source_step))
            def run(begin,rows,columns,source_stride,output_stride):
                consume(LayoutRun(builder.op('add',source,builder.op('mul',begin,step)),
                    builder.op('add',first,begin),rows,columns,source_stride,output_stride,
                    reads='exact_original_projected_source',writes='complete_distinct_original_result'))
            if contiguous:
                panel=self.tile*self.tile
                full=extent//panel*panel
                if full:
                    builder.loop(0,full,panel,lambda begin:run(begin,self.tile,self.tile,self.tile,self.tile))
                rows=(extent-full)//self.tile
                if rows:
                    run(full,rows,self.tile,self.tile,self.tile)
                tail=(extent-full)%self.tile
                if tail:
                    run(full+rows*self.tile,1,tail,tail,tail)
            else:
                full=extent//self.tile*self.tile
                if full:
                    builder.loop(0,full,self.tile,lambda begin:run(begin,self.tile,1,step,1))
                if full<extent:
                    run(full,extent-full,1,step,1)
        builder.loop(0,prod(self.output_shape)//extent,1,outer)
