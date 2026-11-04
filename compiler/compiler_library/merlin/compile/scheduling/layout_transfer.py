"""Exact static tensor permutations as target-neutral strided transfer runs.

No payload arithmetic, target encoding, alias grant or timing is inferred.
The execution owner binds each run to raw same-type transfers and completion.
"""
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class LayoutRun:
    source_element: object
    output_element: object
    rows: int
    columns: int
    source_stride: int
    output_stride: int
    reads: str = 'complete_original_input'
    writes: str = 'disjoint_original_output'
    completion: str = 'synchronous_load_then_store_then_reuse'


@dataclass(frozen=True)
class LayoutTransfer:
    source_shape: tuple[int, ...]
    permutation: tuple[int, ...]
    dtype: str
    tile: int

    def __post_init__(self):
        if (type(self.source_shape) is not tuple or not self.source_shape
            or any(type(x) is not int or not 0 < x < 1 << 31 for x in self.source_shape)
            or type(self.permutation) is not tuple
            or any(type(x) is not int for x in self.permutation)
            or sorted(self.permutation) != list(range(len(self.source_shape)))
            or self.dtype not in ('i8','i32','f32') or type(self.tile) is not int
            or not 0 < self.tile <= 16 or prod(self.source_shape)*self.itemsize >= 1 << 31):
            raise ValueError('exact bounded static same-type permutation required')

    @property
    def itemsize(self):
        return 1 if self.dtype == 'i8' else 4

    @property
    def output_shape(self):
        return tuple(self.source_shape[d] for d in self.permutation)

    @property
    def contiguous_suffix(self):
        # Collapse only adjacent source axes in original row-major order.
        expected = len(self.source_shape)-1
        count = 1
        for axis in reversed(self.permutation):
            if axis != expected:
                break
            count *= self.source_shape[axis]
            expected -= 1
        return count

    def emit(self, builder, consume):
        """Counted integer address loops; every callback is a complete raw run."""
        self.__post_init__()
        shape = self.output_shape
        contiguous = self.contiguous_suffix
        extent = contiguous if contiguous > 1 else shape[-1]
        contiguous_mode = contiguous > 1 or self.permutation[-1] == len(shape)-1
        # The remaining coordinates are decoded from the complete logical output.
        source_pitches = tuple(prod(self.source_shape[i+1:]) for i in range(len(shape)))
        output_pitches = tuple(prod(shape[i+1:]) for i in range(len(shape)))
        def outer(group):
            base = builder.op('mul', group, extent)
            source = 0
            for axis, size in enumerate(shape):
                coordinate = builder.op('urem', builder.op('udiv', base, output_pitches[axis]), size)
                source = builder.op('add', source, builder.op('mul', coordinate, source_pitches[self.permutation[axis]]))
            step = 1 if contiguous_mode else source_pitches[self.permutation[-1]]
            def run(first, count):
                src = builder.op('add', source, builder.op('mul', first, step))
                dst = builder.op('add', base, first)
                consume(LayoutRun(src, dst, 1 if contiguous_mode else count,
                    count if contiguous_mode else 1,
                    count if contiguous_mode else step,
                    count if contiguous_mode else 1))
            full = extent // self.tile * self.tile
            if full:
                builder.loop(0, full, self.tile, lambda first: run(first, self.tile))
            if full < extent:
                run(full, extent-full)
        builder.loop(0, prod(shape)//extent, 1, outer)
