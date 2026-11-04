"""Complete i64 scalar broadcast as two unchanged adjacent raw32 words.

Reuse the existing exact scalar-broadcast/layout geometry. The target binds one
original native-width scalar store, two raw32 lane copies, disjoint output and
completion. No integer arithmetic, byte order change or floating conversion is
licensed by this planner.
"""
from dataclasses import dataclass, replace
from math import prod
from .source_broadcast_transfer import SourceBroadcastTransfer


@dataclass(frozen=True)
class SourceWordBroadcastTransfer:
    shape: tuple[int, ...]
    dtype: str
    tile: int = 16

    def __post_init__(self):
        if (self.dtype != 'i64' or type(self.shape) is not tuple
            or any(type(x) is not int or not 0 < x < 1 << 31 for x in self.shape)
            or prod(self.shape)*8 >= 1 << 31 or self.tile != 16):
            raise ValueError('complete bounded static native i64 scalar storage required')
        self.scalar_geometry.__post_init__()

    @property
    def itemsize(self):
        return 8

    @property
    def scalar_geometry(self):
        # i32 describes transfer geometry only. Original elements and scalar
        # retain their source i64 type; no source tensor is recast to i32.
        return SourceBroadcastTransfer(self.shape, 'i32', self.tile)

    @property
    def movement(self):
        return self.scalar_geometry.movement

    def emit(self, builder, consume):
        self.__post_init__()
        def words(run):
            if (run.columns != 1 or run.source_element != 0
                or run.source_stride != 0 or run.output_stride != 1
                or not 0 < run.rows <= 16):
                raise ValueError('existing scalar geometry lacks exact word-pair mapping')
            consume(replace(run, source_element=0,
                output_element=builder.op('mul', run.output_element, 2),
                columns=2, output_stride=2,
                reads='original_scalar_native_storage_words_in_address_order',
                writes='complete_distinct_original_i64_result'))
        self.scalar_geometry.emit(builder, words)
