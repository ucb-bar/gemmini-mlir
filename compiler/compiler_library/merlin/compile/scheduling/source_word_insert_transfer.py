"""Native i64 insertion by pairing the proven complete logical insertion runs."""
from dataclasses import dataclass,replace
from math import prod
from .source_insert_transfer import SourceInsertTransfer


@dataclass(frozen=True)
class SourceWordInsertTransfer:
    source_shape: tuple[int,...]
    destination_shape: tuple[int,...]
    offsets: tuple[int,...]
    sizes: tuple[int,...]
    strides: tuple[int,...]
    dtype: str = 'i64'
    tile: int = 8

    def __post_init__(self):
        if self.dtype!='i64' or type(self.tile)is not int or self.tile!=8:
            raise ValueError('native i64 insertion requires adjacent pairs and eight-element logical tiles')
        self.movement.__post_init__()
        if max(prod(self.source_shape),prod(self.destination_shape))*8>=1<<31:
            raise ValueError('native i64 insertion exceeds bounded raw byte addressing')

    @property
    def movement(self):
        # This checks original source, destination, bounds and unit-axis rank
        # expansion before any raw-word expansion is allowed.
        return SourceInsertTransfer(self.source_shape,self.destination_shape,
            self.offsets,self.sizes,self.strides,'i32',self.tile)

    def emit(self,builder,consume_seed,consume_patch):
        self.__post_init__()
        def pair(consume,run):
            consume(replace(run,source_element=builder.op('mul',run.source_element,2),
                output_element=builder.op('mul',run.output_element,2),
                columns=run.columns*2,source_stride=run.source_stride*2,
                output_stride=run.output_stride*2,
                reads='both adjacent raw32 words of unchanged original logical input',
                writes='both adjacent raw32 words of fresh native i64 result'))
        self.movement.emit(builder,lambda run:pair(consume_seed,run),
            lambda run:pair(consume_patch,run))
