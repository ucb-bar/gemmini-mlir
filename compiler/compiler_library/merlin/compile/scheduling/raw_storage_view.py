"""Typed payload storage for pure movement; no numeric conversion authority.

Source dtype is preserved by all source geometry planners. A target adapter may
move f32 using a raw32 storage view only if load/store preserve every bit and
perform no payload arithmetic. This law grants neither aliasing nor completion.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RawStorageView:
    source_dtype: str
    storage_dtype: str
    itemsize: int
    interpretation: str

    def __post_init__(self):
        expected = {'i8': ('i8',1,'identity'), 'i32': ('i32',4,'identity'),
                    'f32': ('i32',4,'bitcast_raw32')}
        if expected.get(self.source_dtype) != (self.storage_dtype,self.itemsize,self.interpretation):
            raise ValueError('unsupported exact raw storage view')

    def record(self):
        return dict(asdict(self),law='same-address same-width payload bits; no numeric cast',
            all_payload_bit_patterns=True, arithmetic_allowed=False,
            signed_zero='preserved',nan_payloads='preserved')


def raw_storage_view(dtype):
    if dtype == 'f32':
        return RawStorageView(dtype,'i32',4,'bitcast_raw32')
    if dtype in ('i8','i32'):
        return RawStorageView(dtype,dtype,1 if dtype=='i8' else 4,'identity')
    raise ValueError('pure movement dtype has no exact raw storage view')
