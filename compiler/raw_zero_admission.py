"""Typed positive-f32-zero binding over the shared complete zero initializer.

This is initialization with raw zero bits, never arithmetic or f32 rounding.
The selected zero-source instruction and raw32 view require separate admission.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
from merlin.compile.scheduling.source_zero_transfer import SourceZeroTransfer
from merlin.compile.scheduling.raw_storage_view import raw_storage_view
from raw_storage_transfer import raw_storage_admission, storage_receipt


@dataclass(frozen=True)
class Raw32ZeroTransfer(SourceZeroTransfer):
    """Keep source dtype f32; delegate full-domain geometry to original i32 zero."""
    def __post_init__(self):
        if self.dtype!='f32':
            raise ValueError('raw32 zero source requires original f32')
        view=raw_storage_view(self.dtype)
        SourceZeroTransfer(self.shape,view.storage_dtype,self.tile)
        self.movement.__post_init__()


def positive_zero(value):
    """Only the exact f32 encoding0; -0, NaNs, infinities and nonzero refuse."""
    try:
        return struct.pack('<f',value)==b'\0\0\0\0'
    except (OverflowError,TypeError,struct.error):
        return False


def zero_admission(facts):
    raw=raw_storage_admission('f32',facts)
    if raw is None:
        return None
    from encoding import ACC,ACCUMULATE,FULL
    if ACC!=1<<31 or ACCUMULATE!=1<<30 or FULL!=1<<29:
        return None
    return dict(raw,source_dtype='f32',storage_dtype='i32',literal_bits='0x00000000',
        zero_source_address=0,accumulator_destination=ACC,accumulate=False,
        load_shrink=False,full_width_store=True,
        zero_semantics='existing selected NULL-source MVIN zero initializer; native raw32 zero execution qualified separately',
        existing_zero_owner_sha256=sha256(Path(__file__).with_name('dense_pool.py').read_bytes()).hexdigest(),
        shared_zero_planner_sha256=sha256(Path(__file__).with_name('compiler_library').joinpath('merlin/compile/scheduling/source_zero_transfer.py').read_bytes()).hexdigest(),
        adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def zero_storage_receipt():
    return dict(storage_receipt('f32'),literal_bits='0x00000000',
        negative_zero_admitted=False,source_read='none: selected zero source',
        operation='complete initialization, not floating arithmetic')
