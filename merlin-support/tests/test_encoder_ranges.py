"""The command encoder refuses any field it would otherwise mask to a legal value."""

import pytest

from mlir_oot.core_aten.emit import encode
from mlir_oot.core_aten.stream import ACC, SP, Stream


def _commands():
    s = Stream([])
    s.config_ex()
    s.config_ld(0, 32)
    s.config_st(16)
    s.mvin(None, SP(0), 16, 16)
    s.mvout(None, ACC(0), 16, 16)
    s.preload(SP(0), ACC(0), 16, 16, 16, 16)
    s.compute(SP(16), 16, 16)
    return {ins.kind: ins for ins in s.ins}


@pytest.mark.parametrize('kind,field,value', [
    ('mvin', 'cols', 0x80000010), ('mvout', 'rows', 1 << 16), ('mvin', 'local', 1 << 32),
    ('config_ex', 'dataflow', -15), ('config_ex', 'a_transpose', 2), ('config_ex', 'c_stride', 1 << 16),
    ('config_ld', 'load_id', 4), ('config_ld', 'pixel_repeats', 256), ('config_st', 'stride', -1),
    ('config_st', 'acc_act', 4), ('compute', 'accumulate', 2), ('preload', 'c_cols', -1),
])
def test_encoder_refuses_out_of_range_fields(kind, field, value):
    """Masking a field would issue a different, legal command than the stream names."""
    ins = _commands()[kind]
    encode(ins)
    tampered = ins._replace(attrs=tuple((k, value if k == field else v) for k, v in ins.attrs))
    with pytest.raises(ValueError, match='does not fit|names no load state'):
        encode(tampered)


def test_dram_offset_must_fit_the_register():
    ins = _commands()['mvout']
    with pytest.raises(ValueError, match='does not fit'):
        encode(ins._replace(off=-1))
