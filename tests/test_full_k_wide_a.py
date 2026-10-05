from dataclasses import replace
import pytest
from mlir_oot.golden_gemm import Shape,GoldenGemm
from mlir_oot.golden_tuning import estimate


def control():
    return Shape(3136,128,256,output_dtype='i8',bm=8,bn=8,scale=.003538144286721945,relu=True,wide_store=True,reuse_b=True,cache_b=True,wide_b=True,separate_b_bank=True)


def test_full_panel_loads_cover_each_tile_once_and_remain_in_bank():
    s=replace(control(),wide_a=True);module=GoldenGemm(s).build();module.verify()
    loads=[op for op in module.walk() if op.name=='gemmini.mvin' and op.attributes['load_id'].value.data==0]
    assert loads
    for op in loads:
        assert op.attributes['cols'].value.data<=64
        assert op.attributes['local'].value.data<4096
    a,b=estimate(control()),estimate(s)
    assert a['a_mvin_commands']==3136 and b['a_mvin_commands']==784
    assert a['mesh_compute_commands']==b['mesh_compute_commands']==25088


def test_partial_final_wide_group_and_resource_overflow():
    s=replace(control(),m=17,k=80,wide_a=True,cache_b=False)
    module=GoldenGemm(s).build();module.verify()
    cols={op.attributes['cols'].value.data for op in module.walk() if op.name=='gemmini.mvin' and op.attributes['load_id'].value.data==0}
    assert cols=={16,64}
    with pytest.raises(ValueError):replace(control(),k=8192,wide_a=True).validate()
    with pytest.raises(ValueError):replace(control(),k=257,wide_a=True).validate()
