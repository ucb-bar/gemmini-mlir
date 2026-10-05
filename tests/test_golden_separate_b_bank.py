from dataclasses import replace
import pytest
from mlir_oot.golden_gemm import Shape, GoldenGemm
from mlir_oot.golden_tuning import estimate
from mlir_oot.tables import rtl_facts as F


def test_placement_does_not_change_commands():
    base=Shape(49,2048,512,bm=4,bn=16,wide_b=True,reuse_b=True)
    candidate=replace(base,separate_b_bank=True)
    assert estimate(base)==estimate(candidate)
    module=GoldenGemm(candidate).build()
    loads=[op for op in module.walk() if op.name=='gemmini.mvin']
    b=[op for op in loads if op.attributes['load_id'].value.data==1]
    assert b
    assert all(op.attributes['local'].value.data>=2*F.SPAD_BANK_ROWS for op in b)


def test_capacity_and_competing_placement_refuse():
    with pytest.raises(ValueError,match='lower two'):
        Shape(8,32,8208,bm=1,bn=2,cache_a=True,separate_b_bank=True).validate()
    with pytest.raises(ValueError,match='accumulator'):
        Shape(16,8208,16,bm=1,bn=513,separate_b_bank=True).validate()
    with pytest.raises(ValueError,match='redundant'):
        Shape(32,64,64,bm=1,bn=4,cache_b=True,wide_a=True,pipeline_m=True,
              banked_m=True,separate_b_bank=True).validate()


def test_cached_b_placement():
    module=GoldenGemm(Shape(33,48,32,bm=2,bn=3,cache_b=True,
                           wide_b=True,separate_b_bank=True)).build()
    b=[op for op in module.walk() if op.name=='gemmini.mvin'
       and op.attributes['load_id'].value.data==1]
    assert len(b)==2
    assert [op.attributes['local'].value.data for op in b]==[
        2*F.SPAD_BANK_ROWS,2*F.SPAD_BANK_ROWS+48]


def test_selection_is_explicit_and_preserves_semantics():
    from mlir_oot.dense_schedule import select_kernel
    base=Shape(49,2048,512,bm=4,bn=16,scale=.03125,relu=True)
    assert select_kernel(base)[0].shape is base
    chosen,policy=select_kernel(base,separate_b_bank=True)
    assert chosen.shape==replace(base,separate_b_bank=True)
    assert policy=='dense_gemm:separate_b_bank'
    too_large=Shape(8,32,8208,bm=1,bn=2,cache_a=True)
    assert select_kernel(too_large,separate_b_bank=True)[0].shape is too_large
