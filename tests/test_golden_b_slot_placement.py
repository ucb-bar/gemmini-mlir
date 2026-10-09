"""Explicit B slots prove complete live extents before next-K prefetch."""

from dataclasses import replace

import pytest

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.tables import isa, rtl_facts as F
from mlir_oot.b_slot_placement import select_remaining_b_slots


def shape():
    return Shape(196,512,1024,'i8',bm=13,bn=4,cache_a=True,
                 reuse_b=True,wide_b=True,wide_store=True,
                 scale=.0018682164372876287,relu=True,prefetch_b=True)


def test_two_slots_after_complete_a_preserve_dynamic_bounds_and_source_fields():
    s=shape()
    a_end=s.bm*(s.k//F.DIM)*F.DIM
    assert a_end==13312
    with pytest.raises(ValueError,match='lower two banks'):
        s.validate()
    placement=(a_end,a_end+s.bn*F.DIM)
    module=GoldenGemm(s,prefetch_b_rows=placement).build()
    assert module.attributes['gemmini.prefetch_b_rows'].data=='13312,13376'
    loads=[op for op in module.walk() if op.name=='gemmini.mvin' and op.a('load_id')==1]
    assert {op.a('local') for op in loads}==set(placement)
    assert all(op.a('local')>=a_end and op.a('local')+s.bn*F.DIM<=F.SPAD_ROWS for op in loads)
    preloads=[op for op in module.walk() if op.name=='gemmini.preload' and op.a('bd')!=isa.GARBAGE_ADDR]
    assert {op.a('bd') for op in preloads}=={base+d*F.DIM for base in placement for d in range(s.bn)}
    computes=[op for op in module.walk() if op.name=='gemmini.compute']
    assert all(op.a('a_reserved_rows')==a_end and op.a('a_max')<a_end for op in computes)
    assert any(op.a('a_rows')==4 for op in computes)
    module.verify();lower(module)
    assert not any(op.name.startswith('gemmini.') for op in module.walk())
    assert s==shape()


@pytest.mark.parametrize('placement,error',[
    ((8192,13376),'overlaps reserved A'),
    ((13312,13344),'slots overlap'),
    ((16320,16384),'exceeds the scratchpad'),
    ((16352,13312),'exceeds the scratchpad'),
    ((13313,13376),'DIM-aligned'),
    ((-16,13376),'DIM-aligned'),
    ((True,13376),'DIM-aligned'),
    ((13312.0,13376),'DIM-aligned'),
    ([13312,13376],'DIM-aligned'),
    ((13312,),'DIM-aligned'),
])
def test_refuses_incomplete_or_overlapping_slot_proofs(placement,error):
    with pytest.raises(ValueError,match=error):
        GoldenGemm(shape(),prefetch_b_rows=placement)


def test_explicit_slot_proof_requires_prefetch_and_retains_default_placement():
    with pytest.raises(ValueError,match='require B prefetch'):
        GoldenGemm(replace(shape(),prefetch_b=False),prefetch_b_rows=(13312,13376))
    small=Shape(17,73,65,'i32',bm=2,bn=4,cache_a=True,wide_b=True,reuse_b=True,prefetch_b=True)
    with pytest.raises(ValueError,match='within one scratchpad bank'):
        GoldenGemm(small,prefetch_b_rows=(12256,13376))
    implicit=GoldenGemm(small).build()
    explicit=GoldenGemm(small,prefetch_b_rows=(2*F.SPAD_BANK_ROWS,3*F.SPAD_BANK_ROWS)).build()
    del explicit.attributes['gemmini.prefetch_b_rows']
    assert str(implicit)==str(explicit)


def test_independent_m_n_k_tails_and_multiple_channel_blocks_are_admitted():
    s=Shape(123,73,1041,'i32',bm=8,bn=4,cache_a=True,wide_b=True,reuse_b=True,prefetch_b=True)
    a_end=s.bm*((s.k+F.DIM-1)//F.DIM)*F.DIM
    module=GoldenGemm(s,prefetch_b_rows=(a_end,a_end+64)).build()
    bloads=[op for op in module.walk() if op.name=='gemmini.mvin' and op.a('load_id')==1]
    assert any(op.a('rows')==1 for op in bloads)
    assert any(op.a('cols')==9 for op in bloads)
    computes=[op for op in module.walk() if op.name=='gemmini.compute']
    assert any(op.a('a_rows')==11 for op in computes)
    assert all(op.a('a_reserved_rows')==a_end for op in computes)
    module.verify();lower(module)
    assert not any(op.name.startswith('gemmini.') for op in module.walk())


@pytest.mark.parametrize('m,n,k,bm,bn',[(196,512,1024,13,4),(123,73,1041,8,4),(784,512,256,49,1)])
def test_general_selector_preserves_source_semantics_and_records_lifetimes(m,n,k,bm,bn):
    s=Shape(m,n,k,'i8',bm=bm,bn=bn,cache_a=True,reuse_b=True,wide_b=True,
            wide_store=True,scale=.017,relu=True)
    original=GoldenGemm(s)
    generator,decision=select_remaining_b_slots(original)
    assert decision['applied'] and not decision['timing_claim']
    assert generator.shape==replace(s,prefetch_b=True)
    assert generator.prefetch_b_rows==tuple(decision['prefetch_b_rows'])
    assert decision['reserved_a_rows']==bm*((k+15)//16)*16
    kept,again=select_remaining_b_slots(generator)
    assert kept is generator and not again['applied']
    assert 'retained' in again['refusal']


def test_selector_preserves_resource_and_competing_schedule_fallbacks():
    for s in [Shape(33,64,64,'i32',bm=2,bn=4),
              Shape(8,64,16,'i32',bm=1,bn=4,cache_a=True),
              Shape(32,64,64,'i32',bm=2,bn=4,cache_a=True,separate_b_bank=True),
              Shape(8,64,16320,'i32',bm=1,bn=4,cache_a=True)]:
        control=GoldenGemm(s)
        result,decision=select_remaining_b_slots(control)
        assert result is control and not decision['applied'] and decision['refusal']


def test_unknown_capture_policy_refuses_before_reading_source_or_creating_output(tmp_path):
    from mlir_oot.captured_requant_bundle import build
    output=tmp_path/'result'
    with pytest.raises(ValueError,match='unknown B slot compiler policy'):
        build(tmp_path/'absent',tmp_path/'llvm',output,dense_b_slot_policy='unknown')
    assert not output.exists()
