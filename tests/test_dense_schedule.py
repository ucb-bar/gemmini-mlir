from dataclasses import replace
from mlir_oot.golden_gemm import Shape
from mlir_oot.dense_schedule import select_kernel,choose_banked_by_command_cost,choose_resident_a_by_command_cost,choose_transfer_by_command_cost
from mlir_oot.golden_tuning import estimate


def measured_shape():
 return Shape(3136,64,64,bm=16,bn=4,output_dtype='i8',scale=.0038317402359098196,relu=True,wide_store=True,reuse_b=True,cache_b=True,wide_a=True,wide_b=True)


def test_default_is_unchanged_and_measured_schedule_preserves_epilogue():
 s=measured_shape();generator,kind=select_kernel(s)
 assert generator.shape is s and kind=='dense_gemm'
 generator,kind=select_kernel(s,banked_prefetch=True);c=generator.shape
 assert (c.m,c.n,c.k,c.scale,c.relu,c.bias)==(s.m,s.n,s.k,s.scale,s.relu,s.bias)
 assert c.bm==1 and c.bn==4 and c.banked_m and c.prefetch_m and c.pipeline_m
 assert c.cache_b and c.wide_a and c.wide_b and c.wide_store and not c.reuse_b
 assert estimate(c)['primitive_command_count']==estimate(s)['primitive_command_count']
 assert kind.endswith('banked_prefetch_single_store')


def test_general_shape_family_and_resource_refusals():
 # No model/region identity or exact scale participates in admission.
 s=Shape(47,48,32,bm=2,bn=3,scale=.03125,relu=False)
 g,_=select_kernel(s,banked_prefetch=True);assert g.shape.banked_m
 for candidate in [replace(s,m=16),replace(s,n=128),replace(s,k=80),replace(s,n=47),replace(s,bias=True),replace(s,output_dtype='i32',scale=1.)]:
  g,k=select_kernel(candidate,banked_prefetch=True)
  assert g.shape is candidate and k=='dense_gemm'


def test_grouped_b_is_independent_explicit_and_reduces_commands():
 s=Shape(49,2048,512,output_dtype='i8',bm=4,bn=16)
 g,k=select_kernel(s,banked_prefetch=True);assert g.shape is s
 g,k=select_kernel(s,grouped_b=True)
 assert g.shape.wide_b and not g.shape.banked_m
 assert estimate(g.shape)['primitive_command_count']<estimate(s)['primitive_command_count']


def test_full_k_banked_explicit_resources_and_epilogue():
 import pytest
 s=Shape(3136,128,256,bm=8,bn=8,output_dtype='i8',scale=.003538144286721945,relu=True,cache_b=True,wide_b=True,reuse_b=True,separate_b_bank=True)
 g,_=select_kernel(s);assert g.shape is s
 g,k=select_kernel(s,full_k_banked=True);c=g.shape
 assert (c.m,c.n,c.k,c.scale,c.relu,c.bias)==(s.m,s.n,s.k,s.scale,s.relu,s.bias)
 assert c.bm==1 and c.bn==8 and c.banked_m and c.prefetch_m and c.wide_a
 assert not c.separate_b_bank and not c.reuse_b
 # Cached B and entire A must fit their reserved physical banks.
 for bad in [replace(s,k=8192),replace(s,n=1024),replace(s,k=255),replace(s,bias=True)]:
  with pytest.raises(ValueError):select_kernel(bad,full_k_banked=True)


def test_command_policy_uses_resources_and_cost_without_shape_lookup():
 for m,n,k in ((33,48,32),(80,256,64),(176,128,256),(95,64,80)):
  s=Shape(m,n,k,bm=2,bn=3,scale=.017,relu=True)
  c,decision=choose_banked_by_command_cost(s)
  assert decision['applied'] and c.banked_m
  assert decision['cost_unit']=='primitive_commands_not_cycles'
  assert decision['candidate']['primitive_command_count']<decision['control']['primitive_command_count']
  assert (c.m,c.n,c.k,c.scale,c.relu,c.output_dtype,c.bias)==(s.m,s.n,s.k,s.scale,s.relu,s.output_dtype,s.bias)
  c.validate()
  g,kind=select_kernel(s,banked_command_policy=True)
  assert g.shape==c and kind=='dense_gemm:full_k_banked_command_cost'


def test_command_policy_records_legal_fallback_for_unsupported_or_over_capacity():
 s=Shape(95,64,64,bm=2,bn=4)
 for bad in (replace(s,m=16),replace(s,k=16),replace(s,k=63),replace(s,n=63),
             replace(s,k=8192),replace(s,n=1024),replace(s,bias=True),
             replace(s,output_dtype='i32')):
  c,decision=choose_banked_by_command_cost(bad)
  assert c is bad and not decision['applied'] and decision['refusal']
  g,kind=select_kernel(bad,banked_command_policy=True)
  assert g.shape is bad and kind=='dense_gemm'


def test_command_policy_retains_equal_cost_schedule_and_compares_composed_flags():
 s=Shape(80,256,64,bm=8,bn=8,wide_store=True,wide_a=True,wide_b=True,separate_b_bank=True)
 g,_=select_kernel(s,banked_command_policy=True)
 c,decision=choose_banked_by_command_cost(g.shape)
 assert c is g.shape and not decision['applied']
 assert decision['refusal']=='primitive command count does not decrease'
 # The preexisting smaller-K policy already chooses the same banked schedule.
 g,kind=select_kernel(measured_shape(),banked_prefetch=True,banked_command_policy=True)
 assert kind=='dense_gemm:banked_prefetch_single_store'
 assert g.shape.banked_m


def test_command_policy_cannot_mix_with_explicit_full_k_selection():
 import pytest
 with pytest.raises(ValueError,match='cannot mix'):
  select_kernel(measured_shape(),full_k_banked=True,banked_command_policy=True)


def test_resident_a_policy_preserves_epilogue_and_generalizes_row_and_channel_tails():
 for m,n,k in ((49,2048,512),(17,73,65),(95,48,80),(196,1024,256)):
  s=Shape(m,n,k,'i8',bm=2,bn=3,scale=.017,relu=True)
  c,d=choose_resident_a_by_command_cost(s)
  assert d['applied'] and c.cache_a and c.reuse_b
  assert c.bm==(m+15)//16 and c.bm*c.bn*16<=1024
  assert (c.m,c.n,c.k,c.scale,c.relu,c.output_dtype,c.bias)==(s.m,s.n,s.k,s.scale,s.relu,s.output_dtype,s.bias)
  assert d['candidate']['primitive_command_count']<d['control']['primitive_command_count']
  c.validate()
  g,kind=select_kernel(s,resident_a_command_policy=True)
  assert g.shape==c and kind.endswith('resident_a_command_cost')


def test_resident_a_policy_full_resource_refusal_and_prefetch_fallback():
 for s in (Shape(8,256,64,'i32',bm=1,bn=4),
           Shape(1040,256,64,'i32',bm=4,bn=4),
           Shape(49,2048,4112,'i32',bm=4,bn=16)):
  c,d=choose_resident_a_by_command_cost(s)
  assert c is s and not d['applied'] and d['refusal']
 s=Shape(49,2048,2064,'i32',bm=4,bn=16)
 c,d=choose_resident_a_by_command_cost(s)
 assert d['applied'] and c.cache_a and not c.prefetch_b
 assert 'lower two banks' in d['prefetch_b_refusal']
 c.validate()


def test_transfer_policy_compares_families_without_source_selection():
 for s,family in ((Shape(176,256,64,'i8',bm=8,bn=8,wide_a=True,wide_b=True,wide_store=True),'banked'),
                  (Shape(49,2048,512,'i8',bm=4,bn=16,wide_b=True,wide_store=True,reuse_b=True),'resident_a')):
  c,d=choose_transfer_by_command_cost(s)
  assert d['applied'] and d['selected_family']==family
  assert estimate(c)['primitive_command_count']<estimate(s)['primitive_command_count']
  g,kind=select_kernel(s,transfer_command_policy=True)
  assert g.shape==c and kind.endswith('transfer_command_cost_'+family)
  assert not choose_transfer_by_command_cost(c)[1]['applied']


def test_multiple_command_cost_policies_are_refused():
 import pytest
 with pytest.raises(ValueError,match='one explicit'):
  select_kernel(measured_shape(),resident_a_command_policy=True,banked_command_policy=True)
 with pytest.raises(ValueError,match='cannot mix'):
  select_kernel(measured_shape(),transfer_command_policy=True,full_k_banked=True)
