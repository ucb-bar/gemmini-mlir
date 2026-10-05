from dataclasses import replace
from mlir_oot.golden_gemm import Shape
from mlir_oot.dense_schedule import select_kernel
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
