import pytest
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_flat_conv import padding_segments,virtual_band_groups,spatial_runs,GoldenFlatConv

@pytest.mark.parametrize('stride',[1,2])
@pytest.mark.parametrize('hw',[(1,1),(2,3),(7,7),(14,14),(56,56),(3,19)])
def test_partition_proves_every_source_address_and_zero_lane(hw,stride):
 s=ConvShape(*hw,20,19,stride)
 for height in set((1,min(4,s.oh),s.oh)):
  groups=virtual_band_groups(s,height);visited=[]
  for start,stop,count in groups:
   for y0 in range(start,stop,height):
    visited.extend(range(y0,y0+count))
    for _,_,y,x,rows in spatial_runs(s,count):
     for kh in range(3):
      for kw in range(3):
       sample=padding_segments(s,start+y,x,rows,kh,kw)
       assert sample==padding_segments(s,y0+y,x,rows,kh,kw)
       lanes=[]
       for offset,n,zero in sample:
        for lane in range(offset,offset+n):
         iy=(y0+y)*stride+kh-1;ix=(x+lane)*stride+kw-1
         assert zero != (0<=iy<s.h and 0<=ix<s.w)
         lanes.append(lane)
       assert lanes==list(range(rows))
  assert visited==list(range(s.oh))

def test_virtual_schedule_is_explicit_and_rejects_padded_shape():
 s=ConvShape(3,19,20,19)
 with pytest.raises(ValueError):GoldenFlatConv(s)
 module=GoldenFlatConv(s,wide_a=True,separate_b_bank=True,virtual_padding=True).build()
 module.verify()
 assert 'gemmini.virtual_padding' in module.attributes
 with pytest.raises(ValueError):GoldenFlatConv(ConvShape(3,19,20,19,explicit_halo=True),virtual_padding=True)


def pad_fixture(zero=0,offset=1):
 from mlir_oot.frontend.parse import parse_module
 return parse_module(f'''builtin.module {{
 func.func @pad(%a: tensor<1x20x3x19xi8>) -> tensor<1x20x5x21xi8> {{
 %z = arith.constant {zero} : i8
 %e = tensor.splat %z : tensor<1x20x5x21xi8>
 %p = "tensor.insert_slice"(%a, %e) <{{static_offsets = array<i64: 0,0,{offset},1>, static_sizes = array<i64: 1,20,3,19>, static_strides = array<i64: 1,1,1,1>, operandSegmentSizes = array<i32: 1,1,0,0,0>}}> : (tensor<1x20x3x19xi8>, tensor<1x20x5x21xi8>) -> tensor<1x20x5x21xi8>
 func.return %p : tensor<1x20x5x21xi8>
 }} }}''')


def test_pad_removal_requires_exact_geometry_and_literal_zero():
 from mlir_oot.virtual_padding import strip_zero_pad1
 s=ConvShape(3,19,20,19,explicit_halo=True)
 def output(m):return next(o for o in m.walk() if o.name=='func.return').operands[0]
 m=pad_fixture();m.verify();source,proof=strip_zero_pad1(output(m),s)
 assert tuple(source.type.get_shape())==(1,20,3,19)
 assert proof['kind']=='explicit_static_i8_zero_pad1'
 with pytest.raises(ValueError,match='padding value'):strip_zero_pad1(output(pad_fixture(1)),s)
 with pytest.raises(ValueError,match='geometry'):strip_zero_pad1(output(pad_fixture(offset=0)),s)
 with pytest.raises(ValueError,match='explicit zero'):strip_zero_pad1(source,s)
