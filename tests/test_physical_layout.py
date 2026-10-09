"""Regression at the NHWC -> scalar epilogue -> NHWC seam."""
import unittest
from pathlib import Path
import tempfile
from mlir_oot.frontend.parse import parse_module
from mlir_oot.physical_layout import rewrite, census

SOURCE='''builtin.module {
 func.func @forward(%a: tensor<1x2x3x4xi8>) -> tensor<1x2x3x4xi8> {
 %e = tensor.empty() : tensor<1x4x2x3xi8>
 %t = linalg.transpose ins(%a : tensor<1x2x3x4xi8>) outs(%e : tensor<1x4x2x3xi8>) permutation = [0, 3, 1, 2]
 %r = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%t : tensor<1x4x2x3xi8>) outs(%e : tensor<1x4x2x3xi8>) {
 ^bb0(%x: i8, %y: i8):
 %v = arith.addi %x, %x : i8
 linalg.yield %v : i8
 } -> tensor<1x4x2x3xi8>
 %o = tensor.empty() : tensor<1x2x3x4xi8>
 %z = linalg.transpose ins(%r : tensor<1x4x2x3xi8>) outs(%o : tensor<1x2x3x4xi8>) permutation = [0, 2, 3, 1]
 func.return %z : tensor<1x2x3x4xi8>
 }
}'''

class TestPhysicalLayout(unittest.TestCase):
 def test_scalar_order_preserved_and_layout_roundtrip_removed(self):
  m=parse_module(SOURCE);self.assertEqual(census(m)['copies'],2)
  report=rewrite(m);self.assertEqual(report['after']['copies'],0)
  result=next(o for o in m.walk() if o.name=='func.return').operands[0]
  self.assertEqual(result.owner.name,'linalg.generic')
  body=list(result.owner.regions[0].block.ops)
  self.assertEqual([o.name for o in body],['arith.addi','linalg.yield'])
  self.assertIs(result.owner.operands[0],next(o for o in m.walk() if o.name=='func.func').body.block.args[0])
 def test_unknown_call_retains_boundary_and_signature(self):
  m=parse_module('''builtin.module {
  func.func private @opaque(tensor<2x3xi8>) -> tensor<2x3xi8>
  func.func @forward(%a: tensor<2x3xi8>) -> tensor<3x2xi8> {
   %v = func.call @opaque(%a) : (tensor<2x3xi8>) -> tensor<2x3xi8>
   %e = tensor.empty() : tensor<3x2xi8>
   %t = linalg.transpose ins(%v : tensor<2x3xi8>) outs(%e : tensor<3x2xi8>) permutation = [1, 0]
   func.return %t : tensor<3x2xi8>
  }}''')
  before=str(next(o for o in m.walk() if o.name=='func.call'))
  report=rewrite(m);self.assertEqual(report['after']['copies'],1)
  self.assertEqual(str(next(o for o in m.walk() if o.name=='func.call')),before)

 def test_generic_external_argument_contract_roundtrips(self):
  import tempfile
  from pathlib import Path
  from xdsl.dialects.builtin import ArrayAttr,DictionaryAttr,StringAttr
  from mlir_oot.direct_conv_binding import serialize
  from mlir_oot.physical_layout import rewrite_file
  m=parse_module(SOURCE)
  block=parse_module('builtin.module { func.func private @device(tensor<2x3xi8>) -> tensor<2x3xi8> }').body.block
  declaration=block.detach_op(block.first_op)
  declaration.properties['arg_attrs']=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr('read')})])
  m.body.block.add_op(declaration)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'input.mlir';p.write_text(serialize(m,[]))
   result,_=rewrite_file(p,Path(d)/'out')
   parsed=parse_module(result.read_text())
   decl=next(o for o in parsed.walk() if o.name=='func.func' and o.sym_name.data=='device')
   self.assertEqual(decl.arg_attrs.data[0].data['bufferization.access'].data,'read')

 def test_explicit_reduction_block_retains_order_and_reports_selection(self):
  source='''builtin.module {
   func.func @forward(%x: tensor<1x2x3x4xf32>) -> tensor<1x4xf32> {
    %e = tensor.empty() : tensor<1x4x2x3xf32>
    %t = linalg.transpose ins(%x: tensor<1x2x3x4xf32>) outs(%e: tensor<1x4x2x3xf32>) permutation = [0, 3, 1, 2]
    %zero = arith.constant 0.0 : f32
    %init = tensor.splat %zero : tensor<1x4xf32>
    %r = linalg.reduce ins(%t: tensor<1x4x2x3xf32>) outs(%init: tensor<1x4xf32>) dimensions = [2, 3]
    (%v: f32, %acc: f32) {
     %sum = arith.addf %v, %acc : f32
     linalg.yield %sum : f32
    }
    func.return %r : tensor<1x4xf32>
   }
  }'''
  from mlir_oot.physical_layout import rewrite_file
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);original=root/'source.mlir';original.write_text(source)
   selected,report=rewrite_file(original,root/'blocked',reduction_channel_block=2)
   self.assertEqual(original.read_text(),source)
   self.assertEqual(report['reduction_channel_block'],2)
   self.assertEqual(report['rewrite']['blocked_reductions'],1)
   module=parse_module(selected.read_text());module.verify()
   result=next(o for o in module.walk() if o.name=='func.return').operands[0]
   self.assertEqual(result.type.get_shape(),(1,4))
   reduction=result.owner.operands[0].owner
   self.assertEqual([o.name for o in reduction.regions[0].block.ops],['arith.addf','linalg.yield'])
   self.assertEqual([i.data.value for i in reduction.iterator_types],['parallel','parallel','reduction','reduction','parallel'])
   unblocked,default=rewrite_file(original,root/'default')
   explicit,zero=rewrite_file(original,root/'zero',reduction_channel_block=0)
   self.assertEqual(unblocked.read_bytes(),explicit.read_bytes())
   self.assertNotIn('reduction_channel_block',default)
   self.assertEqual(default['rewrite'],zero['rewrite'])
   _,tail=rewrite_file(original,root/'tail',reduction_channel_block=3)
   self.assertEqual(tail['rewrite']['blocked_reductions'],0)

 def test_catalog_refuses_invalid_or_disabled_layout_selection_before_io(self):
  from mlir_oot.fused_mixed_catalog import merlin_callbacks as fused
  from mlir_oot.stem_pool_mixed_catalog import merlin_callbacks as pooled
  for callback,arguments in [(fused,('absent-llvm','absent-requant')),(pooled,('absent-llvm','absent-requant','absent-pool'))]:
   for options in ({'reduction_channel_block':2}, {'propagate_layout':True,'reduction_channel_block':-1}, {'propagate_layout':True,'reduction_channel_block':True}, {'propagate_layout':True,'reduction_channel_block':2.5}):
    with self.subTest(callback=callback.__module__,options=options),self.assertRaises(ValueError):
     callback(*arguments,**options)
