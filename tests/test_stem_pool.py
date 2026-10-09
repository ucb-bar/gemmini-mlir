import unittest
from pathlib import Path
from xdsl.dialects.builtin import FloatAttr,Float32Type,IntegerAttr,i64,DenseArrayBase
from xdsl.utils.exceptions import VerifyException
from mlir_oot.frontend.parse import parse_module
from mlir_oot.contraction_patterns import match_integer_gemm
from mlir_oot.captured_requant import inspect_chain
from mlir_oot.stem_binding import match
from mlir_oot.stem_pool_binding import rewrite,validate_layout
from mlir_oot.direct_conv_binding import serialize
from mlir_oot.golden_stem_pool import StemPoolShape,GoldenStemPool,command_counts
from mlir_oot.golden_device_lower import _encoded
from mlir_oot.tables import isa

class TestStemPool(unittest.TestCase):
    def fixture(self):
        m=parse_module(Path(__file__).with_name('fixtures').joinpath('stem_pool.mlir').read_text());op=next(x for x in m.walk() if match_integer_gemm(x));return m,op
    def test_capture_pool_boundary_rewrites_exact_layout(self):
        m,op=self.fixture();chain=inspect_chain(op,allow_maxpool=True);self.assertEqual(chain['pool']['output_shape'],[1,64,56,56]);declaration=rewrite(match(op),chain,StemPoolShape(224,224,64,.0020730062387883663),'proven_stem_pool');m.verify();parse_module(serialize(m,[declaration])).verify()
    def test_source_pad_other_than_negative_infinity_refused(self):
        _,op=self.fixture();chain=inspect_chain(op,allow_maxpool=True);insert=next(x for x in chain['operations'] if x.name=='tensor.insert_slice');constant=insert.operands[1].owner.operands[0].owner;constant.properties['value']=FloatAttr(0.0,Float32Type())
        with self.assertRaisesRegex(ValueError,'negative infinity'):inspect_chain(op,allow_maxpool=True)
    def test_wrong_pool_reducer_operand_refused(self):
        _,op=self.fixture();chain=inspect_chain(op,allow_maxpool=True);pool=chain['operations'][-2];reducer=pool.body.block.first_op;reducer.operands=[pool.body.block.args[1],pool.body.block.args[2]]
        with self.assertRaisesRegex(ValueError,'exact maximum'):inspect_chain(op,allow_maxpool=True)
    def test_band_halos_cover_each_valid_window(self):
        for h,w in [(5,35),(17,35),(224,224),(225,224)]:
            s=StemPoolShape(h,w);s.validate();covered=[]
            for p,count,start,end in s.bands():
                self.assertLessEqual((end-start)*s.ow,1024)
                for py in range(p,p+count):
                    covered.append(py)
                    for y in range(py*2-1,py*2+2):
                        if 0<=y<s.oh:self.assertTrue(start<=y<end)
            self.assertEqual(covered,list(range(s.ph)))
    def test_all_store_pool_fields_survive_lowering(self):
        m=GoldenStemPool(StemPoolShape(17,35,19,.03125)).build();op=next(x for x in m.walk() if x.name=='gemmini.config_st' and x.a('pool_stride',0))
        keys=['stride','acc_act','acc_scale','pool_stride','pool_size','pool_out_dim','porows','pocols','orows','ocols','upad','lpad']
        self.assertEqual(_encoded(op),isa.config_st(**{k:op.a(k) for k in keys}))
        op.attributes['pool_out_dim']=IntegerAttr(256,i64)
        with self.assertRaises(VerifyException):op.verify()
    def test_changed_feature_layout_refused(self):
        _,op=self.fixture();chain=inspect_chain(op,allow_maxpool=True);binding=match(op);validate_layout(binding,chain)
        chain['layouts'][2].properties['permutation']=DenseArrayBase.from_list(i64,[0,2,1,3])
        with self.assertRaisesRegex(ValueError,'layout proof'):validate_layout(binding,chain)
    def test_commands_expose_overlap_recomputation(self):
        c=command_counts(StemPoolShape(224,224));self.assertEqual(c['compute'],49000);self.assertEqual(c['convolution_rows_computed'],125);self.assertEqual(c['mvin_b'],14)
