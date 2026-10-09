import unittest
from pathlib import Path
from xdsl.dialects.builtin import IntegerAttr,i64
from mlir_oot.frontend.parse import parse_module
from mlir_oot.contraction_patterns import match_integer_gemm
from mlir_oot.captured_requant import inspect_chain
from mlir_oot.captured_requant_bundle import rewrite_path
from mlir_oot.direct_conv_binding import serialize

class TestCapturedRequant(unittest.TestCase):
    def fixture(self):
        module=parse_module(Path(__file__).with_name('fixtures').joinpath('captured_requant.mlir').read_text())
        op=next(x for x in module.walk() if match_integer_gemm(x))
        return module,op

    def test_real_capture_chain_rewrites_and_roundtrips(self):
        module,op=self.fixture();chain=inspect_chain(op)
        self.assertEqual(len(chain['scales']),2)
        self.assertEqual(chain['bias'].index,2)
        declaration=rewrite_path(op,chain,'proven_requant',None)
        module.verify();parsed=parse_module(serialize(module,[]));parsed.verify()
        self.assertFalse(any(match_integer_gemm(x) for x in parsed.walk()))
        decl=next(x for x in parsed.walk() if x.name=='func.func' and x.sym_name.data=='proven_requant')
        self.assertEqual(len(decl.arg_attrs),3)

    def test_wrong_scalar_operand_refused(self):
        _,op=self.fixture();chain=inspect_chain(op)
        mul=next(x for u in chain['operations'] for r in u.regions for x in r.block.ops if x.name=='arith.mulf')
        mul.operands=[mul.operands[0],mul.parent.args[-1]]
        with self.assertRaisesRegex(ValueError,'operand mismatch'):inspect_chain(op)

    def test_nonzero_zero_point_refused(self):
        _,op=self.fixture();chain=inspect_chain(op)
        constant=chain['quantize'].operands[2].owner.operands[0].owner
        constant.properties['value']=IntegerAttr(1,i64)
        with self.assertRaisesRegex(ValueError,'zero point'):inspect_chain(op)

    def test_fanout_refused(self):
        _,op=self.fixture();chain=inspect_chain(op)
        consumer=chain['operations'][0];op.parent.insert_op_after(consumer.clone(),consumer)
        with self.assertRaisesRegex(ValueError,'fanout'):inspect_chain(op)

    def test_explicit_numeric_contract_survives_structural_serialization(self):
        module,op=self.fixture();chain=inspect_chain(op)
        contract=dict(max_output_lsb_error=1,selected_policy_limit=1,source_region='fixture')
        rewrite_path(op,chain,'bounded_requant',None,numeric_contract=contract)
        parsed=parse_module(serialize(module,[]));parsed.verify()
        declaration=next(x for x in parsed.walk() if x.name=='func.func' and x.sym_name.data=='bounded_requant')
        actual=declaration.attributes['merlin.numeric_contract'].data
        self.assertEqual(actual['max_abs_error'].value.data,1)
        self.assertEqual(actual['selected_policy_limit'].value.data,1)
        self.assertEqual(actual['source_region'].data,'fixture')

    def test_invalid_numeric_policy_refused_before_reading_capture(self):
        from mlir_oot.captured_requant_bundle import build
        for policy in (-1,2,True,0.5):
            with self.assertRaisesRegex(ValueError,'local output error policy'):
                build(Path('absent'),Path('absent'),Path('absent'),max_output_lsb=policy)

    def test_invalid_dense_policy_refused_before_io(self):
        from mlir_oot.captured_requant_bundle import build
        with self.assertRaisesRegex(ValueError,'unknown dense compiler policy'):
            build(Path('absent'),Path('absent'),Path('absent'),dense_input_policy='model_name')
        with self.assertRaisesRegex(ValueError,'cannot mix with source selections'):
            build(Path('absent'),Path('absent'),Path('absent'),dense_input_policy='banked_command_cost',full_k_banked_regions=('capture_id',))
