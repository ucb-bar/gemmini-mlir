import unittest
from pathlib import Path
import numpy as np
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_requant import match,prove_scale,synthesize_bias,quantized


class TestGoldenRequant(unittest.TestCase):
    def fixture(self,text=None):
        text=text or Path(__file__).with_name('fixtures').joinpath('fused_requant.mlir').read_text()
        module=parse_module(text)
        return next(op for op in module.walk() if op.name=='linalg.generic')

    def test_source_bias_gets_proved_integer_table(self):
        fused=match(self.fixture())
        self.assertEqual(fused.integer_bias,[j%5-2 for j in range(19)])
        self.assertTrue(fused.shape.bias)
        self.assertEqual(fused.shape.scale,.03125)

    def test_transition_proof_matches_exhaustive_domain(self):
        for scales in [[.5,.0625],[.7,.7],[.123456,.234567]]:
            proof=prove_scale(scales,-10000,10000)
            self.assertTrue(all(quantized(a,scales)==quantized(a,[proof['scale']]) for a in range(-10000,10001)))

    def test_float_reassociation_refused_with_concrete_witness(self):
        with self.assertRaisesRegex(ValueError,'-12650'):
            prove_scale([.1,.1],-100000,100000)

    def test_nonintegral_bias_cannot_be_silently_rounded(self):
        proof=synthesize_bias([.5,.0625],[0,1,.01],1,-10000,10000)
        self.assertEqual(proof['integer_bias'],[0,32,None])
        for j,bias in enumerate([0,1]):
            for acc in range(-10000,10001):
                source=int(np.clip(np.rint(np.float32(np.float32(acc)*np.float32(.03125))+np.float32(bias)),-128,127))
                self.assertEqual(source,quantized(acc+proof['integer_bias'][j],[proof['scale']]))

    def test_alternate_rounding_refused(self):
        text=Path(__file__).with_name('fixtures').joinpath('fused_requant.mlir').read_text()
        with self.assertRaisesRegex(ValueError,'nearest-even'):
            match(self.fixture(text.replace('math.roundeven','math.round')))

    def test_runtime_bias_refused(self):
        op=self.fixture();runtime_bias=op.parent.insert_arg(op.operands[1].type,len(op.parent.args))
        op.operands=[op.operands[0],runtime_bias,op.operands[2]]
        with self.assertRaisesRegex(ValueError,'immutable dense'):
            match(op)
