import unittest
from pathlib import Path
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_requant import match


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




    def test_alternate_rounding_refused(self):
        text=Path(__file__).with_name('fixtures').joinpath('fused_requant.mlir').read_text()
        with self.assertRaisesRegex(ValueError,'nearest-even'):
            match(self.fixture(text.replace('math.roundeven','math.round')))

    def test_runtime_bias_refused(self):
        op=self.fixture();runtime_bias=op.parent.insert_arg(op.operands[1].type,len(op.parent.args))
        op.operands=[op.operands[0],runtime_bias,op.operands[2]]
        with self.assertRaisesRegex(ValueError,'immutable dense'):
            match(op)
