"""Full accumulator DMA readout does not imply full-width execute operands."""

import unittest

from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.utils.exceptions import VerifyException

from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.ir.gemmini_dialect import ComputeOp, MvoutOp, PreloadOp
from mlir_oot.tables import isa


class TestExecuteAccumulatorRead(unittest.TestCase):
    def kernel(self):
        return GoldenGemm(Shape(16, 16, 16, 'i32', bm=1, bn=1)).build()

    def test_full_accumulator_execute_sources_refused(self):
        for op_type, key in [(ComputeOp, 'a'), (ComputeOp, 'bd'), (PreloadOp, 'bd')]:
            with self.subTest(op=op_type.name, operand=key):
                module = self.kernel()
                op = next(x for x in module.walk() if isinstance(x, op_type))
                op.attributes[key] = IntegerAttr(isa.acc_addr(0, full_row=True), i64)
                with self.assertRaisesRegex(VerifyException, 'unsupported full-width accumulator execute read'):
                    module.verify()

    def test_narrow_accumulator_and_garbage_sources_remain_legal(self):
        for op_type, key in [(ComputeOp, 'a'), (ComputeOp, 'bd'), (PreloadOp, 'bd')]:
            for addr in [0, isa.acc_addr(0), isa.GARBAGE_ADDR]:
                with self.subTest(op=op_type.name, operand=key, address=addr):
                    module = self.kernel()
                    op = next(x for x in module.walk() if isinstance(x, op_type))
                    op.attributes[key] = IntegerAttr(addr, i64)
                    module.verify()

    def test_full_accumulator_dma_readout_remains_legal(self):
        module = self.kernel()
        op = next(x for x in module.walk() if isinstance(x, MvoutOp))
        self.assertEqual(op.a('local'), isa.acc_addr(0, full_row=True))
        module.verify()


if __name__ == '__main__':
    unittest.main()
