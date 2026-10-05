"""A cached A panel uses a bounded dynamic local address and an ordinary CPU loop."""

import unittest

from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin, IntegerAttr, i64
from xdsl.parser import Parser
from xdsl.utils.exceptions import VerifyException

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.ir.gemmini_dialect import ComputeOp, GEMMINI
from mlir_oot.tables import rtl_facts as F


class TestDynamicCompute(unittest.TestCase):
    def test_cached_a_compiles_to_runtime_rs1_packing(self):
        module = GoldenGemm(Shape(8, 512, 256, "i32", bm=1, bn=4,
                                  cache_a=True)).build()
        dynamic = [op for op in module.walk()
                   if isinstance(op, ComputeOp) and len(op.operands_) == 1]
        self.assertTrue(dynamic)
        self.assertEqual(dynamic[0].a("a_max"), 240)
        lowered = lower(module)
        self.assertFalse(any(op.name.startswith("gemmini.") for op in lowered.walk()))
        self.assertTrue(any(op.name == "llvm.or" for op in lowered.walk()))

    def test_dynamic_address_survives_target_ir_roundtrip(self):
        module = GoldenGemm(Shape(8, 512, 256, "i32", bm=1, bn=4,
                                  cache_a=True)).build()
        context = Context()
        for dialect in (Builtin, llvm.LLVM, GEMMINI):
            context.load_dialect(dialect)
        parsed = Parser(context, str(module)).parse_module()
        parsed.verify()
        self.assertTrue(any(isinstance(op, ComputeOp) and len(op.operands_) == 1
                            for op in parsed.walk()))
        lower(parsed)

    def test_dynamic_address_must_fit_reserved_scratchpad(self):
        module = GoldenGemm(Shape(8, 128, 64, "i32", bm=1, bn=1,
                                  cache_a=True)).build()
        op = next(op for op in module.walk() if isinstance(op, ComputeOp))
        op.attributes["a_max"] = IntegerAttr(F.SPAD_ROWS, i64)
        with self.assertRaisesRegex(VerifyException, "exceeds reserved scratchpad"):
            module.verify()

    def test_short_row_read_can_end_at_reserved_boundary(self):
        module = GoldenGemm(Shape(8, 128, 64, "i32", bm=1, bn=1,
                                  cache_a=True)).build()
        op = next(op for op in module.walk() if isinstance(op, ComputeOp))
        op.attributes["a_max"] = IntegerAttr(F.SPAD_ROWS - 7, i64)
        op.attributes["a_reserved_rows"] = IntegerAttr(F.SPAD_ROWS, i64)
        op.attributes["a_rows"] = IntegerAttr(7, i64)
        module.verify()
        # The execute controller zeroes omitted rows; only the encoded row
        # extent is read. One more requested row would cross the reserved end.
        op.attributes["a_rows"] = IntegerAttr(8, i64)
        with self.assertRaisesRegex(VerifyException, "exceeds reserved scratchpad"):
            module.verify()


if __name__ == "__main__":
    unittest.main()
