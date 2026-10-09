"""Batched attention should not reconfigure the accelerator per head."""

import unittest

from mlir_oot.golden_batched_gemm import build
from mlir_oot.golden_gemm import Shape


class TestBatchedConfigOnce(unittest.TestCase):
    def test_one_configuration_for_all_batches(self):
        module = build(32, Shape(8, 64, 8, output_dtype="i32", bm=1, bn=4,
                                 wide_b=True))
        names = [op.name for op in module.walk()]
        self.assertEqual(names.count("gemmini.flush"), 1)
        self.assertEqual(names.count("gemmini.config_ex"), 1)
        self.assertEqual(names.count("gemmini.config_st"), 1)
        self.assertEqual(names.count("gemmini.fence"), 2)
        self.assertEqual(names.count("llvm.call"), 0)
        self.assertEqual(len([op for op in module.body.blocks[0].ops
                              if op.name == "llvm.func"]), 1)


if __name__ == "__main__":
    unittest.main()
