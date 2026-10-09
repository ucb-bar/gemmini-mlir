"""A rewritten i8×i8→i32 matmul must verify and route to Gemmini."""

import unittest

from mlir_oot.frontend.parse import parse_module
from mlir_oot.frontend.linalg_reader import read
from mlir_oot.lowering.model_lane import mesh_eligible


MODULE = """module {
 func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<2x4xi32>
  %f = linalg.fill ins(%z : i32) outs(%e : tensor<2x4xi32>) -> tensor<2x4xi32>
  %y = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.orig_dtype = "float32"} ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%f : tensor<2x4xi32>) -> tensor<2x4xi32>
  func.return %y : tensor<2x4xi32>
 }
}"""


class TestMixedMatmul(unittest.TestCase):
    def test_widens_before_multiply_and_places_current_dtype(self):
        module = parse_module(MODULE)
        matmul = next(op for op in module.walk() if op.name == "linalg.matmul")
        body_ops = [op.name for op in matmul.regions[0].blocks[0].ops]
        self.assertEqual(body_ops.count("arith.extsi"), 2)
        self.assertEqual(body_ops.count("arith.muli"), 1)
        wl = read(module)
        self.assertEqual(len(wl.mesh_regions), 1)
        self.assertEqual(wl.mesh_regions[0].dtype, "i8")
        self.assertTrue(mesh_eligible(matmul,
                                      {r.region_id: r.lane for r in wl.regions}))


if __name__ == "__main__":
    unittest.main()
