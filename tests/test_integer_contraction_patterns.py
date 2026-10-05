"""Canonical batched attention contraction and an inexact accumulator."""

import unittest

from mlir_oot.contraction_patterns import IntegerGemm, match_integer_gemm
from mlir_oot.frontend.parse import parse_module


SOURCE = """module {
  func.func @forward(%a: tensor<2x3x5xi8>, %b: tensor<2x5x4xi8>) -> tensor<2x3x4xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<2x3x4xi32>
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
    %result = linalg.generic {indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<2x3x5xi8>, tensor<2x5x4xi8>)
      outs(%init : tensor<2x3x4xi32>) {
      ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
        %la = arith.extsi %lhs : i8 to i32
        %rb = arith.extsi %rhs : i8 to i32
        %product = arith.muli %la, %rb : i32
        %sum = arith.addi %acc, %product : i32
        linalg.yield %sum : i32
    } -> tensor<2x3x4xi32>
    func.return %result : tensor<2x3x4xi32>
  }
}"""


class TestIntegerContractionPatterns(unittest.TestCase):
    def test_exact_batched_gemm(self):
        module = parse_module(SOURCE)
        op = next(x for x in module.walk() if x.name == "linalg.generic")
        self.assertEqual(match_integer_gemm(op), IntegerGemm(2, 3, 4, 5))

    def test_nonzero_accumulator_declines(self):
        module = parse_module(SOURCE.replace("arith.constant 0 : i32",
                                             "arith.constant 1 : i32"))
        op = next(x for x in module.walk() if x.name == "linalg.generic")
        self.assertIsNone(match_integer_gemm(op))

    def test_transposed_rhs_declines(self):
        wrong = SOURCE.replace("(d0, d3, d2)", "(d0, d2, d3)")
        module = parse_module(wrong)
        op = next(x for x in module.walk() if x.name == "linalg.generic")
        self.assertIsNone(match_integer_gemm(op))


if __name__ == "__main__":
    unittest.main()
