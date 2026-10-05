"""Regression for the two-result linalg.generic spelling in full SmolVLA."""

import unittest

from mlir_oot.frontend.parse import _normalize_multi_generic_results


class TestMultiResultLinalgGeneric(unittest.TestCase):
    def test_normalizes_only_multi_tensor_result_spelling(self):
        source = ("  } -> (tensor<1xi64>, tensor<1xi64>)\n"
                  "  } -> tensor<4xf32>\n"
                  "  } -> (tensor<1xi64>)\n")
        expected = ("  } -> tensor<1xi64>, tensor<1xi64>\n"
                    "  } -> tensor<4xf32>\n"
                    "  } -> (tensor<1xi64>)\n")
        self.assertEqual(_normalize_multi_generic_results(source), expected)
        self.assertEqual(_normalize_multi_generic_results(expected), expected)


if __name__ == "__main__":
    unittest.main()
