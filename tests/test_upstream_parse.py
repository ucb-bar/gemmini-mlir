"""Regression for the two-result linalg.generic spelling in full SmolVLA."""

import unittest

from mlir_oot.frontend.parse import _normalize_multi_generic_results, parse_module
from xdsl.dialects import bufferization, memref


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


class TestBufferizedHostWriter(unittest.TestCase):
    def test_registered_fresh_borrowed_writer_roundtrip(self):
        source = """builtin.module {
          func.func private @borrowed(memref<2x3xbf16>, memref<2x3xbf16>)
          func.func @writer(%a: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
            %read = bufferization.to_buffer %a read_only : tensor<2x3xbf16> to memref<2x3xbf16>
            %out = memref.alloc() {alignment = 64 : i64} : memref<2x3xbf16>
            func.call @borrowed(%read, %out) : (memref<2x3xbf16>, memref<2x3xbf16>) -> ()
            %result = bufferization.to_tensor %out restrict writable : memref<2x3xbf16> to tensor<2x3xbf16>
            func.return %result : tensor<2x3xbf16>
          }
        }"""
        module = parse_module(source)
        module.verify()
        self.assertTrue(any(isinstance(op, bufferization.ToBufferOp) for op in module.walk()))
        self.assertTrue(any(isinstance(op, bufferization.ToTensorOp) for op in module.walk()))
        self.assertTrue(any(isinstance(op, memref.AllocOp) for op in module.walk()))


class TestFreshWriterBufferization(unittest.TestCase):
    def test_read_only_tensor_buffer_roundtrip(self):
        from io import StringIO
        from xdsl.dialects.bufferization import ToBufferOp
        from xdsl.printer import Printer
        from mlir_oot.frontend.parse import parse_module

        source = '''module {
          func.func @borrow(%arg: tensor<2x3xbf16>) {
            %buffer = bufferization.to_buffer %arg read_only : tensor<2x3xbf16> to memref<2x3xbf16>
            func.return
          }
        }'''
        module = parse_module(source)
        operations = [op for op in module.walk() if isinstance(op, ToBufferOp)]
        self.assertEqual(len(operations), 1)
        self.assertIsNotNone(operations[0].read_only)
        stream = StringIO()
        Printer(stream=stream).print_op(module)
        reparsed = parse_module(stream.getvalue())
        self.assertTrue(module.is_structurally_equivalent(reparsed))


if __name__ == "__main__":
    unittest.main()
