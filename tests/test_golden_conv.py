import unittest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from mlir_oot.golden_conv import ConvShape, GoldenConv, command_counts
from mlir_oot.golden_device_lower import lower
from mlir_oot.ir.gemmini_dialect import GEMMINI


class TestGoldenConv(unittest.TestCase):
    def test_shapes_and_limits(self):
        self.assertEqual((ConvShape(7,19,20,19).oh,ConvShape(7,19,20,19).ow),(7,19))
        self.assertEqual((ConvShape(7,19,20,19,2).oh,ConvShape(7,19,20,19,2).ow),(4,10))
        with self.assertRaises(ValueError):
            GoldenConv(ConvShape(56,2048,64,64))
        with self.assertRaises(ValueError):
            GoldenConv(ConvShape(7,7,64,64,3))

    def test_border_and_channel_tails_roundtrip_and_lower(self):
        for stride in (1,2):
            module = GoldenConv(ConvShape(3,19,20,19,stride)).build()
            context = Context()
            for dialect in (Builtin,llvm.LLVM,GEMMINI):
                context.load_dialect(dialect)
            parsed = Parser(context,str(module)).parse_module()
            parsed.verify()
            lowered = lower(parsed)
            self.assertFalse(any(op.name.startswith('gemmini.') for op in lowered.walk()))
            self.assertFalse(any(op.name in ('llvm.alloca','llvm.load','llvm.store') for op in lowered.walk()))

    def test_command_model(self):
        counts = command_counts(ConvShape(3,19,20,19))
        self.assertEqual(counts['compute'],7*3*2*2*2)
        self.assertEqual(counts['mvin_b'],7*3*2*2)
        self.assertEqual(counts['mvout'],3*2*2)
        self.assertEqual(counts['host_im2col_bytes'],0)
        wide = command_counts(ConvShape(56,56,64,64,wide_b=True))
        narrow = command_counts(ConvShape(56,56,64,64))
        self.assertEqual(wide['compute'],narrow['compute'])
        self.assertEqual(wide['mvin_b']*4,narrow['mvin_b'])
