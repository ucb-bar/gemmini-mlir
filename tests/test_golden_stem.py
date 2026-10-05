import unittest
from pathlib import Path
from mlir_oot.golden_stem import StemShape,GoldenStem,command_counts
from mlir_oot.stem_binding import match,rewrite
from mlir_oot.frontend.parse import parse_module
from mlir_oot.contraction_patterns import match_integer_gemm
from mlir_oot.direct_conv_binding import serialize

class TestGoldenStem(unittest.TestCase):
    def fixture(self):
        m=parse_module(Path(__file__).with_name('fixtures').joinpath('stem_matmul0.mlir').read_text())
        return m,next(o for o in m.walk() if match_integer_gemm(o))
    def test_real_stem_matches_and_rewrites(self):
        m,op=self.fixture();b=match(op);self.assertEqual(b.shape,StemShape(224,224,64))
        d=rewrite(b);m.verify();parse_module(serialize(m,[d])).verify()
        self.assertFalse(any(match_integer_gemm(x) for x in m.walk()))
    def test_wrong_tap_order_refused(self):
        _,op=self.fixture();concat=op.operands[0].owner;operands=list(concat.operands);operands[0],operands[1]=operands[1],operands[0];concat.operands=operands
        with self.assertRaisesRegex(ValueError,'tap order'):match(op)
    def test_reference_compute_count_without_im2col(self):
        counts=command_counts(StemShape(224,224,64));self.assertEqual(counts['compute'],43904);self.assertEqual(counts['host_im2col_bytes'],0);self.assertEqual(counts['host_weight_pack_bytes'],0)
    def test_odd_extents_and_channel_tails_verify(self):
        GoldenStem(StemShape(5,35,19)).build().verify()
        GoldenStem(StemShape(5,35,19,output_dtype='i8',scale=.03125,relu=True)).build().verify()
    def test_row_capacity_refused(self):
        with self.assertRaisesRegex(ValueError,'accumulator'):StemShape(224,4096,64).validate()
