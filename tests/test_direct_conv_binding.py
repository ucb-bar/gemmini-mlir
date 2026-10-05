import unittest
from pathlib import Path
from xdsl.dialects.builtin import AffineMapAttr,ArrayAttr
from xdsl.ir.affine import AffineMap,AffineExpr
from mlir_oot.frontend.parse import parse_module
from mlir_oot.direct_conv_binding import match,rewrite,capture,serialize


class TestDirectConvBinding(unittest.TestCase):
    def get(self):
        m=parse_module(Path(__file__).with_name('fixtures').joinpath('direct_conv2.mlir').read_text())
        op=next(x for x in m.walk() if x.name=='linalg.matmul')
        return m,op

    def test_capture_and_rewrite_exact_boundary(self):
        _,op=self.get();b=match(op)
        self.assertEqual((b.shape.h,b.shape.w,b.shape.cin,b.shape.cout),(56,56,64,64))
        self.assertTrue(b.shape.explicit_halo)
        m=capture(b);op=next(x for x in m.walk() if x.name=='linalg.matmul');oldtype=op.results[0].type
        declaration=rewrite(match(op));m.verify()
        printed=serialize(m,declaration)
        parsed=parse_module(printed)
        decl=next(x for x in parsed.body.block.ops if x.name=='func.func' and x.sym_name.data==declaration.sym_name.data)
        self.assertEqual(printed.count('bufferization.access = "read"'),2)
        self.assertEqual(printed.count('bufferization.access = "write"'),1)
        ret=next(x for x in m.walk() if x.name=='func.return')
        self.assertEqual(ret.operands[0].type,oldtype)
        self.assertFalse(any(x.name=='linalg.matmul' for x in m.walk()))
        self.assertTrue(any(x.name=='func.call' for x in m.walk()))

    def test_rejects_wrong_gather_access_despite_provenance(self):
        _,op=self.get();b=match(op)
        d=[AffineExpr.dimension(i) for i in range(6)]
        b.gather.properties['indexing_maps']=ArrayAttr([
            AffineMapAttr(AffineMap(6,0,(d[3],d[0],d[4]+d[1]*2,d[5]+d[2]))),
            AffineMapAttr(AffineMap.identity(6))])
        with self.assertRaisesRegex(ValueError,'affine map'):match(op)

    def test_rejects_gather_that_yields_initializer(self):
        _,op=self.get();b=match(op);block=b.gather.regions[0].blocks[0]
        block.last_op.operands=[block.args[1]]
        with self.assertRaisesRegex(ValueError,'copy its input'):match(op)

    def test_rejects_missing_flattening(self):
        _,op=self.get();op.operands=[op.operands[0],op.operands[0],op.operands[2]]
        with self.assertRaises(ValueError):match(op)

    def test_current_concat_capture_rewrites_without_output_transpose(self):
        m=parse_module(Path(__file__).with_name('fixtures').joinpath('direct_concat_matmul2.mlir').read_text())
        from mlir_oot.contraction_patterns import match_integer_gemm
        op=next(x for x in m.walk() if match_integer_gemm(x))
        b=match(op)
        self.assertEqual(b.orientation,'spatial_first')
        rewrite(b);m.verify()
        ret=next(x for x in m.walk() if x.name=='func.return')
        self.assertEqual(ret.operands[0].owner.name,'func.call')

    def test_concat_tap_reordering_refused(self):
        m=parse_module(Path(__file__).with_name('fixtures').joinpath('direct_concat_matmul2.mlir').read_text())
        from mlir_oot.contraction_patterns import match_integer_gemm
        op=next(x for x in m.walk() if match_integer_gemm(x))
        b=match(op);operands=list(b.gather.operands);operands[0],operands[1]=operands[1],operands[0];b.gather.operands=operands
        with self.assertRaisesRegex(ValueError,'tap order'):match(op)
