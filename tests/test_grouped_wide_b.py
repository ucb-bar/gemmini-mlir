import unittest
from dataclasses import replace
from mlir_oot.golden_gemm import Shape,GoldenGemm
from mlir_oot.golden_tuning import estimate
from mlir_oot.golden_device_lower import lower

class TestGroupedWideB(unittest.TestCase):
    def test_tail_group_command_counts(self):
        s=Shape(23,173,37,'i32',bm=1,bn=5)
        a,b=estimate(s),estimate(replace(s,wide_b=True))
        self.assertEqual(a['b_mvin_commands'],66)
        self.assertEqual(b['b_mvin_commands'],30)
        self.assertEqual(a['b_panel_loads'],b['b_panel_loads'])
        self.assertEqual(a['dma_request_bytes_upper'],b['dma_request_bytes_upper'])
        self.assertEqual(a['mesh_compute_commands'],b['mesh_compute_commands'])

    def test_cached_b_tail_group_counts_and_lowering(self):
        s=Shape(23,173,37,'i32',bm=2,bn=11,cache_b=True,wide_b=True)
        self.assertEqual(estimate(s)['b_mvin_commands'],9)
        module=GoldenGemm(s).build();module.verify()
        loads=[op for op in module.walk() if op.name=='gemmini.mvin' and op.a('load_id')==1]
        self.assertEqual(len(loads),9)
        self.assertEqual([op.a('cols') for op in loads],[64,64,45]*3)
        self.assertEqual([op.a('rows') for op in loads],[16]*6+[5]*3)
        lower(module).verify()

    def test_large_n_with_cached_a_capacity_and_partial_shapes(self):
        for n,k in [(173,37),(2048,2048),(32000,2048)]:
            s=Shape(8,n,k,'i32',bm=1,bn=64,cache_a=True,wide_b=True)
            s.validate()
            module=GoldenGemm(s).build();module.verify();lower(module).verify()
        s=Shape(8,2048,2048,'i32',bm=1,bn=64)
        estimate_wide=estimate(replace(s,wide_b=True,cache_a=True))
        self.assertEqual(estimate_wide['b_mvin_commands'],4096)
        self.assertEqual(estimate_wide['a_mvin_commands'],128)
        self.assertEqual(estimate_wide['mesh_compute_commands'],16384)

    def test_opt_in_policy_only_changes_short_m(self):
        from mlir_oot.contraction_patterns import IntegerGemm
        from mlir_oot.golden_contraction_upstream import choose_shape
        dims=IntegerGemm(1,8,2048,2048)
        old,new=choose_shape(dims),choose_shape(dims,large_n=True)
        self.assertFalse(old.wide_b);self.assertFalse(old.cache_a)
        self.assertTrue(new.wide_b);self.assertTrue(new.cache_a)
        self.assertLess(estimate(new)['primitive_command_count'],estimate(old)['primitive_command_count'])
        dims=IntegerGemm(1,3136,64,64)
        self.assertEqual(choose_shape(dims),choose_shape(dims,large_n=True))

    def test_catalog_policy_preserves_structural_binding(self):
        from test_mixed_matmul_parse import MODULE
        from mlir_oot.golden_device_catalog import build_catalog
        source=MODULE.replace('2x3xi8','8x2048xi8').replace('3x4xi8','2048x2048xi8').replace('2x4xi32','8x2048xi32')
        _,old=build_catalog(source)
        _,new=build_catalog(source,large_n=True)
        self.assertEqual(old['source_sha256'],new['source_sha256'])
        self.assertEqual(old['covered_contractions'],new['covered_contractions'])
        self.assertEqual(new['covered_contractions'],1)
        self.assertFalse(old['kernels'][0]['schedule']['wide_b'])
        self.assertTrue(new['kernels'][0]['schedule']['wide_b'])
        self.assertEqual(new['schedule_policy'],'large_n_grouped_b_v1')
        _,refused=build_catalog(source.replace('arith.constant 0','arith.constant 1'),large_n=True)
        self.assertEqual(refused['covered_contractions'],0)
