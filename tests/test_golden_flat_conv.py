import unittest

from mlir_oot.golden_conv import ConvShape, command_counts as row_counts
from mlir_oot.golden_flat_conv import GoldenFlatConv, command_counts, spatial_runs
from mlir_oot.golden_device_lower import lower


class TestGoldenFlatConv(unittest.TestCase):
    def test_gather_coordinates_cover_exact_output_and_halo(self):
        for h, w, stride in ((7, 7, 1), (14, 14, 2), (3, 19, 1), (7, 19, 2)):
            s = ConvShape(h, w, 20, 19, stride, explicit_halo=True)
            covered = []
            for tile, lane, y, x, rows in spatial_runs(s):
                self.assertLessEqual(lane + rows, 16)
                self.assertLessEqual(x + rows, s.ow)
                for r in range(rows):
                    self.assertEqual(tile*16 + lane + r, y*s.ow + x + r)
                    covered.append((y, x+r))
                    for kh in range(3):
                        for kw in range(3):
                            iy, ix = y*stride+kh, (x+r)*stride+kw
                            self.assertTrue(0 <= iy < h+2 and 0 <= ix < w+2)
            self.assertEqual(covered, [(y, x) for y in range(s.oh) for x in range(s.ow)])

    def test_refuse_capacity_and_unproven_halo(self):
        for s in (ConvShape(7, 7, 64, 64), ConvShape(56, 56, 64, 64, explicit_halo=True)):
            with self.assertRaises(ValueError):
                GoldenFlatConv(s)

    def test_late_shape_command_reduction(self):
        s = ConvShape(7, 7, 512, 512, explicit_halo=True, wide_b=True)
        counts = command_counts(s)
        row = row_counts(s)
        self.assertEqual(counts['compute'], 36864)
        self.assertEqual(counts['compute']*7, row['compute']*4)
        self.assertEqual(counts['mvin_b']*7, row['mvin_b'])
        self.assertEqual(counts['weight_bytes'], 9*512*512)

    def test_schedule_selection_preserves_semantics(self):
        from mlir_oot.conv_schedule import select_kernel
        from dataclasses import asdict
        s = ConvShape(7, 7, 512, 512, explicit_halo=True)
        kernel, kind = select_kernel(s, flat_spatial=True)
        self.assertEqual(kind, 'spatial_flat_wide_a_separate_b')
        self.assertTrue(kernel.separate_b_bank)
        self.assertTrue(kernel.wide_a)
        self.assertEqual(kernel.conv.bn, 16)
        original, selected = asdict(s), asdict(kernel.conv)
        for key in ('bn', 'wide_b'):
            original.pop(key); selected.pop(key)
        self.assertEqual(original, selected)
        self.assertEqual(select_kernel(s)[1], 'output_row')
        self.assertEqual(select_kernel(ConvShape(56,56,64,64,explicit_halo=True),flat_spatial=True)[1], 'spatial_banded_wide_a_separate_b')

    def test_wide_a_channels_and_capacity(self):
        s = ConvShape(7, 7, 512, 512, bn=16, explicit_halo=True)
        narrow, wide = command_counts(s), command_counts(s,wide_a=True)
        self.assertEqual(wide['mvin_a']*4, narrow['mvin_a'])
        self.assertEqual(wide['compute'], narrow['compute'])
        self.assertEqual(wide['mvin_a'], 1440)
        for channels in (1, 16, 20, 64, 84):
            tail = ConvShape(3,19,channels,79,2,bn=8,explicit_halo=True)
            module = GoldenFlatConv(tail,wide_a=True).build()
            module.verify()
            lower(module).verify()

    def test_separate_bank_ranges(self):
        s = ConvShape(7,7,512,512,bn=16,explicit_halo=True)
        module = GoldenFlatConv(s,wide_a=True,separate_b_bank=True).build()
        module.verify()
        loads = [op for op in module.walk() if op.name=='gemmini.mvin']
        a = [op for op in loads if op.a('load_id')==0]
        b = [op for op in loads if op.a('load_id')==1]
        self.assertTrue(a and b)
        self.assertTrue(all(op.a('local')+4*16 <= 8192 for op in a))
        self.assertTrue(all(8192 <= op.a('local') < 16384 for op in b))
        lower(module).verify()

    def test_complete_row_bands_and_tail_cover_source(self):
        from mlir_oot.golden_flat_conv import choose_band_rows
        for h,w,stride,expected in [(56,56,1,4),(28,28,1,8),(56,56,2,8),(31,19,1,None)]:
            s = ConvShape(h,w,20,79,stride,explicit_halo=True)
            rows = choose_band_rows(s)
            if expected is not None:self.assertEqual(rows,expected)
            pixels=[]
            for start in range(0,s.oh,rows):
                for tile,lane,y,x,count in spatial_runs(s,min(rows,s.oh-start)):
                    for i in range(count):
                        pixels.append((start+y,x+i))
                        for kh in range(3):
                            for kw in range(3):
                                self.assertLess((start+y)*stride+kh,h+2)
                                self.assertLess((x+i)*stride+kw,w+2)
            self.assertEqual(pixels,[(y,x) for y in range(s.oh) for x in range(s.ow)])
            module=GoldenFlatConv(s,wide_a=True,separate_b_bank=True,band_rows=rows).build()
            module.verify();lower(module).verify()
        for h,c in [(56,64),(28,128)]:
            s=ConvShape(h,h,c,c,explicit_halo=True)
            counts=command_counts(s,wide_a=True,band_rows=choose_band_rows(s))
            self.assertEqual(counts['compute'],28224)
            self.assertEqual(counts['padded_array_issue_cycles'],451584)

    def test_numeric_probe_shapes_lower_without_host_tensor_operations(self):
        for stride in (1, 2):
            s = ConvShape(3, 19, 20, 19, stride, explicit_halo=True)
            module = GoldenFlatConv(s).build()
            module.verify()
            result = lower(module)
            result.verify()
            self.assertFalse(any(op.name.startswith('gemmini.') for op in result.walk()))
            self.assertFalse(any(op.name in ('llvm.alloca', 'llvm.load', 'llvm.store') for op in result.walk()))
