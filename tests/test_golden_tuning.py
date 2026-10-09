"""The geometric model must count the DMA requests its schedule emits."""

import unittest
from dataclasses import replace

from mlir_oot.golden_gemm import Shape
from mlir_oot.contraction_patterns import IntegerGemm
from mlir_oot.golden_contraction_upstream import choose_shape
from mlir_oot.golden_tuning import estimate


class TestGoldenTuning(unittest.TestCase):
    def test_cached_b_is_loaded_once(self):
        shape = Shape(512, 64, 64, bm=16, bn=4)
        ordinary = estimate(shape)
        cached = estimate(replace(shape, cache_b=True))
        self.assertEqual(ordinary["b_panel_loads"], 32)
        self.assertEqual(cached["b_panel_loads"], 16)
        self.assertEqual(ordinary["a_panel_loads"], cached["a_panel_loads"])

    def test_cache_requires_one_channel_block(self):
        with self.assertRaisesRegex(ValueError, "one output-channel block"):
            Shape(512, 80, 64, bm=16, bn=4, cache_b=True).validate()

    def test_cached_short_row_a_is_loaded_once_across_n_blocks(self):
        ordinary = Shape(8, 512, 256, "i32", bm=1, bn=4)
        cached = replace(ordinary, cache_a=True)
        self.assertEqual(estimate(ordinary)["a_mvin_commands"], 128)
        self.assertEqual(estimate(cached)["a_mvin_commands"], 16)
        self.assertEqual(estimate(ordinary)["mesh_compute_commands"],
                         estimate(cached)["mesh_compute_commands"])
        with self.assertRaisesRegex(ValueError, "one complete M block"):
            replace(cached, m=32).validate()

    def test_model_rule_selects_cached_a_only_for_repeated_n_blocks(self):
        self.assertTrue(choose_shape(IntegerGemm(1, 8, 5632, 2048)).cache_a)
        self.assertTrue(choose_shape(IntegerGemm(1, 8, 32000, 2048)).cache_a)
        self.assertFalse(choose_shape(IntegerGemm(1, 8, 2048, 2048)).cache_a)

    def test_wide_loads_reduce_commands_but_not_payload(self):
        base = Shape(512, 64, 64, bm=16, bn=4, cache_b=True)
        narrow = estimate(base)
        wide = estimate(replace(base, wide_a=True, wide_b=True))
        self.assertEqual(wide["a_panel_loads"], narrow["a_panel_loads"])
        self.assertEqual(wide["b_panel_loads"], narrow["b_panel_loads"])
        self.assertEqual(wide["a_mvin_commands"], 32)
        self.assertEqual(wide["b_mvin_commands"], 4)
        self.assertEqual(narrow["a_mvin_commands"], 128)
        self.assertEqual(narrow["b_mvin_commands"], 16)
        self.assertEqual(wide["mesh_compute_commands"], 512)
        self.assertEqual(wide["padded_array_floor_cycles"], 8192)
        self.assertEqual(wide["unique_tensor_bytes"], narrow["unique_tensor_bytes"])
        self.assertEqual(wide["dma_request_bytes_upper"],
                         narrow["dma_request_bytes_upper"])


if __name__ == "__main__":
    unittest.main()
