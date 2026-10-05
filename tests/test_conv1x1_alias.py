"""Unit 1x1 NHWC convs use their input tensor directly as a GEMM lhs."""

import unittest

from mlir_oot.frontend.reader import Node, TensorDecl, Workload
from mlir_oot.lowering.plan import Builder, Contraction


def plan_for(kernel, padding, stride):
    kh, kw, ci, co = kernel
    h, w = 6, 6
    ho = (h + padding[0] + padding[2] - kh) // stride[0] + 1
    wo = (w + padding[1] + padding[3] - kw) // stride[1] + 1
    wl = Workload(
        "0.1", "gemmini", "0.1",
        {"IFM": TensorDecl("IFM", [1, h, w, ci], "i8", "input"),
         "W": TensorDecl("W", [kh * kw * ci, co], "i8", "weight")},
        [Node("conv2d", "Y0", ["IFM", "W"], [ho * wo, co], "i32",
              {"kernel": list(kernel), "padding": list(padding),
               "stride": list(stride), "dilation": [1, 1],
               "layout": "nhwc"})], {})
    return Builder(wl).build()


class TestConv1x1Alias(unittest.TestCase):
    def test_unit_1x1_has_no_gather(self):
        plan = plan_for((1, 1, 16, 16), (0, 0, 0, 0), (1, 1))
        contraction = next(t for t in plan.tasks if isinstance(t, Contraction))
        self.assertEqual(contraction.lhs, "IFM")
        self.assertNotIn("im2col_recipes", plan.command_buffer.get("params", {}))

    def test_spatial_and_strided_cases_keep_gather(self):
        for kernel, padding, stride in (
            ((3, 3, 16, 16), (1, 1, 1, 1), (1, 1)),
            ((1, 1, 16, 16), (0, 0, 0, 0), (2, 2)),
        ):
            with self.subTest(kernel=kernel, stride=stride):
                plan = plan_for(kernel, padding, stride)
                contraction = next(t for t in plan.tasks if isinstance(t, Contraction))
                self.assertEqual(contraction.lhs, "IFM_im2col")
                self.assertEqual(len(plan.command_buffer["params"]["im2col_recipes"]), 1)


if __name__ == "__main__":
    unittest.main()
