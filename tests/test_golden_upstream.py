"""Upstream capsule semantics must be preserved by the golden bridge."""

import unittest

from mlir_oot.golden_upstream import extract
from mlir_oot.lowering.plan import LoweringDeclined


SOURCE = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x6x6x16xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<16xi32>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [1, 1, 16, 16], stride = [1, 1], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = ["bias_add", "acc_scale", "relu"], output_dtype = "i8", acc_scale = 0.125 : f32, bias = "B", layout = "nhwc"} : (tensor<1x6x6x16xi8>, !merlin_iface.resident) -> tensor<36x16xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}"""


class TestGoldenUpstream(unittest.TestCase):
    def test_typed_capsule_to_golden_binding(self):
        shape, binding = extract(SOURCE)
        self.assertEqual((shape.m, shape.n, shape.k), (36, 16, 16))
        self.assertEqual((shape.bias, shape.scale, shape.relu), (True, 0.125, True))
        self.assertEqual(binding["pointer_binding"],
                         {"A": "IFM", "B": "W", "C": "Y0", "bias": "B"})

    def test_padded_1x1_requires_gather_and_declines(self):
        padded = SOURCE.replace("padding = [0, 0, 0, 0]",
                                "padding = [1, 1, 1, 1]").replace(
                                    "tensor<36x16xi8>", "tensor<64x16xi8>")
        with self.assertRaisesRegex(LoweringDeclined, "needs a gather"):
            extract(padded)

    def test_reordered_epilogue_declines(self):
        reordered = SOURCE.replace('"bias_add", "acc_scale", "relu"',
                                   '"relu", "bias_add", "acc_scale"')
        with self.assertRaisesRegex(LoweringDeclined, "in that order"):
            extract(reordered)


if __name__ == "__main__":
    unittest.main()
