"""Edge-tile grouping must keep the last partial tile out of full loops."""

import unittest

from mlir_oot.golden_gemm import _groups


class TestGroups(unittest.TestCase):
    def test_block_one_after_full_tile(self):
        self.assertEqual(_groups(17, 1),
                         [(0, 1, 1, (16,)), (1, 2, 1, (1,))])

    def test_full_group_followed_by_partial_group(self):
        self.assertEqual(_groups(80, 4),
                         [(0, 4, 4, (16, 16, 16, 16)), (4, 5, 1, (16,))])
        self.assertEqual(_groups(81, 4),
                         [(0, 4, 4, (16, 16, 16, 16)), (4, 6, 2, (16, 1))])

    def test_no_phantom_edge_for_exact_multiple(self):
        self.assertEqual(_groups(64, 4), [(0, 4, 4, (16, 16, 16, 16))])


if __name__ == "__main__":
    unittest.main()
