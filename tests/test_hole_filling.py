import unittest

import os
import os.path as osp
import numpy as np
import igl
from hole_filling import triangulate_refine_fair


class TestHoleFilling(unittest.TestCase):

    def test_hole_filling(self):
        vs, fs, _ = igl.read_off(osp.join(osp.dirname(__file__), 'examples', 'bunny_holes.off'))
        out_vs, out_fs = triangulate_refine_fair(vs, fs)

        self.assertTrue(len(out_vs) > len(vs))


if __name__ == '__main__':
    unittest.main()
