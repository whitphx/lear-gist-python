import unittest

import numpy as np

import gist


class GistTestCase(unittest.TestCase):
    def test_with_zero_array(self):
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        gist.extract(black_image)

    def test_with_None(self):
        with self.assertRaises(TypeError):
            self.assertIsNone(gist.extract(None))
