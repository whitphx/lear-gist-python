import unittest
import os.path

import numpy as np

import gist


DATADIR = os.path.join(os.path.dirname(__file__), 'test/data')


class FunctionalityTestCase(unittest.TestCase):
    def test_with_zero_array(self):
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        gist.extract(black_image)

    def test_with_None(self):
        with self.assertRaises(TypeError):
            self.assertIsNone(gist.extract(None))


class ValueTestCase(unittest.TestCase):
    def load_reference(self, path):
        with open(path) as f:
            content = f.read()

        arr = np.array([float(elem) for elem in content.split()])
        return arr

    def test(self):
        arr = np.load(os.path.join(DATADIR, 'scene.npy'))
        result = gist.extract(arr)

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.no_arg.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)


if __name__ == '__main__':
    unittest.main()
