import unittest
import os.path

from parameterized import parameterized
from memory_profiler import LineProfiler
import numpy as np

import gist


DATADIR = os.path.join(os.path.dirname(__file__), 'test/data')


class FunctionalityTestCase(unittest.TestCase):
    def test_with_zero_array(self):
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        gist.extract(black_image)

    @parameterized.expand([
        ((), ),
        ((0, 0, 3), ),  # Both width and height are 0
        ((0, 32, 3), ),  # Width is 0
        ((32, 0, 3), ),  # Height is 0
        ((32, 32, 0), ),  # Invalid color channels
        ((32, 32, 1), ),  # Invalid color channels
        ((32, 32, 2), ),  # Invalid color channels
        ((32, 32, 4), ),  # Invalid color channels
    ])
    def test_with_invalid_size_array(self, shape):
        arr = np.zeros(shape, dtype=np.uint8)
        with self.assertRaises(ValueError):
            gist.extract(arr)

    def test_with_None(self):
        with self.assertRaises(TypeError):
            self.assertIsNone(gist.extract(None))


class ValueTestCase(unittest.TestCase):
    def load_npy(self, relpath):
        return np.load(os.path.join(DATADIR, relpath))

    def load_reference(self, path):
        with open(path) as f:
            content = f.read()

        arr = np.array([float(elem) for elem in content.split()])
        return arr

    def test(self):
        arr = self.load_npy('scene.npy')
        result = gist.extract(arr)

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.no_arg.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)

    def test_with_nblocks_2_as_positional_argument(self):
        arr = self.load_npy('scene.npy')
        result = gist.extract(arr, 2)

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.nblocks2.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)

    def test_with_nblocks_4_as_positional_argument(self):
        arr = self.load_npy('scene.npy')
        result = gist.extract(arr, 4)

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.nblocks4.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)

    def test_with_nblocks_2_as_keyword_argument(self):
        arr = self.load_npy('scene.npy')
        result = gist.extract(arr, nblocks=2)

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.nblocks2.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)

    def test_with_nblocks_4_as_keyword_argument(self):
        arr = self.load_npy('scene.npy')
        result = gist.extract(arr, nblocks=4)

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.nblocks4.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)

    def test_with_orientations_per_scale_4_4_4_as_keyword_argument(self):
        arr = self.load_npy('scene.npy')
        result = gist.extract(arr, orientations_per_scale=(4, 4, 4))

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.ops_4_4_4.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)

    def test_with_orientations_per_scale_8_8_8_as_keyword_argument(self):
        arr = self.load_npy('scene.npy')
        result = gist.extract(arr, orientations_per_scale=(8, 8, 8))

        reference = self.load_reference(
            os.path.join(DATADIR, 'scene.ops_8_8_8.result'))
        np.testing.assert_allclose(reference, result, rtol=1e-04, atol=1e-04)


class MemoryUsageTestCase(unittest.TestCase):
    def load_npy(self, relpath):
        return np.load(os.path.join(DATADIR, relpath))

    def test(self):
        def run(img):
            gist.extract(img)
            gist.extract(img)
            gist.extract(img)
            gist.extract(img)
            gist.extract(img)

        img = self.load_npy('scene_640.npy')
        img_mb_as_float = img.shape[0] * img.shape[1] * img.shape[2] * 4 / 1024 / 1024

        prof = LineProfiler()
        run = prof(run)
        run(img)

        _, lines = list(prof.code_map.items())[0]
        mem_list = [mem for lineo, mem in list(lines)]
        mem_inc_list = [inc for inc, mem in mem_list[1:]]  # First line of `run` function is definition ("def run (img):")
        mem_inc_list[1:]  # Ignore the first because it includes an overhead of loading module.

        for mem_inc in mem_inc_list:
            # Assert that at least the memory used for the loaded image is released.
            self.assertTrue(mem_inc < img_mb_as_float)


if __name__ == '__main__':
    unittest.main()
