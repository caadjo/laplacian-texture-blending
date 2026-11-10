import unittest

import numpy as np

from src.laplacian_blend import laplacian_blend, mip_schedule, naive_linear_blend


class BlendTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.a = rng.random((31, 35, 3), dtype=np.float32)
        self.b = rng.random((31, 35, 3), dtype=np.float32)
        self.mask = rng.random((31, 35, 1), dtype=np.float32)

    def test_schedule_keeps_residual(self):
        self.assertEqual(mip_schedule(6, 2), (0, 2, 4, 6))
        self.assertEqual(mip_schedule(5, 2), (0, 2, 4, 5))
        self.assertEqual(mip_schedule(0, 2), (0,))
        with self.assertRaises(ValueError):
            mip_schedule(6, 0)

    def test_constant_mask_telescopes_to_lerp(self):
        for value in (0.0, 128 / 255, 1.0):
            mask = np.full_like(self.mask, value)
            expected = naive_linear_blend(self.a, self.b, mask)
            for stride in (1, 2):
                with self.subTest(value=value, stride=stride):
                    actual = laplacian_blend(self.a, self.b, mask, 5, stride)
                    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)

    def test_zero_bands_equals_baseline(self):
        actual = laplacian_blend(self.a, self.b, self.mask, 0, 2)
        np.testing.assert_array_equal(actual, naive_linear_blend(self.a, self.b, self.mask))

    def test_spatial_mask_changes_skipping_result(self):
        full = laplacian_blend(self.a, self.b, self.mask, 6)
        skipped = laplacian_blend(self.a, self.b, self.mask, 6, 2)
        self.assertGreater(np.max(np.abs(full - skipped)), 0)


if __name__ == "__main__":
    unittest.main()
