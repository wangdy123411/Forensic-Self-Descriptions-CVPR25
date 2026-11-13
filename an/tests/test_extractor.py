"""Numerical regression tests for tiled and FFT FSD extraction.

Run from repository root: an/.venv/Scripts/python.exe -m unittest discover -s an/tests -p test_extractor.py
"""

from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fsd.fre import FRE
from fsd.fsd_computation import compute_fsd
from an.robustness.extractor import (
    _accumulate_fft,
    _accumulate_tiled,
    _prepare_residuals,
    extract_fsd,
)


class ExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fft_gram_matches_explicit_reflected_patches(self):
        generator = torch.Generator().manual_seed(817)
        # Odd/non-square dimensions catch incorrect FFT size or lag conventions.
        for shape, kernel, levels in [((3, 31, 37), 5, 2), ((8, 40, 44), 11, 2)]:
            with self.subTest(shape=shape):
                residuals = torch.randn(shape, generator=generator, dtype=torch.float64)
                direct_gram, direct_rhs, count = _accumulate_tiled(residuals, kernel, levels, 3)
                fft_gram, fft_rhs, fft_count = _accumulate_fft(residuals, kernel, levels, 113)
                self.assertEqual(count, fft_count)
                torch.testing.assert_close(fft_gram, direct_gram, rtol=1e-11, atol=1e-10)
                torch.testing.assert_close(fft_rhs, direct_rhs, rtol=1e-11, atol=1e-10)

    def test_end_to_end_matches_upstream(self):
        torch.manual_seed(541)
        fre = FRE(out_channels=2, kernel_size=3).eval()
        array = np.random.default_rng(41).integers(0, 256, (59, 67, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        for mode in ["resize", "crop", "resize_and_crop"]:
            with self.subTest(mode=mode):
                kwargs = dict(kernel_size=5, num_scales=3, max_size=32, resize_mode=mode)
                reference = compute_fsd(image, fre, **kwargs)
                for method in ["tiled", "fft"]:
                    result = extract_fsd(
                        image,
                        fre,
                        **kwargs,
                        method=method,
                        tile_rows=3,
                        boundary_chunk=71,
                        return_preview=True,
                    )
                    torch.testing.assert_close(result.fsd, reference, atol=1e-7, rtol=1e-6)
                    self.assertLess(result.diagnostics["constraint_max_abs_error"], 1e-8)
                    self.assertEqual(result.residual_preview.shape, (2, 64, 64))

    def test_full_960_dimensions_and_channel_constraints(self):
        torch.manual_seed(702)
        fre = FRE().eval()
        image = torch.randint(0, 256, (1, 64, 80), generator=torch.Generator().manual_seed(62))
        result = extract_fsd(image, fre, max_size=40, num_scales=2, method="fft")
        self.assertEqual(tuple(result.fsd.shape), (960,))
        self.assertTrue(torch.isfinite(result.fsd).all())
        torch.testing.assert_close(
            result.fsd.reshape(8, 120).sum(1),
            torch.ones(8, dtype=torch.float64),
            atol=1e-7,
            rtol=1e-7,
        )

    def test_no_artificial_input_normalization(self):
        torch.manual_seed(553)
        fre = FRE(out_channels=2, kernel_size=3).eval()
        image = torch.arange(400, dtype=torch.float64).reshape(20, 20)
        residuals, diagnostics = _prepare_residuals(image, fre, 18, "crop")
        expected = fre(image.unsqueeze(0))[:, 1:-1, 1:-1]
        torch.testing.assert_close(residuals, expected)
        self.assertEqual(diagnostics["input_height"], 20)

    def test_rejects_invalid_geometry(self):
        fre = FRE(out_channels=2, kernel_size=3).eval()
        with self.assertRaises(ValueError):
            extract_fsd(torch.zeros(1, 1), fre)
        with self.assertRaises(ValueError):
            extract_fsd(torch.zeros(20, 20), fre, kernel_size=4)


if __name__ == "__main__":
    unittest.main()
