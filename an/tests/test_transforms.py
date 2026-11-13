"""Behavioral invariants for the paired image-processing study."""

from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from an.robustness.transforms import CONDITIONS, CONDITION_MAP, apply_condition


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.array = np.random.default_rng(992).integers(0, 256, (80, 96, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.array)

    def test_jpeg_is_actual_deterministic_encode_decode(self):
        blobs = []
        original_save = Image.Image.save

        def capture_save(image, fp, *args, **kwargs):
            result = original_save(image, fp, *args, **kwargs)
            if hasattr(fp, "getvalue"):
                blobs.append(fp.getvalue())
            return result

        with patch.object(Image.Image, "save", new=capture_save):
            first = apply_condition(self.image, CONDITION_MAP["jpeg_q50"])
            second = apply_condition(self.image, CONDITION_MAP["jpeg_q50"])
        self.assertEqual(len(blobs), 2)
        self.assertEqual(blobs[0], blobs[1])
        self.assertTrue(blobs[0].startswith(b"\xff\xd8"))
        self.assertTrue(blobs[0].endswith(b"\xff\xd9"))
        with Image.open(BytesIO(blobs[0])) as encoded:
            self.assertEqual(encoded.format, "JPEG")
            self.assertEqual(encoded.size, self.image.size)
            expected = np.asarray(encoded.convert("RGB"))
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.mode, "RGB")
        self.assertFalse(np.array_equal(first, self.array))

    def test_crop_keeps_quarter_area_without_interpolation(self):
        # Unique spatial RGB codes make any resampling or off-center crop visible.
        yy, xx = np.indices((160, 200))
        pixels = np.stack([xx, yy, (xx + yy) % 256], axis=-1).astype(np.uint8)
        result = apply_condition(Image.fromarray(pixels), CONDITION_MAP["crop_area025"])
        self.assertEqual(result.size, (100, 80))
        self.assertEqual(result.width * result.height, pixels.shape[0] * pixels.shape[1] // 4)
        np.testing.assert_array_equal(result, pixels[40:120, 50:150])

    def test_constants_preserved_and_resize_does_not_resize_back(self):
        constant = Image.new("RGB", (96, 80), (128, 128, 128))
        for condition in CONDITIONS:
            with self.subTest(condition=condition.name):
                result = apply_condition(constant, condition)
                self.assertTrue(np.all(np.asarray(result) == 128))
        self.assertEqual(apply_condition(constant, CONDITION_MAP["resize_025"]).size, (24, 20))
        self.assertEqual(
            apply_condition(constant, CONDITION_MAP["resize050_jpeg50"]).size, (48, 40)
        )

    def test_paired_conditions_are_label_and_execution_order_independent(self):
        # The same pixels with opposing experimental labels must receive exactly
        # the same processing. A condition sweep must not mutate its baseline.
        samples = [{"label": 0, "image": self.image}, {"label": 1, "image": self.image.copy()}]
        outputs = {}
        for sample, schedule in zip(samples, [CONDITIONS, list(reversed(CONDITIONS))]):
            outputs[sample["label"]] = {
                condition.name: np.asarray(apply_condition(sample["image"], condition)).copy()
                for condition in schedule
            }
        for condition in CONDITIONS:
            np.testing.assert_array_equal(outputs[0][condition.name], outputs[1][condition.name])
        np.testing.assert_array_equal(self.image, self.array)
        np.testing.assert_array_equal(outputs[0]["clean"], self.array)

    def test_blur_reduces_fine_texture_and_sharpen_increases_edge_contrast(self):
        yy, xx = np.indices((80, 96))
        pattern = np.round(128 + 40 * np.sin(xx * np.pi / 4)).astype(np.uint8)
        image = Image.fromarray(pattern).convert("RGB")
        blurred = np.asarray(apply_condition(image, CONDITION_MAP["blur_sigma20"]))[10:-10, 10:-10]
        sharpened = np.asarray(apply_condition(image, CONDITION_MAP["sharpen_200"]))[10:-10, 10:-10]
        original = np.asarray(image)[10:-10, 10:-10]
        self.assertLess(blurred.var(), original.var() / 3)
        self.assertGreater(sharpened.var(), original.var())


if __name__ == "__main__":
    unittest.main()
