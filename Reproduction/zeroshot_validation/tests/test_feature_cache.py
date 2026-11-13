from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_cache import FeatureCache


def item(key="one", source="COCO2017", label="0", sha="a" * 64):
    return {"image_id": key, "sha256": sha, "source": source, "label": label}


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.identity = {"model": "fsd1.2.0", "weights_sha256": {"fre.pt": "b" * 64}, "tf32": False}
        self.vector = np.linspace(-4, 3, 960, dtype=np.float64)
        self.cache = FeatureCache(self.directory, self.identity)

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    def mutate(self, sql, params=()):
        with self.cache._connection:
            self.cache._connection.execute(sql, params)

    def test_exact_roundtrip_reopen_and_journal_policy(self):
        self.assertTrue(self.cache.put(item(), self.vector))
        self.assertEqual(self.cache.get(item()).tobytes(), self.vector.tobytes())
        self.cache.close()
        self.cache = FeatureCache(self.directory, self.identity)
        self.assertEqual(self.cache.get(item()).tobytes(), self.vector.tobytes())
        self.assertEqual(self.cache._connection.execute("PRAGMA journal_mode").fetchone()[0], "delete")
        self.assertEqual(self.cache._connection.execute("PRAGMA synchronous").fetchone()[0], 2)
        self.assertTrue(self.cache.integrity_check())

    def test_rejects_other_extraction_configuration(self):
        self.cache.put(item(), self.vector)
        with self.assertRaisesRegex(ValueError, "identity"):
            FeatureCache(self.directory, {**self.identity, "tf32": True})
        self.assertEqual(self.cache.get(item()).tobytes(), self.vector.tobytes())

    def test_rejects_image_identity_and_immutable_vector_conflicts(self):
        self.cache.put(item(), self.vector)
        for change in (item(sha="c" * 64), item(source="BigGAN"), item(label="1")):
            with self.assertRaisesRegex(ValueError, "image identity"):
                self.cache.get(change)
        self.assertFalse(self.cache.put(item(), self.vector, "backfill"))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.cache.put(item(), self.vector + 1)
        phase = self.cache._connection.execute("SELECT phase FROM features").fetchone()[0]
        self.assertEqual(phase, "online")

    def test_rejects_nonfinite_and_bad_shape_without_writes(self):
        for invalid in (np.zeros(959), np.zeros((1, 960)), np.full(960, np.nan), np.full(960, np.inf), ["x"] * 960):
            with self.assertRaises(ValueError):
                self.cache.put(item(), invalid)
        self.assertIsNone(self.cache.get(item()))

    def test_rejects_precision_conversion_but_accepts_float64_byte_orders(self):
        for dtype in (np.float32, np.float16, np.int64, np.uint64, np.complex128):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "float64"):
                self.cache.put(item(), np.arange(960, dtype=dtype))
        self.assertIsNone(self.cache.get(item()))
        self.assertTrue(self.cache.put(item(), self.vector.astype(">f8")))
        self.assertEqual(self.cache.get(item()).tobytes(), self.vector.astype("<f8").tobytes())
        self.assertFalse(self.cache.put(item(), self.vector.astype("<f8")))

    def test_corrupt_blob_dtype_dimensions_signature_and_nonfinite_rejected(self):
        changes = [("feature", b"wrong", "length"), ("dtype", "<f4", "dtype"),
                   ("dimensions", 12, "shape"), ("extraction_signature", "c" * 64, "signature"),
                   ("feature_sha256", "c" * 64, "checksum")]
        for field, value, pattern in changes:
            self.cache.put(item(), self.vector)
            self.mutate(f"UPDATE features SET {field}=?", (value,))
            with self.assertRaisesRegex(ValueError, pattern):
                self.cache.get(item())
            self.mutate("DELETE FROM features")
        self.cache.put(item(), self.vector)
        bad = np.full(960, np.nan, dtype="<f8").tobytes()
        self.mutate("UPDATE features SET feature=?,feature_sha256=?", (bad, hashlib.sha256(bad).hexdigest()))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.cache.get(item())

    def test_coverage_distinguishes_online_backfill_missing_and_full_verification(self):
        self.cache.put(item(), self.vector)
        self.cache.put(item("two"), self.vector + 1, phase="backfill")
        report = self.cache.coverage([item(), item("two"), item("three")], verify_payloads=True)
        self.assertEqual((report["cached_count"], report["missing_count"]), (2, 1))
        self.assertEqual(report["missing_image_ids"], ["three"])
        self.assertEqual(report["by_phase"], {"online": 1, "backfill": 1})
        self.assertTrue(report["payloads_verified"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.cache.coverage([item(), item()])


if __name__ == "__main__":
    unittest.main()
