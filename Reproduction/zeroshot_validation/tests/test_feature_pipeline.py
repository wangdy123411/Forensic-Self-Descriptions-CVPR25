from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from feature_cache import FeatureCache
from feature_pipeline import score_and_cache, score_from_raw


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.item = dict(image_id='sample', sha256='a' * 64, source='COCO2017', label='0')
        self.identity = {'purpose': 'pipeline_restart_test'}
        self.calls = 0

        def extract(_path):
            self.calls += 1
            return torch.tensor([1.25, -2.5, 4.75], dtype=torch.float64)

        self.detector = SimpleNamespace(
            config={'fre': {'out_channels': 1}, 'fsd': {'kernel_size': 2}},
            fre=SimpleNamespace(device='cpu'), projections=[],
            gmm=SimpleNamespace(score_samples=lambda value: value.sum(dim=1)),
            train_mean=0.5, train_std=2.0, threshold=-2.0, compute_fsd=extract,
        )

    def cache(self):
        return FeatureCache(Path(self.directory.name), self.identity, dimensions=3)

    def test_reopened_cache_avoids_extraction_and_preserves_score(self):
        with self.cache() as cache:
            cold, hit = score_and_cache(self.detector, 'unused', self.item, cache)
            self.assertFalse(hit)
            self.assertEqual(cold.raw_score, 3.5)
            self.assertEqual(cold.z_score, 1.5)
        with self.cache() as cache:
            warm, hit = score_and_cache(self.detector, 'unused', self.item, cache)
            self.assertTrue(hit)
            self.assertEqual(cold, warm)
            np.testing.assert_array_equal(cache.get(self.item), [1.25, -2.5, 4.75])
        self.assertEqual(self.calls, 1)

    def test_scoring_failure_keeps_committed_feature_for_retry(self):
        def fail(_value):
            raise RuntimeError('simulated failure after extraction')

        self.detector.gmm.score_samples = fail
        with self.cache() as cache:
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                score_and_cache(self.detector, 'unused', self.item, cache)
        self.detector.gmm.score_samples = lambda value: value.sum(dim=1)
        with self.cache() as cache:
            _result, hit = score_and_cache(self.detector, 'unused', self.item, cache)
            self.assertTrue(hit)
        self.assertEqual(self.calls, 1)

    def test_invalid_raw_rejected(self):
        for value in (torch.ones(3, dtype=torch.float32), torch.ones(1, 3, dtype=torch.float64),
                      torch.full((3,), float('nan'), dtype=torch.float64)):
            with self.assertRaises(ValueError):
                score_from_raw(self.detector, value)


if __name__ == '__main__':
    unittest.main()
