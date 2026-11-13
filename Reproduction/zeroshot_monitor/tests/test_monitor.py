import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collector import Collector, LivePredictions, process_alive
from publisher import can_stop_after_upload


FIELDS = ['image_id', 'relative_path', 'sha256', 'source', 'status', 'elapsed_seconds', 'z_score', 'error']


def row(index, source='COCO2017', status='ok'):
    return [str(index), 'C:/private/image-secret.jpg', 'secret-hash', source, status, '15', '-3', 'Secret path C:/private/model.pt' if status == 'error' else '']


def csv_bytes(rows, header=True):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    if header:
        writer.writerow(FIELDS)
    writer.writerows(rows)
    return stream.getvalue().encode()


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name) / 'zeroshot_test'
        self.run.mkdir()
        self.predictions = self.run / 'predictions.csv'
        (self.run / 'progress.json').write_text(json.dumps({'pid': os.getpid(), 'status': 'running', 'expected_by_source': {'COCO2017': 2, 'BigGAN': 2}}))
        (self.run / 'run_metadata.json').write_text(json.dumps({'created_utc': '2026-01-01T00:00:00Z', 'identity': {'selected_count': 4, 'data_root': 'C:/SECRET'}}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_live_unterminated_csv_is_not_counted_then_counted_once(self):
        prefix = csv_bytes([row(1)])
        second = csv_bytes([row(2)], header=False)
        self.predictions.write_bytes(prefix + second[:-2])
        reader = LivePredictions(self.predictions)
        reader.update()
        self.assertEqual(reader.count, 1)
        reader.update()
        self.assertEqual(reader.count, 1)
        with self.predictions.open('ab') as stream:
            stream.write(second[-2:])
        reader.update()
        self.assertEqual(reader.count, 2)
        self.assertEqual(reader.seconds, 30)

    def test_atomic_resume_rewrite_resets_cache(self):
        self.predictions.write_bytes(csv_bytes([row(1), row(2)]))
        reader = LivePredictions(self.predictions)
        reader.update()
        replacement = self.predictions.with_suffix('.tmp')
        replacement.write_bytes(csv_bytes([row(1), row(2), row(3, 'BigGAN')]))
        replacement.replace(self.predictions)
        reader.update()
        self.assertEqual(reader.count, 3)
        self.assertEqual(reader.ok, 3)
        self.assertEqual(reader.by_source['BigGAN'], 1)

    def test_dead_process_overrides_stale_running_checkpoint(self):
        self.predictions.write_bytes(csv_bytes([row(1)]))
        snapshot = Collector(self.run).snapshot(alive_check=lambda _: False, gpu_provider=lambda: None)
        self.assertEqual(snapshot['status'], 'stopped')
        self.assertIsNone(snapshot['eta_seconds'])

    def test_alive_process_without_new_rows_is_stalled(self):
        self.predictions.write_bytes(csv_bytes([row(1)]))
        snapshot = Collector(self.run).snapshot(now=self.predictions.stat().st_mtime + 200, alive_check=lambda _: True, gpu_provider=lambda: None)
        self.assertEqual(snapshot['status'], 'stalled')

    def test_completion_and_privacy_boundary(self):
        self.predictions.write_bytes(csv_bytes([row(1), row(2), row(3, 'BigGAN'), row(4, 'BigGAN', 'error')]))
        snapshot = Collector(self.run).snapshot(alive_check=lambda _: False, gpu_provider=lambda: None)
        self.assertEqual(snapshot['status'], 'completed_with_errors')
        self.assertEqual(snapshot['completed'], 4)
        self.assertEqual(snapshot['errors'], 1)
        self.assertEqual(snapshot['completed_pairs'], 0)
        encoded = json.dumps(snapshot)
        for secret in ('C:/', 'image-secret', 'secret-hash', 'model.pt', 'sha256', 'relative_path', 'data_root'):
            self.assertNotIn(secret, encoded)

    def test_complete_pairs_require_whole_successful_sources(self):
        self.predictions.write_bytes(csv_bytes([row(1), row(2), row(3, 'BigGAN'), row(4, 'BigGAN')]))
        snapshot = Collector(self.run).snapshot(alive_check=lambda _: True, gpu_provider=lambda: None)
        self.assertEqual(snapshot['status'], 'completed')
        self.assertEqual(snapshot['completed_pairs'], 1)
        self.assertEqual(snapshot['available_pairs'], 1)
        self.assertEqual(snapshot['remaining'], 0)

    def test_actual_process_liveness(self):
        self.assertTrue(process_alive(os.getpid()))

    def test_final_csv_waits_for_final_tables_and_process_exit(self):
        snapshot = {'status': 'completed', 'table_status': 'PARTIAL', 'process_alive': True}
        self.assertFalse(can_stop_after_upload(snapshot))
        snapshot['table_status'] = 'COMPLETE_AVAILABLE_DATA'
        self.assertFalse(can_stop_after_upload(snapshot))
        snapshot['process_alive'] = False
        self.assertTrue(can_stop_after_upload(snapshot))
        snapshot.update(status='completed_with_errors', table_status='COMPLETED_WITH_ERRORS')
        self.assertTrue(can_stop_after_upload(snapshot))


if __name__ == '__main__':
    unittest.main()
