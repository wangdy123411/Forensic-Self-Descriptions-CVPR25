import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import restore_hydrated_coco as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / 'data'
        self.run = self.root / 'run'
        self.stage = self.root / 'stage'
        for folder in (self.data / 'manifests', self.data / 'real/COCO2017', self.run, self.stage):
            folder.mkdir(parents=True, exist_ok=True)
        (self.run / 'progress.json').write_text(json.dumps({'pid': 12345}), encoding='utf-8')
        rows, provenance = [], []
        for index, name in enumerate(('aa.jpg', 'bb.jpg')):
            content = ('verified-original-' + name).encode()
            row = {'image_id': str(index), 'relative_path': 'real/COCO2017/' + name,
                   'label': '0', 'source': 'COCO2017', 'split': 'test',
                   'sha256': hashlib.sha256(content).hexdigest()}
            rows.append(row)
            provenance.append({**row, 'original_relative_path': f'COCO2017/images/val2017/{index:012}.jpg', 'file_bytes': len(content)})
            (self.stage / name).write_bytes(content)
            (self.data / row['relative_path']).write_bytes(b'old-placeholder')
        for filename, content in (('eval_unique.csv', rows), ('provenance.csv', provenance)):
            with (self.data / 'manifests' / filename).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(content[0]))
                writer.writeheader()
                writer.writerows(content)
        plan = {'manifest_sha256': recovery.digest(self.data / 'manifests/eval_unique.csv'),
                'rows': recovery.checked_rows(self.data)}
        (self.stage / 'stage_plan.json').write_text(json.dumps(plan), encoding='utf-8')
        self.args = SimpleNamespace(data_root=self.data, run_dir=self.run, stage_dir=self.stage,
                                    confirm_inference_stopped=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_refuses_restoration_when_recorded_process_is_alive(self):
        with patch.object(recovery, 'process_alive', return_value=True):
            with self.assertRaises(RuntimeError):
                recovery.restore(self.args)
        self.assertEqual((self.data / 'real/COCO2017/aa.jpg').read_bytes(), b'old-placeholder')

    def test_all_staged_hashes_are_verified_before_first_destination_changes(self):
        (self.stage / 'bb.jpg').write_bytes(b'corrupt')
        with patch.object(recovery, 'process_alive', return_value=False):
            with self.assertRaises(ValueError):
                recovery.restore(self.args)
        self.assertEqual((self.data / 'real/COCO2017/aa.jpg').read_bytes(), b'old-placeholder')

    def test_stopped_verified_recovery_preserves_manifest_bytes(self):
        manifest = self.data / 'manifests/eval_unique.csv'
        before = manifest.read_bytes()
        with patch.object(recovery, 'process_alive', return_value=False):
            self.assertEqual(recovery.restore(self.args), 0)
        for name in ('aa.jpg', 'bb.jpg'):
            self.assertEqual((self.data / 'real/COCO2017' / name).read_bytes(), (self.stage / name).read_bytes())
        self.assertEqual(before, manifest.read_bytes())


if __name__ == '__main__':
    unittest.main()
