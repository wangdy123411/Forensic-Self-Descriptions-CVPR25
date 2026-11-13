import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_remote_zip_fast import run
from test_sample_remote_zip import LocalServer, config
from zeroshot_common import write_json


class FastSamplerTests(unittest.TestCase):
    def test_checkpoint_survives_transient_windows_sharing_lock(self):
        real_replace = Path.replace
        calls = []
        def transient(path, target):
            calls.append(str(path))
            if len(calls) == 1:
                raise PermissionError('Simulated reader lock')
            return real_replace(path, target)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'state.json'
            with patch.object(Path, 'replace', transient), patch('zeroshot_common.time.sleep'):
                write_json(path, {'complete': True})
            self.assertIn('true', path.read_text())
            self.assertEqual(len(calls), 2)

    def test_parallel_stored_and_deflated_members_resume(self):
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            buffer = io.BytesIO()
            originals = {}
            with zipfile.ZipFile(buffer, 'w', compression=compression) as archive:
                for source in ('real', 'fake'):
                    for number in range(4):
                        name = f'dataset/{source}/{number}.png'
                        data = name.encode() * 500
                        originals[name] = data
                        archive.writestr(name, data)
            payload = buffer.getvalue()
            with LocalServer(payload) as server, tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cfg = config(server.url, len(payload))
                state = run(cfg, root, count=3, workers=2)
                self.assertEqual(state['status'], 'complete')
                self.assertEqual(len(state['images']), 6)
                for row in state['images']:
                    self.assertEqual((root / row['relative_path']).read_bytes(), originals[row['member']])
                repeated = run(cfg, root, count=3, workers=2, resume=True)
                self.assertEqual(state['images'], repeated['images'])
                self.assertEqual(repeated['overview']['pending_images'], 0)
                (root / repeated['images'][0]['relative_path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    run(cfg, root, count=3, workers=2, resume=True)


if __name__ == '__main__':
    unittest.main()
