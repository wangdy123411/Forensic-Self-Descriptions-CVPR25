"""Guard historical-run comparisons against unverified or executable source changes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from an.baseline_comparison.compare_upstream import verify_main_source


class SourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "robustness" / "extractor.py"
        self.path.parent.mkdir()
        self.original = b'def score(x):\n    """Original description."""\n    return x * 2\n'
        self.original_hash = hashlib.sha256(self.original).hexdigest()
        self.path.write_bytes(self.original)
        self.archive = self.root / "original_source.zip"
        self.write_archive(self.original)

    def write_archive(self, payload, member="robustness/extractor.py"):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(member, payload)

    def verify(self, *, with_archive=True):
        return verify_main_source(
            self.path,
            self.original_hash,
            self.archive if with_archive else None,
            source_root=self.root,
        )

    def test_exact_bytes_need_no_archive(self):
        self.archive.unlink()
        evidence = self.verify(with_archive=False)
        self.assertEqual(evidence["verification"], "exact_bytes")
        self.assertEqual(evidence["current_sha256"], self.original_hash)

    def test_formatting_change_still_requires_explicit_archive(self):
        self.path.write_bytes(self.original + b"\n# Editorial comment.\n")
        with self.assertRaisesRegex(ValueError, "implementation hash changed"):
            self.verify(with_archive=False)

    def test_documentation_and_formatting_changes_have_recorded_evidence(self):
        self.path.write_bytes(
            b'def score( x ):\n    """Clearer description."""\n    # Preserve the formula.\n    return x*2\n'
        )
        evidence = self.verify()
        self.assertEqual(evidence["verification"], "executable_ast")
        self.assertTrue(evidence["executable_ast_equal"])
        self.assertEqual(evidence["archived_sha256"], self.original_hash)
        self.assertNotEqual(evidence["current_sha256"], self.original_hash)
        self.assertEqual(
            evidence["archive_sha256"], hashlib.sha256(self.archive.read_bytes()).hexdigest()
        )
        self.assertEqual(evidence["archive_member"], "robustness/extractor.py")

    def test_executable_changes_are_rejected(self):
        variants = [
            self.original.replace(b"x * 2", b"x * 3"),
            self.original.replace(b"return x * 2", b"return abs(x) * 2"),
            b"import math\n" + self.original,
            self.original.replace(
                b"return x * 2", b"if x < 0:\n        return 0\n    return x * 2"
            ),
        ]
        for source in variants:
            with self.subTest(source=source):
                self.path.write_bytes(source)
                with self.assertRaisesRegex(ValueError, "Executable source differs"):
                    self.verify()

    def test_archive_must_match_frozen_hash_even_when_current_matches_it(self):
        self.write_archive(self.original + b"\n# Not the recorded bytes.\n")
        with self.assertRaisesRegex(ValueError, "does not match the recorded run hash"):
            self.verify()

    def test_archive_uses_exact_relative_path(self):
        self.write_archive(self.original, member="extractor.py")
        with self.assertRaisesRegex(ValueError, "exactly one file"):
            self.verify()

    def test_duplicate_archive_members_are_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr("robustness/extractor.py", self.original)
        with self.assertRaisesRegex(ValueError, "exactly one file"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
