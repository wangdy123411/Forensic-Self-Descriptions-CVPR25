"""Merge safety checks with fabricated records; these are not benchmark results."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

AN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AN))
from merge_runs import digest_file, merge_runs

PROTOCOL = {
    "conditions": [{"name": "clean"}, {"name": "jpeg_q50"}],
    "fsd_config": {"max_size": 1024, "num_scales": 3},
    "weights": {"fre.pt": "a" * 64},
    "method": "fft",
    "seed": 20261108,
    "code_sha256": {"run_experiment.py": "b" * 64},
    "attribution": True,
    "calibration_all_conditions": False,
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class MergeRunsTests(unittest.TestCase):
    def setUp(self):
        cache = AN / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="merge-tests-", dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        # Cleanup only ever receives a directory created within this workspace.
        self.assertTrue(self.root.is_relative_to(AN))
        self.real = self.make_phase("real", [("cal", 0, "calibration"), ("real", 0, "test")])
        self.fake = self.make_phase("fake", [("fake", 1, "test")], missing={("fake", "jpeg_q50")})
        self.output = self.root / "merged"

    def make_phase(self, name, specs, missing=()):
        phase = self.root / name
        phase.mkdir()
        conditions = [item["name"] for item in PROTOCOL["conditions"]]
        manifest, expected, records = [], {}, []
        for identifier, label, split in specs:
            original = {
                "image_id": identifier,
                "label": label,
                "split": split,
                "source": "DALL-E 3" if label else "COCO2017",
                "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
                "path": f"fixture/{identifier}.png",
            }
            manifest.append(original)
            expected[identifier] = ["clean"] if split == "calibration" else conditions
            for condition in expected[identifier]:
                if (identifier, condition) in missing:
                    continue
                stem = f"{identifier}__{condition}"
                # Opaque byte fixtures suffice: merge copies/hashes, while the
                # separate numerical verifier validates real NPZ contents.
                for directory, suffix in (
                    ("features", ".npz"),
                    ("previews", ".npz"),
                    ("examples", ".png"),
                ):
                    destination = phase / directory / (stem + suffix)
                    destination.parent.mkdir(exist_ok=True)
                    destination.write_bytes((directory + stem).encode())
                records.append(
                    {
                        **original,
                        "condition": condition,
                        "z_score": -float(label),
                        "anomaly_score": float(label),
                        "raw_score": -float(label),
                        "attribution_loglik": {"Real": 0.0, "DALL-E 3": 1.0},
                        "feature_path": f"features/{stem}.npz",
                    }
                )
        signature = {
            **deepcopy(PROTOCOL),
            "manifest_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "selected_image_ids": [item["image_id"] for item in manifest],
        }
        write_json(
            phase / "run_metadata.json",
            {
                "signature": signature,
                "selected_manifest": manifest,
                "expected_conditions": conditions,
                "expected_image_conditions": expected,
                "environment": {"source_gmm_names": ["Real", "DALL-E 3"], "dtype": "float64"},
            },
        )
        self.write_records(phase, records)
        write_json(phase / "progress.json", {"state": "complete" if not missing else "stopped"})
        return phase

    @staticmethod
    def write_records(phase, records):
        (phase / "records.jsonl").write_text(
            "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
        )

    def test_incomplete_plan_and_provenance_survive_merging(self):
        original_hashes = {
            phase.name: digest_file(phase / "records.jsonl") for phase in (self.real, self.fake)
        }
        result = merge_runs([self.real, self.fake], self.output)
        self.assertEqual((result["expected"], result["completed"], result["missing"]), (5, 4, 1))
        self.assertEqual(result["state"], "incomplete_merged_phases")
        metadata = read_json(self.output / "run_metadata.json")
        self.assertEqual(
            metadata["expected_image_conditions"],
            {
                "cal": ["clean"],
                "real": ["clean", "jpeg_q50"],
                "fake": ["clean", "jpeg_q50"],
            },
        )
        self.assertTrue(metadata["merged_run"])
        self.assertFalse(metadata["scoring_resume_supported"])
        self.assertTrue(metadata["signature"]["merged_analysis_only"])
        self.assertEqual(metadata["source_mapping"], {"COCO2017": "Real", "DALL-E 3": "DALL-E 3"})
        self.assertEqual([item["missing_pairs"] for item in metadata["phase_provenance"]], [0, 1])
        self.assertEqual(len(metadata["selected_manifest"]), 3)
        for phase, provenance in zip((self.real, self.fake), metadata["phase_provenance"]):
            self.assertEqual(provenance["hashes"]["records.jsonl"], original_hashes[phase.name])
            self.assertEqual(digest_file(phase / "records.jsonl"), original_hashes[phase.name])
            for filename, digest in provenance["hashes"].items():
                self.assertEqual(
                    digest_file(self.output / provenance["snapshot_directory"] / filename), digest
                )
        inventory = read_json(self.output / "artifact_hashes.json")
        self.assertEqual(len(inventory), 12)
        for item in inventory:
            self.assertEqual(digest_file(self.output / item["path"]), item["sha256"])
        self.assertEqual(
            digest_file(self.output / "records.jsonl"), metadata["merged_records_sha256"]
        )
        with self.assertRaisesRegex(ValueError, "already a merged run"):
            merge_runs([self.output, self.real], self.root / "remerged")

    def test_protocol_mismatches_are_rejected_before_output_creation(self):
        path = self.fake / "run_metadata.json"
        original = read_json(path)
        changes = {
            "fsd_config": {"max_size": 512, "num_scales": 3},
            "weights": {"fre.pt": "c" * 64},
            "code_sha256": {"run_experiment.py": "d" * 64},
            "seed": 17,
            "method": "tiled",
            "attribution": False,
        }
        for key, value in changes.items():
            with self.subTest(protocol_key=key):
                metadata = deepcopy(original)
                metadata["signature"][key] = value
                write_json(path, metadata)
                with self.assertRaisesRegex(ValueError, "Incompatible scoring protocol"):
                    merge_runs([self.real, self.fake], self.output)
                self.assertFalse(self.output.exists())
        write_json(path, original)

    def test_duplicate_original_hash_across_calibration_and_test_is_rejected(self):
        real_metadata = read_json(self.real / "run_metadata.json")
        calibration = real_metadata["selected_manifest"][0]
        self.assertEqual(calibration["split"], "calibration")
        fake_metadata = read_json(self.fake / "run_metadata.json")
        self.assertEqual(fake_metadata["selected_manifest"][0]["split"], "test")
        fake_metadata["selected_manifest"][0]["sha256"] = calibration["sha256"]
        write_json(self.fake / "run_metadata.json", fake_metadata)
        records = [
            json.loads(line) for line in (self.fake / "records.jsonl").read_text().splitlines()
        ]
        for record in records:
            record["sha256"] = calibration["sha256"]
        self.write_records(self.fake, records)
        with self.assertRaisesRegex(ValueError, "Byte-identical original"):
            merge_runs([self.real, self.fake], self.output)
        self.assertFalse(self.output.exists())

    def test_running_phase_is_rejected_even_when_existing_records_are_valid(self):
        write_json(self.real / "progress.json", {"state": "running", "completed": 3})
        with self.assertRaisesRegex(ValueError, "still reports running"):
            merge_runs([self.real, self.fake], self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("merged.merge-staging-*")), [])

    def test_numerical_runtime_mismatch_is_rejected(self):
        path = self.fake / "run_metadata.json"
        metadata = read_json(path)
        metadata["environment"]["dtype"] = "float32"
        write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "Incompatible numerical runtime"):
            merge_runs([self.real, self.fake], self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
