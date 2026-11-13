"""Audit completed experiment coverage, numeric records, artifacts, and links."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a separate audit; defaults to verification.json inside the run directory",
    )
    parser.add_argument(
        "--wait-for-pipeline",
        type=Path,
        help="Wait for this local pipeline status to be complete before verifying",
    )
    parser.add_argument("--timeout-hours", type=float, default=3)
    args = parser.parse_args()
    if args.wait_for_pipeline:
        deadline = time.monotonic() + args.timeout_hours * 3600
        print("Waiting for the local scoring/analysis pipeline to finish.", flush=True)
        while True:
            if args.wait_for_pipeline.is_file():
                try:
                    state = json.loads(args.wait_for_pipeline.read_text())
                except json.JSONDecodeError:
                    state = {}
                if state.get("stage") == "complete":
                    break
                if state.get("stage") == "error":
                    raise RuntimeError(f"Pipeline reported an error: {state.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the scoring/analysis pipeline")
            time.sleep(15)
    run = args.run.resolve()
    root = Path(__file__).resolve().parent.parent
    started = time.perf_counter()
    meta = json.loads((run / "run_metadata.json").read_text())
    summary = json.loads((run / "analysis_summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (run / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = {
        (key, condition)
        for key, values in meta["expected_image_conditions"].items()
        for condition in values
    }
    observed = [(row["image_id"], row["condition"]) for row in rows]
    assert len(observed) == len(set(observed)), "Duplicate image-condition records"
    assert set(observed) == expected, "Missing/unexpected scored pairs"
    assert summary["status"] == "complete_for_expected_conditions", summary["status"]
    originals = meta["selected_manifest"]
    assert len({r["sha256"] for r in originals}) == len(
        originals
    ), "Duplicate original bytes across split/source"
    for row in originals:
        assert digest(root / row["path"]) == row["sha256"], f"Changed original {row['image_id']}"
    env = meta["environment"]
    assert (
        env["device"].split(":", 1)[0] == "cuda" and "2060" in env["gpu"]
    ), "Unexpected computation device"
    assert env["dtype"] == "float64" and env["tf32"] is False
    config = env["model_config"]
    feature_errors, z_errors, constraints = [], [], []
    for row in rows:
        derived_z = (row["raw_score"] - config["scoring"]["train_mean"]) / config["scoring"][
            "train_std"
        ]
        z_errors.append(abs(derived_z - row["z_score"]))
        assert abs(row["anomaly_score"] + row["z_score"]) < 1e-10
        diagnostics = row["diagnostics"]
        assert diagnostics["dtype"] == "float64" and diagnostics["feature_dim"] == 960
        assert diagnostics["max_size"] == 1024 and diagnostics["num_scales"] == 3
        constraints.append(diagnostics["constraint_max_abs_error"])
        with np.load(run / row["feature_path"]) as data:
            for name in ("raw", "projected", "attribution"):
                feature = data[name]
                assert feature.shape == (960,) and feature.dtype == np.float64
                assert np.isfinite(feature).all()
            if row["condition"] == "clean":
                assert row["fsd_l2_relative"] == 0 and row["projected_l2_relative"] == 0
                assert abs(row["residual_rms_ratio"] - 1) < 1e-12
            # Audits all descriptor constraints using cached vectors as well.
            feature_errors.append(float(np.abs(data["raw"].reshape(8, 120).sum(1) - 1).max()))
    assert max(z_errors) < 1e-10, "z-score normalization mismatch"
    assert max(feature_errors) < 1e-4, "Descriptor channel constraints violated"
    assert max(constraints) < 1e-4, "Recorded channel constraints violated"
    inventory = run / "artifact_hashes.json"
    if inventory.exists():
        for item in json.loads(inventory.read_text()):
            assert (
                digest(run / item["path"]) == item["sha256"]
            ), f"Changed merged artifact {item['path']}"
    figures = list((run / "plots").glob("*.png"))
    for path in figures:
        with Image.open(path) as image:
            assert min(image.size) >= 100
            image.verify()
    report = (run / "report.md").read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", report)
    for link in links:
        if not re.match(r"[a-z]+://|#", link):
            assert (run / link.split("#")[0]).exists(), f"Broken report link: {link}"
    result = dict(
        passed=True,
        scored_pairs=len(rows),
        originals=len(originals),
        test_originals=sum(r["split"] == "test" for r in originals),
        calibration_originals=sum(r["split"] == "calibration" for r in originals),
        condition_counts=dict(Counter(r["condition"] for r in rows if r["split"] == "test")),
        source_counts=dict(Counter(r["source"] for r in originals if r["split"] == "test")),
        max_z_normalization_error=max(z_errors),
        max_feature_constraint_error=max(feature_errors),
        verified_figures=len(figures),
        report_links_checked=len(links),
        seconds=time.perf_counter() - started,
    )
    output = args.output.resolve() if args.output else run / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
