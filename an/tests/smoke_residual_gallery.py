"""Render a clearly labeled numerical-fixture gallery for visual QA only."""

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from an.robustness.residual_gallery import build_residual_galleries, DEFAULT_CONDITIONS


def main():
    output = ROOT / "an/results/gallery_render_qa"
    for name in ["examples", "features", "previews"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(487)
    image = Image.fromarray(rng.integers(0, 256, (80, 120, 3), dtype=np.uint8))
    reference = rng.normal(size=960)
    residual = rng.normal(size=(8, 128, 128))
    records = []
    for index, condition in enumerate(DEFAULT_CONDITIONS):
        stem = f"qa_fixture__{condition}"
        np.savez_compressed(
            output / "features" / f"{stem}.npz", raw=reference + index * 0.05 * rng.normal(size=960)
        )
        np.savez_compressed(
            output / "previews" / f"{stem}.npz", residual_patch=residual / (index + 1)
        )
        image.save(output / "examples" / f"{stem}.png")
        records.append(
            {
                "image_id": "qa_fixture",
                "source": "NUMERICAL_FIXTURE_QA_ONLY",
                "label": 0,
                "split": "test",
                "condition": condition,
                "z_score": -float(index),
                "processed_size": [120, 80],
            }
        )
    (output / "records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    paths = build_residual_galleries(output)
    assert len(paths) == 1
    with Image.open(output / paths[0]) as rendered:
        rendered.verify()
    print(output / paths[0])


if __name__ == "__main__":
    main()
