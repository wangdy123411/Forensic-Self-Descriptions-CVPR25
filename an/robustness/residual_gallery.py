"""Shared-scale residual/descriptor figures from already-scored AN originals.

No new inference is performed. Native residual patches remain native pixels;
they are not upscaled model previews. These panels show amplitude/texture, not
pixelwise residual differences across geometric transforms.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image

DEFAULT_CONDITIONS = (
    "clean",
    "jpeg_q50",
    "resize_025",
    "crop_area025",
    "blur_sigma20",
    "sharpen_200",
    "resize050_jpeg50",
)


def build_residual_galleries(run_dir: str | Path, max_sources: int = 4) -> list[str]:
    """Return paths relative to run_dir for galleries actually written.

    One saved example per source is selected in prediction-record order. Panels
    use identical limits within each original, and colorbars state clipping.
    Missing previews/features are omitted rather than reconstructed or invented.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    records_path = run_dir / "records.jsonl"
    if not records_path.is_file():
        return []
    rows = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("split") == "test":
            rows.append(row)
    indexed = {(row["image_id"], row["condition"]): row for row in rows}
    candidates = {}
    for row in rows:
        if row["condition"] != "clean" or row["source"] in candidates:
            continue
        identifier = row["image_id"]
        if (run_dir / "previews" / f"{identifier}__clean.npz").is_file():
            candidates[row["source"]] = row
    selected = sorted(candidates.values(), key=lambda row: (int(row["label"]), row["source"]))[
        :max_sources
    ]
    output = run_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for original in selected:
        identifier = original["image_id"]
        panels = []
        clean_path = run_dir / "features" / f"{identifier}__clean.npz"
        if not clean_path.is_file():
            continue
        with np.load(clean_path) as loaded:
            reference = np.asarray(loaded["raw"], dtype=float)
        for condition in DEFAULT_CONDITIONS:
            row = indexed.get((identifier, condition))
            stem = f"{identifier}__{condition}"
            preview_path = run_dir / "previews" / f"{stem}.npz"
            feature_path = run_dir / "features" / f"{stem}.npz"
            image_path = run_dir / "examples" / f"{stem}.png"
            if row is None or not all(
                path.is_file() for path in (preview_path, feature_path, image_path)
            ):
                continue
            with np.load(preview_path) as loaded:
                if "residual_patch" not in loaded:
                    continue
                residual = np.asarray(loaded["residual_patch"][0], dtype=float)
            with np.load(feature_path) as loaded:
                descriptor = np.asarray(loaded["raw"], dtype=float)
            if descriptor.size != 960 or reference.size != 960:
                continue
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            panels.append(
                (condition, row, image, residual, (descriptor - reference).reshape(8, 120))
            )
        if len(panels) < 2:
            continue
        residual_limit = max(
            float(np.quantile(np.abs(np.concatenate([p[3].ravel() for p in panels])), 0.99)), 1e-12
        )
        delta_limit = max(
            float(np.quantile(np.abs(np.concatenate([p[4].ravel() for p in panels[1:]])), 0.99)),
            1e-12,
        )
        fig, axes = plt.subplots(
            3,
            len(panels),
            figsize=(3 * len(panels), 8.3),
            squeeze=False,
            gridspec_kw={"height_ratios": [1.1, 1, 0.75]},
            layout="constrained",
        )
        for column, (condition, row, image, residual, delta) in enumerate(panels):
            axes[0, column].imshow(image)
            processed = row.get("processed_size", image.size)
            axes[0, column].set_title(
                f"{condition}\nz={row['z_score']:.2f} | {processed[0]} x {processed[1]}", fontsize=9
            )
            axes[0, column].set_xticks([])
            axes[0, column].set_yticks([])
            residual_artist = axes[1, column].imshow(
                residual,
                cmap="RdBu_r",
                vmin=-residual_limit,
                vmax=residual_limit,
                interpolation="nearest",
            )
            axes[1, column].set_title(
                f"{residual.shape[1]} x {residual.shape[0]} native pixels", fontsize=8
            )
            axes[1, column].set_xticks([])
            axes[1, column].set_yticks([])
            delta_artist = axes[2, column].imshow(
                delta,
                cmap="PuOr_r",
                vmin=-delta_limit,
                vmax=delta_limit,
                aspect="auto",
                interpolation="nearest",
            )
            axes[2, column].set_xticks([0, 59, 119], [1, 60, 120], fontsize=7)
            axes[2, column].set_yticks([0, 3, 7], [1, 4, 8], fontsize=7)
            axes[2, column].set_xlabel("Neighbor coefficient", fontsize=8)
        axes[0, 0].set_ylabel("Processed image", fontsize=10)
        axes[1, 0].set_ylabel("Native residual\nfirst FRE channel", fontsize=10)
        axes[2, 0].set_ylabel("FSD change\nresidual channel", fontsize=10)
        fig.colorbar(
            residual_artist,
            ax=axes[1].tolist(),
            fraction=0.015,
            pad=0.01,
            label="Residual amplitude",
        )
        fig.colorbar(
            delta_artist,
            ax=axes[2].tolist(),
            fraction=0.015,
            pad=0.01,
            label="FSD coefficient change",
        )
        kind = "real" if int(original["label"]) == 0 else "synthetic"
        wrap_width = max(52, len(panels) * 25)
        title = textwrap.fill(
            f"{original['source']} ({kind}) | original {identifier}", width=wrap_width
        )
        subtitle = textwrap.fill(
            "Native patches are not geometrically aligned across crop/resize; residual and FSD rows use shared symmetric limits (99% absolute range).",
            width=wrap_width,
        )
        fig.suptitle(title + "\n" + subtitle, fontsize=12)
        filename = "residual_gallery_" + re.sub(r"[^A-Za-z0-9_-]", "_", original["source"]) + ".png"
        fig.savefig(output / filename, dpi=140, facecolor="white")
        plt.close(fig)
        written.append(f"plots/{filename}")
    return written
