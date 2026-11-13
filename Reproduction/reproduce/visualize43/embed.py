import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

SOURCES = ["ImageNet-1k", "ProGAN", "StyleGAN3", "GLIDE", "SD1.5", "DALLE 3", "MJ v6", "FireFly"]
SEED = 2026


def main():
    parser = argparse.ArgumentParser(description='embed')
    parser.add_argument("--integrated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = np.load(args.integrated / "raw_fsd_960.npy", mmap_mode="r", allow_pickle=False)
    with (args.integrated / "feature_index.csv").open(encoding="utf-8-sig", newline="") as stream:
        index = list(csv.DictReader(stream))
    if matrix.shape != (len(index), 960) or len({r["image_id"] for r in index}) != len(index):
        raise ValueError("Feature matrix/index mismatch")
    selected = []
    for i, row in enumerate(index):
        if row["source"] in SOURCES:
            if int(row["row"]) != i or row["feature_sha256"] != hashlib.sha256(matrix[i].tobytes()).hexdigest():
                raise ValueError("Feature identity mismatch: "+row["image_id"])
            selected.append(row)
    data = matrix[[int(r["row"]) for r in selected]]
    with threadpool_limits(limits=4):
        pca = PCA(n_components=50, whiten=False, svd_solver="randomized", random_state=SEED)
        reduced = pca.fit_transform(data)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca",
                    method="barnes_hut", random_state=SEED, max_iter=1000, n_jobs=4)
        coordinates = tsne.fit_transform(reduced)
    if not np.isfinite(coordinates).all():
        raise ValueError("Non-finite t-SNE coordinates")
    args.output.mkdir(exist_ok=True, parents=True)
    with (args.output / "tsne_coordinates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "source", "tsne_1", "tsne_2"])
        for row, (x,y) in zip(selected, coordinates):
            writer.writerow([row["image_id"], row["source"], float(x), float(y)])
    metadata = {
        "feature_file": str(args.integrated / "raw_fsd_960.npy"), "raw_dimensions": 960,
        "sample_count": len(selected), "sampling": "all available images from the eight paper sources",
        "counts": {s: sum(r["source"] == s for r in selected) for s in SOURCES},
        "seed": SEED, "pca_visualization_only": {"n_components": 50, "whiten": False,
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()), "solver": "randomized"},
        "t_sne": {"perplexity": 30, "init": "pca", "barnes_hut_angle": 0.5, "learning_rate": float(tsne.learning_rate_),
                  "max_iter": 1000, "iterations_reported": int(tsne.n_iter_), "kl_divergence": float(tsne.kl_divergence_)},
        "sklearn_version": sklearn.__version__
    }
    (args.output / "embedding.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
