import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
from scipy.cluster.vq import kmeans2, vq
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_fsd(root):
    integrated = root / "section41_final_20261110" / "integrated"
    run = root / "section43_ossia_progan_20261110" / "4.3"
    records = read_csv(run / "input_index.csv")
    assignments = read_csv(run / "assignments.csv")
    if [(r["image_id"], r["source"]) for r in records] != [(r["image_id"], r["source"]) for r in assignments]:
        raise ValueError("The input index and archived assignments differ")
    lookup = {r["image_id"]: int(r["row"]) for r in read_csv(integrated / "feature_index.csv")}
    replacement_lookup = {r["image_id"]: i for i, r in enumerate(read_csv(run / "replacement_manifest.csv"))}
    raw_fsd = np.load(integrated / "raw_fsd_960.npy", mmap_mode="r")
    replacement = np.load(run / "replacement_features.npy", mmap_mode="r")
    x = np.empty((len(records), 960), dtype="<f8")
    for i, r in enumerate(records):
        image_id = r["image_id"]
        x[i] = replacement[replacement_lookup[image_id]] if r["source"] == "ProGAN" else raw_fsd[lookup[image_id]]
    digest = hashlib.sha256(memoryview(x).cast("B")).hexdigest()
    if digest != read_json(run / "run.json")["input_matrix_bytes_sha256"]:
        raise ValueError("Input feature digest differs from the archived 960-D clustering run")
    return x, assignments, read_json(run / "metrics.json"), digest


def pca_three(x):
    x -= x.mean(0)
    covariance = (x.T @ x) / (len(x) - 1)
    values, vectors = eigh(covariance, subset_by_index=(covariance.shape[0]-3, covariance.shape[0]-1))
    components = vectors[:, ::-1]
    for vector in components.T:
        if vector[np.abs(vector).argmax()] < 0:
            vector *= -1
    points = x @ components
    return points, values[::-1] / np.trace(covariance)


def fit_clusters(points, k, seed, n_init):
    best_labels = None
    best_inertia = np.inf
    generator = np.random.RandomState(seed)
    for _ in range(n_init):
        centroids, _ = kmeans2(points, k, iter=300, minit="++", missing="raise", seed=generator)
        labels, distances = vq(points, centroids)
        inertia = float(distances @ distances)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels, best_inertia


def contingency(a, b, shape):
    counts = np.zeros(shape, dtype=int)
    np.add.at(counts, (a, b), 1)
    return counts


def normalized_mutual_information(counts):
    joint = counts / counts.sum()
    marginal_a = joint.sum(1)
    marginal_b = joint.sum(0)
    ent_a = -xlogy(marginal_a, marginal_a).sum()
    ent_b = -xlogy(marginal_b, marginal_b).sum()
    return float(2 * (ent_a + ent_b + xlogy(joint,joint).sum()) / (ent_a + ent_b))


def main(root, output):
    x, records, archived, digest = load_fsd(root)
    points, variance = pca_three(x)
    k = 6
    labels = np.array([int(r["cluster_k6"]) for r in records])
    raw_clusters, inertia = fit_clusters(points, k, archived["seed"], archived["n_init"])
    rows, cols = linear_sum_assignment(-contingency(raw_clusters, labels, (k, k)))
    cluster_names = np.empty(k, dtype=int)
    cluster_names[rows] = cols
    projected_clusters = cluster_names[raw_clusters]
    names = archived["sources"]
    truth = np.array([names.index(r["source"]) for r in records])
    original = contingency(truth, labels, (len(names), k))
    projected = contingency(truth, projected_clusters, (len(names), k))
    nmi_original = normalized_mutual_information(original)
    archived_nmi = next(r["nmi"] for r in archived["metrics"] if r["k"] == k)
    np.testing.assert_allclose(nmi_original, archived_nmi, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.var(points,axis=0,ddof=1) / np.sum(np.var(x,axis=0,ddof=1)), variance, rtol=1e-10)
    if not np.isfinite(points).all() or len(np.unique(projected_clusters)) != k:
        raise ValueError("Invalid coordinates or an empty 3-D cluster")
    metadata = dict(n_images=len(records), k=k, explained_variance_ratio=variance.tolist(),
                    nmi_960d=nmi_original, nmi_3d=normalized_mutual_information(projected),
                    counts_960d=dict(zip(names,original.tolist())), counts_3d=dict(zip(names,projected.tolist())),
                    fit_agreement_after_label_alignment=float(np.mean(projected_clusters==labels)),
                    inertia_3d=inertia, coordinates="unwhitened PCA, centered raw FSD without rescaling",
                    kmeans_3d="Lloyd, Euclidean, k-means++ initialization, minimum inertia over ten starts",
                    seed=archived["seed"], n_init=archived["n_init"], cluster_names="C0-C5; 3D IDs aligned for display by maximum overlap with 960D assignments",
                    feature_digest=digest, corpus="section43_ossia_progan_20261110", missing_sources=archived["missing_sources"])
    output.mkdir(exist_ok=True, parents=True)
    with (output / "coordinates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id","source","pc1","pc2","pc3","cluster_960d","cluster_3d"))
        for r, xyz, before, after in zip(records, points, labels, projected_clusters):
            writer.writerow((r["image_id"],r["source"], *xyz, before, after))
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute')
    parser.add_argument("--results", type=Path, required=True, help="Project2/Results/reproduction directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"output")
    options = parser.parse_args()
    main(options.results, options.output)
