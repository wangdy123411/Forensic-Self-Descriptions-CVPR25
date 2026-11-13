import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from compute import load_fsd


def split_savings(x, sources, a, b, mask):
    xa = x[mask & (sources == a)]
    xb = x[mask & (sources == b)]
    n, m = len(xa), len(xb)
    return {
        "counts": {a: n, b: m},
        "sse_saved": float(n*m/(n+m) * np.sum((xa.mean(0) - xb.mean(0))**2)),
    }


def audit(results):
    x, assignments, archive, digest = load_fsd(results)
    sources = np.array([row["source"] for row in assignments])
    names = sorted(archive["sources"])
    means = {s: x[sources == s].mean(0) for s in names}
    distances = sorted(
        ({"sources": [a, b], "distance": float(np.linalg.norm(means[a] - means[b]))}
         for a, b in combinations(names, 2)), key=lambda p: p["distance"])
    fits = {}
    for m in archive["metrics"]:
        k = m["k"]
        labels = np.array([int(row[f"cluster_k{k}"]) for row in assignments])
        np.testing.assert_array_equal(np.unique(labels), np.arange(k))
        centres = np.array([x[labels == j].mean(0) for j in range(k)])
        error = np.sum((x-centres[labels])**2, axis=1)
        inertia = float(error.sum())
        np.testing.assert_allclose(inertia, m["inertia"], rtol=1e-11, atol=1e-6)
        fits[k] = {"inertia": inertia, "sse_by_source": {
            s: float(error[sources == s].sum()) for s in names}}
        if k == 6:
            firefly_clusters = np.unique(labels[sources == "FireFly"])
            np.testing.assert_equal(len(firefly_clusters), 1)
            mixed = labels == firefly_clusters[0]
            np.testing.assert_array_equal(np.unique(sources[mixed]), ["FireFly", "ImageNet-1k"])
    dual = split_savings(x, sources, "ProGAN", "GLIDE", np.isin(sources, ["ProGAN", "GLIDE"]))
    mixed_dual = split_savings(x, sources, "ImageNet-1k", "FireFly", mixed)
    reduction = {s: fits[6]["sse_by_source"][s] - fits[24]["sse_by_source"][s] for s in names}
    total = fits[6]["inertia"] - fits[24]["inertia"]
    np.testing.assert_allclose(sum(reduction.values()), total, rtol=1e-11)
    return {
        "input_matrix_bytes_sha256": digest, "shape": list(x.shape),
        "whole_source_centroid_distances": distances,
        "fits": fits, "k6_to_k24_inertia_decrease": total,
        "k6_to_k24_sse_decrease_by_source": reduction,
        "fraction_of_decrease_by_source": {s: v/total for s, v in reduction.items()},
        "fraction_of_decrease_from_progan_and_glide": (reduction["ProGAN"]+reduction["GLIDE"])/total,
        "label_guided_two_centroid_comparison": {
            "definition": "For each specified pair of subsets, compare its own pooled mean to one mean per true label.",
            "all_progan_and_glide": dual, "imagenet_and_firefly_in_mixed_k6_cluster": mixed_dual,
            "ratio_of_savings_imagenet_firefly_to_progan_glide": mixed_dual["sse_saved"]/dual["sse_saved"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='audit k refinement')
    parser.add_argument("--results", type=Path, required=True, help="Project2/Results/reproduction")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent/"output"/"k_refinement_audit.json")
    args = parser.parse_args()
    report = audit(args.results)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
