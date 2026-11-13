import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reference_neighbors import BANK_SIZE, HEADS, HERE, N_FOLDS, SOURCE_TO_HEAD, read_input, rows


def nearest_counts(distance, mask):
    winner = distance[mask].argmin(axis=1)
    return dict(zip(HEADS, np.bincount(winner, minlength=len(HEADS)).tolist()))


def run(project, manifest, saved, out):
    ids, sources, families, variants, fold, features, _, winner, flagged = read_input(project, manifest)
    decisions = rows(saved / "decisions.csv")
    np.testing.assert_array_equal(ids, [r["image_id"] for r in decisions])
    np.testing.assert_array_equal(fold, [int(r["fold"]) for r in decisions])
    np.testing.assert_array_equal(flagged, [bool(int(r["historical_150"])) for r in decisions])
    index = {key: i for i, key in enumerate(ids)}
    gallery = rows(saved / "reference_banks.csv")
    n_heads = len(HEADS)
    assert len(gallery) == N_FOLDS * n_heads * BANK_SIZE
    norm = np.linalg.norm(features, axis=1)
    if not np.isfinite(norm).all() or np.any(norm == 0):
        raise ValueError("Cosine distance requires finite nonzero descriptor norms")
    unit = features / norm[:, None]
    dist = {key: np.full((len(ids), n_heads), np.nan) for key in
            ("test_cosine", "cal_cosine", "test_centroid", "cal_centroid")}
    radii = np.empty((N_FOLDS, n_heads))
    for gallery_fold in range(N_FOLDS):
        cal_fold = (gallery_fold + 1) % N_FOLDS
        queries = np.flatnonzero((fold == gallery_fold) | (fold == cal_fold))
        query_unit = unit[queries]
        query_raw = features[queries]
        held_families = set(families[queries])
        for source, head in SOURCE_TO_HEAD.items():
            col = HEADS.index(head)
            bank_rows = [r for r in gallery if int(r["test_fold"]) == gallery_fold and r["source"] == source]
            bank = np.array([index[r["image_id"]] for r in bank_rows])
            assert len(bank) == len(set(bank)) == BANK_SIZE
            assert np.all(sources[bank] == source)
            np.testing.assert_array_equal(families[bank], [r["family"] for r in bank_rows])
            np.testing.assert_array_equal(variants[bank], [r["variant"] for r in bank_rows])
            assert held_families.isdisjoint(families[bank])
            bank_unit = unit[bank]
            max_sim = (query_unit @ bank_unit.T).max(axis=1)
            if np.any(np.abs(max_sim) > 1 + 1.e-10):
                raise ValueError("Cosine similarity exceeded numerical tolerance")
            cosine = 1 - np.clip(max_sim, -1, 1)
            centroid = features[bank].mean(axis=0)
            euclidean = np.linalg.norm(query_raw - centroid, axis=1)
            radii[gallery_fold, col] = np.sqrt(np.square(features[bank] - centroid).sum() / (len(bank)-1))

            for i in [0, len(queries)//2, len(queries)-1]:
                direct = 0.5 * np.square(bank_unit - query_unit[i]).sum(axis=1).min()
                np.testing.assert_allclose(cosine[i], direct, atol=1.e-10, rtol=0)
            for role, selected in (("test", fold[queries] == gallery_fold), ("cal", fold[queries] == cal_fold)):
                dist[role + "_cosine"][queries[selected], col] = cosine[selected]
                dist[role + "_centroid"][queries[selected], col] = euclidean[selected]
    for values in dist.values():
        assert np.isfinite(values).all() and values.min() >= 0
    assert np.isfinite(radii).all() and radii.min() > 0
    targets = np.array([SOURCE_TO_HEAD.get(s, "") for s in sources])
    known = targets != ""
    assert known.sum() == 6000 and flagged.sum() == 150 and np.all(winner[flagged] == "GigaGAN")
    masks = {"historical_firefly": flagged, "genuine_gigagan": sources == "GigaGAN", "all_known": known}
    masks.update({src: sources == src for src in SOURCE_TO_HEAD})
    summary = {
        "heads": HEADS, "images": len(ids), "ordered_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "input_sha256": {file: hashlib.sha256((saved/file).read_bytes()).hexdigest() for file in
                         ("decisions.csv", "reference_banks.csv")},
        "radii_formula": "sqrt(sum of squared raw Euclidean deviations from each gallery mean divided by (n-1))",
        "calibration_alignment": "For row i, test distances use its fold f gallery and calibration distances use gallery (f-1) mod 5.",
        "distance_range": {k: [float(v.min()), float(v.max())] for k,v in dist.items()},
        "radius_range": [float(radii.min()), float(radii.max())],
        "test_gallery_winners": {},
    }
    for method in ("cosine", "centroid"):
        distances = dist["test_" + method]
        winning_heads = np.array(HEADS)[distances.argmin(axis=1)]
        summary["test_gallery_winners"][method] = {
            "by_group": {group: nearest_counts(distances, mask) for group, mask in masks.items()},
            "all_known_correct": int((known & (winning_heads == targets)).sum()),
        }
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "distances.npz"
    temp = out / "distances.tmp.npz"
    np.savez_compressed(temp, ids=ids, heads=np.array(HEADS), fold=fold, radii=radii, **dist)
    with np.load(temp, allow_pickle=False) as check:
        for k in dist:
            np.testing.assert_array_equal(check[k], dist[k])
    temp.replace(archive)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["test_gallery_winners"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='all gallery diagnostic')
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--saved", type=Path, default=HERE / "reference_neighbors")
    parser.add_argument("--out", type=Path, default=HERE / "all_gallery_diagnostic")
    args = parser.parse_args()
    run(args.project, args.manifest, args.saved, args.out)
