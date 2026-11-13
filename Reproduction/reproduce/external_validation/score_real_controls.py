import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_external import (HEADS, LOCK, SOURCES, assignments, gmm_likelihoods,
                               mean5, read_csv, sha, write_csv)


def prepare(project, output):
    base = project / "Results/reproduction"
    integrated = base / "section41_final_20261110/integrated"
    previous = base / "section42_reference_neighbors_20261112"
    for name, expected in LOCK.items():
        assert sha(previous / name) == expected
    index = {r["image_id"]: r for r in read_csv(integrated / "feature_index.csv")}
    historical = read_csv(previous / "decisions.csv")
    used = [index[r["image_id"]] for r in historical]
    used_bytes = {r["image_sha256"] for r in used}
    used_features = {r["feature_sha256"] for r in used}
    selected = []
    excluded = []
    for r in index.values():
        if r["source"] not in ("COCO2017", "ImageNet-22k"):
            continue
        if r["image_sha256"] in used_bytes or r["feature_sha256"] in used_features:
            excluded.append(r["image_id"])
            continue
        used_bytes.add(r["image_sha256"])
        used_features.add(r["feature_sha256"])
        selected.append(r)

    anchors = [next(r for r in historical if int(r["fold"]) == f and r["source"] == "ImageNet-1k")
               for f in range(5)]
    records = selected + [index[r["image_id"]] for r in anchors]
    raw = np.load(integrated / "raw_fsd_960.npy", mmap_mode="r", allow_pickle=False)
    z = raw[[int(r["row"]) for r in records]]
    for item, row in zip(records, z):
        assert hashlib.sha256(row.tobytes()).hexdigest() == item["feature_sha256"]
    scores = gmm_likelihoods(project, z)
    winners = scores.argmax(axis=1)
    maximum = scores.max(axis=1)
    fold = np.r_[assignments([{"sha256": r["image_sha256"]} for r in selected]), np.arange(5)]
    rel = np.zeros(len(records))
    giga = np.zeros((len(records), 2))
    gallery = read_csv(previous / "reference_banks.csv")
    with np.load(previous / "gigagan_two_centers/coordinates.npz", allow_pickle=False) as coords:
        for f in range(5):
            held = np.flatnonzero(fold == f)
            giga[held] = (np.linalg.norm(z[held,None,:] - coords["two_centers"][f], axis=2)
                          / coords["two_radii"][f])
            for head, source in enumerate(SOURCES):
                ix = held[winners[held] == head]
                if ix.size:
                    ids = [r["image_id"] for r in gallery if int(r["test_fold"]) == f and r["source"] == source]
                    assert len(set(ids)) == 200
                    bank = raw[[int(index[i]["row"]) for i in ids]]
                    rel[ix] = mean5(z[ix], bank) / coords["rel5_scale"][f,head]
    setting = json.loads((previous / "gigagan_two_centers/results.json").read_text())["protocols"]["original_three_known"]
    gates = sorted((r for r in setting["fold_cutoffs"] if r["rule"] == "two_centers"), key=lambda r:r["test_fold"])
    gm = setting["matched95_gmm_cutoffs"]["two_centers"]
    accepted = ((maximum >= gm) & (rel <= np.array([r["common_ceiling"] for r in gates])[fold]) &
                ((winners != HEADS.index("GigaGAN")) |
                 (giga.min(axis=1) <= np.array([r["gigagan_ceiling"] for r in gates])[fold])))
    np.testing.assert_array_equal(np.array(HEADS)[winners[-5:]], [r["candidate"] for r in anchors])
    np.testing.assert_allclose(maximum[-5:], [float(r["gmm_max"]) for r in anchors], rtol=0, atol=1e-6)
    np.testing.assert_allclose(rel[-5:], [float(r["normalized_distance"]) for r in anchors], rtol=1e-9, atol=1e-9)
    out = [dict(image_id=r["image_id"], sha256=r["image_sha256"], source=r["source"], fold=int(fold[i]),
                candidate=HEADS[winners[i]], max_log_likelihood=float(maximum[i]), winner_relative5=float(rel[i]),
                normalized_imagenet_distance=float(giga[i,0]), normalized_coco_distance=float(giga[i,1]),
                baseline=int(accepted[i])) for i, r in enumerate(selected)]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, out)
    print(f"Saved {len(out)} disjoint photo controls, excluded duplicate IDs: {excluded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='score real controls')
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.project, args.out)
