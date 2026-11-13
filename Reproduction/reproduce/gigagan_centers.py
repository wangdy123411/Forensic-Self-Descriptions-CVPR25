import argparse
from pathlib import Path

import numpy as np

from reference_neighbors import BANK_SIZE, HEADS, HERE, N_FOLDS, SOURCE_TO_HEAD, mean_five, read_input, rows


TWO_NAMES = ("GigaGAN_cond_imagenet256", "GigaGAN_t2i_coco256_pooled")
GIGA_VARIANTS = {
    TWO_NAMES[0]: 100,
    "GigaGAN_t2i_coco256_diff_noised": 50,
    "GigaGAN_t2i_coco256_rep": 50,
}


def centroid_and_radius(reference):
    center = reference.mean(axis=0)
    radius = np.sqrt(np.square(reference - center).sum() / (len(reference) - 1))
    return center, radius


def run(project, manifest, saved, out):
    ids, sources, families, variants, fold, raw, _, winner, _ = read_input(project, manifest)
    decisions = rows(saved / "decisions.csv")
    gallery = rows(saved / "reference_banks.csv")
    np.testing.assert_array_equal(ids, [r["image_id"] for r in decisions])
    np.testing.assert_array_equal(fold, [int(r["fold"]) for r in decisions])
    np.testing.assert_array_equal(winner, [r["candidate"] for r in decisions])
    lookup = {key: i for i, key in enumerate(ids)}
    assert len(gallery) == N_FOLDS * len(HEADS) * BANK_SIZE

    arrays = {role + "_" + coordinate: np.full(shape, np.nan)
              for role in ("test", "cal")
              for coordinate, shape in (("two", (len(ids), 2)), ("one", (len(ids),)),
                                        ("rel5", (len(ids),)), ("giga5", (len(ids),)))}
    two_radii = np.empty((N_FOLDS, len(TWO_NAMES)))
    one_radii = np.empty(N_FOLDS)
    two_centers = np.empty((N_FOLDS, len(TWO_NAMES), raw.shape[1]))
    one_center = np.empty((N_FOLDS, raw.shape[1]))
    scales = np.empty((N_FOLDS, len(HEADS)))
    for f in range(N_FOLDS):
        next_fold = (f + 1) % N_FOLDS
        query = np.flatnonzero((fold == f) | (fold == next_fold))
        excluded_families = set(families[query])
        for source, head in SOURCE_TO_HEAD.items():
            records = [r for r in gallery if int(r["test_fold"]) == f and r["source"] == source]
            bank = np.array([lookup[r["image_id"]] for r in records])
            assert len(bank) == len(set(bank)) == BANK_SIZE and np.all(sources[bank] == source)
            np.testing.assert_array_equal(families[bank], [r["family"] for r in records])
            np.testing.assert_array_equal(variants[bank], [r["variant"] for r in records])
            assert excluded_families.isdisjoint(families[bank])
            reference = raw[bank]
            scale = float(np.median(mean_five(reference, reference, self_query=True)))
            assert scale > 0
            np.testing.assert_allclose(scale, [float(r["reference_distance_median"]) for r in records],
                                       atol=1e-8, rtol=1e-12)
            scales[f, HEADS.index(head)] = scale

            ix = query if head == "GigaGAN" else query[winner[query] == head]
            five = mean_five(raw[ix], reference)
            if head == "GigaGAN":
                names, counts = np.unique(variants[bank], return_counts=True)
                assert dict(zip(names, counts)) == GIGA_VARIANTS
                class_cond = variants[bank] == TWO_NAMES[0]
                center, one_radii[f] = centroid_and_radius(reference)
                one_center[f] = center
                one = np.linalg.norm(raw[ix] - center, axis=1)
                two = np.empty((len(ix), len(TWO_NAMES)))
                for j, mask in enumerate((class_cond, ~class_cond)):
                    center, two_radii[f, j] = centroid_and_radius(reference[mask])
                    two_centers[f,j] = center
                    two[:, j] = np.linalg.norm(raw[ix] - center, axis=1)
            for role, target_fold in (("test", f), ("cal", next_fold)):
                selected = fold[ix] == target_fold
                nominated = selected & (winner[ix] == head)
                arrays[role + "_rel5"][ix[nominated]] = five[nominated] / scale
                if head == "GigaGAN":
                    arrays[role + "_giga5"][ix[selected]] = five[selected]
                    arrays[role + "_one"][ix[selected]] = one[selected]
                    arrays[role + "_two"][ix[selected]] = two[selected]

    for values in arrays.values():
        assert np.isfinite(values).all() and values.min() >= 0
    assert (two_radii > 0).all() and (one_radii > 0).all()
    expected = np.array([float(r["normalized_distance"]) for r in decisions])
    np.testing.assert_allclose(arrays["test_rel5"], expected, atol=1e-8, rtol=1e-12)
    previous_giga = rows(saved / "distances.csv")
    np.testing.assert_allclose(arrays["test_giga5"][[lookup[r["image_id"]] for r in previous_giga]],
                               [float(r["distance"]) for r in previous_giga], atol=1e-8, rtol=1e-12)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "coordinates.tmp.npz"
    np.savez_compressed(tmp, ids=ids, fold=fold, heads=np.array(HEADS), two_names=np.array(TWO_NAMES),
                        two_centers=two_centers, one_center=one_center, two_radii=two_radii,
                        one_radii=one_radii, rel5_scale=scales, **arrays)
    with np.load(tmp, allow_pickle=False) as packed:
        for key, values in arrays.items():
            np.testing.assert_array_equal(packed[key], values)
    tmp.replace(out / "coordinates.npz")
    print(f"Verified {len(ids)} rows. Maximum absolute test_rel5 replay difference: "
          f"{np.max(np.abs(arrays['test_rel5'] - expected)):.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gigagan centers')
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--saved", type=Path, default=HERE / "reference_neighbors")
    parser.add_argument("--out", type=Path, default=HERE / "gigagan_centers")
    args = parser.parse_args()
    run(args.project, args.manifest, args.saved, args.out)
