import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from evaluate_external import LOCK, read_csv, sha, write_csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_neighbors import accept_top95


KNOWN = ("ImageNet-1k", "StyleGAN3", "ProGAN")
PROTOCOL = "original_three_known"
REAL_QUANTILE = 0.95


def distance_pass(terms, fold, candidate, relative, giga):
    return ((relative <= terms["common"][fold]) &
            ((candidate != "GigaGAN") | (giga <= terms["giga"][fold])))


def real_pass(terms, fold, candidate, relative):
    return (candidate != "Real") | (relative <= terms["real"][fold])


def add_rows(output, ids, sources, split, fold, candidate, maximum, relative,
             old, new, terms, hashes=None):
    for i, key in enumerate(ids):
        f = int(fold[i])
        real = candidate[i] == "Real"
        output.append(dict(image_id=key, sha256=hashes[i] if hashes is not None else "",
                           cohort=split, source=sources[i], fold=f, candidate=candidate[i],
                           gmm_max=float(maximum[i]), winner_relative5=float(relative[i]),
                           real_ceiling=float(terms["real"][f]) if real else "",
                           passes_real_distance=int(relative[i] <= terms["real"][f]) if real else "",
                           previous_accepted=int(old[i]), new_accepted=int(new[i])))


def count(label, rows, accepted_head=None):
    data = list(rows)
    old = np.array([bool(r["previous_accepted"]) for r in data])
    new = np.array([bool(r["new_accepted"]) for r in data])
    if accepted_head:
        matches = np.array([r["candidate"] == accepted_head for r in data])
        old &= matches
        new &= matches
    return dict(group=label, n=len(data), previous=int(old.sum()), new=int(new.sum()),
                accepted_to_rejected=int((old & ~new).sum()), rejected_to_accepted=int((~old & new).sum()))


def run(project, output, controls):
    results = project / "Results/reproduction"
    ref = results / "section42_reference_neighbors_20261112"
    previous_external = results / "section42_external_validation_20261112"
    for name, digest in LOCK.items():
        if sha(ref / name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    with np.load(ref / "gigagan_two_centers/coordinates.npz", allow_pickle=False) as file:
        coords = {key: file[key] for key in file.files}
    hist = read_csv(ref / "decisions.csv")
    archived = read_csv(ref / "gigagan_two_centers/decisions.csv")
    ids = [r["image_id"] for r in hist]
    np.testing.assert_array_equal(ids, coords["ids"])
    np.testing.assert_array_equal(ids, [r["image_id"] for r in archived])
    cand = np.array([r["candidate"] for r in hist])
    src = np.array([r["source"] for r in hist])
    maximum = np.array([float(r["gmm_max"]) for r in hist])
    fold = coords["fold"]
    known = np.isin(src, KNOWN)
    real = (src == "ImageNet-1k") & (cand == "Real")
    setting = json.loads((ref / "gigagan_two_centers/results.json").read_text())["protocols"][PROTOCOL]
    gates = sorted((r for r in setting["fold_cutoffs"] if r["rule"] == "two_centers"),
                   key=lambda r: r["test_fold"])
    assert [r["test_fold"] for r in gates] == list(range(5))
    terms = {key: np.array([r[name] for r in gates]) for key, name in
             (("common", "common_ceiling"), ("giga", "gigagan_ceiling"))}
    cal_gallery = (fold - 1) % 5
    calibration_counts = []
    ceilings = []
    for f in range(5):
        pool = real & (cal_gallery == f)
        ceilings.append(np.quantile(coords["cal_rel5"][pool], REAL_QUANTILE, method="higher"))
        calibration_counts.append(int(pool.sum()))
    terms["real"] = np.array(ceilings)
    rel = coords["test_rel5"]
    giga = (coords["test_two"] / coords["two_radii"][fold]).min(axis=1)
    eligible = distance_pass(terms, fold, cand, rel, giga)
    old_cut = setting["matched95_gmm_cutoffs"]["two_centers"]
    old = eligible & (maximum >= old_cut)
    np.testing.assert_array_equal(old, [r[PROTOCOL+"_matched95_two_centers"] == "1" for r in archived])
    new, new_cut = accept_top95(maximum, eligible & real_pass(terms, fold, cand, rel), known)
    assert int(old[known].sum()) == int(new[known].sum()) == 2850 and known.sum() == 3000
    all_rows = []
    add_rows(all_rows, ids, src, "original", fold, cand, maximum, rel, old, new, terms)

    cal_rel = coords["cal_rel5"]
    cal_giga = (coords["cal_two"] / coords["two_radii"][cal_gallery]).min(axis=1)
    cal_eligible = distance_pass(terms, cal_gallery, cand, cal_rel, cal_giga)
    cal_eligible &= real_pass(terms, cal_gallery, cand, cal_rel)
    fold_cut = np.array([accept_top95(maximum, cal_eligible, known & (cal_gallery == f))[1]
                         for f in range(5)])
    cv_old = np.array([r[PROTOCOL+"_heldout_two_centers"] == "1" for r in archived])
    cv_new = eligible & real_pass(terms, fold, cand, rel) & (maximum >= fold_cut[fold])

    report = dict(protocol=PROTOCOL, frozen_sha256=LOCK,
                  method="Mean Euclidean distance to the five nearest of 200 ImageNet-1k references, divided by the median within-bank reference distance. Each fold uses the 95th percentile (higher) among genuine, Real-nominated calibration images. Acceptance also requires both original distance gates and the recalibrated maximum GMM score.",
                  known_sources=KNOWN, known_count=int(known.sum()), known_accepted=int(new[known].sum()),
                  original_gmm_cutoff=old_cut, new_gmm_cutoff=new_cut,
                  real_quantile=REAL_QUANTILE, real_ceilings=ceilings, real_calibration_counts=calibration_counts,
                  common_ceilings=terms["common"].tolist(), heldout_gmm_cutoffs=fold_cut.tolist(),
                  heldout_known=[int(cv_old[known].sum()), int(cv_new[known].sum())],
                  heldout_correct_real=[int(cv_old[real].sum()), int(cv_new[real].sum())])
    for split, file in [("external_main", previous_external/"main/decisions.csv"),
                        ("external_gallery", previous_external/"gallery/decisions.csv"),
                        ("real_controls", controls)]:
        if file is None:
            continue
        external = read_csv(file)
        control = split == "real_controls"
        report.setdefault("query_sha256", {})[split] = sha(file)
        ext_fold = np.array([int(r["fold"]) for r in external])
        ext_cand = np.array([r["candidate"] for r in external])
        ext_rel = np.array([float(r["winner_relative5"]) for r in external])
        ext_score = np.array([float(r["max_log_likelihood"]) for r in external])
        center_fields = (("normalized_imagenet_distance","normalized_coco_distance") if control else
                         ("gigagan_imagenet_distance","gigagan_coco_distance"))
        ext_giga = np.array([min(float(r[col]) for col in center_fields) for r in external])
        ext_eligible = distance_pass(terms, ext_fold, ext_cand, ext_rel, ext_giga)
        before = ext_eligible & (ext_score >= old_cut)
        old_key = "baseline" if control else PROTOCOL+"_matched95_frozen_two_centers"
        np.testing.assert_array_equal(before, [int(r[old_key]) for r in external])
        after = ext_eligible & real_pass(terms, ext_fold, ext_cand, ext_rel) & (ext_score >= new_cut)
        add_rows(all_rows, [r["image_id"] for r in external], [r["source"] for r in external],
                 split, ext_fold, ext_cand, ext_score, ext_rel, before, after, terms,
                 [r["sha256"] for r in external])

    historical = [r for r in all_rows if r["cohort"] == "original"]
    firefly = [r for r in all_rows if r["cohort"] == "external_main" and r["source"] == "Firefly"]
    targets = [r for r in firefly if r["previous_accepted"] and r["candidate"] == "Real"]
    gallery = [r for r in all_rows if r["cohort"] == "external_gallery"]
    assert len(targets) == 43 and len(gallery) == 25
    assert all(r["candidate"] == "Real" and r["previous_accepted"] for r in gallery)
    summaries = [count("Original three known, accepted", (r for r in historical if r["source"] in KNOWN)),
                 count("ImageNet-1k, correctly accepted as Real",
                       (r for r in historical if r["source"] == "ImageNet-1k"), "Real"),
                 count("External Firefly, all", firefly), count("Previously accepted Firefly as Real", targets),
                 count("GigaGAN website, accepted as Real", gallery, "Real"),
                 count("External GigaGAN archive, correctly accepted",
                       (r for r in all_rows if r["cohort"] == "external_main" and r["source"] == "GigaGAN"), "GigaGAN")]
    unknown_sources = {"FireFly", "DALLE 3", "StyleGAN2"}
    summaries.append(count("Original unknown, accepted",
                           (r for r in historical if r["source"] in unknown_sources)))
    for source in sorted(unknown_sources):
        summaries.append(count(source+" original, accepted", (r for r in historical if r["source"] == source)))
    independent = [r for r in all_rows if r["cohort"] == "real_controls"]
    independent_sources = {r["source"] for r in independent}
    for source in sorted(independent_sources):
        summaries.append(count(source+", correctly accepted as Real", (r for r in independent if r["source"] == source), "Real"))
    real_sources = independent_sources | {"ImageNet-1k"}
    report["real_nominations"] = dict(Counter(r["source"] for r in all_rows
                                             if r["candidate"] == "Real" and r["source"] in real_sources))
    assert len({r["image_id"] for r in all_rows}) == len(all_rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"summary.csv", summaries)
    write_csv(output/"decisions.csv", all_rows)
    write_csv(output/"target_68.csv", targets + gallery)
    (output/"operating_point.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for r in summaries:
        print(r["group"], r["previous"], r["new"], "/", r["n"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='evaluate real distance')
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controls", type=Path)
    args = p.parse_args()
    run(args.project, args.out, args.controls)
