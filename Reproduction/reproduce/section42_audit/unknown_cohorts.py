from project_paths import CODE_ROOT

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rank_metrics(unknown_scores, known_scores, correctly_classified, threshold):

    sorted_unknown = np.sort(unknown_scores)
    left = np.searchsorted(sorted_unknown, known_scores, side="left")
    right = np.searchsorted(sorted_unknown, known_scores, side="right")
    wins = (left + right) / (2 * len(sorted_unknown))
    return {
        "unknown_count": len(unknown_scores),
        "au_oscr": float(np.mean(wins * correctly_classified)),
        "known_unknown_roc_auc": float(np.mean(wins)),
        "unknown_accepted": int(np.sum(unknown_scores >= threshold)),
        "unknown_acceptance": float(np.mean(unknown_scores >= threshold)),
    }


def compare(project, output):
    sys.path.insert(0, str(CODE_ROOT))
    from fsd.attribution import load_source_gmms
    from fsd.projection import apply_projections, load_transforms

    root = project / "Results/reproduction"
    released = root / "section42_author_v120_20261110/4.2"
    integrated = root / "section41_final_20261110/integrated"
    rows = read_csv(released / "attribution.csv")
    index = read_csv(integrated / "feature_index.csv")
    features = np.load(integrated / "raw_fsd_960.npy", mmap_mode="r")
    assert len(index) == len(rows) == len(features) == 28992
    assert all(a["image_id"] == b["image_id"] and a["source"] == b["source"] and int(b["row"]) == n
               for n, (a, b) in enumerate(zip(rows, index)))
    model = json.loads((released / "model.json").read_text())

    def weight(name):
        directories = [project / "Weights/fsd_v1.2.0", CODE_ROOT]
        path = next(d / name for d in directories if (d / name).is_file())
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == model["weight_sha256"][name]
        return path

    names = ["ProGAN", "StyleGAN3", "Real"]
    source = np.array([r["source"] for r in rows])
    known = np.isin(source, ["ProGAN", "StyleGAN3", "ImageNet-1k"])
    paper_unknown = np.isin(source, ["DALLE 3", "FireFly", "StyleGAN2"])
    coverage_unknown = np.array([r["supported"] == "0" for r in rows])
    assert (sum(known), sum(paper_unknown), sum(coverage_unknown)) == (3000, 3997, 16995)
    assert not np.any(known & paper_unknown | known & coverage_unknown | paper_unknown & coverage_unknown)
    transforms = load_transforms(weight("fsd_transforms.pt")) + load_transforms(weight("attribution_transforms.pt"))
    models = load_source_gmms(weight("source_gmms.pt"))
    scores = np.full(len(rows), np.nan)
    best = np.full(len(rows), -1)
    indices = np.flatnonzero(known | paper_unknown | coverage_unknown)
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            batch = indices[start:start + 256]
            x = apply_projections(torch.from_numpy(features[batch]), transforms)
            raw = torch.stack([models[name].score_samples(x) for name in names], 1)
            values, heads = raw.max(dim=1)
            scores[batch], best[batch] = values.numpy(), heads.numpy()
    assert np.isfinite(scores[indices]).all()
    known_scores = scores[known]
    target = np.where(source[known] == "ImageNet-1k", "Real", source[known])
    correct = np.asarray(names)[best[known]] == target
    threshold = float(np.quantile(known_scores, .05, method="lower"))
    report = {
        "candidates": names,
        "known_count": int(sum(known)),
        "known_correct": int(np.sum(correct)),
        "known_accepted_at_threshold": int(sum(known_scores >= threshold)),
        "threshold_from_test_known_scores_descriptive_only": threshold,
        "cohorts": {},
        "weight_sha256": model["weight_sha256"],
    }
    for cohort, mask in [("paper_overlap", paper_unknown), ("release_unsupported", coverage_unknown)]:
        per_source = {}
        for label in sorted(set(source[mask])):
            grouped = scores[mask & (source == label)]
            per_source[label] = rank_metrics(grouped, known_scores, correct, threshold)
        all_images = rank_metrics(scores[mask], known_scores, correct, threshold)
        balanced = {}
        for key in ["au_oscr", "known_unknown_roc_auc", "unknown_acceptance"]:
            balanced[key] = float(np.mean([m[key] for m in per_source.values()]))
            weighted = sum(m[key] * m["unknown_count"] for m in per_source.values()) / sum(mask)
            assert np.isclose(all_images[key], weighted, rtol=0, atol=1e-12)
        report["cohorts"][cohort] = {"all_images": all_images, "equal_weight_per_unknown_source": balanced,
                                     "by_unknown_source": per_source}

    archived_scores = np.array([float(r["max_log_likelihood"]) for r in rows])
    archived_target = np.array([r["target"] for r in rows])
    archived_prediction = np.array([r["candidate"] for r in rows])
    archived_correct = archived_target == archived_prediction
    report["released_14_way_on_same_unsupported_unknowns"] = {}
    for subset, mask in [("same_3000_known", known), ("all_release_supported_known", ~coverage_unknown)]:
        subset_scores = archived_scores[mask]
        subset_threshold = float(np.quantile(subset_scores, .05, method="lower"))
        metrics = rank_metrics(archived_scores[coverage_unknown], subset_scores, archived_correct[mask], subset_threshold)
        metrics["known_count"] = int(sum(mask))
        metrics["known_correct"] = int(sum(archived_correct[mask]))
        metrics["threshold_from_test_known_scores_descriptive_only"] = subset_threshold
        report["released_14_way_on_same_unsupported_unknowns"][subset] = metrics
    reference = json.loads((Path(__file__).with_name("candidate_audit.json")).read_text())
    base = reference["comparisons"]["paper_overlap_3_vs_10"]["before"]
    observed = report["cohorts"]["paper_overlap"]["all_images"]
    assert np.allclose([observed["au_oscr"], observed["known_unknown_roc_auc"], np.mean(correct), threshold],
                       [base["au_oscr"], base["known_unknown_auc"], base["macro_accuracy"],
                        base["at_same_baseline_threshold"]["tau"]], rtol=0, atol=1e-8)
    released_all = reference["comparisons"]["represented_8_vs_all_14"]["after"]
    full = report["released_14_way_on_same_unsupported_unknowns"]["all_release_supported_known"]
    assert np.allclose([full["au_oscr"], full["known_unknown_roc_auc"]],
                       [released_all["au_oscr"], released_all["known_unknown_auc"]], rtol=0, atol=1e-12)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for cohort, results in report["cohorts"].items():
        print(cohort, results["all_images"], results["equal_weight_per_unknown_source"])
    print("Archived 14-way comparisons:", report["released_14_way_on_same_unsupported_unknowns"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unknown cohorts')
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("unknown_cohorts.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    compare(args.project, args.output)
