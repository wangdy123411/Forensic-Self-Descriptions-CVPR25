from project_paths import PROJECT_ROOT, CODE_ROOT

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def replay(project, output):
    sys.path.insert(0, str(CODE_ROOT))
    from fsd.attribution import load_source_gmms
    from fsd.projection import apply_projections, load_transforms

    results = project / "Results/reproduction"
    release = results / "section42_author_v120_20261110/4.2"
    base = results / "section41_final_20261110/integrated"
    model = json.loads((release / "model.json").read_text())
    classes = model["classes"]
    weights = {}
    for filename, expected in model["weight_sha256"].items():
        path = next(folder / filename for folder in (project / "Weights/fsd_v1.2.0", CODE_ROOT)
                    if (folder / filename).is_file())
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Weight digest mismatch: {filename}")
        weights[filename] = path
    rows = read_rows(release / "attribution.csv")
    index = read_rows(base / "feature_index.csv")
    features = np.load(base / "raw_fsd_960.npy", mmap_mode="r")
    if len(rows) != len(index) or features.shape != (len(rows), 960) or len(rows) != model["n"]:
        raise ValueError("Archived feature dimensions do not match the predictions")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs in archived predictions")
    for i, (row, ref) in enumerate(zip(rows, index)):
        if ref["image_id"] != row["image_id"] or ref["source"] != row["source"] or int(ref["row"]) != i:
            raise ValueError(f"Feature/prediction row alignment error at {i}")

    torch.set_num_threads(4)
    transforms = load_transforms(weights["fsd_transforms.pt"], "cpu")
    transforms += load_transforms(weights["attribution_transforms.pt"], "cpu")
    gmms = load_source_gmms(weights["source_gmms.pt"], "cpu")
    if list(gmms) != classes:
        raise ValueError("Release and GMM class orders differ")
    scores = np.empty((len(rows), len(classes)), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            projected = apply_projections(torch.from_numpy(features[start:start + 256].copy()), transforms)
            scores[start:start + len(projected)] = torch.stack(
                [gmm.score_samples(projected) for gmm in gmms.values()], dim=1
            ).numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite GMM scores")
    predicted = np.asarray(classes)[scores.argmax(axis=1)]
    maxima = scores.max(axis=1)
    if not np.array_equal(predicted, [r["candidate"] for r in rows]):
        raise ValueError("Replay winners differ from the archive")
    expected = np.array([float(r["max_log_likelihood"]) for r in rows])
    delta = float(np.max(abs(maxima - expected)))
    if delta >= 1e-6:
        raise ValueError(f"Replay maximum difference exceeds tolerance: {delta}")

    fields = ["image_id", "source", "target", "candidate", "max_log_likelihood",
              "true_score", "true_rank", "winner_minus_true", *classes]
    counts = Counter()
    ntied = 0
    supported = 0
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "full_scores.csv.part"
    with partial.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r, s, candidate, maximum in zip(rows, scores, predicted, maxima):
            result = {k: r[k] for k in ("image_id", "source", "target")}
            result.update(candidate=candidate, max_log_likelihood=maximum)
            result.update(zip(classes, s))
            if r["target"]:
                value = s[classes.index(r["target"])]
                rank = 1 + int(np.count_nonzero(s > value))
                supported += 1
                ntied += int(np.count_nonzero(s == value) > 1)
                if candidate != r["target"]:
                    counts[rank] += 1
                result.update(true_score=value, true_rank=rank, winner_minus_true=maximum - value)
            writer.writerow(result)
    partial.replace(output / "full_scores.csv")
    meta = {
        "protocol": model["mode"], "classes": classes, "rows": len(rows), "known": supported,
        "correct_top1_parity_count": len(rows), "max_abs_winner_loglik_difference": delta,
        "known_with_tied_true_score": ntied, "misclassified_known": sum(counts.values()),
        "misclassified_known_by_true_rank": dict(sorted(counts.items())),
        "rank_definition": "1 + number of candidate raw log likelihoods strictly above the true source score",
        "unsupported": "empty target, true_score, true_rank and winner_minus_true",
        "weights_sha256": model["weight_sha256"],
        "archived_feature_index": str(base / "feature_index.csv"),
        "archived_predictions": str(release / "attribution.csv"),
    }
    (output / "full_scores_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(meta["rows"], meta["max_abs_winner_loglik_difference"], meta["misclassified_known_by_true_rank"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='replay full')
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    replay(args.project, args.output)
