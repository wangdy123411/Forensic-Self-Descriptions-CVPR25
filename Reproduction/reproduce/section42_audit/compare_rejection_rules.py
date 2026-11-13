import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def split_known(ids, sources, known):

    folds = np.full(len(ids), "unknown", dtype=object)
    for source in sorted(set(sources[known])):
        ix = np.flatnonzero(known & (sources == source))
        ordered = sorted(ix, key=lambda i: hashlib.sha256(
            ("fsd-rejection-v1/" + ids[i]).encode("utf-8")).digest())
        quarter = len(ordered) // 4
        folds[ordered[:quarter]] = "reference"
        folds[ordered[quarter:2 * quarter]] = "calibrate"
        folds[ordered[2 * quarter:]] = "evaluate"
    return folds


def percentile(reference, values):
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + right) / (2 * len(ordered))


def acceptance_by_source(accepted, mask, sources):
    return {
        str(source): float(accepted[mask & (sources == source)].mean())
        for source in sorted(set(sources[mask]))
    }


def evaluate(score, folds, correct, sources):
    known = folds == "evaluate"
    unknown = folds == "unknown"
    threshold = float(np.quantile(score[folds == "calibrate"], 0.05, method="lower"))
    accepted = score >= threshold
    known_rates = acceptance_by_source(accepted, known, sources)
    unknown_rates = acceptance_by_source(accepted, unknown, sources)
    ranking = percentile(score[unknown], score[known])


    known_scores = score[known]
    cutoff = np.sort(known_scores)[len(known_scores) - int(np.ceil(0.95 * len(known_scores)))]
    tied = (0.95 * len(known_scores) - np.count_nonzero(known_scores > cutoff)) / np.count_nonzero(known_scores == cutoff)
    unknown_at_95 = (np.count_nonzero(score[unknown] > cutoff)
                     + tied * np.count_nonzero(score[unknown] == cutoff)) / unknown.sum()
    return {
        "threshold": threshold,
        "known_acceptance": float(accepted[known].mean()),
        "unknown_acceptance": float(accepted[unknown].mean()),
        "unknown_acceptance_at_matched_95_known": float(unknown_at_95),
        "known_macro_acceptance": float(np.mean(list(known_rates.values()))),
        "unknown_macro_acceptance": float(np.mean(list(unknown_rates.values()))),
        "unknown_accepted_by_source": unknown_rates,
        "known_accepted_by_source": known_rates,
        "correct_known_and_accepted": float((correct[known] & accepted[known]).mean()),
        "known_unknown_auroc": float(ranking.mean()),
        "au_oscr": float((ranking * correct[known]).mean()),
    }


def compare(ids, sources, known, correct, likelihoods):
    if len(set(ids)) != len(ids) or not np.isfinite(likelihoods).all():
        raise ValueError("Expected unique image IDs and finite scores")
    folds = split_known(ids, sources, known)
    ordered = np.sort(likelihoods, axis=1)
    maximum = ordered[:, -1]
    margin = maximum - ordered[:, -2]
    reference = folds == "reference"

    joint = np.minimum(percentile(maximum[reference], maximum),
                       percentile(margin[reference], margin))
    author_confidence = 1 / np.exp((likelihoods - maximum[:, None]) / 750).sum(axis=1)
    scores = {"max": maximum, "author_confidence": author_confidence,
              "margin": margin, "max_and_margin": joint}
    return {
        "counts": {name: int(np.count_nonzero(folds == name))
                   for name in ("reference", "calibrate", "evaluate", "unknown")},
        "rules": {name: evaluate(s, folds, correct, sources) for name, s in scores.items()},
    }


def read_masked(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        candidates = reader.fieldnames[reader.fieldnames.index("max_log_likelihood") + 1:]
        rows = list(reader)
    likelihoods = np.array([[float(r[c]) for c in candidates] for r in rows])
    winners = np.array(candidates)[np.argmax(likelihoods, axis=1)]
    if not np.array_equal(winners, [r["candidate"] for r in rows]):
        raise ValueError("Saved winners differ from the maximum-likelihood source")
    targets = np.array([r["target"] for r in rows])
    return compare([r["image_id"] for r in rows], np.array([r["source"] for r in rows]),
                   targets != "unknown", winners == targets, likelihoods)


if __name__ == "__main__":
    results = {name: read_masked(HERE.parent / folder / "predictions.csv")
               for name, folder in (("3+3", "section42_paper_overlap"),
                                    ("5+4", "section42_paper_substituted"))}
    path = HERE / "heldout_rejection_rules.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for protocol, result in results.items():
        print(protocol, result["counts"])
        for rule, metrics in result["rules"].items():
            print(rule, {k: round(metrics[k], 4) for k in
                         ("known_acceptance", "unknown_acceptance",
                          "unknown_acceptance_at_matched_95_known", "known_unknown_auroc", "au_oscr")})
