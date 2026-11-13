from project_paths import DATA_ROOT

import csv
import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def analyze(predictions, provenance, metrics):
    known_map = metrics["known_target_mapping"]
    class_names = list(known_map.values())
    source = np.array([row["source"] for row in predictions])
    candidate = np.array([row["candidate"] for row in predictions])
    scores = np.array([float(row["max_log_likelihood"]) for row in predictions])
    target = np.array([known_map.get(name, "") for name in source])
    known = target != ""
    correct = target == candidate
    thresholds = np.unique(scores)

    def count_accepts(selection):
        values = np.sort(scores[selection])
        return len(values) - np.searchsorted(values, thresholds, side="left")

    false_positive_rate = count_accepts(~known) / sum(~known)
    per_class_f1 = []
    for name in class_names:
        predicted = candidate == name
        labelled = target == name
        counts = count_accepts(predicted & labelled)
        per_class_f1.append(2 * counts / (sum(labelled) + count_accepts(predicted)))
    def integrate(y):
        return float(np.trapezoid(np.r_[0., y[::-1]], np.r_[0., false_positive_rate[::-1]]))

    f1_area = integrate(np.mean(per_class_f1, axis=0))
    oscr = integrate(count_accepts(correct) / sum(known))
    assert abs(f1_area - metrics["au_crr_fang_macro_f1_vs_crr"]) < 1e-10
    assert abs(oscr - metrics["au_oscr"]) < 1e-10
    threshold = float(np.quantile(scores[known], .05, method="lower"))
    accepted = scores >= threshold
    packages = np.array([provenance[row["image_id"]] for row in predictions])
    by_source = {}
    for name in sorted(set(source)):
        selection = source == name
        by_source[name] = {
            "n": int(sum(selection)),
            "top1_by_candidate": dict(Counter(candidate[selection]).most_common()),
            "accepted_by_candidate": dict(Counter(candidate[selection & accepted]).most_common()),
            "accepted_by_package": {
                package: {
                    "accepted": int(sum(selection & (packages == package) & accepted)),
                    "n": int(sum(selection & (packages == package))),
                    "top1_by_candidate": dict(Counter(candidate[selection & (packages == package)]).most_common()),
                }
                for package in sorted(set(packages[selection]))
            },
        }
    return {
        "independent_curve_integrals": {"area_macro_f1_vs_crr": f1_area, "au_oscr": oscr},
        "threshold_derived_on_test_for_description_only": threshold,
        "known_accepted": int(sum(known & accepted)),
        "unknown_accepted": int(sum(~known & accepted)),
        "by_source": by_source,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='diagnose overlap')
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parent.parent / "section42_paper_overlap")
    data = parser.parse_args().run
    database = DATA_ROOT / 'manifests/provenance.csv'
    packages = {row["image_id"]: row["original_relative_path"].split("/")[0] for row in read_rows(database)}
    supplemental = data / "supplemental_firefly_mnw" / "manifest.csv"
    if supplemental.exists():
        packages.update({r["image_id"]: r["package"] for r in read_rows(supplemental)})
    audit = analyze(read_rows(data / "predictions.csv"), packages, json.loads((data / "metrics.json").read_text()))
    (data / "source_diagnostics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
