"""Compare literal upstream public-API probes with cached default-grid results."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "an/baseline_comparison"


def main():
    main_rows = [
        json.loads(line)
        for line in (ROOT / "an/results/main/records.jsonl").read_text().splitlines()
        if line
    ]
    by_key = {(row["sha256"], row["condition"]): row for row in main_rows}
    results = []
    for path in sorted(BASE.glob("upstream_default1024_*.json")):
        probe = json.loads(path.read_text())
        if probe["status"] != "success":
            results.append(dict(probe=path.name, passed=False, status=probe["status"]))
            continue
        row = by_key[(probe["image_sha256"], probe["condition"])]
        upstream = probe["result"]
        names = list(row["attribution_loglik"])
        values = np.array([row["attribution_loglik"][name] for name in names])
        probs = np.exp((values - values.max()) / 750)
        probs /= probs.sum()
        max_prob = max(abs(upstream["scores"][name] - value) for name, value in zip(names, probs))
        delta_z = abs(upstream["z_score"] - row["z_score"])
        source_match = upstream["source"] == row["attribution_pred"]
        decision_match = upstream["is_fake"] == (row["z_score"] < -2)
        result = dict(
            probe=path.name,
            image_id=row["image_id"],
            condition=row["condition"],
            upstream_z=upstream["z_score"],
            an_z=row["z_score"],
            absolute_z_difference=delta_z,
            source_match=source_match,
            released_decision_match=decision_match,
            maximum_source_probability_difference=max_prob,
            confidence_difference=abs(upstream["confidence"] - row["attribution_confidence"]),
            passed=bool(delta_z < 1e-3 and max_prob < 1e-4 and source_match and decision_match),
        )
        results.append(result)
    report = dict(
        passed=bool(results) and all(row["passed"] for row in results),
        purpose="Literal upstream public API checks at default 1024; same original weights and processed inputs",
        probability_conversion="AN raw source likelihoods converted with original temperature750 softmax",
        results=results,
    )
    (BASE / "api_probe_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
