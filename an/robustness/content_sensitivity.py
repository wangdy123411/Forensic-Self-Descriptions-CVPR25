"""Supplementary paired AUROC bootstrap using archive stems as content proxies.

This leaves the main source-stratified estimator and files unchanged. Synthbuster
members sharing a filename stem receive one joint bootstrap multiplicity across
generators. Stems are unconfirmed reference-content identifiers, not verified
prompt IDs. Fake counts per generator may vary; each macro statistic still
weights generators equally. Real images are resampled within real-source strata.

From the repository root:
    python -m an.robustness.content_sensitivity an/results/main --bootstrap 1000
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .analysis import _clean_json, _conditions, _interval, _read_records, _table, _weighted_auroc


def _proxy_cluster_ids(frame: pd.DataFrame) -> np.ndarray:
    """Use Synthbuster archive stems; all other originals remain independent."""
    result = []
    for row in frame.to_dict("records"):
        dataset = str(row.get("dataset", "")).strip().casefold()
        member = row.get("archive_member")
        if (
            int(row["label"]) == 1
            and dataset == "synthbuster"
            and isinstance(member, str)
            and member.strip()
        ):
            stem = PurePosixPath(member.replace("\\", "/")).stem
            if not stem:
                raise ValueError(f"Empty archive-member stem for {row['image_id']}")
            result.append(json.dumps(["archive_stem_proxy", dataset, stem]))
        else:
            result.append(json.dumps(["independent_original", str(row["image_id"])]))
    return np.asarray(result, dtype=str)


def _cluster_weights(
    labels: np.ndarray,
    sources: np.ndarray,
    clusters: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One multiplicity per fake proxy cluster; real source sizes stay fixed."""
    if len(labels) != len(sources) or len(labels) != len(clusters):
        raise ValueError("Labels, sources, and clusters must have identical lengths")
    if draws < 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Require positive bootstrap draws and binary labels")
    weights = np.zeros((draws, len(labels)), dtype=np.float64)
    for source in np.unique(sources[labels == 0]):
        indices = np.flatnonzero((labels == 0) & (sources == source))
        weights[:, indices] = rng.multinomial(
            len(indices), np.full(len(indices), 1 / len(indices)), size=draws
        )
    fake = np.flatnonzero(labels == 1)
    if len(fake):
        unique, inverse = np.unique(clusters[fake], return_inverse=True)
        multiplicities = rng.multinomial(
            len(unique), np.full(len(unique), 1 / len(unique)), size=draws
        )
        weights[:, fake] = multiplicities[:, inverse]
    return weights


def _cluster_audit(frame: pd.DataFrame) -> dict:
    fake = frame.loc[frame["label"] == 1].copy()
    fake["proxy_cluster"] = _proxy_cluster_ids(fake)
    counts = fake.groupby("proxy_cluster").size()
    source_counts = fake.groupby("proxy_cluster")["source"].nunique()
    shared = source_counts[source_counts > 1].index
    uses_proxy = fake["proxy_cluster"].map(
        lambda value: json.loads(value)[0] == "archive_stem_proxy"
    )
    return {
        "n_fake_originals": len(fake),
        "n_fake_proxy_clusters": len(counts),
        "n_fake_with_archive_stem_proxy": int(uses_proxy.sum()),
        "n_cross_generator_shared_proxy_clusters": len(shared),
        "n_fake_in_cross_generator_shared_proxy_clusters": int(counts.loc[shared].sum()),
        "largest_fake_proxy_cluster": int(counts.max()) if len(counts) else 0,
    }


def _condition_metrics(
    paired: pd.DataFrame, condition: str, bootstrap: int, rng: np.random.Generator
) -> list[dict]:
    labels = paired["label"].to_numpy(int)
    sources = paired["source"].to_numpy(str)
    current = paired["anomaly_score"].to_numpy(float)
    clean = paired["clean_anomaly_score"].to_numpy(float)
    clusters = _proxy_cluster_ids(paired)
    weights = _cluster_weights(labels, sources, clusters, bootstrap, rng)
    source_names = sorted(set(sources[labels == 1]))
    if not (labels == 0).any() or not source_names:
        return []
    result, current_by_source, clean_by_source = [], [], []
    audit = _cluster_audit(paired)
    for group, mask in [("pooled", np.ones(len(labels), dtype=bool))] + [
        (source, (labels == 0) | (sources == source)) for source in source_names
    ]:
        point = _weighted_auroc(labels[mask], current[mask], np.ones((1, mask.sum())))[0]
        clean_point = _weighted_auroc(labels[mask], clean[mask], np.ones((1, mask.sum())))[0]
        current_boot = _weighted_auroc(labels[mask], current[mask], weights[:, mask])
        clean_boot = _weighted_auroc(labels[mask], clean[mask], weights[:, mask])
        row = dict(
            condition=condition,
            group=group,
            n_real=int(((labels == 0) & mask).sum()),
            n_fake=int(((labels == 1) & mask).sum()),
            n_sources=len(source_names) if group == "pooled" else 1,
            auroc=float(point),
            clean_auroc=float(clean_point),
            auroc_delta=float(point - clean_point),
            **audit,
        )
        _add_intervals(row, current_boot, clean_boot, bootstrap)
        result.append(row)
        if group != "pooled":
            current_by_source.append(current_boot)
            clean_by_source.append(clean_boot)
    # np.mean intentionally propagates a missing source's NaN. Dropping that
    # source would change which macro statistic a draw estimates.
    macro_current = np.mean(current_by_source, axis=0)
    macro_clean = np.mean(clean_by_source, axis=0)
    macro = dict(
        condition=condition,
        group="macro",
        n_real=int((labels == 0).sum()),
        n_fake=int((labels == 1).sum()),
        n_sources=len(source_names),
        **{
            name: float(np.mean([row[name] for row in result[1:]]))
            for name in ("auroc", "clean_auroc", "auroc_delta")
        },
        **audit,
    )
    _add_intervals(macro, macro_current, macro_clean, bootstrap)
    result.append(macro)
    return result


def _add_intervals(row: dict, current: np.ndarray, clean: np.ndarray, requested: int) -> None:
    valid = np.isfinite(current) & np.isfinite(clean)
    row["bootstrap_draws_requested"] = requested
    row["bootstrap_draws_valid"] = int(valid.sum())
    row["bootstrap_draws_missing_class_or_source"] = int((~valid).sum())
    for prefix, values in (
        ("auroc", current[valid]),
        ("clean_auroc", clean[valid]),
        ("auroc_delta", (current - clean)[valid]),
    ):
        row[prefix + "_ci_low"], row[prefix + "_ci_high"] = _interval(values)


def analyze_content_sensitivity(
    run_dir: str | Path, bootstrap: int = 1000, seed: int = 20261108
) -> dict:
    """Write only content_sensitivity CSV/JSON/Markdown; never change main results."""
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    run_dir = Path(run_dir)
    progress = run_dir / "progress.json"
    if (
        progress.is_file()
        and json.loads(progress.read_text(encoding="utf-8-sig")).get("state") == "running"
    ):
        raise ValueError(
            "Wait for the scoring phase to stop before creating a sensitivity snapshot"
        )
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8-sig"))
    frame, warnings = _read_records(run_dir / "records.jsonl")
    manifest = pd.DataFrame(metadata["selected_manifest"])
    if manifest["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in selected manifest")
    provenance = manifest.set_index("image_id")
    for field in ("archive_member", "dataset"):
        frame[field] = frame["image_id"].map(provenance[field]) if field in provenance else None
    test = frame.loc[frame["split"] == "test"].copy()
    clean = test.loc[test["condition"] == "clean"].copy()
    if clean.empty:
        raise ValueError("No clean test originals")
    expected_conditions = _conditions(frame, metadata)
    clean_reference = clean[["image_id", "anomaly_score"]].rename(
        columns={"anomaly_score": "clean_anomaly_score"}
    )
    rng = np.random.default_rng(seed)
    output, coverage = [], []
    expected_pairs = metadata.get("expected_image_conditions", {})
    declared_expectations = isinstance(expected_pairs, dict) and bool(expected_pairs)
    if not declared_expectations:
        expected_pairs = {}
        warnings.append(
            "No expected-image condition plan is available; completeness describes observed pairs only."
        )
    test_ids = set(manifest.loc[manifest["split"] == "test", "image_id"])
    for condition in expected_conditions:
        current = test.loc[test["condition"] == condition]
        paired = current.merge(clean_reference, on="image_id", how="inner", validate="one_to_one")
        expected = {
            identifier for identifier in test_ids if condition in expected_pairs.get(identifier, [])
        }
        missing = sorted(expected - set(paired["image_id"]))
        coverage.append(
            dict(
                condition=condition,
                paired_test=len(paired),
                expected_test=len(expected),
                missing_expected_test=len(missing),
            )
        )
        if missing:
            warnings.append(
                f"{condition}: {len(missing)} expected test originals lack a clean/processed pair."
            )
        if paired.empty or paired["label"].nunique() < 2:
            warnings.append(
                f"{condition}: insufficient paired real and fake scores for AUROC sensitivity."
            )
            continue
        output.extend(_condition_metrics(paired, condition, bootstrap, rng))
    metrics = pd.DataFrame(output)
    if metrics.empty:
        raise ValueError("No condition has both real and fake paired scores")
    main_path = run_dir / "metrics.csv"
    comparisons = []
    if main_path.is_file():
        main = pd.read_csv(main_path)
        if main.duplicated(["condition", "group"]).any():
            raise ValueError("Main metrics have ambiguous condition/group rows")
        keys = ["condition", "group"]
        values = [
            name
            for name in (
                "auroc",
                "clean_auroc",
                "auroc_delta",
                "auroc_ci_low",
                "auroc_ci_high",
                "auroc_delta_ci_low",
                "auroc_delta_ci_high",
            )
            if name in main
        ]
        metrics = metrics.merge(
            main[keys + values].rename(columns={name: "main_" + name for name in values}),
            on=keys,
            how="left",
            validate="one_to_one",
        )
        for name in ("auroc", "clean_auroc", "auroc_delta"):
            if "main_" + name in metrics:
                comparable = metrics[name].notna() & metrics["main_" + name].notna()
                if not np.allclose(
                    metrics.loc[comparable, name],
                    metrics.loc[comparable, "main_" + name],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Sensitivity point estimate differs from main {name}; check matched cohorts"
                    )
                comparisons.append(name)
    invalid = int(metrics["bootstrap_draws_missing_class_or_source"].max())
    if invalid:
        warnings.append(
            f"Up to {invalid}/{bootstrap} cluster draws lack a required source/class; intervals use valid draws only."
        )
    summary = {
        "status": (
            "complete_for_observed_test_pairs_only"
            if not declared_expectations
            else (
                "incomplete"
                if any(row["missing_expected_test"] for row in coverage)
                else "complete_for_expected_test_pairs"
            )
        ),
        "analysis": "supplementary_archive_stem_proxy_cluster_bootstrap",
        "bootstrap_draws": bootstrap,
        "seed": seed,
        "cluster_definition": "Synthbuster dataset + archive-member filename stem, shared across generator directories; other original IDs are independent",
        "cluster_identity_confirmed": False,
        "real_resampling": "original IDs independently within each real-source stratum",
        "real_fake_content_dependence_modeled": False,
        "fake_resampling": "sample the observed number of fake proxy clusters with replacement; shared multiplicity across all images in a cluster",
        "fake_source_counts_fixed": False,
        "macro_weighting": "equal generator AUROC in every valid draw",
        "paired_clean_processed_weights": True,
        "main_analysis_unchanged": True,
        "main_point_estimates_verified": comparisons,
        "records_sha256": hashlib.sha256((run_dir / "records.jsonl").read_bytes()).hexdigest(),
        "test_real_source_counts": dict(Counter(clean.loc[clean["label"] == 0, "source"])),
        **_cluster_audit(clean),
        "coverage": coverage,
        "warnings": warnings,
    }
    metrics.to_csv(run_dir / "content_sensitivity_metrics.csv", index=False)
    (run_dir / "content_sensitivity_summary.json").write_text(
        json.dumps(_clean_json(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    macro = metrics.loc[metrics["group"] == "macro"]
    columns = {
        "condition": "Condition",
        "auroc": "Macro AUROC",
        "auroc_delta": "Paired change",
        "main_auroc_delta_ci_low": "Main lower",
        "main_auroc_delta_ci_high": "Main upper",
        "auroc_delta_ci_low": "Proxy-cluster lower",
        "auroc_delta_ci_high": "Proxy-cluster upper",
        "bootstrap_draws_valid": "Valid draws",
    }
    report = [
        "# Supplementary content-proxy sensitivity",
        "",
        f"Status: **{summary['status']}**.",
        "",
        f"The clean fake cohort contains {summary['n_fake_originals']} distinct original images and "
        f"{summary['n_fake_proxy_clusters']} proxy clusters. "
        f"{summary['n_cross_generator_shared_proxy_clusters']} stems are shared across generators "
        f"({summary['n_fake_in_cross_generator_shared_proxy_clusters']} images).",
        "",
        "Archive stems may identify shared reference content, but they are not verified prompt IDs. "
        "Distinct image bytes do not establish independent content. This is a dependence-assumption "
        "sensitivity analysis, not a replacement for the main source-stratified bootstrap.",
        "",
        f"Each of {bootstrap} draws resamples fake proxy clusters with replacement and assigns the same "
        "multiplicity to every image in the selected cluster. Real originals are resampled within each "
        "real source. Clean and processed scores reuse identical weights; generator AUROCs are averaged "
        "equally. Fake counts per generator vary in this supplement, unlike the main bootstrap. "
        "Differences between intervals therefore reflect the resampling-design change as well as "
        "potential cross-generator dependence. A missing generator makes the macro draw invalid; "
        "the number of valid draws is disclosed. Intervals are pointwise 95% percentiles, without "
        "multiple-comparison adjustment.",
        "",
        "No real-to-fake content correspondence is inferred or modeled. If a future real-image cohort "
        "contains reference images used to generate the synthetic cohort, those relationships require "
        "verified identifiers and a joint real/fake cluster design.",
        "",
        "Point estimates retain the main image-based, equally weighted generator estimand; no scores, "
        "thresholds, checkpoint parameters, or main confidence intervals are changed.",
        "",
        _table(macro, columns),
        "",
        "[Per-generator, pooled, and macro sensitivity metrics](content_sensitivity_metrics.csv) | "
        "[Cluster definitions, counts, coverage, and reproducibility](content_sensitivity_summary.json)",
        "",
        "Interpret interval-width changes descriptively. The proxy can miss semantically related images "
        "with different stems, and identical stems are not confirmed identical prompts. A stronger future "
        "study would sample verified reference-content or prompt IDs jointly across generators, then "
        "bootstrap those verified groups.",
    ]
    if warnings:
        report.extend(["", "Warnings:", ""] + [f"- {warning}" for warning in warnings])
    (run_dir / "content_sensitivity.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return _clean_json(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    print(
        json.dumps(analyze_content_sensitivity(args.run_dir, args.bootstrap, args.seed), indent=2)
    )


if __name__ == "__main__":
    main()
