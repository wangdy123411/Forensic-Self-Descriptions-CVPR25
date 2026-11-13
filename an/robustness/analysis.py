"""Paired, source-stratified robustness analysis of released FSD predictions.

Only clean real calibration records select the operating threshold. Test rows
never select thresholds. Bootstrap samples retain each label/source count and
reuse the same image weights for clean and processed scores.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

NUMERIC_COLUMNS = (
    "label",
    "severity",
    "z_score",
    "anomaly_score",
    "raw_score",
    "attribution_confidence",
    "fsd_l2_relative",
    "fsd_cosine_distance",
    "projected_l2_relative",
    "projected_cosine_distance",
    "residual_rms",
    "residual_rms_ratio",
    "seconds",
    "peak_cuda_mb",
    "raw_residual_rms",
    "native_residual_patch_hf_fraction",
    *(f"raw_residual_rms_ch{channel}" for channel in range(8)),
    *(f"residual_rms_ch{channel}" for channel in range(8)),
    *(f"native_hf_fraction_ch{channel}" for channel in range(8)),
)
DRIFT_COLUMNS = (
    "fsd_l2_relative",
    "fsd_cosine_distance",
    "projected_l2_relative",
    "projected_cosine_distance",
    "residual_rms",
    "residual_rms_ratio",
    "raw_residual_rms",
    "raw_residual_rms_ratio",
    "native_residual_patch_hf_fraction",
    "native_hf_fraction_delta",
    *(f"raw_residual_rms_ch{channel}" for channel in range(8)),
    *(f"residual_rms_ch{channel}" for channel in range(8)),
    *(f"native_hf_fraction_ch{channel}" for channel in range(8)),
)


def _read_records(path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings, rows = [], []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"Skipped malformed/incomplete JSONL line {number}.")
                continue
            if not isinstance(row, dict) or not all(
                key in row for key in ("image_id", "label", "source", "split", "condition")
            ):
                warnings.append(f"Skipped line {number}: missing record identifiers.")
                continue
            diagnostics = row.get("diagnostics", {})
            if isinstance(diagnostics, dict):
                for key in ("raw_residual_rms", "native_residual_patch_hf_fraction"):
                    if key in diagnostics:
                        row[key] = diagnostics[key]
                for key, prefix in (
                    ("raw_residual_rms_channels", "raw_residual_rms"),
                    ("residual_rms_channels", "residual_rms"),
                    ("native_residual_patch_hf_fraction_channels", "native_hf_fraction"),
                ):
                    values = diagnostics.get(key, [])
                    if isinstance(values, list):
                        for channel, value in enumerate(values[:8]):
                            row[f"{prefix}_ch{channel}"] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable records in {path}")
    frame = pd.DataFrame(rows)
    for name in NUMERIC_COLUMNS:
        frame[name] = pd.to_numeric(frame.get(name, np.nan), errors="coerce")
    missing_score = frame["anomaly_score"].isna()
    frame.loc[missing_score, "anomaly_score"] = -frame.loc[missing_score, "z_score"]
    frame["z_score"] = -frame["anomaly_score"]
    valid = np.isfinite(frame["anomaly_score"]) & frame["label"].isin([0, 1])
    if (~valid).any():
        warnings.append(
            f"Excluded {int((~valid).sum())} records without finite scores or binary labels."
        )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No finite scored records are available yet.")
    for name in ("image_id", "source", "split", "condition"):
        frame[name] = frame[name].astype(str)
    frame["label"] = frame["label"].astype(int)
    if "family" not in frame:
        frame["family"] = frame["condition"]
    frame["family"] = frame["family"].fillna(frame["condition"]).astype(str)
    for name in ("attribution_pred", "attribution_loglik"):
        if name not in frame:
            frame[name] = None
    inconsistent = frame.groupby("image_id")[["label", "source", "split"]].nunique().max(axis=1) > 1
    if inconsistent.any():
        raise ValueError(
            f"Image IDs have inconsistent labels/source/split: {inconsistent[inconsistent].index[:5].tolist()}"
        )
    duplicate = frame.duplicated(["image_id", "condition"], keep="last")
    if duplicate.any():
        warnings.append(
            f"Used the last finite record for {int(duplicate.sum())} duplicate image/condition rows."
        )
        frame = frame.loc[~duplicate].copy()
    return frame, warnings


def _metadata(run_dir: Path) -> dict:
    result = {}
    for filename in (
        "config.json",
        "run_config.json",
        "metadata.json",
        "run_metadata.json",
        "run.json",
    ):
        path = run_dir / filename
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(value, dict):
                    result.update(value)
            except (OSError, json.JSONDecodeError):
                pass
    return result


def _conditions(frame: pd.DataFrame, metadata: dict) -> list[str]:
    requested = metadata.get("conditions", metadata.get("expected_conditions", []))
    if isinstance(requested, dict):
        requested = list(requested)
    names = []
    for item in requested if isinstance(requested, list) else []:
        value = item.get("name", item.get("condition")) if isinstance(item, dict) else item
        if isinstance(value, str) and value not in names:
            names.append(value)
    observed = frame[["condition", "family", "severity"]].drop_duplicates("condition")
    observed = observed.sort_values(["family", "severity", "condition"], na_position="last")
    names.extend(name for name in observed["condition"] if name not in names)
    return (["clean"] if "clean" in names else []) + [name for name in names if name != "clean"]


def _point_metrics(y: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict[str, float]:
    real, fake = y == 0, y == 1
    both = real.any() and fake.any()
    out = {
        "n_real": int(real.sum()),
        "n_fake": int(fake.sum()),
        "auroc": float(roc_auc_score(y, scores)) if both else np.nan,
        "average_precision": float(average_precision_score(y, scores)) if both else np.nan,
    }
    for prefix, cutoff in (("released", 2.0), ("calibrated", threshold)):
        if cutoff is None or not np.isfinite(cutoff):
            tpr = fpr = np.nan
        else:
            prediction = scores > cutoff
            tpr = float(prediction[fake].mean()) if fake.any() else np.nan
            fpr = float(prediction[real].mean()) if real.any() else np.nan
        out[f"{prefix}_tpr"] = tpr
        out[f"{prefix}_fpr"] = fpr
        out[f"{prefix}_balanced_accuracy"] = (tpr + 1 - fpr) / 2
    return out


def _resample_weights(
    y: np.ndarray, source: np.ndarray, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """One multiplicity per original ID, separately resampled in each stratum."""
    weights = np.zeros((draws, len(y)), dtype=np.float64)
    strata = np.array([f"{label}\0{name}" for label, name in zip(y, source)])
    for stratum in np.unique(strata):
        index = np.flatnonzero(strata == stratum)
        weights[:, index] = rng.multinomial(
            len(index), np.full(len(index), 1 / len(index)), size=draws
        )
    return weights


def _weighted_auroc(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact weighted Mann–Whitney AUROC; ties receive half credit.

    Vectorizing bootstrap multiplicities avoids repeatedly sorting the same
    score arrays and produces the same metric as explicit resampled IDs.
    """
    if weights.shape[0] == 0:
        return np.empty(0)
    order = np.argsort(score, kind="stable")
    y, score, weights = y[order], score[order], weights[:, order]
    start = np.r_[0, np.flatnonzero(np.diff(score) != 0) + 1]
    positive = np.add.reduceat(weights * (y == 1), start, axis=1)
    negative = np.add.reduceat(weights * (y == 0), start, axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    numerator = (positive * (negative.cumsum(axis=1) - 0.5 * negative)).sum(axis=1)
    return np.divide(
        numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0
    )


def _interval(value: np.ndarray) -> tuple[float, float]:
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return np.nan, np.nan
    low, high = np.quantile(value, [0.025, 0.975])
    return float(low), float(high)


def _cohort_metrics(
    paired: pd.DataFrame,
    condition: str,
    threshold: float | None,
    bootstrap: int,
    rng: np.random.Generator,
    cohort: str,
) -> list[dict]:
    if paired.empty:
        return []
    y = paired["label"].to_numpy(int)
    source = paired["source"].to_numpy(str)
    scores = paired["anomaly_score"].to_numpy(float)
    clean = paired["clean_anomaly_score"].to_numpy(float)
    weights = _resample_weights(y, source, bootstrap, rng) if bootstrap else np.empty((0, len(y)))
    fake_sources = sorted(set(source[y == 1]))
    selections = [("pooled", np.ones(len(y), dtype=bool))]
    selections.extend((name, (y == 0) | (source == name)) for name in fake_sources)
    output, per_source_boot = [], []
    for group, use in selections:
        metrics = _point_metrics(y[use], scores[use], threshold)
        baseline = _point_metrics(y[use], clean[use], threshold)
        row = {
            "cohort": cohort,
            "condition": condition,
            "group": group,
            **metrics,
            **{
                f"clean_{key}": val
                for key, val in baseline.items()
                if key not in {"n_real", "n_fake"}
            },
            "auroc_delta": metrics["auroc"] - baseline["auroc"],
            "n_sources": len(fake_sources) if group == "pooled" else 1,
        }
        for prefix, cutoff in (("released", 2.0), ("calibrated", threshold)):
            for label_name, label_mask in (
                ("", np.ones(use.sum(), dtype=bool)),
                ("real_", y[use] == 0),
                ("fake_", y[use] == 1),
            ):
                row[f"{prefix}_{label_name}prediction_flip_rate"] = (
                    float(
                        (
                            (scores[use][label_mask] > cutoff) != (clean[use][label_mask] > cutoff)
                        ).mean()
                    )
                    if cutoff is not None and np.isfinite(cutoff) and label_mask.any()
                    else np.nan
                )
        current_boot = _weighted_auroc(y[use], scores[use], weights[:, use])
        clean_boot = _weighted_auroc(y[use], clean[use], weights[:, use])
        for prefix, values in (
            ("auroc", current_boot),
            ("clean_auroc", clean_boot),
            ("auroc_delta", current_boot - clean_boot),
        ):
            row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"] = _interval(values)
        output.append(row)
        if group != "pooled":
            per_source_boot.append((current_boot, clean_boot))
    if fake_sources:
        items = output[1:]
        numeric = [
            key
            for key in items[0]
            if key not in {"cohort", "condition", "group", "n_real", "n_fake", "n_sources"}
        ]
        macro = {
            "cohort": cohort,
            "condition": condition,
            "group": "macro",
            "n_real": int((y == 0).sum()),
            "n_fake": int((y == 1).sum()),
            "n_sources": len(fake_sources),
        }
        # Macro point estimates average generators equally. CI endpoints must
        # instead come from the jointly resampled macro statistic.
        for key in numeric:
            values = np.asarray([item[key] for item in items], dtype=float)
            macro[key] = float(np.mean(values))
        current_boot = np.mean([item[0] for item in per_source_boot], axis=0)
        clean_boot = np.mean([item[1] for item in per_source_boot], axis=0)
        for prefix, values in (
            ("auroc", current_boot),
            ("clean_auroc", clean_boot),
            ("auroc_delta", current_boot - clean_boot),
        ):
            macro[f"{prefix}_ci_low"], macro[f"{prefix}_ci_high"] = _interval(values)
        output.append(macro)
    return output


def _real_domain_metrics(
    pairs: list[pd.DataFrame],
    conditions: list[str],
    real_sources: list[str],
    threshold: float | None,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Repeat detection comparisons separately for each real-image domain.

    All domains retain the same released and pooled-clean-calibration cutoffs.
    Domain-specific threshold tuning would answer a different question.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for condition, paired in zip(conditions, pairs):
        for real_source in real_sources:
            subset = paired.loc[
                (paired["label"] == 1)
                | ((paired["label"] == 0) & (paired["source"] == real_source))
            ]
            for row in _cohort_metrics(
                subset, condition, threshold, bootstrap, rng, "condition_paired"
            ):
                rows.append({"real_domain": real_source, **row})
    return pd.DataFrame(rows)


def _input_domain_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Describe the source/encoding mix without making provenance guesses."""
    data = clean.copy()
    encoding = pd.Series("unknown", index=data.index, dtype="str")
    if "path" in data:
        encoding = data["path"].map(
            lambda value: Path(str(value)).suffix.lower().lstrip(".") or "unknown"
        )
    for key in ("format", "image_format", "original_format"):
        if key in data:
            valid = data[key].notna() & (data[key].astype(str).str.len() > 0)
            encoding.loc[valid] = data.loc[valid, key].astype(str).str.lower()
    data["input_encoding"] = encoding.replace({"jpg": "jpeg", "tif": "tiff"})
    for key in ("width", "height"):
        data[key] = pd.to_numeric(data.get(key, np.nan), errors="coerce")
    return (
        data.groupby(["source", "label", "input_encoding"], dropna=False)
        .agg(
            originals=("image_id", "size"),
            median_width=("width", "median"),
            median_height=("height", "median"),
        )
        .reset_index()
    )


def _paired_rows(test: pd.DataFrame, clean: pd.DataFrame, condition: str) -> pd.DataFrame:
    current = test.loc[test["condition"] == condition].copy()
    columns = [
        "image_id",
        "anomaly_score",
        "attribution_pred",
        "attribution_confidence",
        "raw_residual_rms",
        "native_residual_patch_hf_fraction",
    ]
    reference = clean[columns].rename(
        columns={name: f"clean_{name}" for name in columns if name != "image_id"}
    )
    result = current.merge(reference, on="image_id", how="inner", validate="one_to_one")
    result["raw_residual_rms_ratio"] = result["raw_residual_rms"] / result[
        "clean_raw_residual_rms"
    ].clip(lower=1e-30)
    result["native_hf_fraction_delta"] = (
        result["native_residual_patch_hf_fraction"]
        - result["clean_native_residual_patch_hf_fraction"]
    )
    return result


def _source_mapping(
    run_dir: Path, supplied: dict | str | Path | None, frame: pd.DataFrame
) -> dict[str, str]:
    if isinstance(supplied, dict):
        value = supplied
    else:
        path = Path(supplied) if supplied else run_dir / "source_mapping.json"
        value = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    if "source_mapping" in value and isinstance(value["source_mapping"], dict):
        value = value["source_mapping"]
    mapping = {str(key): str(val) for key, val in value.items() if isinstance(val, str) and val}
    if not mapping:
        # Exact names only; guesses between versioned generators are unsafe.
        available = set()
        for item in frame["attribution_loglik"]:
            if isinstance(item, dict):
                available.update(item)
        mapping = {source: source for source in set(frame["source"]) & available}
    return mapping


def _attribution(
    paired: pd.DataFrame, mapping: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame()
    fake = paired.loc[paired["label"] == 1].copy()
    fake["mapped_source"] = fake["source"].map(mapping)
    known = fake.loc[
        fake["mapped_source"].notna()
        & fake["attribution_pred"].notna()
        & fake["clean_attribution_pred"].notna()
    ].copy()
    records = []
    for condition, subset in known.groupby("condition", sort=False):
        source_rows = []
        for source, data in subset.groupby("source"):
            current = (data["attribution_pred"] == data["mapped_source"]).mean()
            clean = (data["clean_attribution_pred"] == data["mapped_source"]).mean()
            row = {
                "condition": condition,
                "source": source,
                "n": len(data),
                "accuracy": current,
                "clean_accuracy": clean,
                "accuracy_delta": current - clean,
                "prediction_flip_rate": (
                    data["attribution_pred"] != data["clean_attribution_pred"]
                ).mean(),
                "mean_confidence": data["attribution_confidence"].mean(),
            }
            records.append(row)
            source_rows.append(row)
        for group, equal in (("pooled", False), ("macro", True)):
            weights = (
                np.ones(len(source_rows)) if equal else np.array([row["n"] for row in source_rows])
            )
            records.append(
                {
                    "condition": condition,
                    "source": group,
                    "n": len(subset),
                    **{
                        key: np.average([row[key] for row in source_rows], weights=weights)
                        for key in (
                            "accuracy",
                            "clean_accuracy",
                            "accuracy_delta",
                            "prediction_flip_rate",
                            "mean_confidence",
                        )
                    },
                }
            )
    return pd.DataFrame(records), known


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(val) for key, val in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"
    return str(value)


def _table(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    header = "| " + " | ".join(columns.values()) + " |\n"
    header += "| " + " | ".join("---" for _ in columns) + " |\n"
    return header + "\n".join(
        "| " + " | ".join(_fmt(row.get(key, "n/a")) for key in columns) + " |"
        for row in frame.to_dict("records")
    )


def _draw_pointwise_intervals(
    ax,
    center: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    positions: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Render actual percentile endpoints, even when they exclude the estimate."""
    valid = np.isfinite(low) & np.isfinite(high) & (low <= high)
    ax.hlines(positions[valid], low[valid], high[valid], color=color, linewidth=1.2)
    ax.plot(center, positions, "o", markersize=4, label=label, color=color)


def _plot_results(
    output: Path,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    conditions: list[str],
    threshold: float | None,
    attribution_known: pd.DataFrame,
    real_domains: pd.DataFrame | None = None,
) -> list[str]:
    # Backend selection happens only when plotting is requested, not on import.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {0: "#2166ac", 1: "#b2182b"}

    def save(fig, name):
        fig.savefig(output / name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(f"plots/{name}")

    valid_conditions = [name for name in conditions if name in set(metrics["condition"])]
    macro = (
        metrics.loc[metrics["group"] == "macro"].set_index("condition").reindex(valid_conditions)
    )
    pooled = (
        metrics.loc[metrics["group"] == "pooled"].set_index("condition").reindex(valid_conditions)
    )
    if not macro.empty:
        y = np.arange(len(valid_conditions))
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(y) * 0.29)), sharey=True)
        for data, label, offset, color in (
            (macro, "Macro over generators", -0.12, "#087f8c"),
            (pooled, "Pooled images", 0.12, "#a35c32"),
        ):
            for ax, column in zip(axes, ("auroc", "auroc_delta")):
                center = data[column].to_numpy(float)
                low, high = data[f"{column}_ci_low"].to_numpy(float), data[
                    f"{column}_ci_high"
                ].to_numpy(float)
                _draw_pointwise_intervals(ax, center, low, high, y + offset, label, color)
        axes[0].set(yticks=y, yticklabels=valid_conditions, xlabel="AUROC", xlim=(0, 1.02))
        axes[0].invert_yaxis()
        axes[1].set_xlabel("AUROC change vs the same clean images")
        axes[1].axvline(0, color="0.5", lw=1)
        for ax in axes:
            ax.grid(axis="x", alpha=0.2)
        axes[0].legend(loc="lower left", fontsize=8)
        fig.suptitle(
            "Detection robustness · paired test cohorts · 95% percentile bootstrap intervals"
        )
        fig.tight_layout()
        save(fig, "auroc_by_condition.png")

        fig, ax = plt.subplots(figsize=(max(9, len(valid_conditions) * 0.45), 4.5))
        x = np.arange(len(valid_conditions))
        for col, label, style, color in (
            ("released_tpr", "Synthetic TPR · release threshold", "o-", "#b2182b"),
            ("released_fpr", "Real FPR · release threshold", "s-", "#2166ac"),
            ("calibrated_tpr", "Synthetic TPR · clean calibration", "o--", "#dd826f"),
            ("calibrated_fpr", "Real FPR · clean calibration", "s--", "#71a3cc"),
        ):
            if macro[col].notna().any():
                ax.plot(x, macro[col], style, color=color, label=label, ms=4)
        ax.set(
            xticks=x, xticklabels=valid_conditions, ylim=(-0.02, 1.03), ylabel="Fraction of images"
        )
        ax.tick_params(axis="x", rotation=70)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
        ax.set_title("Frozen operating thresholds; TPR macro-averaged over generators", pad=50)
        fig.tight_layout()
        save(fig, "fixed_threshold_errors.png")

    source_metrics = metrics.loc[~metrics["group"].isin(["pooled", "macro"])]
    if not source_metrics.empty:
        matrix = source_metrics.pivot(index="group", columns="condition", values="auroc").reindex(
            columns=valid_conditions
        )
        fig, ax = plt.subplots(
            figsize=(max(9, len(valid_conditions) * 0.5), max(3, len(matrix) * 0.6))
        )
        plot = ax.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set(
            xticks=np.arange(len(matrix.columns)),
            xticklabels=matrix.columns,
            yticks=np.arange(len(matrix)),
            yticklabels=matrix.index,
        )
        ax.tick_params(axis="x", rotation=70)
        for row in range(len(matrix)):
            for col in range(len(matrix.columns)):
                val = matrix.iloc[row, col]
                if np.isfinite(val):
                    ax.text(
                        col,
                        row,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val < 0.55 else "black",
                    )
        ax.set_title("Each synthetic source versus all available paired real test images")
        fig.colorbar(plot, ax=ax, label="AUROC")
        fig.tight_layout()
        save(fig, "source_auroc_heatmap.png")

    if not paired.empty:
        fig, axes = plt.subplots(
            2, 1, figsize=(max(10, len(valid_conditions) * 0.5), 7), sharex=True
        )
        for label in (0, 1):
            subset = paired.loc[paired["label"] == label]
            summary = (
                subset.groupby("condition")[list(DRIFT_COLUMNS)].median().reindex(valid_conditions)
            )
            for ax, col in zip(axes, ("fsd_l2_relative", "residual_rms_ratio")):
                ax.plot(
                    np.arange(len(valid_conditions)),
                    summary[col],
                    "o-",
                    ms=4,
                    color=colors[label],
                    label="Real" if label == 0 else "Synthetic",
                )
        axes[0].set_ylabel("Median raw 960D FSD\nrelative L2 drift")
        axes[0].set_yscale("symlog", linthresh=0.1)
        axes[1].set_ylabel("Median residual RMS ratio\nto the clean original")
        axes[1].axhline(1, color=".5", lw=1)
        axes[1].set(xticks=np.arange(len(valid_conditions)), xticklabels=valid_conditions)
        axes[1].tick_params(axis="x", rotation=70)
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        axes[0].set_title("Feature and residual changes on matched originals")
        fig.tight_layout()
        save(fig, "feature_residual_drift.png")

        if paired["native_residual_patch_hf_fraction"].notna().any():
            fig, axes = plt.subplots(
                2, 1, figsize=(max(10, len(valid_conditions) * 0.5), 7), sharex=True
            )
            for label in (0, 1):
                subset = paired.loc[paired["label"] == label]
                stats = (
                    subset.groupby("condition")[
                        ["native_residual_patch_hf_fraction", "raw_residual_rms_ratio"]
                    ]
                    .median()
                    .reindex(valid_conditions)
                )
                for ax, col in zip(
                    axes, ("native_residual_patch_hf_fraction", "raw_residual_rms_ratio")
                ):
                    ax.plot(
                        np.arange(len(valid_conditions)),
                        stats[col],
                        "o-",
                        ms=4,
                        color=colors[label],
                        label="Real" if label == 0 else "Synthetic",
                    )
            axes[0].set_ylabel("Median native-patch\nhigh-frequency energy fraction")
            axes[1].set_ylabel("Median native RMS ratio\nto the clean original")
            axes[1].axhline(1, color=".5", lw=1)
            axes[1].set(xticks=np.arange(len(valid_conditions)), xticklabels=valid_conditions)
            axes[1].tick_params(axis="x", rotation=70)
            for ax in axes:
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
            axes[0].set_title(
                "Native residual diagnostics before model resizing · center patch spectrum"
            )
            fig.tight_layout()
            save(fig, "native_residual_spectrum.png")

        runtime = (
            paired.groupby("condition")
            .agg(mean_seconds=("seconds", "mean"), max_cuda_mb=("peak_cuda_mb", "max"))
            .reindex(valid_conditions)
        )
        fig, axes = plt.subplots(
            1, 2, figsize=(11, max(5, len(valid_conditions) * 0.27)), sharey=True
        )
        y = np.arange(len(valid_conditions))
        axes[0].barh(y, runtime["mean_seconds"], color="#54748c")
        axes[1].barh(y, runtime["max_cuda_mb"], color="#887297")
        axes[0].set(yticks=y, yticklabels=valid_conditions, xlabel="Mean seconds / image")
        axes[0].invert_yaxis()
        axes[1].set_xlabel("Peak allocated GPU tensor memory (MiB)")
        fig.suptitle("Costs for clean-paired test images; calibration excluded")
        fig.tight_layout()
        save(fig, "runtime_memory.png")

        # One most-severe completed condition per family keeps ROC curves legible.
        observed = paired[["condition", "family", "severity"]].drop_duplicates("condition")
        selected = ["clean"] if "clean" in set(observed["condition"]) else []
        for family, rows in observed.loc[observed["condition"] != "clean"].groupby("family"):
            selected.append(rows.sort_values(["severity", "condition"]).iloc[-1]["condition"])
        fig, ax = plt.subplots(figsize=(7, 6))
        for condition in selected:
            data = paired.loc[paired["condition"] == condition]
            if data["label"].nunique() == 2:
                fpr, tpr, _ = roc_curve(data["label"], data["anomaly_score"])
                ax.plot(fpr, tpr, label=f"{condition} (n={len(data)})", lw=1.6)
        ax.plot([0, 1], [0, 1], "--", color=".65", lw=1)
        ax.set(
            xlabel="Real false-positive rate",
            ylabel="Synthetic true-positive rate",
            xlim=(0, 1),
            ylim=(0, 1.01),
            title="Pooled ROC curves · highest available severity per family",
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        save(fig, "roc_curves.png")

    if not attribution_known.empty:
        clean_data = attribution_known.loc[attribution_known["condition"] == "clean"]
        if not clean_data.empty:
            matrix = pd.crosstab(clean_data["mapped_source"], clean_data["attribution_pred"])
            normalized = matrix.div(matrix.sum(axis=1), axis=0)
            fig, ax = plt.subplots(
                figsize=(max(6, len(matrix.columns) * 0.7), max(3, len(matrix) * 0.6))
            )
            plot = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set(
                xticks=np.arange(len(matrix.columns)),
                xticklabels=matrix.columns,
                yticks=np.arange(len(matrix)),
                yticklabels=matrix.index,
                xlabel="Released classifier prediction",
                ylabel="Known test source",
                title="Clean source attribution · row-normalized · no unknown rejection",
            )
            ax.tick_params(axis="x", rotation=65)
            fig.colorbar(plot, ax=ax, label="Fraction")
            fig.tight_layout()
            save(fig, "attribution_clean_confusion.png")
    if real_domains is not None and not real_domains.empty:
        values = real_domains.loc[real_domains["group"] == "macro"]
        if not values.empty:
            fig, axes = plt.subplots(
                3, 1, figsize=(max(10, len(valid_conditions) * 0.5), 9), sharex=True
            )
            x = np.arange(len(valid_conditions))
            for source, group in values.groupby("real_domain", sort=True):
                group = group.set_index("condition").reindex(valid_conditions)
                axes[0].plot(x, group["auroc"], "o-", ms=3, label=source)
                axes[0].fill_between(
                    x,
                    group["auroc_ci_low"].to_numpy(float),
                    group["auroc_ci_high"].to_numpy(float),
                    alpha=0.12,
                )
                axes[1].plot(x, group["released_fpr"], "o-", ms=3, label=source)
                axes[2].plot(x, group["calibrated_fpr"], "o-", ms=3, label=source)
            for ax in axes:
                ax.set_ylim(-0.02, 1.02)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8, loc="best")
            axes[0].set_ylabel("Generator-macro AUROC")
            axes[0].set_title(
                "Real domains analyzed separately · AUROC bands are 95% bootstrap intervals"
            )
            axes[1].set_ylabel("Real FPR\nreleased threshold")
            axes[2].set_ylabel("Real FPR\nshared clean-calibrated threshold")
            axes[2].set(xticks=x, xticklabels=valid_conditions)
            axes[2].tick_params(axis="x", rotation=70)
            fig.tight_layout()
            save(fig, "real_domain_comparison.png")
    return paths


def analyze(
    run_dir: str | Path,
    bootstrap: int = 1000,
    *,
    seed: int = 20261108,
    source_mapping: dict | str | Path | None = None,
) -> dict:
    """Write analysis CSVs, PNG figures, JSON summary, and report.md into run_dir.

    Files are snapshots: calling again after a resumed run refreshes all outputs.
    Condition-specific comparisons use only IDs also scored clean. A separate
    common-cohort table compares exactly the same IDs across every requested
    condition. An absent requested condition makes that common cohort empty.
    """
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
    )
    run_dir = Path(run_dir).resolve()
    frame, warnings = _read_records(run_dir / "records.jsonl")
    metadata = _metadata(run_dir)
    conditions = _conditions(frame, metadata)
    test = frame.loc[frame["split"] == "test"].copy()
    clean = test.loc[test["condition"] == "clean"].copy()
    if clean.empty:
        raise ValueError("No clean test records; paired analysis cannot run yet.")
    calibration = frame.loc[
        (frame["split"] == "calibration") & (frame["condition"] == "clean") & (frame["label"] == 0)
    ]
    calibration_threshold = (
        float(np.quantile(calibration["anomaly_score"], 0.95, method="higher"))
        if len(calibration)
        else None
    )
    if len(calibration) < 100:
        warnings.append(
            f"Only {len(calibration)} clean real calibration images: a nominal 5% FPR threshold has substantial sampling uncertainty."
        )
    if calibration_threshold is None:
        warnings.append(
            "Calibrated operating-point metrics are unavailable; no clean real calibration records."
        )
    # Never expose calibration images to any test metric or bootstrap.
    rng = np.random.default_rng(seed)
    metrics_rows, coverage_rows, pairs = [], [], []
    common_ids = set(clean["image_id"])
    manifest = metadata.get("selected_manifest", [])
    expected_test = metadata.get("test_image_ids")
    if not isinstance(expected_test, list) and isinstance(manifest, list):
        expected_test = [
            str(row["image_id"])
            for row in manifest
            if isinstance(row, dict) and row.get("split") == "test" and "image_id" in row
        ]
    expected_test_ids = (
        set(map(str, expected_test)) if isinstance(expected_test, list) and expected_test else None
    )
    expected_count = len(expected_test_ids) if expected_test_ids is not None else len(clean)
    for condition in conditions:
        paired = _paired_rows(test, clean, condition)
        pairs.append(paired)
        current = test.loc[test["condition"] == condition]
        ids = set(paired["image_id"])
        common_ids &= ids
        coverage_rows.append(
            {
                "condition": condition,
                "scored_test": len(current),
                "paired_test": len(paired),
                "clean_test_available": len(clean),
                "missing_vs_clean": len(clean) - len(paired),
                "expected_test": expected_count,
                "missing_expected_test": (
                    expected_count - len(set(current["image_id"]) & expected_test_ids)
                    if expected_test_ids is not None
                    else np.nan
                ),
                "unpaired_processed": len(current) - len(paired),
                "n_real": int((paired["label"] == 0).sum()),
                "n_fake": int((paired["label"] == 1).sum()),
            }
        )
        metrics_rows.extend(
            _cohort_metrics(
                paired, condition, calibration_threshold, bootstrap, rng, "condition_paired"
            )
        )
    paired_all = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    metrics = pd.DataFrame(metrics_rows)
    coverage = pd.DataFrame(coverage_rows)
    common_rows = []
    for condition, paired in zip(conditions, pairs):
        subset = paired.loc[paired["image_id"].isin(common_ids)]
        common_rows.extend(
            _cohort_metrics(
                subset, condition, calibration_threshold, 0, rng, "all_conditions_common"
            )
        )
    common_metrics = pd.DataFrame(common_rows, columns=metrics.columns)
    real_sources = sorted(clean.loc[clean["label"] == 0, "source"].unique())
    real_domain_metrics = _real_domain_metrics(
        pairs, conditions, real_sources, calibration_threshold, bootstrap, seed + 1
    )
    input_domains = _input_domain_summary(clean)
    incomplete = bool((coverage["missing_vs_clean"] > 0).any())
    if incomplete:
        warnings.append(
            "Some requested/observed conditions lack clean-paired test images. Condition-specific cohorts differ; consult coverage.csv and common-cohort metrics."
        )
    requested_known = bool(metadata.get("conditions") or metadata.get("expected_conditions"))
    if not requested_known:
        warnings.append(
            "Expected-condition metadata is absent; wholly absent conditions cannot be detected. Coverage refers to observed conditions only."
        )
    if expected_test_ids is not None and set(clean["image_id"]) != expected_test_ids:
        warnings.append(
            f"Clean scoring covers {len(set(clean['image_id']) & expected_test_ids)} of {len(expected_test_ids)} expected test IDs; observed clean total is {len(clean)}."
        )
        incomplete = True
    expected_pairs = metadata.get("expected_image_conditions")
    missing_expected_pairs = None
    if isinstance(expected_pairs, dict):
        expected_set = {
            (str(image_id), str(condition))
            for image_id, names in expected_pairs.items()
            if isinstance(names, list)
            for condition in names
        }
        observed_set = set(zip(frame["image_id"], frame["condition"]))
        missing_expected_pairs = len(expected_set - observed_set)
        if missing_expected_pairs:
            warnings.append(
                f"{missing_expected_pairs} expected image/condition predictions are absent or invalid, including any calibration records."
            )
            incomplete = True

    mapping = _source_mapping(run_dir, source_mapping, frame)
    likelihood_names = set()
    for values in frame["attribution_loglik"]:
        if isinstance(values, dict):
            likelihood_names.update(values)
    if likelihood_names:
        unsupported = {
            key: value for key, value in mapping.items() if value not in likelihood_names
        }
        if unsupported:
            warnings.append(f"Excluded unsupported attribution label mappings: {unsupported}.")
            mapping = {key: value for key, value in mapping.items() if value in likelihood_names}
    attribution, known = _attribution(paired_all, mapping)
    if not mapping:
        warnings.append(
            "No unambiguous source-to-checkpoint mapping: source accuracy is not reported."
        )
    drift = paired_all.groupby(["condition", "source", "label"], sort=False)[
        list(DRIFT_COLUMNS)
    ].agg(["count", "median", "mean", "std"])
    drift.columns = [f"{name}_{stat}" for name, stat in drift.columns]
    drift = drift.reset_index()
    runtime = paired_all.groupby("condition", sort=False).agg(
        n=("image_id", "size"),
        mean_seconds=("seconds", "mean"),
        median_seconds=("seconds", "median"),
        max_cuda_mb=("peak_cuda_mb", "max"),
    )
    runtime_calibration = (
        frame.loc[frame["split"] == "calibration"]
        .groupby("condition", sort=False)
        .agg(
            n=("image_id", "size"),
            mean_seconds=("seconds", "mean"),
            median_seconds=("seconds", "median"),
            max_cuda_mb=("peak_cuda_mb", "max"),
        )
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    real_domain_metrics.to_csv(run_dir / "real_domain_metrics.csv", index=False)
    input_domains.to_csv(run_dir / "input_domain_summary.csv", index=False)
    common_metrics.to_csv(run_dir / "metrics_common_cohort.csv", index=False)
    coverage.to_csv(run_dir / "coverage.csv", index=False)
    drift.to_csv(run_dir / "feature_drift.csv", index=False)
    runtime.to_csv(run_dir / "runtime.csv")
    runtime_calibration.to_csv(run_dir / "runtime_calibration.csv")
    attribution.to_csv(run_dir / "attribution_metrics.csv", index=False)
    ci_columns = [
        key
        for key in metrics
        if key
        in {
            "cohort",
            "condition",
            "group",
            "n_real",
            "n_fake",
            "n_sources",
            "auroc",
            "clean_auroc",
            "auroc_delta",
        }
        or "_ci_" in key
    ]
    metrics[ci_columns].to_csv(run_dir / "auroc_intervals.csv", index=False)
    pred_summary = (
        paired_all.loc[paired_all["label"] == 1]
        .groupby(["condition", "source", "attribution_pred"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    pred_summary["known_source_mapping"] = pred_summary["source"].isin(mapping)
    pred_summary.to_csv(run_dir / "attribution_predictions.csv", index=False)
    figures = _plot_results(
        run_dir / "plots",
        metrics,
        paired_all,
        conditions,
        calibration_threshold,
        known,
        real_domain_metrics,
    )
    from .residual_gallery import build_residual_galleries

    try:
        figures.extend(build_residual_galleries(run_dir, max_sources=8))
    except (OSError, ValueError, KeyError) as exc:
        warnings.append(f"Residual example gallery could not be completed: {exc}")

    summary = {
        "run_dir": str(run_dir),
        "bootstrap_draws": bootstrap,
        "bootstrap_seed": seed,
        "real_domain_bootstrap_seed": seed + 1,
        "status": (
            "incomplete"
            if incomplete
            else (
                "complete_for_expected_conditions"
                if requested_known
                else "complete_for_observed_conditions_only"
            )
        ),
        "n_finite_records": len(frame),
        "n_clean_test": len(clean),
        "n_common_test": len(common_ids),
        "n_expected_test": expected_count,
        "missing_expected_image_conditions": missing_expected_pairs,
        "n_calibration_real_clean": len(calibration),
        "calibrated_anomaly_threshold": calibration_threshold,
        "calibrated_z_threshold": (
            -calibration_threshold if calibration_threshold is not None else None
        ),
        "calibration_empirical_fpr": (
            float((calibration["anomaly_score"] > calibration_threshold).mean())
            if calibration_threshold is not None
            else None
        ),
        "threshold_rule": "fake iff anomaly_score > threshold; clean calibration 95th percentile, method=higher",
        "released_z_threshold": -2.0,
        "source_mapping": mapping,
        "conditions": conditions,
        "real_domains": real_sources,
        "calibration_real_domains": calibration.groupby("source").size().to_dict(),
        "runtime_cohort": "condition-specific clean-paired test originals; calibration excluded",
        "runtime_timer_scope": "Per-condition image processing, feature extraction, scoring, compressed feature-cache writing, and drift calculation; excludes initial image decoding and later preview/record writing.",
        "warnings": warnings,
        "figures": figures,
        "macro_results": metrics.loc[metrics["group"] == "macro"].to_dict("records"),
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(_clean_json(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    report = _report(
        summary,
        metrics,
        common_metrics,
        coverage,
        drift,
        attribution,
        metadata,
        real_domain_metrics,
        input_domains,
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return _clean_json(summary)


def _report(
    summary: dict,
    metrics: pd.DataFrame,
    common: pd.DataFrame,
    coverage: pd.DataFrame,
    drift: pd.DataFrame,
    attribution: pd.DataFrame,
    metadata: dict,
    real_domains: pd.DataFrame,
    input_domains: pd.DataFrame,
) -> str:
    macro = metrics.loc[metrics["group"] == "macro"].copy()
    clean = macro.loc[macro["condition"] == "clean"]
    variants = macro.loc[(macro["condition"] != "clean") & np.isfinite(macro["auroc_delta"])]
    lines = [
        "# AN robustness experiment",
        "",
        f"Analysis status: **{summary['status']}**.",
        "",
        "This evaluates the released FSD checkpoint under deterministic image processing. It is not a reproduction of the complete CVPR 2025 benchmark, nor a measured social-media upload/download pipeline.",
        "",
        f"Scored clean test originals: **{summary['n_clean_test']}**. Originals present in every requested/observed condition: **{summary['n_common_test']}**. Finite prediction records: **{summary['n_finite_records']}**.",
        "",
    ]
    if not clean.empty:
        row = clean.iloc[0]
        lines.append(
            f"Clean macro AUROC: **{row['auroc']:.3f}** (95% interval {_fmt(row['auroc_ci_low'])}–{_fmt(row['auroc_ci_high'])}); {int(row['n_real'])} real and {int(row['n_fake'])} synthetic test images across {int(row['n_sources'])} synthetic sources."
        )
        lines.append("")
    if not variants.empty:
        worst = variants.sort_values("auroc_delta").iloc[0]
        best = variants.sort_values("auroc_delta", ascending=False).iloc[0]
        lines.extend(
            [
                f"The lowest observed macro AUROC change is **{worst['condition']}**: {_fmt(worst['auroc_delta'])} versus its matched clean cohort (95% paired interval {_fmt(worst['auroc_delta_ci_low'])}–{_fmt(worst['auroc_delta_ci_high'])}). The highest observed change is **{best['condition']}** ({_fmt(best['auroc_delta'])}). These are descriptive comparisons across this sweep; intervals are not corrected for multiple testing.",
                "",
            ]
        )
    lines.extend(
        [
            "## Detection results",
            "",
            "Macro values weight synthetic sources equally; real images are pooled in their observed proportions. Each condition uses only test originals with both clean and processed scores. The paired clean AUROC column exposes differences in cohort composition. Macro AP averages separate generator-versus-real comparisons; its class balance is the balance within each comparison, not the combined real/fake counts shown on the macro row. Pooled AP is reported separately in metrics.csv.",
            "",
            _table(
                macro,
                {
                    "condition": "Condition",
                    "n_real": "Real",
                    "n_fake": "Fake",
                    "auroc": "Macro AUROC",
                    "clean_auroc": "Matched clean",
                    "auroc_delta": "Change",
                    "average_precision": "Macro AP",
                    "released_fpr": "Release FPR",
                    "released_tpr": "Release TPR",
                    "calibrated_fpr": "Calibrated FPR",
                    "calibrated_tpr": "Calibrated TPR",
                },
            ),
            "",
            "[All per-source and pooled metrics](metrics.csv) · [95% AUROC/delta intervals](auroc_intervals.csv) · [Common-cohort metrics](metrics_common_cohort.csv) · [Coverage](coverage.csv)",
            "",
            "![Detection performance](plots/auroc_by_condition.png)",
            "",
            "![Per-source AUROC](plots/source_auroc_heatmap.png)",
            "",
            "## Real-image domains and existing encoding",
            "",
            "Each available real-image domain is analyzed separately. The table below records the domains actually evaluated; no result is claimed for an absent dataset. COCO images already carry JPEG history, so an additional JPEG save is recompression. Native encoding is read from file contents: a `.png` filename does not prove lossless encoding or an uncompressed history. Existing encoding, resolution, and camera populations can confound cross-source comparisons and remain relevant when interpreting processing sensitivity.",
            "",
            _table(
                input_domains,
                {
                    "source": "Source",
                    "label": "Fake label",
                    "input_encoding": "Current encoding",
                    "originals": "Originals",
                    "median_width": "Median width",
                    "median_height": "Median height",
                },
            ),
            "",
            "[Every real domain versus every generator, with intervals and domain-specific FPR](real_domain_metrics.csv) · [Original domain/encoding summary](input_domain_summary.csv)",
            "",
            "Each real-domain comparison retains the same frozen release and shared clean-calibrated thresholds; no domain-specific test threshold is fitted. Domain rows include per-generator, pooled-fake, and equally weighted generator-macro results.",
            "",
            (
                _table(
                    real_domains.loc[
                        (real_domains["group"] == "macro") & (real_domains["condition"] == "clean")
                    ],
                    {
                        "real_domain": "Clean real domain",
                        "n_real": "Real",
                        "n_fake": "Fake",
                        "auroc": "Macro AUROC",
                        "released_fpr": "Release FPR",
                        "calibrated_fpr": "Calibrated FPR",
                    },
                )
                if not real_domains.empty
                else "No real-domain comparisons are available."
            ),
            "",
            "![Real domain AUROC and false-positive rates](plots/real_domain_comparison.png)",
            "",
            "## Thresholds and uncertainty",
            "",
            "The release comparison uses z < −2 (equivalently anomaly > 2). The second threshold is frozen using only the 95th percentile of clean real calibration anomaly scores, with NumPy's `higher` quantile and a strict `>` decision. No transformed test images select a threshold.",
            "",
            f"Calibration images: **{summary['n_calibration_real_clean']}**; anomaly cutoff: **{_fmt(summary['calibrated_anomaly_threshold'])}**; empirical calibration FPR: **{_fmt(summary['calibration_empirical_fpr'])}**. A target 5% calibration FPR does not guarantee 5% FPR on another domain or on processed images.",
            "",
            "Calibration domain counts: "
            + ", ".join(
                f"{name}={count}" for name, count in summary["calibration_real_domains"].items()
            )
            + ".",
            "",
            f"Intervals use **{summary['bootstrap_draws']}** percentile bootstrap draws, seed **{summary['bootstrap_seed']}**, resampling original image IDs separately within each label/source stratum. Each draw reuses identical multiplicities for clean and processed scores. Macro intervals are computed from jointly resampled source AUROCs. They describe sampling uncertainty conditional on these selected sources and the fixed threshold; they do not cover checkpoint training uncertainty or unseen-source variation.",
            "",
            "![Frozen-threshold errors](plots/fixed_threshold_errors.png)",
            "",
            "## Residuals and 960D descriptors",
            "",
            "Raw descriptor drift is separate from the learned projected-feature drift. Ratios are relative to the same original image; residual RMS summarizes amplitude and does not require pixelwise alignment of different-size images. Associations between drift and performance are descriptive, not proof of a causal mechanism.",
            "",
            "[Per-source feature/residual summaries](feature_drift.csv)",
            "",
            "![Feature and residual changes](plots/feature_residual_drift.png)",
            "",
        ]
    )
    if "plots/native_residual_spectrum.png" in summary["figures"]:
        lines.extend(
            [
                "Native residual RMS and high-frequency energy are measured before internal model resizing. The spectrum statistic uses a demeaned, Hann-windowed native center patch of up to 128×128 pixels and a cutoff of 0.25 cycles/pixel. Smaller processed images use smaller patches, so frequency resolution differs; actual patch shapes are recorded per image. Changes in crop/resize conditions are not aligned pixelwise residual differences.",
                "",
                "![Native residual spectrum and amplitude](plots/native_residual_spectrum.png)",
                "",
            ]
        )
    galleries = [path for path in summary["figures"] if "/residual_gallery_" in path]
    if galleries:
        lines.extend(
            [
                "The following examples were selected in record order among originals with saved previews, rather than selected by correctness. Within each original, residual and coefficient panels share display limits across conditions; the color ranges are clipped at the 99th absolute percentile for visibility. These examples illustrate the aggregate measurements and do not establish typical performance on their own.",
                "",
            ]
        )
        for path in galleries:
            lines.extend(
                [f"![Matched image, native residual, and FSD coefficient changes]({path})", ""]
            )
    if not attribution.empty:
        attr_macro = attribution.loc[attribution["source"] == "macro"]
        lines.extend(
            [
                "## Source attribution",
                "",
                "Accuracy is restricted to test sources with explicit checkpoint-label mappings. Predictions still range over the full released classifier's source set. These are closed-set top-1 measurements on supported sources; there is no unknown rejection or open-set performance claim. The confidence output is not a validated probability of correctness.",
                "",
                _table(
                    attr_macro,
                    {
                        "condition": "Condition",
                        "n": "Known fake images",
                        "accuracy": "Macro accuracy",
                        "clean_accuracy": "Matched clean",
                        "accuracy_delta": "Change",
                        "prediction_flip_rate": "Prediction flips",
                    },
                ),
                "",
                "[Attribution metrics](attribution_metrics.csv) · [Predictions for mapped and unmapped sources](attribution_predictions.csv)",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Source attribution",
                "",
                "No supported, mapped attribution cohort was available. Source accuracy and open-set performance are not claimed. Descriptive predicted labels, if present, are retained in [attribution_predictions.csv](attribution_predictions.csv).",
                "",
            ]
        )
    lines.extend(["## Measured implications and next experiments", ""])
    if not variants.empty:
        worst = variants.sort_values("auroc_delta").iloc[0]
        lines.append(
            f"Prioritize **{worst['condition']}** for a larger held-out replication: it has the largest measured macro ranking loss in this run. Inspect source-level outcomes before treating the aggregate as a property of every generator."
        )
        high_fpr = variants.loc[np.isfinite(variants["calibrated_fpr"])].sort_values(
            "calibrated_fpr", ascending=False
        )
        if not high_fpr.empty:
            row = high_fpr.iloc[0]
            lines.append(
                f"The highest processed real-image FPR at the frozen clean-calibrated threshold is **{row['condition']}** ({row['calibrated_fpr']:.1%}). A separate follow-up can calibrate on processed real development images, then evaluate untouched test images; do not retroactively optimize this run's test threshold."
            )
        stable = variants.loc[variants["auroc_delta_ci_low"] > -0.02, "condition"].tolist()
        if stable:
            lines.append(
                "Within this sample, the lower 95% delta bound is above −0.02 for: "
                + ", ".join(stable)
                + ". This supports only a sample-conditional small-loss statement, not universal robustness."
            )
    lines.extend(
        [
            "Use additional real-camera domains, unseen generators, more independent originals, and held-out processing severities for confirmation. Native-resolution crop aggregation or a real-only density model covering processing conditions are candidates for a separately trained and independently tested mitigation study.",
            "",
            "## Runtime and reproducibility",
            "",
            "[Test runtime summary](runtime.csv) · [Separate calibration timing](runtime_calibration.csv) · [Machine-readable analysis configuration](analysis_summary.json) · [Raw per-image predictions](records.jsonl)",
            "",
            "The runtime table and plot use the same condition-specific clean-paired test originals. Calibration records are excluded and summarized separately. Counts can differ across conditions when paired test records are missing.",
            "",
            summary["runtime_timer_scope"],
            "",
            "![Measured runtime and memory](plots/runtime_memory.png)",
            "",
            "All derivative conditions share their original image identity. Checkpoint training overlap with public evaluation images cannot be ruled out without the checkpoint authors' image-level training manifest. A local calibration/test split establishes only separation within this experiment.",
            "",
        ]
    )
    if metadata:
        selected = {
            key: metadata[key]
            for key in (
                "dataset",
                "manifest",
                "manifest_path",
                "device",
                "git_commit",
                "weights_dir",
                "started_at",
                "elapsed_seconds",
            )
            if key in metadata
        }
        if selected:
            lines.extend(
                [
                    "Recorded run metadata:",
                    "",
                    "```json",
                    json.dumps(_clean_json(selected), indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## Coverage and limitations",
            "",
            _table(
                coverage,
                {
                    "condition": "Condition",
                    "scored_test": "Scored test",
                    "paired_test": "Paired",
                    "expected_test": "Expected test",
                    "missing_expected_test": "Missing expected",
                    "missing_vs_clean": "Missing vs clean",
                    "unpaired_processed": "No clean match",
                },
            ),
            "",
        ]
    )
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "This report is an analysis snapshot. If the experiment resumes, rerun `an/analyze_results.py` to refresh it. Absent or failed records are not counted as correct predictions.",
            "",
            "Research context and primary-source links: [research notes](../../docs/research.md).",
            "",
        ]
    )
    return "\n".join(lines)
