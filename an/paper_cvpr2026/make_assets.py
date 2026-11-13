"""Create the paper table and vector figure directly from frozen experiment CSVs."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
AN = BASE.parent
metrics_path = AN / "results/main/metrics.csv"
metrics = pd.read_csv(metrics_path)
metrics = metrics[metrics.cohort == "condition_paired"]
macro = metrics[metrics.group == "macro"].set_index("condition")
conditions = [
    ("clean", "Clean"),
    ("jpeg_q50", "JPEG Q50"),
    ("resize_050", r"Resize $0.5\times$"),
    ("crop_area025", r"Crop, 25\% area"),
    ("blur_sigma20", "Blur, radius 2"),
    ("sharpen_200", r"Sharpen, 200\%"),
    ("resize050_jpeg50", r"Resize $0.5\times$ + JPEG Q50"),
]
rows = []
for condition, label in conditions:
    row = macro.loc[condition]
    assert row.n_real == 100 and row.n_fake == 200
    rows.append(f"{label} & {row.auroc:.4f} & {row.released_tpr*100:.1f} & {row.released_fpr*100:.1f} " + r"\\")
table = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Processing & AUROC & TPR & FPR \\", r"\midrule", *rows,
         r"\bottomrule", r"\end{tabular}"]
(BASE / "processing_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

sources = ["Adobe Firefly", "DALL-E 3", "Stable Diffusion XL", "GLIDE"]
labels = ["Firefly", "DALL-E 3", "SDXL", "GLIDE"]
clean = metrics[(metrics.condition == "clean") & metrics.group.isin(sources)].set_index("group").loc[sources]
resized = metrics[(metrics.condition == "resize_050") & metrics.group.isin(sources)].set_index("group").loc[sources]
assert (resized.n_fake == 50).all() and (resized.n_real == 100).all()
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.30, 1.48))
y = np.arange(4)
for i in y:
    ax.plot([resized.auroc.iloc[i], clean.auroc.iloc[i]], [i, i], color="0.72", lw=1, zorder=1)
ax.axvline(0.5, color="0.65", ls="--", lw=0.7)
ax.scatter(clean.auroc, y, s=26, marker="o", facecolors="white", edgecolors="#0072B2", linewidths=1.1, label="Clean", zorder=4)
errors = np.vstack([resized.auroc - resized.auroc_ci_low, resized.auroc_ci_high - resized.auroc])
ax.errorbar(resized.auroc, y, xerr=errors, fmt="D", ms=3.6, color="#D55E00", lw=1, capsize=2, label=r"Resize $0.5\times$", zorder=3)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_ylim(3.45, -0.45)
ax.set_xlim(-0.025, 1.035)
ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
ax.set_xlabel("AUROC (fixed score direction)", labelpad=1)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=3, pad=2)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0, handletextpad=.4, columnspacing=1.4)
fig.subplots_adjust(left=.22, right=.97, bottom=.22, top=.82)
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures/generator_sensitivity.pdf")
fig.savefig(BASE / "figures/generator_sensitivity.png", dpi=300)
plt.close(fig)
selected = metrics[metrics.condition.isin([c for c, _ in conditions])]
selected.to_csv(BASE / "figure_table_data.csv", index=False)
summary = json.loads((AN / "baseline_comparison/run1024/comparison_summary.json").read_text())
assert summary["completed_cases"] == 175 and all(summary["independently_recomputed_numeric_pass"].values())
assert all(a["all_match"] for a in summary["agreement"].values())
(BASE / "data_provenance.json").write_text(json.dumps({
    "metrics": str(metrics_path.relative_to(AN)),
    "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    "main_test_originals": 300, "comparison_originals": 25, "comparison_cases": 175,
    "comparison_summary_sha256": hashlib.sha256((AN / "baseline_comparison/run1024/comparison_summary.json").read_bytes()).hexdigest(),
    "figure": "Clean and half-size resized AUROCs for four generators; pointwise source-stratified bootstrap intervals for resized cases",
    "table": "Main 300-original cohort; macro AUROC and released-threshold rates; no task metric is computed across mismatched cohorts"
}, indent=2), encoding="utf-8")
print("Generated vector figure and seven table rows from frozen CSVs.")
