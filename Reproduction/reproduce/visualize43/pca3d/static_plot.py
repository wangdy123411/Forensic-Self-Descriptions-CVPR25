import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


SOURCE_COLORS = {
    "ImageNet-1k": "#1f77b4", "ProGAN": "#ff7f0e", "StyleGAN3": "#2ca02c",
    "GLIDE": "#d62728", "SD1.5": "#9467bd", "DALLE 3": "#8c564b",
    "MJ v6": "#e377c2", "FireFly": "#7f7f7f",
}
SOURCE_NAMES = {
    "ImageNet-1k": "ImageNet-1k (real)", "DALLE 3": "DALL·E 3", "FireFly": "Firefly",
}
CLUSTER_COLORS = ["#0072B2", "#D55E00", "#009E73", "#8D67B4", "#B18708", "#C95D91"]
INK = "#202832"
SECONDARY = "#495461"
DETAIL_BOUNDS = np.array([[-80, -25, -35], [-45, 25, 35]], dtype=float)


def draw_box(ax, bounds):

    for dimension in range(3):
        fixed = [axis for axis in range(3) if axis != dimension]
        for first in range(2):
            for second in range(2):
                ends = bounds.copy()
                ends[:, fixed[0]] = bounds[first, fixed[0]]
                ends[:, fixed[1]] = bounds[second, fixed[1]]
                ax.plot(*ends.T, color="#4e5965", lw=0.85, linestyle="--", alpha=.90)


def scatter(ax, positions, colors, bounds, detail=False):
    ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=-63)
    ax.set_box_aspect(bounds[1] - bounds[0], zoom=1.02)
    ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2])
    for index, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis), start=1):
        axis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        axis.set_pane_color((.965, .972, .979, 1))
        axis.line.set_color("#8793a0")
        axis._axinfo["grid"].update(color="#cdd5dd", linewidth=.56)
        axis.labelpad = 4 if index == 3 else 0
        axis.set_label_text(f"PC{index}", fontsize=11, color=INK)
    ax.tick_params(axis="both", labelsize=9, colors=SECONDARY, pad=-1, width=.6)
    ax.scatter(*positions.T, c=colors, s=6.8 if detail else 5.5,
               linewidths=0, alpha=.78, depthshade=False, rasterized=True)
    if not detail:
        draw_box(ax, DETAIL_BOUNDS)


def legend(fig, x, values, colors, names):
    handles = [Line2D([], [], linestyle="", marker="o", markersize=7.3,
                      markerfacecolor=colors[v], markeredgewidth=0,
                      label=names.get(v, str(v))) for v in values]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x, .002),
               ncol=3, fontsize=10.7, labelcolor=INK, frameon=False,
               handletextpad=.34, handlelength=.9, columnspacing=1.4, labelspacing=.55)


def main():
    parser = argparse.ArgumentParser(description='static plot')
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--output", type=Path,
                        help="Output PNG. Defaults to cluster_3d.png in the data directory.")
    args = parser.parse_args()
    metrics = json.loads((args.data_dir / "metrics.json").read_text(encoding="utf-8"))
    with (args.data_dir / "coordinates.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metrics["n_images"] or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Coordinate rows disagree with the metric count or contain duplicate images")
    xyz = np.array([[float(r[f"pc{i}"]) for i in range(1, 4)] for r in rows])
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite PCA coordinates")
    sources = [s for s in SOURCE_COLORS if any(r["source"] == s for r in rows)]
    k = metrics["k"]
    if k != len(CLUSTER_COLORS):
        raise ValueError("The cluster palette is defined for K=6")
    cluster_ids = list(range(k))
    colors = [np.array([SOURCE_COLORS[r["source"]] for r in rows])]
    for key in ("cluster_960d", "cluster_3d"):
        labels = np.array([int(r[key]) for r in rows])
        if set(labels) != set(cluster_ids):
            raise ValueError(f"Unexpected labels in {key}")
        colors.append(np.array(CLUSTER_COLORS)[labels])
    bounds = np.array([xyz.min(axis=0), xyz.max(axis=0)])
    room = (bounds[1] - bounds[0]) * .035
    bounds += [-room, room]
    detail = np.all((xyz >= DETAIL_BOUNDS[0]) & (xyz <= DETAIL_BOUNDS[1]), axis=1)
    full = np.ones(len(rows), dtype=bool)
    order = np.random.default_rng(2026).permutation(len(rows))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(14.5, 9.2))
    fig.text(.037, .956, "FSD clustering in three principal components", fontsize=24,
             color=INK, fontweight="bold", va="center")
    ratio = sum(metrics["explained_variance_ratio"])
    fig.text(.038, .909, f"{len(rows):,} images  ·  {len(sources)} sources  ·  "
             f"PC1–PC3 retain {ratio:.1%} of FSD variance", fontsize=13.3,
             color=SECONDARY, va="center")

    centers = [.206, .518, .830]
    titles = ["Image source", "K-means on 960 features", "K-means on 3 components"]
    subtitles = ["Known source labels", f"K = {k}   ·   NMI = {metrics['nmi_960d']:.3f}",
                 f"K = {k}   ·   NMI = {metrics['nmi_3d']:.3f}"]
    for i, center in enumerate(centers):
        fig.text(center, .843, titles[i], ha="center", fontsize=15.3,
                 fontweight="bold", color=INK)
        fig.text(center, .812, subtitles[i], ha="center", fontsize=11.5, color=SECONDARY)
        for mask, view_bounds, bottom, height, zoomed in ((full, bounds, .474, .332, False),
                                                  (detail, DETAIL_BOUNDS, .130, .310, True)):
            ax = fig.add_axes([center-.143, bottom, .286, height], projection="3d")
            use = order[mask[order]]
            scatter(ax, xyz[use], colors[i][use], view_bounds, detail=zoomed)

    for y, label in ((.636, f"All {len(rows):,} images"),
                     (.278, f"Boxed region  ·  {detail.sum():,} images")):
        fig.text(.039, y, label, rotation=90, va="center", ha="center",
                 fontsize=12.5, color=SECONDARY)
    fig.add_artist(Line2D([.064, .971], [.453, .453], transform=fig.transFigure,
                          color="#dde2e7", linewidth=.75))
    cluster_palette = dict(enumerate(CLUSTER_COLORS))
    legend(fig, centers[0], sources, SOURCE_COLORS, SOURCE_NAMES)
    for center in centers[1:]:
        legend(fig, center, cluster_ids, cluster_palette, {i:f"C{i}" for i in cluster_ids})
    dest = args.output or args.data_dir / "cluster_3d.png"
    fig.savefig(dest, dpi=260)
    plt.close(fig)
    print(f"{dest} ({len(rows):,} images, {detail.sum():,} in boxed region)")


if __name__ == "__main__":
    main()
