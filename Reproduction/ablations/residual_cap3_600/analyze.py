from project_paths import EXTENSION_ROOT

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


VARIANTS = ("baseline", "cap3", "cap2", "no_upsample")
TITLES = ("Original FSD", "Residual cap 3×", "Residual cap 2×", "No upsampling")
DISPLAY = ("ImageNet-1k", "ProGAN", "StyleGAN3", "GLIDE", "DALL-E 3", "Firefly")
REPORTED = ("nmi", "ari", "purity", "cross_rate", "progan_within_rate", "glide_within_rate")
PARENT_INVENTORY = "478316ca942cce1bfd525e2f3f459786147fc4b9f4ad4e971b417734d2909205"
CAP2_INVENTORY = "aa43bb6843ee0d8b72576a900c9153b9a06e83b92442d75af7691e2c983e15bc"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def peer_order(original, other):
    n = len(DISPLAY)
    agreement = np.zeros((n, n), dtype=int)
    np.add.at(agreement, (original, other), 1)
    _, perm = linear_sum_assignment(-agreement)
    return perm.tolist()


def render(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(9.4, 7.3), layout="constrained", sharey=True)
    n = len(DISPLAY)
    for ax, branch, title in zip(axs.flat, VARIANTS, TITLES):
        counts = np.asarray(report["primary_counts"][branch])[:, report["figure_cluster_order"][branch]]
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i, j in np.argwhere(counts > 0):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if counts[i, j] > 55 else "#14283e")
        ax.set_xticks(range(n), [f"C{i}" for i in range(n)])
        ax.set_yticks(range(n), DISPLAY)
        for setter in (ax.set_xticks, ax.set_yticks):
            setter(np.arange(-.5, n, 1), minor=True)
        ax.tick_params(which="both", length=0, pad=6)
        ax.grid(which="minor", color="white", linewidth=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title}\nNMI {report['primary'][branch]['nmi']:.3f}", fontsize=14)
    fig.supxlabel("Cluster aligned to original FSD by image overlap", fontsize=12)
    fig.savefig(out / "comparison.png", dpi=190, facecolor="white")
    plt.close(fig)


def analyze(parent, cap2, folder, out, code, draw=True):
    import torch
    for archive, pinned, required in (
        (parent, PARENT_INVENTORY, ("scripts/analyze.py", "features/manifest.csv", "analysis/metrics.json")),
        (cap2, CAP2_INVENTORY, ("features/manifest.csv", "analysis/metrics.json", "analysis/assignments.csv",
                               "analysis/per_seed.csv")),
    ):
        if sha(archive / "checksums.json") != pinned:
            raise ValueError(f"Inventory differs from the verified run: {archive}")
        checksum = json.loads((archive / "checksums.json").read_text())
        for name in required:
            if sha(archive / name) != checksum[name]:
                raise ValueError(f"Verified file changed: {archive / name}")
    spec = importlib.util.spec_from_file_location("paired_pilot_metric_functions", parent / "scripts/analyze.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    project = EXTENSION_ROOT / "reproduce" if (EXTENSION_ROOT / "reproduce").is_dir() else code
    sys.path.insert(0, str(project.resolve()))
    import clustering
    prior = json.loads((parent / "analysis/metrics.json").read_text())
    cap2_report = json.loads((cap2 / "analysis/metrics.json").read_text())
    if cap2_report["parent_inventory_sha256"] != PARENT_INVENTORY or any(
        cap2_report[key] != prior[key] for key in ("seeds", "k", "n_init", "solver_sha256", "device")
    ):
        raise ValueError("The previous two runs used different clustering protocols")
    if (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != prior["device"]
            or sha(Path(clustering.__file__)) != prior["solver_sha256"]):
        raise ValueError("Expected the prior Blackwell and CUDA solver")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = records(parent / "features/manifest.csv")
    current = records(folder / "manifest.csv")
    keys = ("row", "image_id", "source", "image_sha256", "width", "height")
    if len(current) != 600 or len(rows) != 600 or any(r[k] != u[k] for r, u in zip(rows, current) for k in keys):
        raise ValueError("New features are not in the original 600-row cohort order")
    truth = np.array([pilot.SOURCES.index(r["source"]) for r in rows])
    if len({r["image_id"] for r in rows}) != 600 or not np.all(np.bincount(truth) == 100):
        raise ValueError("The paired cohort is invalid")
    if sha(folder / "manifest.csv") != sha(cap2 / "features/manifest.csv"):
        raise ValueError("The cap3 cohort differs from the verified cap2 cohort")
    assignments = records(cap2 / "analysis/assignments.csv")
    if [(r["image_id"], r["source"]) for r in assignments] != [(r["image_id"], r["source"]) for r in rows]:
        raise ValueError("Parent assignments are misaligned")
    parent_scores = {(r["variant"], int(r["seed"])): r for r in records(cap2 / "analysis/per_seed.csv")}
    x = np.load(folder / "cap3.npy", allow_pickle=False)
    if x.shape != (600, 960) or x.dtype != np.dtype("<f8") or not np.isfinite(x).all():
        raise ValueError("Expected 600 finite raw float64 FSDs of length 960")
    out.mkdir(parents=True, exist_ok=True)
    scores, tables, primaries, primary_labels, primary_counts = [], [], {}, {}, {}
    for branch in VARIANTS:
        for seed in pilot.SEEDS:
            if branch == "cap3":
                torch.manual_seed(seed)
                labels, loss = clustering.gpu_kmeans(x, pilot.K, seed, pilot.N_INIT)
                for row, label in zip(assignments, labels):
                    row[f"cap3_{seed}"] = int(label)
            else:
                labels = np.array([int(r[f"{branch}_{seed}"]) for r in assignments])
                loss = float(parent_scores[branch, seed]["inertia"])
            counts = pilot.count_clusters(truth, labels)
            measured = pilot.metrics(truth, labels, counts)
            if branch != "cap3" and any(not np.isclose(value, float(parent_scores[branch, seed][key]),
                                                       rtol=1e-12, atol=1e-12)
                                        for key, value in measured.items()):
                raise ValueError(f"Parent labels and reported scores differ for {branch}, {seed}")
            score = dict(variant=branch, seed=seed, inertia=float(loss), **measured)
            scores.append(score)
            for name, values in zip(pilot.SOURCES, counts):
                tables.append(dict(variant=branch, seed=seed, source=name,
                                   **{f"C{i}": int(value) for i, value in enumerate(values)}))
            if seed == pilot.SEEDS[0]:
                primaries[branch] = score
                primary_labels[branch] = labels
                primary_counts[branch] = counts.tolist()
            if branch == "cap3":
                print(f"cap3 seed={seed} NMI={score['nmi']:.6f} cross={score['cross_rate']:.4f}", flush=True)
    scored = {(r["variant"], r["seed"]): r for r in scores}
    differences = []
    for seed in pilot.SEEDS:
        for reference in ("baseline", "cap2", "no_upsample"):
            delta = {k: scored["cap3", seed][k] - scored[reference, seed][k] for k in REPORTED}
            differences.append(dict(seed=seed, variant="cap3", reference=reference, **delta))
    report = {k: prior[k] for k in ("k", "seeds", "primary_seed", "n_init", "max_iter", "solver",
                                     "solver_sha256", "device", "representation", "sources", "images_per_source",
                                     "nmi_average", "cross_rate", "within_rate")}
    report.update(primary=primaries, primary_counts=primary_counts, paired_changes=differences,
                  figure_cluster_order={k: peer_order(primary_labels["baseline"], lab)
                                        for k, lab in primary_labels.items()},
                  parent_inventory_sha256=PARENT_INVENTORY, cap2_inventory_sha256=CAP2_INVENTORY,
                  file_sha256={"cap3.npy": sha(folder / "cap3.npy"), "manifest.csv": sha(folder / "manifest.csv")})
    pilot.write_csv(out / "comparison.csv", [primaries[k] for k in VARIANTS])
    pilot.write_csv(out / "per_seed.csv", scores)
    pilot.write_csv(out / "paired_changes.csv", differences)
    pilot.write_csv(out / "contingency.csv", tables)
    pilot.write_csv(out / "assignments.csv", assignments)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if draw and importlib.util.find_spec("matplotlib") is not None:
        render(report, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze')
    for key in ("parent", "cap2", "features", "code"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--figure-only", action="store_true")
    mode.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    if args.figure_only:
        render(json.loads((args.out / "metrics.json").read_text(encoding="utf-8")), args.out)
    else:
        if any(value is None for value in (args.parent, args.cap2, args.features, args.code)):
            parser.error("Supply --parent --cap2 --features and --code for paired GPU analysis")
        analyze(args.parent, args.cap2, args.features, args.out, args.code, not args.no_figure)
