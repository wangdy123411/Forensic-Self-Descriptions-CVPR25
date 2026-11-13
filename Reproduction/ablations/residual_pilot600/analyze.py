import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SOURCES = ('ImageNet-1k', 'ProGAN', 'StyleGAN3', 'GLIDE', 'DALLE 3', 'FireFly')
DISPLAY = ('ImageNet-1k', 'ProGAN', 'StyleGAN3', 'GLIDE', 'DALL-E 3', 'Firefly')
VARIANTS = ('baseline', 'no_upsample')
SEEDS = (2026, 2027, 2028, 2029, 2030)
K = 6
N_INIT = 10


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count_clusters(truth, labels):
    counts = np.zeros((K, K), dtype=np.int64)
    np.add.at(counts, (truth, labels), 1)
    if np.any(counts.sum(axis=0) == 0):
        raise ValueError('A returned solution has fewer than six populated clusters')
    return counts


def pair_counts(counts):
    return int((counts * (counts - 1)).sum() // 2)


def metrics(truth, labels, counts):
    pg, glide = counts[SOURCES.index('ProGAN')], counts[SOURCES.index('GLIDE')]
    cross = int(pg @ glide)
    within_pg, within_glide = pair_counts(pg), pair_counts(glide)
    n_pg, n_glide = int(pg.sum()), int(glide.sum())
    return dict(nmi=float(normalized_mutual_info_score(truth, labels, average_method='arithmetic')),
                ari=float(adjusted_rand_score(truth, labels)),
                purity=float(counts.max(axis=0).sum() / len(truth)),
                cross_pairs=cross, cross_total=n_pg * n_glide,
                cross_rate=cross / (n_pg * n_glide),
                progan_pairs=within_pg, progan_total=n_pg * (n_pg - 1) // 2,
                progan_within_rate=within_pg / (n_pg * (n_pg - 1) // 2),
                glide_pairs=within_glide, glide_total=n_glide * (n_glide - 1) // 2,
                glide_within_rate=within_glide / (n_glide * (n_glide - 1) // 2))


def plot_comparison(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True,
                             layout='constrained')
    display_order = report['figure_cluster_order']
    for ax, variant, title in zip(axes, VARIANTS, ('Original FSD', 'No residual upsampling')):
        counts = np.asarray(report['primary_counts'][variant])[:, display_order[variant]]
        ax.imshow(counts, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        for row, col in np.argwhere(counts > 0):
            ax.text(col, row, str(counts[row, col]), color='white' if counts[row, col] > 55 else '#152a3b',
                    ha='center', va='center', fontsize=12)
        ax.set_xticks(range(K), [f'C{c}' for c in display_order['baseline']])
        ax.set_yticks(range(K), DISPLAY)
        ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='both', length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title}\nNMI {report['primary'][variant]['nmi']:.3f}", fontsize=14, pad=10)
        ax.set_xlabel('Cluster matched by image overlap', labelpad=8)
    fig.savefig(out / 'comparison.png', dpi=190, facecolor='white')
    plt.close(fig)


def analyze(inputs, out, code_dir, draw=True):
    import torch
    project = code_dir / 'reproduce' if (code_dir / 'reproduce').is_dir() else code_dir
    sys.path.insert(0, str(project.resolve()))
    import clustering

    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA for the original Section 4.3 solver')
    with (inputs / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['image_id'] for r in rows]
    if len(set(ids)) != 600 or Counter(r['source'] for r in rows) != {s: 100 for s in SOURCES}:
        raise ValueError('Expected six sources with 100 distinct original IDs each')
    truth = np.array([SOURCES.index(r['source']) for r in rows])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True, exist_ok=True)
    scores, tallies = [], []
    labels_by_variant = {}
    primary = {}
    primary_counts = {}
    file_hashes = {'manifest.csv': sha256(inputs / 'manifest.csv')}
    assignments = [dict(image_id=r['image_id'], source=r['source']) for r in rows]

    for variant in VARIANTS:
        file = inputs / f'{variant}.npy'
        features = np.load(file, allow_pickle=False)
        if features.shape != (600, 960) or features.dtype != np.dtype('<f8') or not np.isfinite(features).all():
            raise ValueError(f'{file.name} must contain 600 finite raw 960-dimensional float64 FSDs')
        file_hashes[file.name] = sha256(file)
        for seed in SEEDS:
            torch.manual_seed(seed)
            labels, inertia = clustering.gpu_kmeans(features, K, seed, N_INIT)
            counts = count_clusters(truth, labels)
            score = dict(variant=variant, seed=seed, inertia=float(inertia), **metrics(truth, labels, counts))
            scores.append(score)
            for row, label in zip(assignments, labels):
                row[f'{variant}_{seed}'] = int(label)
            for name, source_counts in zip(SOURCES, counts):
                tallies.append(dict(variant=variant, seed=seed, source=name,
                                    **{f'C{i}': int(value) for i, value in enumerate(source_counts)}))
            if seed == SEEDS[0]:
                labels_by_variant[variant] = labels
                primary[variant] = score
                primary_counts[variant] = counts.tolist()
            print(f"{variant} seed={seed} NMI={score['nmi']:.5f} cross={score['cross_rate']:.4f}", flush=True)

    overlap = np.zeros((K, K), dtype=np.int64)
    np.add.at(overlap, (labels_by_variant['baseline'], labels_by_variant['no_upsample']), 1)
    original_columns, modified_columns = linear_sum_assignment(-overlap)
    order = dict(baseline=original_columns.tolist(), no_upsample=modified_columns.tolist())
    reported = ('nmi', 'ari', 'purity', 'cross_rate', 'progan_within_rate', 'glide_within_rate')
    deltas = []
    for seed in SEEDS:
        baseline, modified = [score for score in scores if score['seed'] == seed]
        deltas.append(dict(seed=seed, **{key: modified[key] - baseline[key] for key in reported}))
    report = dict(k=K, seeds=SEEDS, primary_seed=SEEDS[0], n_init=N_INIT, max_iter=300,
                  solver='Previous Section 4.3 GPU Lloyd solver with scikit-learn k-means++ starts',
                  solver_sha256=sha256(Path(clustering.__file__)),
                  device=torch.cuda.get_device_name(0), representation='Raw float64 FSD, Euclidean',
                  sources=SOURCES, images_per_source=100, file_sha256=file_hashes,
                  nmi_average='arithmetic', cross_rate='All ProGAN x GLIDE image pairs assigned to the same cluster / 10,000',
                  within_rate='All distinct unordered pairs from one source assigned to the same cluster / 4,950',
                  primary=primary, primary_counts=primary_counts,
                  paired_changes= deltas, figure_cluster_order=order)
    write_csv(out / 'per_seed.csv', scores)
    write_csv(out / 'paired_changes.csv', deltas)
    write_csv(out / 'assignments.csv', assignments)
    write_csv(out / 'contingency.csv', tallies)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if draw:
        plot_comparison(report, out)


def main():
    parser = argparse.ArgumentParser(description='analyze')
    parser.add_argument('--dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--code', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--figure-only', action='store_true', help='Render an existing metrics.json in --out')
    mode.add_argument('--no-figure', action='store_true', help='Run GPU clustering and write all numerical outputs')
    args = parser.parse_args()
    if args.figure_only:
        plot_comparison(json.loads((args.out / 'metrics.json').read_text(encoding='utf-8')), args.out)
    else:
        if args.dir is None or args.code is None:
            parser.error('--dir and --code are required for clustering')
        analyze(args.dir, args.out, args.code, not args.no_figure)


if __name__ == '__main__':
    main()
