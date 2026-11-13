from project_paths import CODE_ROOT, EXTENSION_ROOT

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


KNOWN = {"ImageNet-1k": "Real", "StyleGAN3": "StyleGAN3", "ProGAN": "ProGAN"}
UNKNOWN = ("StyleGAN2", "DALLE 3", "FireFly")
SUBSTITUTIONS = {"StyleGAN": "GigaGAN", "SD1.4": "SDXL", "SD3.0": "SD2.1"}
CHECKPOINTS = {
    "fsd_transforms.pt": "1e87d792b413101e58d9de71551182a1fab8b879ca6f6ba9780b6adcb9a5a699",
    "attribution_transforms.pt": "90fab44761c138a70ec7de585d4ec998ce1fe1db8a36fd8631b5988207067605",
    "source_gmms.pt": "42b5e4110a4ff867503983a806b2c7c8674c545994d59ff5c4f01369c54e82d0",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def crosscheck_areas(metrics, truth, winner, scores, names):

    known = truth != "unknown"
    unknown_scores = np.sort(scores[~known])
    known_scores = scores[known]
    ranks = (unknown_scores.searchsorted(known_scores, "left")
             + unknown_scores.searchsorted(known_scores, "right")) / (2 * len(unknown_scores))
    direct_auc = ranks.mean()
    direct_oscr = np.mean(ranks * (truth[known] == winner[known]))


    thresholds, weights = np.unique(unknown_scores, return_counts=True)
    mid_f1 = np.zeros(len(thresholds))
    for name in names:
        hits = np.sort(scores[(truth == name) & (winner == name)])
        assigned = np.sort(scores[winner == name])
        n_true = np.count_nonzero(truth == name)
        for side in ("left", "right"):
            n_hits = len(hits) - hits.searchsorted(thresholds, side)
            n_assigned = len(assigned) - assigned.searchsorted(thresholds, side)
            mid_f1 += n_hits / (n_true + n_assigned)
    direct_fang = np.average(mid_f1 / len(names), weights=weights)
    values = {
        "au_oscr": float(direct_oscr),
        "au_crr_fang_macro_f1_vs_crr": float(direct_fang),
        "known_unknown_roc_auc": float(direct_auc),
    }
    for key, value in values.items():
        np.testing.assert_allclose(value, metrics[key], atol=1e-12, rtol=0)
    return values


def extract_firefly(project, manifest, archive):
    from fsd.detector import FSDDetector

    weights = project / "Weights" / "fsd_v1.2.0"
    expected = {
        "fre.pt": "d95b9c50837dbf7b660bbefa20cdaa5db5e59601a9d6544573c10e78e04906bb",
        "config.json": "7cc34433045adb998762e00de7de25c50f9c1e10dbac1c18899c6c63c4cfafe4",
    }
    for filename, digest in expected.items():
        if sha256(weights / filename) != digest:
            raise ValueError(f"Supplemental extractor checksum mismatch for {filename}")
    records = read_csv(manifest)
    deficit = 1000 - sum(r["source"] == "FireFly" for r in archive)
    if not 1 <= len(records) <= deficit or any(r["source"] != "FireFly" for r in records):
        raise ValueError("Supplement must contain unique FireFly images within this source's deficit")
    ids = {r["image_id"] for r in archive}
    hashes = {r["image_sha256"] for r in archive}
    detector = FSDDetector.load(weights, device="cpu")
    result, features, evidence = [], [], []
    for r in records:
        path = manifest.parent / r["path"]
        digest = sha256(path)
        if digest != r["sha256"] or digest in hashes or not r["image_id"] or r["image_id"] in ids:
            raise ValueError(f"Duplicate or mismatching supplemental file: {path.name}")
        ids.add(r["image_id"])
        hashes.add(digest)
        feature = detector.compute_fsd(path).detach().cpu().numpy()
        if feature.shape != (960,) or feature.dtype != np.float64 or not np.isfinite(feature).all():
            raise ValueError(f"Invalid supplemental FSD for {path.name}")
        row = dict(image_id=r["image_id"], source=r["source"], image_sha256=digest,
                   feature_sha256=hashlib.sha256(feature.tobytes()).hexdigest())
        features.append(feature)
        result.append(row)
        evidence.append(dict(row, filename=path.name))
    return result, np.stack(features), dict(manifest_name=manifest.name, manifest_sha256=sha256(manifest),
                                            extractor_files=expected, device="cpu", records=evidence)


def run(project, output, protocol, supplement=None):
    code = CODE_ROOT
    sys.path[:0] = [str(EXTENSION_ROOT / "reproduce"), str(code), str(EXTENSION_ROOT / "zeroshot_validation")]
    from experiments import openset_metrics
    from fsd.attribution import load_source_gmms
    from fsd.projection import load_transforms, apply_projections

    known_mapping = KNOWN.copy()
    unknown_sources = list(UNKNOWN)
    if protocol == "substituted":
        known_mapping.update({"GigaGAN": "GigaGAN", "SDXL": "Stable Diffusion XL"})
        unknown_sources.append("SD2.1")
    elif supplement:
        raise ValueError("Supplemental FireFly files are for the substituted run")
    results = project / "Results" / "reproduction"
    integrated = results / "section41_final_20261110" / "integrated"
    archived = results / "section42_author_v120_20261110" / "4.2"
    checksums = json.loads((archived / "checksums.json").read_text(encoding="utf-8"))
    for file in ("model.json", "attribution.csv"):
        if sha256(archived / file) != checksums[file]:
            raise ValueError(f"Archived checksum mismatch for {file}")
    model = json.loads((archived / "model.json").read_text(encoding="utf-8"))
    if model["weight_sha256"] != CHECKPOINTS:
        raise ValueError("Archived weights differ from the reference release")

    paths = {}
    for file, digest in CHECKPOINTS.items():
        paths[file] = next(p / file for p in (project / "Weights" / "fsd_v1.2.0", code)
                           if (p / file).is_file())
        if sha256(paths[file]) != digest:
            raise ValueError(f"Checkpoint checksum mismatch for {file}")

    index = read_csv(integrated / "feature_index.csv")
    previous = read_csv(archived / "attribution.csv")
    raw = np.load(integrated / "raw_fsd_960.npy", mmap_mode="r", allow_pickle=False)
    if not (len(index) == len(previous) == model["n"] and raw.shape == (len(index), 960)
            and raw.dtype == np.dtype("<f8")):
        raise ValueError("Inconsistent feature archive")

    rows, locations, reference = [], [], []
    for position, (record, old) in enumerate(zip(index, previous)):
        if (record["image_id"], record["source"], int(record["row"])) != (
                old["image_id"], old["source"], position):
            raise ValueError("Archived attribution rows and feature index are misaligned")
        if record["source"] in known_mapping or record["source"] in unknown_sources:
            if (record["label"] == "0") != (record["source"] == "ImageNet-1k"):
                raise ValueError("Real source mapping differs from expected")
            if hashlib.sha256(raw[position].tobytes()).hexdigest() != record["feature_sha256"]:
                raise ValueError(f"Feature checksum mismatch for {record['image_id']}")
            rows.append(record)
            locations.append(position)
            reference.append(old)
    selected = raw[locations]
    supplemental_info = None
    if supplement:
        extra_rows, extra_features, supplemental_info = extract_firefly(project, supplement, index)
        rows.extend(extra_rows)
        selected = np.concatenate([selected, extra_features])
    for key in ("image_id", "image_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key} in selected rows")
    counts = Counter(r["source"] for r in rows)
    if set(counts) != set(known_mapping).union(unknown_sources):
        raise ValueError(f"Incomplete evaluation coverage: {counts}")

    transforms = (load_transforms(paths["fsd_transforms.pt"], "cpu")
                  + load_transforms(paths["attribution_transforms.pt"], "cpu"))
    gmms = load_source_gmms(paths["source_gmms.pt"], "cpu")
    labels = list(gmms)
    names = list(known_mapping.values())
    if labels != model["classes"]:
        raise ValueError("Released class list differs from archived weights")


    ll = np.empty((len(rows), len(labels)), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            block = torch.from_numpy(selected[start:start + 256])
            x = apply_projections(block, transforms)
            ll[start:start + len(x)] = torch.stack([g.score_samples(x) for g in gmms.values()], 1).numpy()
    if not np.isfinite(ll).all():
        raise ValueError("Non-finite GMM log-likelihood")
    archived_ll = ll[:len(reference)]
    full_candidate = np.asarray(labels)[archived_ll.argmax(1)]
    np.testing.assert_array_equal(full_candidate, [r["candidate"] for r in reference])
    delta = float(np.max(np.abs(archived_ll.max(1) - [float(r["max_log_likelihood"]) for r in reference])))
    if delta > 1e-6:
        raise ValueError(f"Archived likelihood parity failed ({delta})")

    masked = ll[:, [labels.index(n) for n in names]]
    sources = np.asarray([r["source"] for r in rows])
    truth = np.asarray([known_mapping.get(s, "unknown") for s in sources])
    known = truth != "unknown"
    winners = masked.argmax(1)
    candidates = np.asarray(names)[winners]
    scores = masked.max(1)
    metrics, curve = openset_metrics(truth, winners, scores, names)
    audit = crosscheck_areas(metrics, truth, candidates, scores, names)

    tau = float(np.quantile(scores[known], 0.05, method="lower"))
    accepted = scores >= tau
    by_source = {}
    for s in (*known_mapping, *unknown_sources):
        mask = sources == s
        n = counts[s]
        by_source[s] = {
            "n": n, "known": s in known_mapping,
            "unthresholded_candidate_counts": {name: int(np.sum(mask & (candidates == name))) for name in names},
            "closed_set_correct": int(np.sum(mask & (candidates == truth))) if s in known_mapping else None,
            "accepted_at_descriptive_tau": int(np.sum(mask & accepted)),
            "rejected_at_descriptive_tau": int(np.sum(mask & ~accepted)),
            "rejection_rate_at_descriptive_tau": float(np.mean(~accepted[mask])),
        }
    correct = candidates == truth
    metrics.update(
        protocol=f"frozen_author_v1.2.0_masked_paper_source_{protocol}",
        known_sources=list(known_mapping), unknown_sources=unknown_sources, source_counts=dict(counts),
        candidates=names, known_target_mapping=known_mapping, training_images_used=0,
        score=("maximum raw log-likelihood over Real, StyleGAN3 and ProGAN" if protocol == "overlap" else
               "maximum raw log-likelihood over the five candidate GMMs"),
        unknown_status=("Held out from candidate set; all three sources have a GMM in the full release."
                        if protocol == "overlap" else "StyleGAN2, DALL-E 3 and Adobe Firefly have GMMs in "
                        "the full release. SD2.1 has no released GMM. All four are held out at inference."),
        paper_table_3_exact_reproduction=False,
        unavailable_paper_known_gmms=["StyleGAN", "SD1.4"], unavailable_paper_unknown_samples=["SD3.0"],
        threshold_for_main_metrics=None,
        descriptive_threshold={
            "selection": "lower empirical 5th percentile of known test scores; descriptive only",
            "tau": tau, "decision_rule": "accept when maximum log-likelihood >= tau",
            "known_accept": float(accepted[known].mean()),
            "correct_known_accept": float((accepted[known] & correct[known]).mean()),
            "unknown_reject": float((~accepted[~known]).mean()),
            "known_accepted": int(accepted[known].sum()),
            "known_correctly_accepted": int((accepted[known] & correct[known]).sum()),
            "unknown_accepted": int(accepted[~known].sum()),
            "unknown_rejected": int((~accepted[~known]).sum()),
        },
        per_source=by_source, numerical_crosscheck=audit,
        reference_archived_14way={"top1_matches": len(reference), "max_abs_loglik_delta": delta},
        weight_sha256=CHECKPOINTS,
        inputs={"feature_run": str(integrated.relative_to(project).as_posix()),
                "reference_run": str(archived.relative_to(project).as_posix()),
                "feature_index_sha256": sha256(integrated / "feature_index.csv"),
                "feature_tensor_sha256": sha256(integrated / "raw_fsd_960.npy"),
                "archived_attribution_sha256": checksums["attribution.csv"]},
        fang_definition_url="https://papers.bmvc2023.org/0659.pdf",
    )
    if protocol == "substituted":
        metrics["substitutions"] = SUBSTITUTIONS
        metrics["target_counts"] = {"known": 6000, "unknown": 5000}
        metrics["supplement"] = supplemental_info
    output.mkdir(parents=True, exist_ok=True)
    predictions = [
        dict(image_id=r["image_id"], source=r["source"], target=target, candidate=candidate,
             max_log_likelihood=float(score), **dict(zip(names, map(float, values))))
        for r, target, candidate, score, values in zip(rows, truth, candidates, scores, masked)
    ]
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "threshold_curve.csv", curve)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in (
        "known_n", "unknown_n", "known_macro_accuracy", "au_crr_fang_macro_f1_vs_crr",
        "au_oscr", "known_unknown_roc_auc", "descriptive_threshold")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='section42 paper sources')
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--protocol", choices=("overlap", "substituted"), default="overlap")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--supplement-manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    default_output = Path(__file__).resolve().parent.parent / ("section42_paper_" + args.protocol)
    run(args.project.resolve(), (args.output or default_output).resolve(), args.protocol, args.supplement_manifest)
