import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
HEADS = ["Real", "StyleGAN3", "ProGAN", "GigaGAN", "Stable Diffusion XL"]
SOURCE_TO_HEAD = {"ImageNet-1k": "Real", "StyleGAN3": "StyleGAN3", "ProGAN": "ProGAN",
                  "GigaGAN": "GigaGAN", "SDXL": "Stable Diffusion XL"}
K = 5
BANK_SIZE = 200
N_FOLDS = 5
GATE_QUANTILE = 0.98


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def family_and_variant(path, sha):
    path = PurePosixPath(path)
    variant = path.parent.name
    if path.parts[0] == "Synthbuster":
        return "synthbuster/" + path.stem, "synthbuster/" + variant
    if variant.startswith("GigaGAN_t2i_coco256_"):
        return "gigagan-t2i/" + path.stem, variant
    if variant in ("dalle3_cocoval", "sdxl_cocoval"):
        match = re.fullmatch(r"(?:ann|captions_val2017_)(\d+)_\d+", path.stem)
        if match:
            return "grip-coco-number/" + str(int(match[1])), variant
    return "sha256/" + sha, variant


def read_input(project, manifest):
    integrated = project / "Results/reproduction/section41_final_20261110/integrated"
    index = {r["image_id"]: r for r in rows(integrated / "feature_index.csv")}
    provenance = {r["image_id"]: r for r in rows(manifest) if r["included_in_unique_eval"] == "1"}
    raw = np.load(integrated / "raw_fsd_960.npy", mmap_mode="r")
    previous = project / "Results/reproduction/section42_paper_sources_20261111"
    paths = {"overlap/predictions.csv":"6470bd248f48d999d7a9e07c224d6cb3fdae9bb096355b8f2a6310b94e8f15ce",
             "substituted/predictions.csv":"264ecd38f1b7a18c5eb5cfd463e67b4e617a9061e6eaf952ce808d94f0a267d9",
             "fixed_overlap_candidate_additions.json":"68716d234ba2ed6b595d982e120249c4a5fc11cfa9b7f3fc8564bf966f558866"}
    for name, expected in paths.items():
        assert hashlib.sha256((previous/name).read_bytes()).hexdigest() == expected
    original = rows(previous / "overlap/predictions.csv")
    expanded = rows(previous / "substituted/predictions.csv")
    original_ids = {r["image_id"] for r in original}
    selected = [r for r in expanded if r["target"] != "unknown" or r["image_id"] in original_ids]
    ids = np.array([r["image_id"] for r in selected])
    sources = np.array([r["source"] for r in selected])
    assert len(original_ids) == 6997 and len(set(ids)) == len(ids) == 9997
    features = np.empty((len(ids), 960), dtype=np.float64)
    families, variants, file_hashes, feature_hashes = [], [], [], []
    for i, (key, source) in enumerate(zip(ids, sources)):
        record, prov = index[key], provenance[key]
        assert record["source"] == prov["source"] == source and record["image_sha256"] == prov["sha256"]
        features[i] = raw[int(record["row"])]
        assert hashlib.sha256(features[i].tobytes()).hexdigest() == record["feature_sha256"]
        family, variant = family_and_variant(prov["original_relative_path"], prov["sha256"])
        families.append(family)
        variants.append(variant)
        file_hashes.append(prov["sha256"])
        feature_hashes.append(record["feature_sha256"])
    assert len(set(file_hashes)) == len(ids) and np.isfinite(features).all()

    root = {}
    def leader(f):
        while f in root:
            f = root[f]
        return f
    first = {}
    for family, h in zip(families, feature_hashes):
        a, b = leader(family), leader(first.setdefault(h, family))
        if a != b:
            root[max(a, b)] = min(a, b)
    families = np.array([leader(f) for f in families])
    fold = np.array([int(digest("fsd-reference-neighbors-v1/" + f)[:16], 16) % N_FOLDS for f in families])
    scores = np.array([[float(r[h]) for h in HEADS] for r in selected])
    winners = np.array(HEADS)[scores.argmax(axis=1)]
    maximum = scores.max(axis=1)
    assert np.isfinite(scores).all() and np.array_equal(winners, [r["candidate"] for r in selected])
    np.testing.assert_allclose(maximum, [float(r["max_log_likelihood"]) for r in selected], atol=1e-8, rtol=0)
    old_threshold = json.loads((previous / "fixed_overlap_candidate_additions.json").read_text())["fixed_three_way_threshold"]
    flag = ((sources == "FireFly") & (winners == "GigaGAN") &
            (scores[:, :3].max(axis=1) < old_threshold) & (maximum >= old_threshold))

    assert flag.sum() == 150 and digest("\n".join(sorted(ids[flag]))) == "4b9febcf7ecd0e978c3b006ed84cfea1aa645a5a86390b3c1e45bed62a12a6a4"
    return ids, sources, families, np.array(variants), fold, features, maximum, winners, flag


def bank_for_source(src, held, ids, sources, variants, families, fold):
    pool = np.flatnonzero((sources == src) & (fold != held) & (fold != (held+1) % N_FOLDS))
    sizes = Counter(variants[sources == src])
    total = sum(sizes.values())
    quotas = {v: BANK_SIZE * n // total for v, n in sizes.items()}
    by_remainder = sorted(sizes, key=lambda v: (-(BANK_SIZE * sizes[v] % total), v))
    for v in by_remainder[:BANK_SIZE - sum(quotas.values())]:
        quotas[v] += 1
    chosen, used = [], set()
    for v, wanted in sorted(quotas.items()):
        candidates = sorted(pool[variants[pool] == v], key=lambda i: digest(f"gallery/{held}/{ids[i]}"))
        group = []
        for i in candidates:
            if families[i] not in used:
                used.add(families[i])
                group.append(i)
                if len(group) == wanted:
                    break
        if len(group) != wanted:
            raise ValueError(f"Too few distinct references for {src}, {v}, fold {held}")
        chosen.extend(group)
    return np.array(chosen)


def mean_five(queries, bank, self_query=False):

    squared = (queries*queries).sum(axis=1)[:, None] + (bank*bank).sum(axis=1) - 2 * queries @ bank.T
    if self_query:
        np.fill_diagonal(squared, np.inf)
    nearest = np.partition(np.maximum(squared, 0), K-1, axis=1)[:, :K]
    return np.sqrt(nearest).mean(axis=1)


def accept_top95(maximum, eligible, calibration):
    n = np.count_nonzero(calibration)
    target = (95*n + 50)//100
    vals = np.sort(maximum[eligible & calibration])
    if len(vals) < target:
        raise ValueError("Distance veto prevents 95% acceptance")
    cutoff = vals[-target]
    accepted = eligible & (maximum >= cutoff)
    assert np.count_nonzero(accepted & calibration) == target
    return accepted, float(cutoff)


def comparison(mask, baseline, method):
    a, b = baseline[mask], method[mask]
    return dict(n=len(a), baseline=int(a.sum()), accepted=int(b.sum()),
                rescued=int((a & ~b).sum()), newly_accepted=int((~a & b).sum()),
                rejected=int((~b).sum()), reduction_pp=float(100*(a.mean()-b.mean())))


def paired_ci(mask, sources, baseline, method):

    rng = np.random.default_rng(7454)
    diff = np.zeros(10000)
    for src in sorted(set(sources[mask])):
        s = comparison(mask & (sources == src), baseline, method)
        n, r, a = s["n"], s["rescued"], s["newly_accepted"]
        counts = rng.multinomial(n, np.array([r, a, n-r-a])/n, size=len(diff))
        diff += counts[:,0] - counts[:,1]
    return (100*np.quantile(diff, [.025,.975])/mask.sum()).tolist()


def write_rows(path, data):
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def run(project, manifest, out):
    out.mkdir(exist_ok=True, parents=True)
    ids, sources, families, variants, fold, features, maximum, winner, flagged = read_input(project, manifest)
    unknown = ~np.isin(sources, list(SOURCE_TO_HEAD))
    assert unknown.sum() == 3997
    masks = {"original_three_known": np.isin(sources, list(SOURCE_TO_HEAD)[:3]),
             "all_five_known": ~unknown}
    glob = np.ones(len(ids), dtype=bool)
    base_distance = np.empty(len(ids))
    giga_distance = np.full(len(ids), np.nan)
    banks, distance_rows = [], []
    protocols = {}
    for name, known in masks.items():
        protocols[name] = dict(known=known, ratio=np.full(len(ids), np.nan),
                               giga_ok=np.ones(len(ids), dtype=bool),
                               decisions={k: np.zeros(len(ids), dtype=bool) for k in
                                          ("baseline", "distance", "combined", "combined_gigagan_gate")},
                               cutoffs=[])
    for f in range(N_FOLDS):
        test = fold == f
        cal = fold == (f+1) % N_FOLDS
        use = test | cal
        norm_distance = np.full(len(ids), np.nan)
        giga_fold = np.full(len(ids), np.nan)
        for src, head in SOURCE_TO_HEAD.items():
            bank = bank_for_source(src, f, ids, sources, variants, families, fold)
            assert not set(families[bank]).intersection(families[use])
            ref = features[bank]
            scale = np.median(mean_five(ref, ref, self_query=True))
            assert scale > 0
            selected = use & (winner == head)
            if head == "GigaGAN":
                selected |= use & (sources == "GigaGAN")
            ix = np.flatnonzero(selected)
            dist = mean_five(features[ix], ref)
            win = winner[ix] == head
            norm_distance[ix[win]] = dist[win] / scale
            if head == "GigaGAN":
                giga_fold[ix] = dist
                giga_distance[ix[test[ix]]] = dist[test[ix]]
            banks.extend(dict(test_fold=f, source=src, image_id=ids[i], variant=variants[i],
                              family=families[i], reference_distance_median=float(scale)) for i in bank)
        assert np.isfinite(norm_distance[use]).all()
        base_distance[test] = norm_distance[test]


        genuine_cal = cal & (sources == "GigaGAN")
        giga_gate = float(np.quantile(giga_fold[genuine_cal], GATE_QUANTILE, method="higher"))
        giga_ok = (winner != "GigaGAN") | (giga_fold <= giga_gate)
        for name, obj in protocols.items():
            known_cal = cal & obj["known"]

            gate = float(np.quantile(norm_distance[known_cal], GATE_QUANTILE, method="higher"))
            ratio = norm_distance / gate
            obj["ratio"][test] = ratio[test]
            obj["giga_ok"][test] = giga_ok[test]
            limits = {"fold": f, "known_calibration_n": int(known_cal.sum()), "distance_gate": gate,
                      "genuine_gigagan_calibration_n": int(genuine_cal.sum()), "genuine_gigagan_raw_gate": giga_gate}
            for label, values, eligible in [("baseline", maximum, glob), ("distance", -ratio, glob),
                                             ("combined", maximum, ratio <= 1),
                                             ("combined_gigagan_gate", maximum, (ratio <= 1) & giga_ok)]:
                accepted, threshold = accept_top95(values, eligible, known_cal)
                obj["decisions"][label][test] = accepted[test]
                limits[label] = threshold
            obj["cutoffs"].append(limits)

    true_giga = sources == "GigaGAN"
    for mask, name in [(true_giga, "genuine_gigagan"), (flagged, "historical_firefly")]:
        for i in np.flatnonzero(mask):
            assert np.isfinite(giga_distance[i])
            distance_rows.append(dict(group=name, image_id=ids[i], fold=int(fold[i]), variant=variants[i],
                                      gmm_candidate=winner[i], distance=float(giga_distance[i])))
    write_rows(out / "distances.csv", distance_rows)
    write_rows(out / "reference_banks.csv", banks)
    ds = {}
    for key, m in [("genuine", true_giga), ("historical_firefly", flagged),
                   ("genuine_with_gigagan_winner", true_giga & (winner == "GigaGAN"))]:
        d = giga_distance[m]
        ds[key] = {"n": len(d), "quartiles": np.quantile(d, [0.25,0.5,0.75]).tolist()}
        if key != "historical_firefly":

            ordered = np.sort(d)
            rank = (ordered.searchsorted(giga_distance[flagged], "left") + ordered.searchsorted(giga_distance[flagged], "right"))/2
            ds[key]["distance_auc"] = float(rank.mean()/len(d))
    ds["genuine_by_variant"] = {v: np.quantile(giga_distance[true_giga & (variants == v)], [.25,.5,.75]).tolist()
                                for v in sorted(set(variants[true_giga]))}
    report = dict(k=K, gallery_size_per_source=BANK_SIZE, known_veto_quantile=GATE_QUANTILE,
                  rule_descriptions={"baseline":"Maximum of five released GMM log-likelihoods.",
                    "distance":"Relative mean distance to five nearest examples of the winning source, calibrated on known queries.",
                    "combined":"Common 98th-percentile known distance ceiling; adjust GMM cutoff to 95% overall known coverage.",
                    "combined_gigagan_gate":"Exploratory. Same combined rule plus a 98th-percentile GigaGAN distance ceiling from disjoint genuine GigaGAN calibration examples."},
                  split_description="Five deterministic folds grouped by exact FSD, shared Synthbuster stem, paired GigaGAN COCO stem, and matching GRIP COCO numeric ID. Each test fold uses the next fold for calibration and the other three for a 200-image, variant-stratified gallery per candidate.",
                  matched_description="Thresholds for GMM and distance-only scores are matched on pooled out-of-fold known queries to obtain exactly 95%. The heldout results instead freeze every threshold before the test fold.",
                  distance_summary=ds, protocols={})
    summary_rows = []
    decision_rows = [dict(image_id=i, source=s, variant=v, family=fam, fold=int(f), candidate=w,
                       gmm_max=float(m), normalized_distance=float(d), historical_150=int(fl))
                     for i,s,v,fam,f,w,m,d,fl in zip(ids,sources,variants,families,fold,winner,maximum,base_distance,flagged)]
    for name, obj in protocols.items():
        known = obj["known"]
        ratio = obj["ratio"]
        frozen = obj["decisions"]
        matched, thresholds = {}, {}
        for label, score, eligible in [("baseline", maximum, glob), ("distance", -ratio, glob),
                                       ("combined", maximum, ratio <= 1),
                                       ("combined_gigagan_gate", maximum, (ratio <= 1) & obj["giga_ok"])]:
            matched[label], thresholds[label] = accept_top95(score, eligible, known)
        result = {"calibrated_by_fold": obj["cutoffs"], "matched_thresholds": thresholds}
        targets = np.array([SOURCE_TO_HEAD.get(s, "") for s in sources])
        for mode, rules in [("matched95", matched), ("heldout", frozen)]:
            res = {}
            for rule, accepted in rules.items():
                a = rules["baseline"]
                items = {"known": known, "unknown": unknown, "historical_150": flagged}
                items.update({s:sources == s for s in sorted(set(sources[unknown | known]))})
                res[rule] = {k:comparison(mask, a, accepted) for k,mask in items.items()}
                res[rule]["known_correct_and_accepted"] = int((accepted & known & (winner == targets)).sum())
                res[rule]["unknown_paired_reduction_ci95_pp_fixed_decisions"] = paired_ci(unknown, sources, a, accepted)
                summary_rows.extend(dict(protocol=name, operating_point=mode, rule=rule, group=group,
                                         **res[rule][group]) for group in items)
                for i, row in enumerate(decision_rows):
                    row[f"{name}_{mode}_{rule}"] = int(accepted[i])
            result[mode] = res
        report["protocols"][name] = result
    original = report["protocols"]["original_three_known"]["matched95"]["baseline"]
    assert (original["unknown"]["accepted"], original["historical_150"]["accepted"]) == (1723, 140)
    write_rows(out / "decisions.csv", decision_rows)
    write_rows(out / "summary.csv", summary_rows)
    write_rows(out / "firefly_150_decisions.csv", [r for r in decision_rows if r["historical_150"]])
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"distances":ds,"primary":report["protocols"]["original_three_known"]["matched95"]["combined"]},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reference neighbors')
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE/"reference_neighbors")
    args = parser.parse_args()
    run(args.project, args.manifest, args.out)
