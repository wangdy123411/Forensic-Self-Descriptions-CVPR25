import argparse
import json
from pathlib import Path

import numpy as np

from reference_neighbors import HEADS, SOURCE_TO_HEAD, accept_top95, comparison, rows, write_rows


def support(distances, gallery, radii, candidate):
    row = np.arange(len(candidate))
    d_center = distances["centroid"]
    angular = distances["cosine"]
    own_nearest = angular[row, candidate]
    other = angular.copy()
    other[row, candidate] = np.inf
    return {"centroid_radius": -d_center[row, candidate]/radii[gallery, candidate],
            "retrieval_support": other.min(axis=1) - own_nearest}


def run(archive, decisions, out):
    records = rows(decisions)
    ids = np.array([r["image_id"] for r in records])
    source = np.array([r["source"] for r in records])
    maximum = np.array([float(r["gmm_max"]) for r in records])
    c = np.array([HEADS.index(r["candidate"]) for r in records])
    flag = np.array([int(r["historical_150"]) == 1 for r in records])
    true = np.array([HEADS.index(SOURCE_TO_HEAD[s]) if s in SOURCE_TO_HEAD else -1 for s in source])
    group = {"unknown": true < 0, "historical_150":flag, "FireFly":source=="FireFly",
             "StyleGAN2":source=="StyleGAN2", "DALLE 3":source=="DALLE 3"}
    with np.load(archive, allow_pickle=False) as data:
        assert np.array_equal(ids, data["ids"]) and list(data["heads"]) == HEADS
        fold = data["fold"]
        assert np.array_equal(fold, [int(r["fold"]) for r in records])
        radii = data["radii"]
        assert radii.shape == (5,5) and np.all(radii > 0)
        test = {k:data["test_"+k] for k in ("centroid","cosine")}
        cal = {k:data["cal_"+k] for k in ("centroid","cosine")}
    test_scores = support(test, fold, radii, c)
    cal_scores = support(cal, (fold-1)%5, radii, c)
    all_rows = np.ones(len(fold), dtype=bool)
    retrieved = test["cosine"].argmin(axis=1)
    prototype = test["centroid"].argmin(axis=1)
    agreement = {}
    variant = np.array([r["variant"] for r in records])
    diagnostic_groups = {"historical_150":flag, "genuine_gigagan":source=="GigaGAN",
                         "original_known":(true>=0)&(true<3), "all_known":true>=0}
    for v in sorted(set(variant[source=="GigaGAN"])):
        diagnostic_groups[v] = (source=="GigaGAN") & (variant==v)
    for label, mask in diagnostic_groups.items():
        agreement[label] = dict(n=int(mask.sum()), gmm_and_retrieval_agree=int((mask & (c==retrieved)).sum()),
                   gmm_and_centroid_agree=int((mask & (c==prototype)).sum()),
                   nearest_image_source={h:int((mask & (retrieved==i)).sum()) for i,h in enumerate(HEADS)},
                   nearest_centroid_source={h:int((mask & (prototype==i)).sum()) for i,h in enumerate(HEADS)},
                   correct_gmm=int((mask & (c==true)).sum()),
                   correct_nearest_image=int((mask & (retrieved==true)).sum()))
    rule_names = ["original", "centroid_radius", "retrieval_support"]
    output = {"description": {"centroid_radius":"GMM nominates; use negative Euclidean distance to its reference centroid divided by the sample RMS source radius (Fang et al., Eqs. 4-8).",
              "retrieval_support":"GMM nominates; cosine similarity of its nearest example minus that of the closest rival source's nearest example. Exemplar retrieval is inspired by LIDA; the margin and reject rule are our adaptations.",
              "common_calibration":"Per-fold known 2nd percentile evidence floor; adjust GMM cutoff on eligible knowns to 95% coverage. Matched95 pools out-of-fold known scores when selecting the GMM threshold. Heldout freezes all thresholds on the separate calibration fold."},
              "gallery_diagnostics":agreement, "protocols":{}}
    output_rows = [dict(image_id=i,source=s,fold=int(f),gmm_candidate=HEADS[j],nearest_exemplar=HEADS[n],
                        nearest_centroid=HEADS[p],centroid_normalized_distance=float(-a),
                        retrieval_support=float(b),historical_150=int(m))
                   for i,s,f,j,n,p,a,b,m in zip(ids,source,fold,c,retrieved,prototype,
                                                 test_scores["centroid_radius"], test_scores["retrieval_support"], flag)]
    summary_rows = []
    for protocol, known in [("original_three_known",(true>=0)&(true<3)),("all_five_known",true>=0)]:
        eligible = {r:all_rows.copy() for r in rule_names}
        heldout = {r:np.zeros(len(fold), dtype=bool) for r in rule_names}
        thresholds = []
        for f in range(5):
            held = fold == f
            calibration = known & (fold == (f+1)%5)
            for rule in rule_names:
                if rule == "original":
                    floor, cal_ok = None, all_rows
                else:
                    floor = float(np.quantile(cal_scores[rule][calibration], 0.02, method="lower"))
                    cal_ok = cal_scores[rule] >= floor
                    eligible[rule][held] = test_scores[rule][held] >= floor
                _, cutoff = accept_top95(maximum, cal_ok, calibration)
                heldout[rule][held] = eligible[rule][held] & (maximum[held] >= cutoff)
                thresholds.append(dict(test_fold=f, rule=rule, evidence_floor=floor, gmm_cutoff=cutoff))
        matched = {}
        matched_cuts = {}
        for rule in rule_names:
            matched[rule], matched_cuts[rule] = accept_top95(maximum, eligible[rule], known)
        report = {"thresholds":thresholds, "matched_gmm_cutoffs":matched_cuts}
        for mode, rules in [("matched95",matched),("heldout",heldout)]:
            refkey = f"{protocol}_{mode}_baseline"
            prior = np.array([r[refkey]=="1" for r in records])
            assert np.array_equal(prior, rules["original"])
            stats = {}
            for rule, keep in rules.items():
                stats[rule] = {}
                for label, mask in {"known":known, **group}.items():
                    counts = comparison(mask, prior, keep)
                    stats[rule][label] = counts
                    summary_rows.append(dict(protocol=protocol,mode=mode,rule=rule,group=label,**counts))
                stats[rule]["correct_known_and_accepted"] = int((known & keep & (true==c)).sum())
                for row, accepted in zip(output_rows, keep):
                    row[f"{protocol}_{mode}_{rule}"] = int(accepted)
            report[mode] = stats
        output["protocols"][protocol] = report
    out.mkdir(exist_ok=True, parents=True)
    write_rows(out/"comparison.csv", summary_rows)
    write_rows(out/"decisions.csv", output_rows)
    (out/"comparison.json").write_text(json.dumps(output,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='reference rule comparison')
    p.add_argument("--distances", required=True, type=Path)
    p.add_argument("--decisions", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    run(args.distances,args.decisions,args.out)
