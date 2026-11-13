import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_rejection_rules import split_known


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paired_margin95"
OVERLAP = ROOT.parent / "section42_paper_overlap" / "predictions.csv"
SUBST = ROOT.parent / "section42_paper_substituted" / "predictions.csv"
SPECIAL = ROOT / "rank_audit" / "firefly_150.csv"
ARCHIVE = ROOT / "fixed_overlap_candidate_additions.json"
HEADS = ["Real", "StyleGAN3", "ProGAN", "GigaGAN", "Stable Diffusion XL"]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(reference, values):

    x, n = np.unique(reference, return_counts=True)
    pos = (np.cumsum(n) - (n + 1) / 2) / (len(reference) - 1)
    return np.interp(values, x, pos, left=0, right=1)


def joint_scores(maximum, gap, reference):
    a = percentile(maximum[reference], maximum)
    b = percentile(gap[reference], gap)


    return np.minimum(a, b), np.maximum(a, b)


def threshold(primary, secondary, calibration):
    chosen = np.flatnonzero(calibration)
    order = np.lexsort((secondary[chosen], primary[chosen]))

    target = (95 * len(chosen) + 99) // 100
    ix = chosen[order[len(chosen) - target]]
    a, b = float(primary[ix]), float(secondary[ix])
    keep = (primary > a) | ((primary == a) & (secondary >= b))
    assert int(keep[chosen].sum()) == target, "Indistinguishable boundary scores"
    return keep, [a, b]


def paired(mask, a, b):
    n = int(mask.sum())
    aa, bb = int((mask & a).sum()), int((mask & b).sum())
    removed = int((mask & a & ~b).sum())
    new = int((mask & ~a & b).sum())
    assert aa - bb == removed - new
    return dict(n=n, max_accept=aa, joint_accept=bb, max_to_reject=removed,
                reject_to_joint_accept=new, accepted_both=int((mask & a & b).sum()),
                max_pct=100*aa/n, joint_pct=100*bb/n, drop_pp=100*(aa-bb)/n)


def paired_ci(mask, sources, a, b):

    rng = np.random.default_rng(7454)
    samples = np.zeros(10000)
    for src in sorted(set(sources[mask])):
        stats = paired(mask & (sources == src), a, b)
        n, removed, new = [stats[k] for k in ("n", "max_to_reject", "reject_to_joint_accept")]
        p = np.array([removed, new, n-removed-new], dtype=float) / n
        bootstrap = rng.multinomial(n, p, size=len(samples))
        samples += bootstrap[:, 0] - bootstrap[:, 1]
    return (np.quantile(samples, [.025, .975]) * 100 / mask.sum()).tolist()


def compute(rs, special, out_details=False):
    ids = [r["image_id"] for r in rs]
    sources = np.array([r["source"] for r in rs])
    targets = np.array([r["target"] for r in rs])
    known = targets != "unknown"
    flagged = np.array([i in special for i in ids])
    assert len(ids) == len(set(ids)) and flagged.sum()==150
    assert np.all((sources[flagged]=="FireFly") & ~known[flagged])
    x = np.array([[float(r[h]) for h in HEADS] for r in rs])
    assert np.isfinite(x).all()
    order = np.argsort(-x, axis=1, kind="stable")
    ix = np.arange(len(rs))
    winner = np.array(HEADS)[order[:,0]]
    runner = np.array(HEADS)[order[:,1]]
    maximum = x[ix, order[:,0]]
    gap = maximum - x[ix, order[:,1]]
    assert np.array_equal(winner, [r["candidate"] for r in rs])
    np.testing.assert_allclose(maximum, [float(r["max_log_likelihood"]) for r in rs], rtol=0, atol=1e-8)
    folds = split_known(ids, sources, known)

    def run(ref, cal, scored):
        score, second = joint_scores(maximum, gap, ref)
        baseline, cut_a = threshold(maximum, np.zeros(len(rs)), cal)
        joint, cut_b = threshold(score, second, cal)
        unknown = ~known
        report = dict(reference_n=int(ref.sum()), calibration_n=int(cal.sum()),
                      max_cutoff=cut_a[0], joint_percentile_cutoffs=cut_b,
                      known=paired(scored, baseline, joint),
                      known_by_source={s: paired(scored & (sources==s),baseline,joint) for s in sorted(set(sources[scored]))},
                      unknown=paired(unknown, baseline, joint),
                      unknown_by_source={s: paired(unknown & (sources==s),baseline,joint) for s in sorted(set(sources[unknown]))},
                      historical_firefly_150=paired(flagged,baseline,joint),
                      paired_unknown_drop_pp_ci95_fixed_thresholds=paired_ci(unknown,sources,baseline,joint),
                      correct_and_accepted_known=dict(n=int(scored.sum()),
                          max=int((scored & baseline & (targets==winner)).sum()),
                          joint=int((scored & joint & (targets==winner)).sum())))
        nominal_floor = lambda v: float(np.interp(cut_b[0], percentile(v,v)[np.argsort(v)], np.sort(v)))
        report["joint_raw_floors"] = {"maximum":nominal_floor(maximum[ref]), "gap":nominal_floor(gap[ref])}
        report["historical_firefly_gap_median"] = float(np.median(gap[flagged]))
        report["historical_firefly_gap_below_floor"] = int(np.count_nonzero(flagged & (gap < report["joint_raw_floors"]["gap"])))
        return report, baseline, joint, score, second

    primary, a, b, score, second = run(known, known, known)


    heldout, _, _, _, _ = run(folds=="reference", folds=="calibrate", folds=="evaluate")
    matched, _, _, _, _ = run(folds=="reference", folds=="evaluate", folds=="evaluate")
    if out_details:
        def export(name, which):
            with (OUT/name).open("w", encoding="utf-8-sig", newline="") as f:
                w=csv.writer(f)
                w.writerow(["image_id","source","target","top1","top2","max_log_likelihood","gap",
                            "joint_min_percentile","joint_max_percentile","baseline_accept","joint_accept",
                            "historical_firefly_150","prior_known_fold"])
                for i in np.flatnonzero(which):
                    w.writerow([ids[i],sources[i],targets[i],winner[i],runner[i],maximum[i],gap[i],
                                score[i],second[i],int(a[i]),int(b[i]),int(flagged[i]),folds[i]])
        export("decisions_6997.csv", np.ones(len(rs),dtype=bool))
        export("firefly_150_decisions.csv",flagged)
    return dict(exact95_pooled=primary, disjoint_calibration_check=heldout,
                heldout_matched95=matched)


def main():
    OUT.mkdir(exist_ok=True)
    small, large = rows(OVERLAP), rows(SUBST)
    archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    special = {r["image_id"] for r in rows(SPECIAL)}
    byid = {r["image_id"]:r for r in large}
    assert len(byid)==len(large)==11000
    fixed = []
    historical = set()
    for r in small:
        s = byid[r["image_id"]]
        assert s["source"]==r["source"] and s["target"]==r["target"]
        assert max(abs(float(s[h])-float(r[h])) for h in HEADS[:3])<1e-7
        fixed.append(s)
        if (s["source"]=="FireFly" and s["candidate"]=="GigaGAN" and
            float(r["max_log_likelihood"])<archive["fixed_three_way_threshold"]<=float(s["max_log_likelihood"])):
            historical.add(r["image_id"])
    assert historical==special and len(special)==150
    assert len(fixed)==6997 and sum(r["target"]!="unknown" for r in fixed)==3000
    hashes = {p.name+"@"+p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (OVERLAP,SUBST,SPECIAL)}
    assert hashes[OVERLAP.name+"@"+OVERLAP.parent.name] in archive["sha256"].values()


    replay = {r["image_id"]:r for r in rows(ROOT/"rank_audit"/"full_scores.csv")}
    assert all(r["image_id"] in replay and replay[r["image_id"]]["source"]==r["source"] for r in fixed)
    max_replay_delta = max(abs(float(r[h])-float(replay[r["image_id"]][h])) for r in fixed for h in HEADS)
    assert max_replay_delta < 1e-7
    historic_tau = archive["fixed_three_way_threshold"]
    computed = {grp: sum(float(r["max_log_likelihood"]) >= historic_tau for r in fixed
                         if (r["target"]!="unknown")==is_known)
                for grp, is_known in (("known", True), ("unknown", False))}
    assert computed=={"known":2852, "unknown":1749}
    result = dict(heads=HEADS, hashes=hashes, original_fixed_three_head_cutoff=archive["fixed_three_way_threshold"],
                  method="Min of linearly interpolated known-only percentiles of highest score and top1-minus-top2 gap. At boundary, higher remaining percentile wins. Cutoff is the lowest score of the top ceil(0.95*n) known images.",
                  input_audit={"matches_overlap_raw_hash":True,"substituted_matches_early_whole_file_hash":
                      hashes[SUBST.name+"@"+SUBST.parent.name] in archive["sha256"].values(),
                      "max_abs_error_vs_independent_score_archive_all_6997_x_5":max_replay_delta,
                      "historical_five_head_accepted":computed,"historical_firefly_ids_exact_match":True},
                  fixed_6997=compute(fixed,special,True),
                  broader_5known_4unknown_11000=compute(large,special))
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for key in ("fixed_6997","broader_5known_4unknown_11000"):
        for regime, out in result[key].items():
            print(key,regime, json.dumps({k:out[k] for k in ("known","unknown","historical_firefly_150","paired_unknown_drop_pp_ci95_fixed_thresholds")}))
        print("by unknown source",json.dumps(result[key]["exact95_pooled"]["unknown_by_source"]))


if __name__ == "__main__":
    main()
