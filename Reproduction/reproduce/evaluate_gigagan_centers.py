import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reference_neighbors import GATE_QUANTILE, HEADS, N_FOLDS, SOURCE_TO_HEAD, accept_top95, rows, write_rows

PREVIOUS = "previous_knn"
ONE = "one_center"
TWO = "two_centers"


def distances(data, phase, gallery):
    return {
        PREVIOUS: data[phase+"_giga5"],
        ONE: data[phase+"_one"] / data["one_radii"][gallery],
        TWO: (data[phase+"_two"] / data["two_radii"][gallery]).min(axis=1),
    }


def paired_count(mask, prior, method):
    p = prior[mask]
    m = method[mask]
    return dict(n=len(p), previous_count=int(p.sum()), method_count=int(m.sum()),
                previous_yes_to_no=int((p & ~m).sum()), previous_no_to_yes=int((~p & m).sum()),
                net_change=int(m.sum()-p.sum()), method_percent=100*float(m.mean()))


def paired_family_ci(mask, old, new, families):


    _, index, counts = np.unique(families[mask], return_inverse=True, return_counts=True)
    deltas = np.bincount(index, weights=new[mask].astype(int)-old[mask].astype(int))
    pairs, freq = np.unique(np.column_stack([counts,deltas]), axis=0, return_counts=True)
    rng = np.random.default_rng(7454)
    samples = rng.multinomial(len(counts), freq/len(counts), size=6000)
    change = 100*(samples @ pairs[:,1])/(samples @ pairs[:,0])
    return np.quantile(change,[.025,.975]).tolist()


def run(coordinates, previous, out):
    records = rows(previous)
    ids = np.array([r["image_id"] for r in records])
    sources = np.array([r["source"] for r in records])
    variants = np.array([r["variant"] for r in records])
    families = np.array([r["family"] for r in records])
    nominations = np.array([r["candidate"] for r in records])
    maximum = np.array([float(r["gmm_max"]) for r in records])
    fold = np.array([int(r["fold"]) for r in records])
    flag = np.array([r["historical_150"] == "1" for r in records])
    genuine = sources == "GigaGAN"
    giga_winner = nominations == "GigaGAN"
    imagenet = genuine & (variants=="GigaGAN_cond_imagenet256")
    coco = genuine & np.char.startswith(variants,"GigaGAN_t2i_coco256_")
    unknown = ~np.isin(sources, list(SOURCE_TO_HEAD))
    assert (int(imagenet.sum()), int(coco.sum()), int(unknown.sum()), int(flag.sum())) == (500,500,3997,150)
    assert (giga_winner & genuine).sum() == 966 and np.all(giga_winner[flag])
    with np.load(coordinates, allow_pickle=False) as npz:
        arrays = {k:npz[k] for k in npz.files}
    np.testing.assert_array_equal(arrays["ids"], ids)
    np.testing.assert_array_equal(arrays["fold"], fold)
    np.testing.assert_array_equal(arrays["heads"], HEADS)
    np.testing.assert_array_equal(arrays["two_names"], ["GigaGAN_cond_imagenet256","GigaGAN_t2i_coco256_pooled"])
    radii = arrays["two_radii"]
    assert radii.shape == (N_FOLDS,2) and np.all(radii>0) and np.all(arrays["one_radii"]>0)
    np.testing.assert_allclose(arrays["test_rel5"], [float(r["normalized_distance"]) for r in records],
                               rtol=1e-10, atol=1e-10)
    test_d = distances(arrays,"test",fold)
    cal_d = distances(arrays,"cal",(fold-1)%N_FOLDS)
    methods = list(test_d)
    norm_to_centers = arrays["test_two"]/radii[fold]
    near = norm_to_centers.argmin(axis=1)
    nearest_names = ["ImageNet_conditioned","COCO_text"]
    decisions = [dict(image_id=key, source=s, variant=v, fold=int(f), candidate=c, historical_150=int(h),
                      nearer_normalized_center=nearest_names[j], normalized_imagenet_distance=float(d[0]),
                      normalized_coco_distance=float(d[1]))
                 for key,s,v,f,c,h,j,d in zip(ids,sources,variants,fold,nominations,flag,near,norm_to_centers)]
    method_note = {"previous_knn":"Exact replay of the previous common 5NN check plus the GigaGAN-specific 5NN ceiling.",
                   "one_center":"Replace only the GigaGAN ceiling by Euclidean distance to its one pooled reference centroid, divided by its sample RMS radius.",
                   "two_centers":"Replace only the GigaGAN ceiling by the smaller of two radius-normalized distances (ImageNet conditional or pooled COCO text-to-image). Query variant is never used to choose the center."}
    report = dict(methods=method_note, protocol="All methods use the same 5 GMM scores, grouped folds, 98th percentile genuine-GigaGAN calibration ceiling, common winner 5-nearest-neighbor veto and 95% known coverage target.",
                  denominators="For GigaGAN we report accepted as any source AND correctly accepted as GigaGAN divided by all genuine images. The unknown metrics count acceptance under any candidate.",
                  cohort_note="The primary original_three_known operating point uses 3,000 known queries; genuine GigaGAN is a separate outcome. The all_five_known operating point includes GigaGAN in its 6,000 known queries. Both use the same 3,997 unknown queries.",
                  reference_counts_per_fold={"ImageNet_conditioned":100,"COCO_text":{"diff_noised":50,"rep":50}},
                  coordinate_sha256=hashlib.sha256(coordinates.read_bytes()).hexdigest(),
                  distance_columns=nearest_names, per_fold_radii=radii.tolist(),
                  previous_predictions_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(), protocols={})
    summaries=[]
    for name,known in [("original_three_known", np.isin(sources,list(SOURCE_TO_HEAD)[:3])),("all_five_known",~unknown)]:
        eligibility = {m:np.zeros(len(ids),dtype=bool) for m in methods}
        heldout = {m:np.zeros(len(ids),dtype=bool) for m in methods}
        cutoffs=[]
        for f in range(N_FOLDS):
            held = fold==f
            calibration = fold==(f+1)%N_FOLDS
            known_cal = known & calibration
            common_ceiling = float(np.quantile(arrays["cal_rel5"][known_cal], GATE_QUANTILE, method="higher"))
            common_cal = arrays["cal_rel5"] <= common_ceiling
            common_test = arrays["test_rel5"] <= common_ceiling
            for m in methods:
                ceiling = float(np.quantile(cal_d[m][calibration & genuine], GATE_QUANTILE, method="higher"))
                eligible_cal = common_cal & (~giga_winner | (cal_d[m]<=ceiling))
                eligible_test = common_test & (~giga_winner | (test_d[m]<=ceiling))
                eligibility[m][held] = eligible_test[held]
                _, cutoff = accept_top95(maximum,eligible_cal,known_cal)
                heldout[m][held] = eligible_test[held] & (maximum[held]>=cutoff)
                cutoffs.append(dict(test_fold=f, rule=m, common_ceiling=common_ceiling, gigagan_ceiling=ceiling,
                                    gmm_cutoff=cutoff, genuine_calibration_n=int((calibration & genuine).sum())))
        matched, matched_cut = {},{}
        for m in methods:
            matched[m],matched_cut[m] = accept_top95(maximum, eligibility[m],known)
        result = dict(fold_cutoffs=cutoffs, matched95_gmm_cutoffs=matched_cut)
        for mode, keep_dict in [("matched95",matched),("heldout",heldout)]:
            prior = keep_dict[PREVIOUS]
            prior_col=f"{name}_{mode}_combined_gigagan_gate"
            np.testing.assert_array_equal(prior,[r[prior_col]=="1" for r in records])
            metrics = {}
            for m, keep in keep_dict.items():
                details={}
                for label, mask in {"known":known,"all_unknown":unknown,"all_firefly":sources=="FireFly",
                                    "historical_150":flag,"rest_firefly":(sources=="FireFly") & ~flag,
                                    "dalle3":sources=="DALLE 3","stylegan2":sources=="StyleGAN2",
                                    "gigagan":genuine,"imagenet_gigagan":imagenet,"coco_gigagan":coco}.items():
                    metrics_to_use=[("accepted",prior,keep)]
                    if label in ("gigagan","imagenet_gigagan","coco_gigagan"):
                        metrics_to_use.append(("correctly_accepted_as_gigagan",prior & giga_winner,keep & giga_winner))
                    for measure, before, after in metrics_to_use:
                        count=paired_count(mask,before,after)
                        if label in ("all_unknown","all_firefly","historical_150","gigagan") and m!=PREVIOUS:
                            count["change_pp_ci95_fixed_family_bootstrap"]=paired_family_ci(mask,before,after,families)
                        details[f"{label}_{measure}"]=count
                        summaries.append(dict(protocol=name,mode=mode,rule=m,group=label,measure=measure,
                                              **{k:v for k,v in count.items() if not k.startswith("change_pp_ci")}))
                metrics[m]=details
                for row, accepted in zip(decisions,keep):
                    row[f"{name}_{mode}_{m}"]=int(accepted)
            result[mode]=metrics
        report["protocols"][name]=result
    out.mkdir(parents=True,exist_ok=True)
    write_rows(out/"summary.csv",summaries)
    write_rows(out/"decisions.csv",decisions)
    write_rows(out/"firefly_150_decisions.csv",[r for r in decisions if r["historical_150"]])
    (out/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    for group in ("known_accepted","all_unknown_accepted","all_firefly_accepted","historical_150_accepted",
                  "gigagan_accepted","gigagan_correctly_accepted_as_gigagan","imagenet_gigagan_correctly_accepted_as_gigagan",
                  "coco_gigagan_correctly_accepted_as_gigagan"):
        result=report["protocols"]["original_three_known"]["matched95"]
        print(group, [(m,result[m][group]["method_count"]) for m in methods])


if __name__ == "__main__":
    p=argparse.ArgumentParser(description='evaluate gigagan centers')
    p.add_argument("--coordinates",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.coordinates,args.previous,args.out)
