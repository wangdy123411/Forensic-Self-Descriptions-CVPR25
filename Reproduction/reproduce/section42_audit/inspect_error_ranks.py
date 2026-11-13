from project_paths import DATA_ROOT

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compare_rejection_rules import split_known


HERE = Path(__file__).resolve().parent
OUT = HERE / "rank_audit"
DATA = DATA_ROOT
SUBST = HERE.parent / "section42_paper_substituted"
OVERLAP = HERE.parent / "section42_paper_overlap"
MNW = SUBST / "supplemental_firefly_mnw"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranked_scores(row, names):
    return sorted(((name, float(row[name])) for name in names),
                  key=lambda item: item[1], reverse=True)


def ranks(rows, key):
    return dict(sorted(Counter(int(r[key]) for r in rows if r[key] != "").items()))


def summarize_unknown(rows):
    supported = [r for r in rows if r["full14_true_rank"] != ""]
    return {
        "count": len(rows),
        "with_true_model_and_full_scores": len(supported),
        "rank_if_only_true_model_added": ranks(supported, "rank_if_only_true_model_added"),
        "rank_in_full14": ranks(supported, "full14_true_rank"),
        "without_full_rank": dict(Counter(r["full_score_status"] for r in rows
                                           if r["full14_true_rank"] == "")),
    }


def run():
    full = {r["image_id"]: r for r in read(OUT / "full_scores.csv")}
    extra = OUT / "supplemental_full_scores.csv"
    if extra.is_file():
        full.update({r["image_id"]: r for r in read(extra)})
    paths = {r["image_id"]: str(DATA / r["relative_path"])
             for r in read(DATA / "manifests/provenance.csv")}
    paths.update({r["image_id"]: str(MNW / r["path"]) for r in read(MNW / "manifest.csv")})
    fixed = json.loads((HERE / "fixed_overlap_candidate_additions.json").read_text())
    fixed_tau = fixed["fixed_three_way_threshold"]
    base = {r["image_id"]: r for r in read(OVERLAP / "predictions.csv")}
    report = {}
    for name, folder in (("3plus3", OVERLAP), ("5plus4", SUBST)):
        predictions = read(folder / "predictions.csv")
        metadata = json.loads((folder / "metrics.json").read_text())
        names = metadata["candidates"]
        tau = metadata["descriptive_threshold"]["tau"]
        ids = [r["image_id"] for r in predictions]
        sources = np.array([r["source"] for r in predictions])
        known = np.array([r["target"] != "unknown" for r in predictions])
        folds = split_known(ids, sources, known)
        heldout = sorted(float(r["max_log_likelihood"]) for r, f in zip(predictions, folds)
                         if f == "evaluate")
        matched_tau = heldout[len(heldout) - int(np.ceil(0.95 * len(heldout)))]
        assert heldout.count(matched_tau) == 1
        known_errors, unknowns, rejected_correct = [], [], []
        for row, fold in zip(predictions, folds):
            scores = ranked_scores(row, names)
            winner, best = scores[0]
            assert winner == row["candidate"] and abs(best - float(row["max_log_likelihood"])) < 1e-7
            accepted = best >= tau
            image_id, source, target = row["image_id"], row["source"], row["target"]
            detail = dict(image_id=image_id, image_path=paths[image_id], source=source,
                          winning_gmm=winner, winning_score=best,
                          accepted_at_original_threshold=int(accepted),
                          heldout_split=fold, accepted_at_matched95_threshold=int(best >= matched_tau))
            if target != "unknown":
                truth = float(row[target])
                detail.update(true_gmm=target, true_score=truth,
                              true_rank=1 + sum(s > truth for _, s in scores),
                              winner_minus_true=best - truth)
                detail.update({f"score_{c}": float(row[c]) for c in names})
                if winner != target:
                    known_errors.append(detail)
                elif not accepted:
                    rejected_correct.append(detail)
                continue
            full_row = full.get(image_id)
            if full_row and full_row["source"] != source:
                raise ValueError(f"Source mismatch on {image_id}")
            true_gmm = full_row["target"] if full_row else ""
            lifted = (name == "5plus4" and image_id in base and source == "FireFly"
                      and float(base[image_id]["max_log_likelihood"]) < fixed_tau
                      and winner == "GigaGAN" and best >= fixed_tau)
            if not (accepted or best >= matched_tau or lifted):
                continue
            if true_gmm:
                assert true_gmm not in names
                if max(abs(float(row[c]) - float(full_row[c])) for c in names) > 1e-5:
                    raise ValueError(f"Different models or image features on {image_id}")
            detail.update(true_gmm_in_protocol="", true_rank_in_protocol="",
                          full_score_status=("available" if true_gmm else
                                             "no released GMM" if full_row else "no full score row"),
                          full14_true_gmm=true_gmm, full14_true_score=full_row["true_score"] if true_gmm else "",
                          full14_true_rank=full_row["true_rank"] if true_gmm else "",
                          full14_winning_gmm=full_row["candidate"] if full_row else "",
                          rank_if_only_true_model_added=(1 + sum(s > float(full_row["true_score"])
                                                                for _, s in scores) if true_gmm else ""),
                          newly_accepted_firefly_as_gigagan_fixed_experiment=int(lifted))
            detail.update({f"score_{c}": float(row[c]) for c in names})
            unknowns.append(detail)
        write(f"known_top1_errors_{name}.csv", known_errors)
        write(f"known_top1_correct_but_rejected_{name}.csv", rejected_correct)
        write(f"unknown_false_accepts_{name}.csv", unknowns)
        original_unknowns = [r for r in unknowns if r["accepted_at_original_threshold"]]
        matched_unknowns = [r for r in unknowns if r["accepted_at_matched95_threshold"]]
        by_source = defaultdict(list)
        for r in original_unknowns:
            by_source[r["source"]].append(r)
        report[name] = dict(
            known_n=int(known.sum()), original_descriptive_threshold=tau, heldout_matched95_threshold=matched_tau,
            known_top1_errors=len(known_errors), true_ranks_before_threshold=ranks(known_errors, "true_rank"),
            true_ranks_among_accepted_wrong=ranks([r for r in known_errors if r["accepted_at_original_threshold"]],
                                                  "true_rank"),
            known_top1_correct_but_rejected=len(rejected_correct),
            unknown_false_accepts_original=summarize_unknown(original_unknowns),
            unknown_false_accepts_heldout_matched95=summarize_unknown(matched_unknowns),
            by_unknown_source_original={k: summarize_unknown(v) for k, v in by_source.items()})
        firefly = [r for r in unknowns if r["newly_accepted_firefly_as_gigagan_fixed_experiment"]]
        if name == "5plus4":
            assert len(firefly) == 150
            write("firefly_150.csv", firefly)
            report[name]["firefly_150"] = summarize_unknown(firefly)
    (OUT / "error_rank_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
