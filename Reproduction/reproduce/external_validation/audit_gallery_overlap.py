import csv
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from audit_external_overlap import HERE, fingerprint, judge, positive_controls, prepare, read_csv


def main():
    folder = HERE / "input" / "gigagan_gallery"
    old_rows, previous_new, old_hashes, n_inventory = prepare()
    for row in previous_new:
        row["scope"] = "main_external"
    with ThreadPoolExecutor(max_workers=6) as pool:
        reference = list(pool.map(fingerprint, old_rows + previous_new))
    deadline = time.monotonic() + 180
    while not (folder / "selection.csv").is_file():
        if time.monotonic() > deadline:
            raise FileNotFoundError(folder / "selection.csv")
        time.sleep(2)
    selection = read_csv(folder / "selection.csv")
    gallery_rows = []
    for row in selection:
        physical = folder / row["path"]
        gallery_rows.append(dict(source="gigagan", image_id=row["image_id"], physical=physical,
                                 sha256=row["sha256"], family="paper_fig_A8/" + row["image_id"], scope="gallery",
                                 prompt_group=row["prompt_group"], example_index=row["example_index"]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        gallery = list(pool.map(fingerprint, gallery_rows))
    results, matches = [], []
    for index, item in enumerate(gallery):
        flags = []
        closest = {}
        if item["sha256"] in old_hashes:
            flags.append("inventory_bytes/" + old_hashes[item["sha256"]])
        for ref in reference + gallery[:index]:
            kind, ph, dh, mae = judge(item, ref)
            scope = ref["scope"]
            near = (ph, dh, round(mae, 3), ref["image_id"])
            if scope not in closest or near < closest[scope]:
                closest[scope] = near
            if kind:
                flags.append(kind + "/" + ref["image_id"])
                matches.append(dict(image_id=item["image_id"], target_id=ref["image_id"], scope=scope,
                                    reason=kind, phash=ph, dhash=dh, color_mae=round(mae, 3)))
        results.append(dict(image_id=item["image_id"], source="GigaGAN", sha256=item["sha256"],
                            prompt_group=item["prompt_group"], example_index=item["example_index"],
                            eligible=int(not flags), overlap_flags="|".join(flags),
                            old_nearest_phash=closest["old"][0], old_nearest_id=closest["old"][3],
                            external_nearest_phash=closest["main_external"][0],
                            external_nearest_id=closest["main_external"][3],
                            gallery_nearest_phash=closest.get("gallery", [""])[0]))
    out = HERE / "overlap_gallery"
    out.mkdir(exist_ok=True)
    tmp = out / "overlap.csv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    tmp.replace(out / "overlap.csv")
    controls = positive_controls(gallery)
    report = dict(figure="GigaGAN supplemental Figure A8", gallery_size=len(gallery),
                  eligible=sum(r["eligible"] for r in results), prompt_group_counts=dict(Counter(r["prompt_group"] for r in results)),
                  previous_full_inventory_rows=n_inventory, inventory_and_prior_supplement_hash_count=len(old_hashes),
                  old_rgb_reference_count=len(old_rows), external_rgb_reference_count=len(previous_new),
                  near_duplicate_rule="Full byte/pixel hashes, (pHash<=8 OR dHash<=6) AND 32x32 RGB MAE<=9. "
                                      "Old/new Firefly and GigaGAN all screened, plus all pairs within the figure. "
                                      "Distinct images from one prompt stay in the cohort. Illustrative selection and "
                                      "five prompt groups limit generalizability.",
                  possible_matches=matches, positive_controls=controls)
    (out / "overlap.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Eligible", report["eligible"], "/", len(gallery), "; matches", len(matches),
          "; group counts", report["prompt_group_counts"], flush=True)


if __name__ == "__main__":
    main()
