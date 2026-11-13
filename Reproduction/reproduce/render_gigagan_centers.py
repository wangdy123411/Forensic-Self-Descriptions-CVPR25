import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Line, PolyLine, String
from reportlab.lib import colors


GROUPS = (
    ("GigaGAN, ImageNet", "#176687", [], 500),
    ("GigaGAN, COCO text", "#795590", [5, 2.5], 500),
    ("Historical Firefly", "#AE4C25", [2, 1.6], 150),
)


def read_ratios(folder):
    report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    cutoffs = {
        name: {r["test_fold"]: r["gigagan_ceiling"] for r in p["fold_cutoffs"] if r["rule"] == "two_centers"}
        for name, p in report["protocols"].items()
    }
    limits = cutoffs["original_three_known"]
    assert len(limits) == 5 and all(other == limits for other in cutoffs.values())
    values = [[] for _ in GROUPS]
    seen = set()
    with (folder / "decisions.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "GigaGAN":
                if r["variant"] == "GigaGAN_cond_imagenet256":
                    group = 0
                elif r["variant"] in {"GigaGAN_t2i_coco256_diff_noised", "GigaGAN_t2i_coco256_rep"}:
                    group = 1
                else:
                    raise ValueError(f"Unexpected GigaGAN variant {r['variant']}")
            elif int(r["historical_150"]):
                assert r["source"] == "FireFly" and r["candidate"] == "GigaGAN"
                group = 2
            else:
                continue
            assert r["image_id"] not in seen
            seen.add(r["image_id"])
            ratio = min(float(r["normalized_imagenet_distance"]), float(r["normalized_coco_distance"])) / limits[int(r["fold"])]
            if not math.isfinite(ratio) or ratio < 0:
                raise ValueError("Expected finite nonnegative distance ratio")
            values[group].append(ratio)
    assert [len(v) for v in values] == [s[3] for s in GROUPS]
    return values


def render(values, folder):
    chart = Drawing(435, 216)
    left, right, low, high = 43, 426, 37, 171
    ink = colors.HexColor("#263544")
    grid = colors.HexColor("#DCE1E6")

    def text(x, y, label, anchor="middle", size=8.3):
        return String(x, y, label, textAnchor=anchor, fontName="Helvetica", fontSize=size, fillColor=ink)

    upper = max(1.5, math.ceil(max(max(v) for v in values) * 2) / 2)

    def xpos(ratio):
        return left + ratio * (right-left) / upper

    for i in range(6):
        y = low + i * (high-low) / 5
        chart.add(Line(left, y, right, y, strokeColor=grid, strokeWidth=0.5))
        chart.add(text(left-5, y-2.8, f"{i/5:.1f}", "end"))
    for i in range(round(upper*2) + 1):
        x = xpos(i / 2)
        chart.add(Line(x, low, x, low-3, strokeColor=ink, strokeWidth=0.7))
        chart.add(text(x, low-13, f"{i/2:g}"))
    chart.add(Line(left, low, right, low, strokeColor=ink, strokeWidth=0.7))
    chart.add(Line(left, low, left, high, strokeColor=ink, strokeWidth=0.7))
    chart.add(text((left+right)/2, 9, "Nearest-center distance / fold-specific GigaGAN limit", size=9))
    y_label = Group(text(0, 0, "Cumulative fraction of images", size=9))
    y_label.translate(11, (low+high)/2)
    y_label.rotate(90)
    chart.add(y_label)
    chart.add(Line(xpos(1), low, xpos(1), high+3, strokeColor=colors.HexColor("#536273"),
                   strokeWidth=0.7, strokeDashArray=[4, 3]))
    chart.add(text(xpos(1), high+8, "GigaGAN distance limit", size=8.3))

    for values_i, (label, hex_color, dash, n), key_x in zip(values, GROUPS, (43, 181, 326)):
        style = dict(strokeColor=colors.HexColor(hex_color), strokeWidth=1.6, strokeDashArray=dash)
        points = [left, low]
        cumulative, last_y = 0, low
        for value, count in sorted(Counter(values_i).items()):
            x = xpos(value)
            points.extend([x, last_y])
            cumulative += count
            last_y = low + (high-low) * cumulative / n
            points.extend([x, last_y])
        points.extend([right, high])
        chart.add(PolyLine(points, fillColor=None, **style))
        chart.add(Line(key_x, 204, key_x+14, 204, **style))
        chart.add(text(key_x+18, 201, f"{label} (n={n})", "start", size=8.15))

    base = folder / "gigagan_two_center_distance"
    renderSVG.drawToFile(chart, str(base.with_suffix(".svg")))
    with pdfium.PdfDocument(renderPDF.drawToString(chart)) as pdf:
        pdf[0].render(scale=4).to_pil().save(base.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render gigagan centers')
    parser.add_argument("comparison_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "gigagan_centers/comparison")
    args = parser.parse_args()
    cohorts = read_ratios(args.comparison_dir)
    render(cohorts, args.comparison_dir)
    for data, (name, _, _, n) in zip(cohorts, GROUPS):
        print(f"{name}: {sum(x <= 1 for x in data)}/{n} within the fold-specific GigaGAN limit")
