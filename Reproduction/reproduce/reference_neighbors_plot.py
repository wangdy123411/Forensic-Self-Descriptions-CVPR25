import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Line, PolyLine, String
from reportlab.lib import colors


STYLES = {
    "genuine_gigagan": ("Held-out GigaGAN", "#176889", []),
    "historical_firefly": ("Misaccepted Firefly", "#B64D26", [4, 2]),
}


def read_distances(path):
    groups = {name: [] for name in STYLES}
    image_ids = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"group", "distance", "image_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected group, distance and image_id columns")
        for row in reader:
            group, ident = row["group"], row["image_id"].strip()
            distance = float(row["distance"])
            if group not in groups or not ident or ident in image_ids:
                raise ValueError(f"Invalid group or duplicate/empty image ID at line {reader.line_num}")
            if not math.isfinite(distance) or distance < 0:
                raise ValueError(f"Invalid distance at line {reader.line_num}")
            groups[group].append(distance)
            image_ids.add(ident)
    if any(not values for values in groups.values()):
        raise ValueError("Each cohort must have at least one observation")
    return groups


def make_plot(groups):
    width, height = 346, 199
    left, right, bottom, top = 45, 336, 39, 167
    graph = Drawing(width, height)
    ink = colors.HexColor("#243241")
    grid = colors.HexColor("#D8E0E5")

    def label(x, y, caption, anchor="middle", size=8):
        return String(x, y, caption, textAnchor=anchor, fontName="Helvetica",
                      fontSize=size, fillColor=ink)

    minimum = min(min(values) for values in groups.values())
    maximum = max(max(values) for values in groups.values())

    rough_step = (maximum - minimum or max(maximum, 1)) / 5
    unit = 10 ** math.floor(math.log10(rough_step))
    step = next(scale for scale in (1, 2, 2.5, 5, 10) if unit * scale >= rough_step) * unit
    first, last = math.floor(minimum / step), math.ceil(maximum / step)
    if first == last:
        last += 1
    lo, hi = first * step, last * step

    def xpos(value):
        return left + (value - lo) * (right - left) / (hi - lo)

    for i in range(6):
        y = bottom + i * (top - bottom) / 5
        graph.add(Line(left, y, right, y, strokeColor=grid, strokeWidth=0.4))
        graph.add(label(left - 5, y - 2.8, f"{i / 5:.1f}", "end"))
    for i in range(first, last + 1):
        x = xpos(i * step)
        graph.add(Line(x, bottom - 3, x, bottom, strokeColor=ink, strokeWidth=0.6))
        graph.add(label(x, bottom - 13, f"{i * step:g}"))
    graph.add(Line(left, bottom, right, bottom, strokeColor=ink, strokeWidth=0.7))
    graph.add(Line(left, bottom, left, top, strokeColor=ink, strokeWidth=0.7))
    graph.add(label((left + right) / 2, 9, "Mean raw FSD distance to 5 nearest GigaGAN references", size=8.8))
    vertical = Group(label(0, 0, "Cumulative fraction of images", size=8.8))
    vertical.translate(11, (top + bottom) / 2)
    vertical.rotate(90)
    graph.add(vertical)

    for (group, (name, hex_color, dash)), legend_x in zip(STYLES.items(), (left, 194)):
        values = groups[group]
        style = dict(strokeColor=colors.HexColor(hex_color), strokeWidth=1.5,
                     strokeDashArray=dash)
        points = [left, bottom]
        y = bottom
        cumulative = 0
        for distance, count in sorted(Counter(values).items()):
            x = xpos(distance)
            points.extend([x, y])
            cumulative += count
            y = bottom + (top - bottom) * cumulative / len(values)
            points.extend([x, y])
        points.extend([right, top])
        graph.add(PolyLine(points, fillColor=None, **style))
        graph.add(Line(legend_x, 184, legend_x + 17, 184, **style))
        graph.add(label(legend_x + 21, 181, f"{name} (n={len(values):,})", "start", size=8.4))
    return graph


def main():
    parser = argparse.ArgumentParser(description='reference neighbors plot')
    parser.add_argument("output_dir", type=Path, help="Directory containing distances.csv")
    args = parser.parse_args()
    graph = make_plot(read_distances(args.output_dir / "distances.csv"))
    base = args.output_dir / "gigagan_firefly_distance"
    renderSVG.drawToFile(graph, str(base.with_suffix(".svg")))
    with pdfium.PdfDocument(renderPDF.drawToString(graph)) as pdf:
        pdf[0].render(scale=4).to_pil().save(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
