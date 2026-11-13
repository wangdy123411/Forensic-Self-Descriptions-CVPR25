from pathlib import Path
import xml.etree.ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
import pypdfium2 as pdfium


ERRORS = [(160, 5, 0, 0, 0), (871, 172, 19, 1, 0), (582, 283, 265, 288, 1014)]
GMM_COUNTS = [3, 5, 14]
RANKS = ["2nd", "3rd", "4th", "5th", "6th–14th"]
COLORS = ["#286485", "#80BDCF", "#F5D785", "#D88751", "#775485"]


def render_svg(output):
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="246pt",
                     height="133pt", viewBox="0 0 246 133", role="img",
                     attrib={"aria-labelledby": "title description"})
    ET.SubElement(svg, "title", id="title").text = "Rank of the true source among attribution errors"
    ET.SubElement(svg, "desc", id="description").text = (
        "Each bar covers known images assigned the wrong top-ranked source before rejection. "
        "Colors show the true source's rank. Percentages use the number of errors in each protocol.")
    ET.SubElement(svg, "rect", width="246", height="133", fill="#FFFFFF")
    group = ET.SubElement(svg, "g", attrib={"font-family": "Helvetica, Arial, sans-serif",
                                           "fill": "#172F3B"})

    def label(x, y, value, *, size=7.7, align="middle", parent=group, **attrs):
        ET.SubElement(parent, "text", x=f"{x:g}", y=f"{y:g}", attrib={
            "font-size": str(size), "text-anchor": align, **attrs}).text = value

    def box(parent, x, y, width, height, color):
        return ET.SubElement(parent, "rect", x=f"{x:.6f}", y=f"{y:.6f}",
                             width=f"{width:.6f}", height=f"{height:.6f}", fill=color)

    for tick in (0, 25, 50, 75, 100):
        y = 109 - 89 * tick/100
        ET.SubElement(group, "line", x1="36", x2="237", y1=str(y), y2=str(y),
                      stroke="#D8DEE3", attrib={"stroke-width": ".4"})
        label(33, y+2.7, str(tick), align="end", fill="#303A41")
    label(9, 64.5, "Share of errors (%)", transform="rotate(-90 9 64.5)")
    for x, n, row in zip((66, 134, 202), GMM_COUNTS, ERRORS):
        total = sum(row)
        protocol = ET.SubElement(group, "g", id=f"gmm-{n}")
        accumulated = 0
        for i, count in enumerate(row):
            if not count:
                continue
            accumulated += count
            top = 109 - 89 * accumulated/total
            h = 89 * count/total
            share = 100 * count/total
            rect = box(protocol, x-21.5, top, 43, h, COLORS[i])
            ET.SubElement(rect, "title").text = f"{RANKS[i]}: {count:,} of {total:,} ({share:.2f}%)"
            if share >= 9:
                label(x, top+h/2+2.6, f"{share:.1f}%", parent=protocol, size=7.6,
                      fill="#FFFFFF" if i in (0, 4) else "#172F3B",
                      **{"font-weight": "bold" if i == 0 else "normal"})
        label(x, 119.7, f"{n} GMMs", parent=protocol, size=8.4)
        label(x, 129.6, f"(n={total:,})", parent=protocol)
    label(12, 10, "True-source rank:", align="start", size=8.2)
    for x, rank, color in zip((82, 111, 138, 165, 191), RANKS, COLORS):
        box(group, x, 3.7, 6.3, 6.3, color)
        label(x+8.3, 10, rank, align="start", size=8.2)
    ET.indent(svg)
    ET.ElementTree(svg).write(output.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)


def render(output):
    pdf = output.with_suffix(".pdf")
    c = Canvas(str(pdf), pagesize=(246, 133))
    c.setTitle("Rank of the true source among attribution errors")
    cx = [66, 134, 202]
    x0, x1, y0, h, w = 36, 237, 24, 89, 43
    def ink(color):
        c.setFillColor(HexColor(color))
    for tick in (0, 25, 50, 75, 100):
        y = y0 + h * tick/100
        c.setLineWidth(.4)
        c.setStrokeColor(HexColor("#D8DEE3"))
        c.line(x0, y, x1, y)
        ink("#303A41")
        c.setFont("Helvetica", 7.7)
        c.drawRightString(x0 - 3, y-2.7, str(tick))
    c.saveState()
    c.translate(9, y0 + h/2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Share of errors (%)")
    c.restoreState()
    for x, n, row in zip(cx, GMM_COUNTS, ERRORS):
        total = sum(row)
        bottom = y0
        for i, count in enumerate(row):
            if count == 0:
                continue
            share = count * 100 / total
            height = h * count / total
            ink(COLORS[i])
            c.rect(x-w/2, bottom, w, height, stroke=0, fill=1)
            if share >= 9:
                ink("#FFFFFF" if i in (0, 4) else "#172F3B")
                c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 7.6)
                c.drawCentredString(x, bottom+height/2-2.6, f"{share:.1f}%")
            bottom += height
        ink("#172F3B")
        c.setFont("Helvetica", 8.4)
        c.drawCentredString(x, 13.3, f"{n} GMMs")
        c.setFont("Helvetica", 7.7)
        c.drawCentredString(x, 3.4, f"(n={total:,})")
    c.setFont("Helvetica", 8.2)
    c.drawString(12, 123, "True-source rank:")
    for x, rank, color in zip([82, 111, 138, 165, 191], RANKS, COLORS):
        ink(color)
        c.rect(x, 123, 6.3, 6.3, fill=1, stroke=0)
        ink("#172F3B")
        c.drawString(x+8.3, 123, rank)
    c.showPage()
    c.save()
    with pdfium.PdfDocument(str(pdf)) as doc:
        doc[0].render(scale=4).to_pil().save(output.with_suffix(".png"))


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "true_source_rank_errors"
    render(output)
    render_svg(output)
