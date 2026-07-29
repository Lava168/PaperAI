from __future__ import annotations

import math
from pathlib import Path


OUT_DIR = Path("/Users/mac/Documents/A2C/figures_vector")
SVG = OUT_DIR / "Figure1_A2C_NODE_framework.svg"

W = 2400
H = 1500

BLUE = "#1f6fb2"
BLUE_DARK = "#0f4e8a"
CYAN = "#4aa6bd"
GREEN = "#168b7a"
ORANGE = "#e56b2f"
PURPLE = "#7a4da3"
GRAY = "#4d5a66"
LIGHT_BG = "#f8fbfd"
INK = "#18202a"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class SVGDoc:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, text: str) -> None:
        self.parts.append(text)

    def rect(self, x, y, w, h, fill="none", stroke="#000", sw=2, rx=10, opacity=1, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{dash_attr}/>'
        )

    def line(self, x1, y1, x2, y2, stroke="#000", sw=2, marker=False, dash=None, opacity=1):
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{dash_attr}{marker_attr}/>'
        )

    def path(self, d, fill="none", stroke="#000", sw=2, marker=False, opacity=1, dash=None):
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"{dash_attr}{marker_attr}/>'
        )

    def circle(self, cx, cy, r, fill="#fff", stroke="#000", sw=2, opacity=1):
        self.add(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def ellipse(self, cx, cy, rx, ry, fill="#fff", stroke="#000", sw=2, opacity=1):
        self.add(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def text(self, x, y, text, size=24, fill=INK, weight=400, anchor="start", italic=False, family="Arial"):
        style = "font-style:italic;" if italic else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" style="{style}">{esc(text)}</text>'
        )

    def multiline(self, x, y, lines, size=22, fill=INK, weight=400, anchor="start", leading=1.2, family="Arial"):
        for i, line in enumerate(lines):
            self.text(x, y + i * size * leading, line, size=size, fill=fill, weight=weight, anchor=anchor, family=family)

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.parts)
        SVG.write_text(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M2,2 L10,6 L2,10 Z" fill="#26323f"/>
  </marker>
  <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="#72808f" flood-opacity="0.22"/>
  </filter>
  <linearGradient id="mriGrad" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0" stop-color="#0f1720"/>
    <stop offset="0.55" stop-color="#566170"/>
    <stop offset="1" stop-color="#d8e0e8"/>
  </linearGradient>
  <linearGradient id="odeCurve" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#1a9850"/>
    <stop offset="0.52" stop-color="#f0b429"/>
    <stop offset="1" stop-color="#d73027"/>
  </linearGradient>
</defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
{content}
</svg>
''',
            encoding="utf-8",
        )


def add_panel(svg: SVGDoc, x, y, w, h, num, title, color=BLUE):
    svg.rect(x, y, w, h, fill="#ffffff", stroke=color, sw=2.2, rx=14)
    svg.add(f'<g filter="url(#softShadow)">')
    svg.add(f'<circle cx="{x+34:.1f}" cy="{y+34:.1f}" r="20" fill="{color}" stroke="none"/>')
    svg.text(x + 34, y + 42, str(num), size=24, fill="#fff", weight=700, anchor="middle")
    svg.add("</g>")
    svg.multiline(x + 68, y + 30, title, size=22, fill=color, weight=700, leading=1.1)


def add_arrow(svg: SVGDoc, x1, y1, x2, y2):
    svg.line(x1, y1, x2, y2, stroke="#26323f", sw=4, marker=True)


def add_mri_stack(svg: SVGDoc, x, y, label, color):
    svg.rect(x - 10, y - 5, 54, 76, fill=color, stroke=color, sw=1.5, rx=8, opacity=0.18)
    svg.text(x + 17, y + 43, label, size=22, fill=color, weight=700, anchor="middle")
    for i in range(5):
        dx = i * 7
        dy = i * -5
        svg.rect(x + 72 + dx, y + 14 + dy, 110, 82, fill="url(#mriGrad)", stroke="#151a20", sw=1.4, rx=5)
        svg.ellipse(x + 127 + dx, y + 55 + dy, 38, 49, fill="none", stroke="#e6eef6", sw=2, opacity=0.8)
        svg.path(
            f"M{x+108+dx},{y+77+dy} C{x+126+dx},{y+25+dy} {x+154+dx},{y+35+dy} {x+150+dx},{y+80+dy}",
            stroke="#e6eef6",
            sw=2,
            opacity=0.65,
        )
    svg.text(x + 232, y + 53, "...", size=28, fill=GRAY)


def add_followup(svg: SVGDoc, x, y):
    svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="#1d242d"/>')
    svg.rect(x - 18, y + 18, 36, 42, fill="#1d242d", stroke="none", rx=16)
    svg.line(x + 36, y + 38, x + 250, y + 38, stroke="#1d242d", sw=2, marker=True)
    labels = ["Baseline", "12m", "24m", "...", "Tn"]
    for i, lab in enumerate(labels):
        cx = x + 58 + i * 43
        svg.circle(cx, y + 38, 10, fill="#fff", stroke="#1d242d", sw=2)
        svg.text(cx, y + 68, lab, size=14, fill=INK, anchor="middle")


def add_brain(svg: SVGDoc, x, y, scale=1, colored=True):
    colors = ["#df7d74", "#e9a44a", "#d1c85b", "#70a95c", "#54a4ad", "#6d8bc2", "#9d6fb1", "#c6709b"]
    svg.path(
        f"M{x},{y+70*scale} C{x+18*scale},{y+18*scale} {x+90*scale},{y} {x+150*scale},{y+22*scale} "
        f"C{x+200*scale},{y+42*scale} {x+202*scale},{y+112*scale} {x+152*scale},{y+135*scale} "
        f"C{x+92*scale},{y+160*scale} {x+15*scale},{y+132*scale} {x},{y+70*scale} Z",
        fill="#edf2f5",
        stroke="#49525c",
        sw=2,
    )
    for i in range(8):
        sx = x + (22 + (i % 4) * 37) * scale
        sy = y + (36 + (i // 4) * 48) * scale
        if colored:
            svg.path(
                f"M{sx},{sy} c{18*scale},-{22*scale} {45*scale},-{8*scale} {45*scale},{20*scale} "
                f"c-{8*scale},{24*scale} -{43*scale},{25*scale} -{58*scale},{4*scale} z",
                fill=colors[i],
                stroke="#ffffff",
                sw=1.2,
                opacity=0.85,
            )
    svg.path(
        f"M{x+45*scale},{y+75*scale} C{x+70*scale},{y+50*scale} {x+112*scale},{y+52*scale} {x+135*scale},{y+82*scale}",
        stroke="#49525c",
        sw=2,
        opacity=0.55,
    )
    svg.path(
        f"M{x+55*scale},{y+103*scale} C{x+92*scale},{y+90*scale} {x+116*scale},{y+98*scale} {x+145*scale},{y+120*scale}",
        stroke="#49525c",
        sw=2,
        opacity=0.45,
    )


def add_region_graph(svg: SVGDoc, cx, cy, r=100, n=21, colors=None, labels=True):
    if colors is None:
        colors = ["#1f9d8a", "#64a762", "#e6b53c", "#e56b2f", "#d9453d", "#7d5fb2", "#396ab1"]
    pts = []
    for i in range(n):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    for i, (x1, y1) in enumerate(pts):
        for j in range(i + 1, n):
            if (i * 7 + j * 3) % 11 in (0, 1, 2):
                x2, y2 = pts[j]
                svg.line(x1, y1, x2, y2, stroke="#a6b0bb", sw=1, opacity=0.45)
    for i, (px, py) in enumerate(pts):
        fill = colors[i % len(colors)]
        svg.circle(px, py, 16, fill=fill, stroke="#fff", sw=2)
        if labels and i in (0, 1, 4, 7, 12, 15, 18, 20):
            svg.text(px, py + 6, str(i + 1), size=13, fill="#fff", weight=700, anchor="middle")


def add_matrix(svg: SVGDoc, x, y, rows=8, cols=8, cell=15, palette="blue"):
    for r in range(rows):
        for c in range(cols):
            base = 0.08 + 0.75 * math.exp(-abs(r - c) / 2.2)
            val = min(1, max(0, base + 0.18 * math.sin((r + 1) * (c + 2))))
            if palette == "blue":
                fill = f"rgb({int(230 - 155*val)},{int(242 - 112*val)},{int(252 - 32*val)})"
            elif palette == "green":
                fill = f"rgb({int(238 - 160*val)},{int(248 - 82*val)},{int(235 - 135*val)})"
            else:
                fill = f"rgb({int(245 - 105*val)},{int(232 - 102*val)},{int(248 - 72*val)})"
            svg.rect(x + c * cell, y + r * cell, cell, cell, fill=fill, stroke="#fff", sw=0.7, rx=0)


def add_encoder_blocks(svg: SVGDoc, x, y):
    for i in range(5):
        w = 52 - i * 5
        h = 120 - i * 15
        svg.rect(x + i * 34, y + i * 14, w, h, fill="#b9d4ef", stroke="#4f88c0", sw=2, rx=4)
    svg.line(x - 48, y + 58, x - 10, y + 58, stroke="#26323f", sw=3, marker=True)


def add_formula_box(svg: SVGDoc, x, y, w, h, lines, stroke="#8fbbe3", fill="#f3f8fe", size=20):
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=2, rx=10)
    svg.multiline(x + w / 2, y + 30, lines, size=size, fill=INK, weight=500, anchor="middle", leading=1.15, family="Times New Roman")


def add_gauge(svg: SVGDoc, cx, cy):
    for i, color in enumerate(["#4fa85b", "#f0c33b", "#f39a2c", "#d94a38"]):
        start = math.pi - i * math.pi / 4
        end = math.pi - (i + 0.85) * math.pi / 4
        x1, y1 = cx + 70 * math.cos(start), cy - 70 * math.sin(start)
        x2, y2 = cx + 70 * math.cos(end), cy - 70 * math.sin(end)
        svg.path(f"M{x1},{y1} A70,70 0 0 1 {x2},{y2}", stroke=color, sw=18, fill="none")
    svg.line(cx, cy, cx + 44, cy - 40, stroke="#1d242d", sw=5)
    svg.circle(cx, cy, 8, fill="#1d242d", stroke="#fff", sw=2)


def add_scatter(svg: SVGDoc, x, y):
    svg.line(x, y + 100, x + 160, y + 100, stroke="#4e5964", sw=2)
    svg.line(x, y + 100, x, y, stroke="#4e5964", sw=2)
    for i in range(26):
        px = x + 12 + i * 5.4
        py = y + 84 - 0.42 * i * 5.4 + 13 * math.sin(i * 1.7)
        svg.circle(px, py, 4, fill="#3277b7", stroke="none", opacity=0.75)
    svg.line(x + 8, y + 84, x + 150, y + 25, stroke="#252a31", sw=2)
    svg.text(x + 80, y + 127, "Actual MMSE", size=13, fill=INK, anchor="middle")
    svg.text(x - 18, y + 58, "Predicted", size=13, fill=INK, anchor="middle")


def add_timeline(svg: SVGDoc, x, y):
    svg.line(x, y, x + 230, y, stroke="#2c3540", sw=2)
    for i, lab in enumerate(["t0", "t1", "t2", "t3", "t4", "...", "T"]):
        cx = x + i * 35
        svg.circle(cx, y, 7, fill="#fff", stroke="#2c3540", sw=2)
        svg.text(cx, y - 17, lab, size=13, fill=INK, anchor="middle")
    svg.text(x + 115, y + 38, "imputation / robustness", size=17, fill=INK, anchor="middle")


def main() -> None:
    svg = SVGDoc()
    svg.text(28, 58, "Figure 1. A2C-NODE: overall longitudinal study workflow and model framework", size=34, fill="#0f1720", weight=800)

    top_y = 96
    gap = 14
    widths = [300, 390, 345, 345, 360, 560]
    xs = [18]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    htop = 1015

    add_panel(svg, xs[0], top_y, widths[0], htop, 1, ["Data input"], BLUE)
    add_panel(svg, xs[1], top_y, widths[1], htop, 2, ["MRI preprocessing +", "atlas segmentation"], BLUE)
    add_panel(svg, xs[2], top_y, widths[2], htop, 3, ["3D feature encoder", "and region pooling"], BLUE)
    add_panel(svg, xs[3], top_y, widths[3], htop, 4, ["Dynamic graph", "construction"], BLUE)
    add_panel(svg, xs[4], top_y, widths[4], htop, 5, ["Graph Neural ODE"], GREEN)
    add_panel(svg, xs[5], top_y, widths[5], 720, 6, ["Longitudinal multi-task outputs"], ORANGE)

    # Panel 1
    x, y, w = xs[0], top_y, widths[0]
    svg.multiline(x + w / 2, y + 130, ["ADNI / AIBL", "longitudinal T1-MRI"], size=23, weight=700, anchor="middle")
    add_mri_stack(svg, x + 22, y + 205, "CN", "#4fa85b")
    add_mri_stack(svg, x + 22, y + 420, "MCI", "#e5a735")
    add_mri_stack(svg, x + 22, y + 635, "AD", "#d94a38")
    add_followup(svg, x + 42, y + 850)
    svg.multiline(
        x + 32,
        y + 965,
        ["CN: Cognitively normal", "MCI: Mild cognitive impairment", "AD: Alzheimer's disease"],
        size=16,
        fill=INK,
        leading=1.4,
    )

    # Panel 2
    x, y, w = xs[1], top_y, widths[1]
    svg.multiline(x + 95, y + 150, ["Skull", "stripping"], size=15, weight=700, anchor="middle")
    svg.multiline(x + 190, y + 150, ["Registration"], size=15, weight=700, anchor="middle")
    svg.multiline(x + 295, y + 150, ["Normalization"], size=15, weight=700, anchor="middle")
    svg.path(f"M{x+68},{y+228} C{x+82},{y+155} {x+145},{y+155} {x+145},{y+228}", fill="#d3d8dc", stroke="#67717b", sw=2)
    add_brain(svg, x + 145, y + 175, scale=0.55, colored=False)
    for i in range(4):
        svg.rect(x + 282 + i * 9, y + 188 - i * 7, 68, 68, fill="#d7dce1", stroke="#59636e", sw=1.5, rx=0)
        for k in range(1, 4):
            svg.line(x + 282 + i * 9 + k * 17, y + 188 - i * 7, x + 282 + i * 9 + k * 17, y + 256 - i * 7, stroke="#87919b", sw=0.8)
            svg.line(x + 282 + i * 9, y + 188 - i * 7 + k * 17, x + 350 + i * 9, y + 188 - i * 7 + k * 17, stroke="#87919b", sw=0.8)
    svg.line(x + 115, y + 210, x + 145, y + 210, stroke="#26323f", sw=2, marker=True)
    svg.line(x + 242, y + 210, x + 278, y + 210, stroke="#26323f", sw=2, marker=True)
    svg.line(x + 20, y + 308, x + w - 20, y + 308, stroke="#9eb6ce", sw=1.5, dash="3 4")
    svg.text(x + w / 2, y + 355, "Atlas-based parcellation", size=21, weight=700, anchor="middle")
    add_brain(svg, x + 78, y + 380, scale=1.15, colored=True)
    svg.text(x + w / 2, y + 620, "21 anatomical brain regions", size=20, weight=700, anchor="middle")
    svg.line(x + 20, y + 675, x + w - 20, y + 675, stroke="#9eb6ce", sw=1.5, dash="3 4")
    svg.text(x + w / 2, y + 725, "Region node construction", size=21, weight=700, anchor="middle")
    add_region_graph(svg, x + w / 2, y + 885, r=115, n=21)
    svg.text(x + w / 2, y + 1010, "21 region nodes (N = 21)", size=18, fill=INK, anchor="middle")

    # Panel 3
    x, y, w = xs[2], top_y, widths[2]
    svg.text(x + w / 2, y + 145, "3D encoder → voxel-level features", size=20, weight=700, anchor="middle")
    for i in range(4):
        svg.rect(x + 50 + i * 8, y + 200 - i * 7, 110, 88, fill="url(#mriGrad)", stroke="#151a20", sw=1.3, rx=5)
        svg.ellipse(x + 105 + i * 8, y + 244 - i * 7, 38, 49, fill="none", stroke="#e6eef6", sw=2, opacity=0.8)
    add_encoder_blocks(svg, x + 220, y + 196)
    add_matrix(svg, x + 85, y + 390, rows=7, cols=12, cell=16, palette="blue")
    svg.text(x + w / 2, y + 535, "φθ(xᵢ): voxel-level feature map", size=21, fill=INK, weight=600, anchor="middle", family="Times New Roman")
    svg.line(x + 20, y + 600, x + w - 20, y + 600, stroke="#9eb6ce", sw=1.5, dash="3 4")
    svg.multiline(x + w / 2, y + 638, ["Region pooling →", "node feature embedding"], size=19, weight=700, anchor="middle")
    for i in range(7):
        cx = x + 62 + i * 39
        c = ["#1f9d8a", "#5b6fb0", "#6a58a9", "#9970ab", "#c66aa2", "#e3a33d", "#e56b2f"][i]
        svg.circle(cx, y + 700, 17, fill=c, stroke="#fff", sw=2)
        label = ["1", "2", "...", "k", "...", "21", ""][i] if i < 6 else ""
        if label:
            svg.text(cx, y + 707, label, size=14, fill="#fff", weight=700, anchor="middle")
        svg.line(cx, y + 725, cx, y + 765, stroke="#26323f", sw=2, marker=True)
        svg.rect(cx - 12, y + 778, 24, 66, fill=c, stroke="#3d4a55", sw=1.2, rx=3, opacity=0.75)
        for k in range(1, 4):
            svg.line(cx - 12, y + 778 + k * 16, cx + 12, y + 778 + k * 16, stroke="#fff", sw=0.8, opacity=0.9)
    add_formula_box(
        svg,
        x + 32,
        y + 885,
        w - 64,
        88,
        ["zᵢ,ₖ = 1/|Rᵢ,ₖ|  Σᵥ∈Rᵢ,ₖ  φθ(xᵢ)ᵥ", "Zᵢ ∈ Rᴺˣᵈ    (N = 21 nodes)"],
        size=19,
    )

    # Panel 4
    x, y, w = xs[3], top_y, widths[3]
    svg.multiline(x + w / 2, y + 130, ["(1) Anatomical prior", "adjacency Aprior"], size=19, weight=700, anchor="middle")
    add_matrix(svg, x + 80, y + 205, rows=8, cols=8, cell=18, palette="blue")
    svg.text(x + 70, y + 205, "1", size=14, anchor="end")
    svg.text(x + 70, y + 343, "21", size=14, anchor="end")
    svg.text(x + 100, y + 188, "1", size=14)
    svg.text(x + 208, y + 188, "21", size=14)
    svg.text(x + 250, y + 235, "1", size=16, weight=700)
    svg.text(x + 250, y + 343, "0", size=16, weight=700)
    svg.add(f'<circle cx="{x+172:.1f}" cy="{y+382:.1f}" r="17" fill="#fff7d6" stroke="#c79518" stroke-width="3"/>')
    svg.text(x + 172, y + 390, "+", size=26, weight=700, anchor="middle")
    svg.line(x + 20, y + 382, x + w - 20, y + 382, stroke="#26323f", sw=2, marker=True)
    svg.multiline(x + w / 2, y + 430, ["(2) Learned attention", "from Qᵢ and Kᵢ"], size=18, weight=700, anchor="middle")
    add_matrix(svg, x + 64, y + 510, rows=5, cols=5, cell=17, palette="green")
    svg.text(x + 112, y + 608, "Qᵢ ∈ Rᴺˣᵈ", size=17, anchor="middle", family="Times New Roman")
    svg.text(x + w / 2, y + 558, "...", size=22, anchor="middle")
    add_matrix(svg, x + 210, y + 510, rows=5, cols=5, cell=17, palette="purple")
    svg.text(x + 258, y + 608, "Kᵢ ∈ Rᴺˣᵈ", size=17, anchor="middle", family="Times New Roman")
    svg.line(x + w / 2, y + 632, x + w / 2, y + 670, stroke="#26323f", sw=2, marker=True)
    svg.text(x + w / 2, y + 710, "Sᵢ = softmax(QᵢKᵢᵀ/√d)", size=19, weight=600, anchor="middle", family="Times New Roman")
    add_matrix(svg, x + 105, y + 735, rows=7, cols=7, cell=18, palette="green")
    svg.add(f'<circle cx="{x+172:.1f}" cy="{y+885:.1f}" r="17" fill="#fff" stroke="#666" stroke-width="3"/>')
    svg.text(x + 172, y + 893, "+", size=25, weight=700, anchor="middle")
    add_formula_box(
        svg,
        x + 32,
        y + 920,
        w - 64,
        90,
        ["Aᵢ = αA_prior + (1−α)", "softmax(QᵢKᵢᵀ/√d)"],
        stroke="#b7a6d9",
        fill="#f8f5ff",
        size=18,
    )

    # Panel 5
    x, y, w = xs[4], top_y, widths[4]
    svg.multiline(x + w / 2, y + 135, ["Continuous-time disease", "progression modeling"], size=19, weight=700, anchor="middle")
    add_formula_box(svg, x + 35, y + 195, w - 70, 82, ["dhᵢ(t)/dt = fθ(hᵢ(t), Aᵢ, e(t))"], stroke="#8ac7ba", fill="#f2fbf8", size=23)
    svg.multiline(x + w / 2, y + 325, ["Initial node states Hᵢ(t₀)=Zᵢ", "(from Stage 3)"], size=18, anchor="middle")
    times = ["t₀", "t₁", "t₂", "t₃"]
    node_cols = [
        ["#68bd6d", "#4db7b0", "#5aa1cf"],
        ["#e7d151", "#c2bf45", "#ea9b42"],
        ["#f0a443", "#e78035", "#dc6547"],
        ["#df5245", "#ce3f36", "#b62f34"],
    ]
    for idx, t in enumerate(times):
        cy = y + 430 + idx * 130
        svg.text(x + 42, cy + 6, t, size=22, fill=INK, anchor="middle", italic=True)
        add_region_graph(svg, x + 155, cy, r=52, n=12, colors=node_cols[idx], labels=False)
    svg.line(x + 42, y + 390, x + 42, y + 832, stroke="#303a44", sw=2, marker=True)
    svg.text(x + 42, y + 862, "time", size=17, fill=INK, anchor="middle")
    svg.path(
        f"M{x+288},{y+410} C{x+252},{y+465} {x+338},{y+520} {x+290},{y+585} "
        f"C{x+242},{y+650} {x+348},{y+708} {x+292},{y+790}",
        stroke="url(#odeCurve)",
        sw=5,
        marker=True,
    )
    svg.text(x + 304, y + 405, "Healthy", size=16, fill="#1a9850", weight=700)
    svg.multiline(x + 300, y + 820, ["More", "pathological"], size=16, fill="#d73027", weight=700)
    svg.multiline(x + w / 2, y + 960, ["Evolved state Hᵢ(tT) at final time T", "used for downstream tasks"], size=16, anchor="middle")

    # Panel 6
    x, y, w = xs[5], top_y, widths[5]
    row_h = 126
    row_y = [y + 80, y + 218, y + 356, y + 494, y + 632]
    titles = ["CN / MCI / AD classification", "pMCI conversion risk", "MMSE prediction", "Regional atrophy estimation", "Missing-follow-up robustness"]
    for i, yy in enumerate(row_y):
        svg.rect(x + 24, yy, w - 48, row_h, fill="#fffaf6", stroke=ORANGE, sw=1.6, rx=10)
        svg.add(f'<circle cx="{x+56:.1f}" cy="{yy+32:.1f}" r="17" fill="{ORANGE}" stroke="none"/>')
        svg.text(x + 56, yy + 39, str(i + 1), size=18, fill="#fff", weight=700, anchor="middle")
        svg.multiline(x + 84, yy + 28, [titles[i]], size=18, fill=INK, weight=700)
    # class output
    for i, (lab, col) in enumerate([("CN", "#4fa85b"), ("MCI", "#e5b932"), ("AD", "#d94a38")]):
        cx = x + 285 + i * 66
        yy = row_y[0] + 48
        svg.circle(cx, yy, 12, fill=col, stroke="#336", sw=1)
        svg.rect(cx - 18, yy + 16, 36, 36, fill=col, stroke="none", rx=18)
        svg.text(cx, yy + 74, lab, size=14, fill=INK, anchor="middle")
    svg.text(x + 470, row_y[0] + 42, "P(class)", size=19, fill=INK, italic=True, family="Times New Roman")
    for i, col in enumerate(["#4fa85b", "#e5b932", "#d94a38"]):
        svg.rect(x + 476 + i * 22, row_y[0] + 75 - i * 13, 16, 44 + i * 13, fill=col, stroke="#333", sw=0.7)
    # gauge
    add_gauge(svg, x + 300, row_y[1] + 88)
    svg.text(x + 450, row_y[1] + 62, "P(convert)", size=19, italic=True, family="Times New Roman")
    svg.text(x + 455, row_y[1] + 94, "= 0.68", size=20, weight=700)
    # scatter
    add_scatter(svg, x + 285, row_y[2] + 18)
    # atrophy
    add_brain(svg, x + 290, row_y[3] + 30, scale=0.75, colored=False)
    svg.path(f"M{x+317},{row_y[3]+86} C{x+345},{row_y[3]+50} {x+388},{row_y[3]+58} {x+416},{row_y[3]+92}", fill="none", stroke="#d94a38", sw=10, opacity=0.48)
    for k, col in enumerate(["#d94a38", "#efb37b", "#d9e5f2", "#6e9fc8"]):
        svg.rect(x + 492, row_y[3] + 35 + k * 17, 18, 17, fill=col, stroke="none")
    svg.multiline(x + 515, row_y[3] + 48, ["High", "", "", "Low"], size=12, fill=INK)
    # timeline
    add_timeline(svg, x + 230, row_y[4] + 58)

    # flow arrows between top panels
    for i in range(5):
        add_arrow(svg, xs[i] + widths[i] + 2, top_y + 515, xs[i + 1] - 6, top_y + 515)

    # bottom counterfactual panel
    bx, by, bw, bh = xs[4], 1160, W - xs[4] - 18, 270
    add_panel(svg, bx, by, bw, bh, 7, ["Counterfactual explanation and clinical report"], PURPLE)
    svg.text(bx + 34, by + 92, "Region-level counterfactual intervention", size=16, weight=700)
    add_region_graph(svg, bx + 130, by + 178, r=70, n=21, colors=["#b989c8", "#9d80c8", "#c29bd6"], labels=False)
    svg.add(f'<circle cx="{bx+168:.1f}" cy="{by+184:.1f}" r="18" fill="#d63c3c" stroke="#fff" stroke-width="3"/>')
    svg.text(bx + 168, by + 190, "k", size=16, fill="#fff", weight=700, anchor="middle")
    add_formula_box(svg, bx + 250, by + 126, 210, 82, ["Clamp region k", "hᶜᶠᵢ,ₖ(t)=cₖ"], stroke="#b7a6d9", fill="#f8f5ff", size=17)
    svg.text(bx + 250, by + 235, "Average Treatment Effect", size=14, anchor="middle")
    svg.line(bx + 472, by + 170, bx + 510, by + 170, stroke="#26323f", sw=3, marker=True)
    # bar chart
    chart_x, chart_y = bx + 520, by + 97
    svg.line(chart_x, chart_y + 130, chart_x + 195, chart_y + 130, stroke=INK, sw=2)
    svg.line(chart_x, chart_y + 130, chart_x, chart_y + 15, stroke=INK, sw=2)
    for i in range(21):
        h = 12 + i * 4 + 18 * abs(math.sin(i * 0.83))
        col = "#7a4da3" if i != 8 else "#d63c3c"
        svg.rect(chart_x + 10 + i * 7.4, chart_y + 130 - h, 5, h, fill=col, stroke="none")
    svg.text(chart_x + 98, chart_y + 160, "Region (k)", size=13, anchor="middle")
    svg.text(chart_x - 22, chart_y + 80, "Δ AD risk", size=13, anchor="middle")
    svg.text(chart_x + 96, chart_y + 10, "ATE", size=17, weight=700, anchor="middle")
    svg.line(bx + 725, by + 170, bx + 772, by + 170, stroke="#26323f", sw=3, marker=True)
    # report sheet
    rx, ry = bx + 790, by + 94
    svg.path(f"M{rx},{ry} h118 l28,28 v145 h-146 z", fill="#f3f4f7", stroke="#a4aab2", sw=2)
    svg.path(f"M{rx+118},{ry} v28 h28", fill="#dce0e6", stroke="#a4aab2", sw=2)
    add_brain(svg, rx + 14, ry + 18, scale=0.36, colored=True)
    for i in range(6):
        svg.line(rx + 22, ry + 94 + i * 20, rx + 112, ry + 94 + i * 20, stroke="#9aa4ad", sw=2)
        svg.path(f"M{rx+120},{ry+89+i*20} l5,6 l12,-14", stroke="#5a9f61", sw=2.1)
    svg.multiline(rx + 73, by + 240, ["Longitudinal", "support report"], size=15, weight=700, anchor="middle")

    # Legend and caption
    ly = 1448
    svg.rect(22, ly - 34, 1200, 44, fill="#ffffff", stroke="#87919b", sw=1.5, rx=8, dash="3 4")
    svg.text(52, ly - 6, "Legend", size=20, weight=700)
    items = [("Data & preprocessing", BLUE), ("Modeling & dynamics", GREEN), ("Outputs", ORANGE), ("Interpretability", PURPLE)]
    lx = 160
    for label, col in items:
        svg.circle(lx, ly - 9, 14, fill=col, stroke="#444", sw=1)
        svg.text(lx + 25, ly - 3, label, size=17)
        lx += 265
    svg.add(f'<circle cx="{lx:.1f}" cy="{ly-9:.1f}" r="14" fill="#fff7d6" stroke="#c79518" stroke-width="3"/>')
    svg.text(lx, ly - 1, "+", size=20, weight=700, anchor="middle")
    svg.text(lx + 25, ly - 3, "Weighted fusion", size=17)
    svg.rect(1248, ly - 36, 1128, 48, fill="#f9fafb", stroke="#c7cdd4", sw=1.5, rx=8)
    svg.text(
        1812,
        ly - 4,
        "A2C-NODE integrates longitudinal structural MRI, anatomy-aware graph construction, continuous-time graph dynamics, and counterfactual interpretability.",
        size=18,
        weight=700,
        anchor="middle",
    )

    svg.save()
    print(SVG)


if __name__ == "__main__":
    main()
