#!/usr/bin/env python3
"""Recreate the user's 'Evaluation Framework / Testing Strategies by Chapter'
roadmap as a single, fully-editable widescreen PPT slide (native shapes + text,
no embedded image). Text and structure mirror the source figure.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "Evaluation-Framework-Roadmap-EN.pptx"

NAVY = RGBColor(0x12, 0x2A, 0x52)
BLUE = RGBColor(0x1F, 0x4E, 0x9B)
LIGHT = RGBColor(0xE9, 0xEF, 0xF7)
BORDER = RGBColor(0x7C, 0x93, 0xB8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x20, 0x2A, 0x3A)
GRAY = RGBColor(0x55, 0x5F, 0x70)
RED = RGBColor(0xC0, 0x1A, 0x1A)

SLIDE_W, SLIDE_H = 13.333, 7.5

ASSET = Path(__file__).resolve().parent / "_ef_formula_assets"


def render_math(key: str, tex: str, *, fontsize: int = 22, color: str = "#122A52") -> Path:
    """Render a small LaTeX (mathtext) snippet to a tight transparent PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["mathtext.fontset"] = "cm"
    ASSET.mkdir(exist_ok=True)
    out = ASSET / f"{key}.png"
    fig = plt.figure(figsize=(4.0, 0.6))
    fig.text(0.5, 0.5, tex, fontsize=fontsize, ha="center", va="center", color=color)
    fig.savefig(out, dpi=240, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def place_img(slide, png, rx, ry, rw, rh):
    """Place a PNG centered inside a region, preserving aspect ratio."""
    from PIL import Image
    with Image.open(png) as im:
        aspect = im.width / im.height
    h = rh
    w = h * aspect
    if w > rw:
        w = rw
        h = w / aspect
    ix = rx + (rw - w) / 2.0
    iy = ry + (rh - h) / 2.0
    slide.shapes.add_picture(str(png), Inches(ix), Inches(iy), Inches(w), Inches(h))


def rect(slide, x, y, w, h, fill, line=None, line_w=1.0, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.shadow.inherit = False
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(line_w)
    return sp


def rich_text(slide, x, y, w, h, paras, *, align=PP_ALIGN.CENTER,
              anchor=MSO_ANCHOR.MIDDLE, pad=0.03):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(pad)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", align)
        para.space_after = Pt(p.get("sa", 1))
        para.space_before = Pt(p.get("sb", 0))
        para.line_spacing = p.get("ls", 1.0)
        run = para.add_run()
        run.text = p["t"]
        f = run.font
        f.name = "Arial"
        f.size = Pt(p["s"])
        f.bold = p.get("b", False)
        f.italic = p.get("i", False)
        f.color.rgb = p.get("c", DARK)
    return box


def bar(slide, x, y, w, h, text, *, fill=NAVY, color=WHITE, size=12, bold=True,
        italic=False, line=None):
    rect(slide, x, y, w, h, fill, line=line, line_w=1.0)
    rich_text(slide, x, y, w, h, [{"t": text, "s": size, "b": bold, "c": color, "i": italic}])


def card(slide, x, y, w, h, spec, *, fill=WHITE, border=BORDER, line_w=1.0,
         anchor=MSO_ANCHOR.MIDDLE):
    rect(slide, x, y, w, h, fill, line=border, line_w=line_w)
    if isinstance(spec, dict) and "formula" in spec:
        paras = spec.get("paras", [])
        png = spec["formula"]
        pos = spec.get("pos", "bottom")
        strip = spec.get("strip", 0.19)
        gap = 0.02
        if pos == "top":
            place_img(slide, png, x + 0.05, y + 0.05, w - 0.10, strip)
            rich_text(slide, x, y + strip + gap, w, h - strip - gap - 0.02, paras,
                      anchor=MSO_ANCHOR.TOP)
        else:
            rich_text(slide, x, y + 0.02, w, h - strip - gap - 0.02, paras,
                      anchor=MSO_ANCHOR.MIDDLE)
            place_img(slide, png, x + 0.05, y + h - strip - 0.02, w - 0.10, strip)
    else:
        rich_text(slide, x, y, w, h, spec, anchor=anchor)


def arrow(slide, x, y, w, h, *, shape=MSO_SHAPE.RIGHT_ARROW, color=BLUE):
    sp = rect(slide, x, y, w, h, color, shape=shape)
    return sp


def subrow(slide, x, y, w, h, items, *, gap=0.08, border=BORDER):
    n = len(items)
    sw = (w - gap * (n - 1)) / n
    for i, paras in enumerate(items):
        card(slide, x + i * (sw + gap), y, sw, h, paras, border=border)


def group(slide, x, y, w, h, header, items, *, footer_paras=None, footer_h=0.0,
          header_h=0.26, pad=0.07, sub_border=BORDER):
    rect(slide, x, y, w, h, WHITE, line=NAVY, line_w=1.25)
    bar(slide, x, y, w, header_h, header, fill=LIGHT, color=NAVY, size=9, line=NAVY)
    body_top = y + header_h + pad
    body_h = h - header_h - 2 * pad - footer_h
    subrow(slide, x + pad, body_top, w - 2 * pad, body_h, items, border=sub_border)
    if footer_paras:
        rich_text(slide, x + pad, y + h - footer_h - 0.02, w - 2 * pad, footer_h,
                  footer_paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)


def title(t, *, c=NAVY, s=8.0):
    return {"t": t, "s": s, "b": True, "c": c, "sa": 1}


def body(t, *, c=DARK, s=6.8, i=False):
    return {"t": t, "s": s, "c": c, "i": i, "ls": 1.0, "sa": 0}


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # small formulas rendered as crisp transparent images (avoid text mojibake)
    f_zazv = render_math("za_zv", r"$z_a,\ \ z_v$")
    f_zvwmh = render_math("zv_wmh", r"$z_v \rightarrow \mathrm{WMH}$")
    f_cf = render_math("cf", r"$\mathrm{do}(z_v = \bar{z}_v^{\,CN})$")
    f_djs = render_math("djs_utot", r"$D_{JS},\ \ U_{total}$")
    f_djs_ch4 = render_math("djs_ch4", r"$(D_{JS},\ U_{total})$")

    # ----------------------------- LEFT PANEL ----------------------------- #
    LX, LW = 0.15, 5.15
    bar(slide, LX, 0.15, LW, 0.32, "Evaluation Framework", size=12)

    # 1. Classification Performance Testing (2 subboxes + 2 footer bullets)
    group(slide, LX, 0.55, LW, 1.36, "1. Classification Performance Testing",
          [
              [title("Three-class (CN / MCI / AD)"),
               body("Accuracy, Balanced Accuracy, Macro-AUC, "
                    "Recall (Sensitivity) per class, Confusion Matrix", s=6.4)],
              [title("Binary (AD vs NC)"),
               body("Accuracy, AUC, F1-score")],
          ],
          footer_paras=[
              {"t": "•  Not comparable across tasks (three-class vs binary)",
               "s": 6.3, "c": GRAY, "sa": 1, "align": PP_ALIGN.LEFT},
              {"t": "•  Each metric reported with Bootstrap 95% CI and "
                    "multi-seed (3–5 seeds) mean ± SD",
               "s": 6.3, "c": GRAY, "align": PP_ALIGN.LEFT},
          ],
          footer_h=0.40)

    # 2. External Generalization Testing
    group(slide, LX, 1.97, LW, 1.02, "2. External Generalization Testing",
          [
              [title("AIBL Held-out"), body("External validation"), body("(locked test)")],
              [title("OASIS Zero-shot"), body("Zero-shot transfer"), body("(no fine-tuning)")],
              [title("IXI (Healthy Controls)"),
               body("CN retention rate /"), body("health-specificity"),
               body("AD→CN misclassification", c=RED, s=6.3),
               body("= 0 (safety constraint)", c=RED, s=6.3)],
          ])

    # 3. Mechanism Disentanglement Testing (Chapter 2)
    group(slide, LX, 3.05, LW, 1.14, "3. Mechanism Disentanglement Testing (Chapter 2)",
          [
              {"paras": [title("Disentanglement /"), title("Branch-balance Score"),
                         body("Separation between"), body("atrophy / vascular latents")],
               "formula": f_zazv, "pos": "bottom", "strip": 0.17},
              {"paras": [title("Prediction"), body("AUC for vascular"),
                         body("branch validity")],
               "formula": f_zvwmh, "pos": "top", "strip": 0.22},
              {"paras": [title("Baselines & Causal Check"),
                         body("Compare with FactorVAE"), body("and other baselines;"),
                         body("Counterfactual intervention:")],
               "formula": f_cf, "pos": "bottom", "strip": 0.19},
          ])

    # 4. Longitudinal & Progression Testing (Chapter 3)
    group(slide, LX, 4.25, LW, 1.12, "4. Longitudinal & Progression Testing (Chapter 3)",
          [
              [title("Trajectory", s=7.3), title("Separation", s=7.3),
               body("pMCI vs sMCI", s=6.3), body("AUC / AP", s=6.3)],
              [title("Cognitive", s=7.3), title("Prediction", s=7.3),
               body("MMSE (calibrated)", s=6.3), body("MAE / RMSE", s=6.3)],
              [title("Probability", s=7.3), title("Calibration", s=7.3),
               body("Brier Score", s=6.3)],
              [title("Robustness", s=7.3), title("Stress Test", s=7.3),
               body("Missing follow-up", s=6.3), body("up to 70% dropout", s=6.3)],
              [title("Anatomical", s=7.3), title("Interpretation", s=7.3),
               body("Regional counterfactual", s=6.3), body("ATE analysis", s=6.3)],
          ])

    # 5. Trustworthy Decision Testing (Chapter 4)
    group(slide, LX, 5.43, LW, 1.22, "5. Trustworthy Decision Testing (Chapter 4)",
          [
              {"paras": [title("Conflict Detection (ROC)"),
                         body("Cross-modal discrepancy &"),
                         body("total uncertainty; report"),
                         body("separation ratio (conflict"),
                         body("vs non-conflict group)")],
               "formula": f_djs, "pos": "bottom", "strip": 0.17},
              [title("Uncertainty-based"), title("Rejection"),
               body("Uncertainty rejection"), body("curve to verify"),
               body("effectiveness of high-"), body("uncertainty triggering"),
               body("human review")],
              [title("Clinical Safety"), title("Constraint"),
               body("AD → CN"), body("misclassification"), body("must be 0")],
          ])

    bar(slide, LX, 6.73, LW, 0.45,
        "All results are declared as research prototypes, not end-to-end "
        "jointly trained, not approved medical devices.",
        size=8.2, bold=True, italic=True)

    # ----------------------------- RIGHT PANEL ---------------------------- #
    RX, RW = 5.45, 7.70
    bar(slide, RX, 0.15, RW, 0.32, "Testing Strategies by Chapter", size=12)

    def chapter_block(oy, oh, num, name):
        rect(slide, RX + 0.05, oy, RW - 0.10, oh, WHITE, line=BORDER, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tagx, tagw = RX + 0.22, 1.0
        rect(slide, tagx, oy + 0.10, tagw, 0.26, NAVY)
        rich_text(slide, tagx, oy + 0.10, tagw, 0.26,
                  [{"t": f"Chapter {num}", "s": 8.5, "b": True, "c": WHITE}])
        rich_text(slide, tagx + tagw + 0.12, oy + 0.10, 3.0, 0.26,
                  [{"t": name, "s": 11, "b": True, "c": NAVY}], align=PP_ALIGN.LEFT)

    # shared flow-box geometry
    bx1, bw = RX + 0.27, 2.15
    aw = 0.33
    bx2 = bx1 + bw + aw
    bx3 = bx2 + bw + aw

    # Chapter 1 — ARA-Net
    o1 = 0.58
    chapter_block(o1, 1.18, 1, "ARA-Net")
    fy, fh = o1 + 0.42, 0.66
    card(slide, bx1, fy, bw, fh, [title("Training / Development"), body("ADNI")])
    arrow(slide, bx1 + bw + 0.02, fy + fh / 2 - 0.09, aw - 0.04, 0.18)
    card(slide, bx2, fy, bw, fh, [title("External Test"), body("AIBL (subject-level held-out)")])
    arrow(slide, bx2 + bw + 0.02, fy + fh / 2 - 0.09, aw - 0.04, 0.18)
    card(slide, bx3, fy, bw, fh, [title("Outputs"),
                                  body("Three-class (CN / MCI / AD)"),
                                  body("Performance Metrics (Section 1)")])

    # Chapter 2 — PathwayPro
    o2 = 1.86
    chapter_block(o2, 1.18, 2, "PathwayPro")
    fy = o2 + 0.42
    card(slide, bx1, fy, bw, fh, [title("Training / Development"),
                                  body("ADNI (paired T1 + FLAIR subset)")])
    arrow(slide, bx1 + bw + 0.02, fy + fh / 2 - 0.09, aw - 0.04, 0.18)
    card(slide, bx2, fy, bw, fh, [title("Zero-shot Test"), body("OASIS (no fine-tuning)")])
    arrow(slide, bx2 + bw + 0.02, fy + fh / 2 - 0.09, aw - 0.04, 0.18)
    card(slide, bx3, fy, bw, fh, [title("Outputs"),
                                  body("Mechanism Disentanglement"),
                                  body("Metrics (Section 3)")])

    # Chapter 3 — A2C-NODE (with subgroup branch under box 1)
    o3 = 3.14
    chapter_block(o3, 1.95, 3, "A2C-NODE")
    fy, fh3 = o3 + 0.42, 0.74
    card(slide, bx1, fy, bw, fh3, [title("Training / Development"),
                                   body("ADNI (longitudinal cohort)")])
    arrow(slide, bx1 + bw + 0.02, fy + fh3 / 2 - 0.09, aw - 0.04, 0.18)
    card(slide, bx2, fy, bw, fh3, [title("External Validation"),
                                   body("AIBL (subject-level held-out)")])
    arrow(slide, bx2 + bw + 0.02, fy + fh3 / 2 - 0.09, aw - 0.04, 0.18)
    card(slide, bx3, fy, bw, fh3, [title("Outputs", s=7.6),
                                   body("pMCI vs sMCI (AUC / AP),", s=6.3),
                                   body("MMSE (MAE / RMSE),", s=6.3),
                                   body("Calibration (Brier),", s=6.3),
                                   body("Robustness / ATE (Section 4)", s=6.3)])
    sgy = fy + fh3 + 0.14
    arrow(slide, bx1 + bw / 2 - 0.09, fy + fh3 + 0.0, 0.18, 0.14, shape=MSO_SHAPE.DOWN_ARROW)
    card(slide, bx1, sgy, bw, 0.52, [title("Subgroup Analysis"),
                                     body("pMCI (progressive)"),
                                     body("vs sMCI (stable)")])

    # Chapter 4 — CUED-AD
    o4 = 5.21
    chapter_block(o4, 1.30, 4, "CUED-AD")
    fy, fh4 = o4 + 0.42, 0.74
    card(slide, bx1, fy, bw, fh4, [title("Evaluation Set"),
                                   body("AIBL Conflict Pool"),
                                   body("(natural + synthetic conflicts)")])
    arrow(slide, bx1 + bw + 0.02, fy + fh4 / 2 - 0.09, aw - 0.04, 0.18)
    out_w = bx3 + bw - bx2
    card(slide, bx2, fy, out_w, fh4,
         {"paras": [title("Outputs"), body("Conflict Detection ROC"),
                    body("Uncertainty Rejection Curve (Section 5)")],
          "formula": f_djs_ch4, "pos": "bottom", "strip": 0.17})

    prs.save(str(OUTPUT))
    print(f"Saved: {OUTPUT}")
    print(f"Slides: {len(prs.slides)} | shapes on slide 1: {len(slide.shapes)}")
    return OUTPUT


if __name__ == "__main__":
    build()
