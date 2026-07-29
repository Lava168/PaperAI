#!/usr/bin/env python3
"""Build an English academic-report PPT that follows the 17-slide outline doc
(基于解剖学引导...PPT大纲文字). Reuses the recovered Wu-style design helpers and
the base utility module; technical chapters follow the outline rule:
one architecture figure + one representative formula + 3-5 measured metrics +
one honest limitation. Background slides carry only literature/industry data.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import build_qual_exam_ppt as base
import build_qual_exam_ppt_en as en
import ppt_design as design

ROOT = base.ROOT
TEMPLATE = base.TEMPLATE
OUTPUT = ROOT / "Atlas-Guided-AD-Qualification-Academic-Report-EN.pptx"

MARGIN = base.MARGIN
MAX_W = base.MAX_W
CONTENT_TOP = base.CONTENT_TOP
FOOTER_TOP = base.FOOTER_TOP

set_speaker_notes = en.set_speaker_notes
add_textbox = en.add_textbox

ASSET_DIR = Path(__file__).parent / "_formula_assets"


def render_formula_png(name: str, tex_lines, fontsize: int = 23) -> Path:
    """Render LaTeX (mathtext) formula lines to a tight, transparent PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["mathtext.fontset"] = "cm"
    ASSET_DIR.mkdir(exist_ok=True)
    out = ASSET_DIR / f"{name}.png"
    n = max(1, len(tex_lines))
    fig = plt.figure(figsize=(9.0, 0.66 * n))
    for i, ln in enumerate(tex_lines):
        fig.text(0.01, 1.0 - (i + 0.5) / n, ln, fontsize=fontsize,
                 ha="left", va="center", color="#16213e")
    fig.savefig(out, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- #
# Custom layouts
# ----------------------------------------------------------------------------- #
def layout_chapter_formula(prs, *, key, section, title, subtitle, arch_img, arch_cap,
                           framework, formula_tex, formula_meaning, oral, metrics,
                           limitation, bridge, notes, slide_num=None):
    """Technical chapter page: left = method + key results + limitation;
    right = one architecture figure on top + one representative-formula box
    (the formula is embedded as a rendered image with a plain-language gloss)."""
    slide = base.new_slide(prs)
    design.sidebar(slide, section, slide_num)
    design.header(slide, title, subtitle, MARGIN, MAX_W)

    area_top, area_bottom = CONTENT_TOP, FOOTER_TOP
    left_w = 4.55
    right_x = MARGIN + left_w + 0.35
    right_w = MAX_W - left_w - 0.35

    # ---- left column: framework + metrics + limitation ----
    lines = ["Method / framework:"] + list(framework) + ["", "Key results:"] + list(metrics)
    lines += ["", f"Main limitation: {limitation}"]
    if bridge:
        lines.append(f"Transition: {bridge}")
    design.text_block(slide, MARGIN, area_top, left_w, area_bottom - area_top, lines,
                      size=9.5)

    # ---- right column: architecture figure ----
    img_h = 2.85
    path = base.img(arch_img)
    if path:
        w, h = base.fit_image(path, right_w, img_h)
        ix = right_x + (right_w - w) / 2.0
        base.add_picture_fitted(slide, path, ix, area_top, w, h)
        design.textbox(slide, right_x, area_top + h + 0.04, right_w, 0.26, arch_cap,
                       font="Arial", size=9, color=design.DARK, align=PP_ALIGN.CENTER)
        ref = design.lookup_ref(arch_img, "")
        if ref:
            design.textbox(slide, right_x, area_top + h + 0.30, right_w, 0.2, ref,
                           font="Arial", size=8, color=design.GRAY, italic=True,
                           align=PP_ALIGN.CENTER)

    # ---- right column: representative-formula box (rendered image + gloss) ----
    box_top = area_top + img_h + 0.58
    box_h = area_bottom - box_top
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(right_x),
                                   Inches(box_top), Inches(right_w), Inches(box_h))
    panel.fill.solid()
    panel.fill.fore_color.rgb = design.BLUE_LIGHT
    panel.line.color.rgb = design.BLUE_MID
    panel.line.width = Pt(1)
    panel.shadow.inherit = False

    pad = 0.16
    inner_x = right_x + pad
    inner_w = right_w - 2 * pad
    design.textbox(slide, inner_x, box_top + 0.07, inner_w, 0.24,
                   "Representative formula — what it computes",
                   font="Arial", size=9, color=design.BLUE, bold=True)

    fpath = render_formula_png(key, formula_tex)
    avail_h = 1.18
    fw, fh = base.fit_image(fpath, inner_w, avail_h)
    fx = right_x + (right_w - fw) / 2.0
    ftop = box_top + 0.34
    base.add_picture_fitted(slide, fpath, fx, ftop, fw, fh)

    txt_top = ftop + fh + 0.05
    mbox = slide.shapes.add_textbox(Inches(inner_x), Inches(txt_top), Inches(inner_w),
                                    Inches(max(0.4, box_h - (txt_top - box_top) - 0.08)))
    tf = mbox.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.02)
    p0 = tf.paragraphs[0]
    r_lbl = p0.add_run()
    r_lbl.text = "What it means: "
    r_lbl.font.name = "Arial"
    r_lbl.font.size = Pt(9)
    r_lbl.font.bold = True
    r_lbl.font.color.rgb = design.BLUE
    r_txt = p0.add_run()
    r_txt.text = formula_meaning
    r_txt.font.name = "Arial"
    r_txt.font.size = Pt(9)
    r_txt.font.color.rgb = design.DARK
    if oral:
        p1 = tf.add_paragraph()
        r1 = p1.add_run()
        r1.text = "▸ " + oral
        r1.font.name = "Arial"
        r1.font.size = Pt(8.8)
        r1.font.italic = True
        r1.font.color.rgb = design.GREEN

    set_speaker_notes(slide, notes)
    return slide


def layout_table(prs, *, section, title, subtitle, headers, rows, note, notes,
                 slide_num=None):
    """Slide 13: four-layer results overview as a real table."""
    slide = base.new_slide(prs)
    design.sidebar(slide, section, slide_num)
    design.header(slide, title, subtitle, MARGIN, MAX_W)

    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_top = CONTENT_TOP + 0.15
    tbl_h = (FOOTER_TOP - tbl_top) - 0.55
    gfx = slide.shapes.add_table(n_rows, n_cols, Inches(MARGIN), Inches(tbl_top),
                                 Inches(MAX_W), Inches(tbl_h))
    table = gfx.table
    col_w = [1.7, 2.2, 2.3, 3.6, MAX_W - (1.7 + 2.2 + 2.3 + 3.6)]
    for i, w in enumerate(col_w[:n_cols]):
        table.columns[i].width = Inches(w)

    for c, htext in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = design.BLUE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = htext
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.name = "Arial"
        r.font.color.rgb = design.WHITE

    for ri, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(ri, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = design.WHITE if ri % 2 else design.BLUE_LIGHT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c < 3 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.size = Pt(9.5)
            r.font.name = "Arial"
            r.font.color.rgb = design.DARK
            if c == 0:
                r.font.bold = True
                r.font.color.rgb = design.BLUE

    design.textbox(slide, MARGIN, FOOTER_TOP - 0.42, MAX_W, 0.4, note,
                   font="Arial", size=11, color=design.ACCENT, bold=True)
    set_speaker_notes(slide, notes)
    return slide


def layout_section_divider(prs, *, number, label, title, subtitle, notes, slide_num=None):
    """Part-divider slide: big part number + title + descriptive subtitle."""
    slide = base.new_slide(prs)
    design.sidebar(slide, label, slide_num)

    design.textbox(slide, 1.2, 2.30, 2.8, 1.8, number, font="Arial", size=96,
                   color=design.BLUE, bold=True, align=PP_ALIGN.LEFT,
                   anchor=MSO_ANCHOR.MIDDLE)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.15), Inches(2.62),
                                  Inches(0.035), Inches(1.55))
    rule.fill.solid()
    rule.fill.fore_color.rgb = design.BLUE_MID
    rule.line.fill.background()
    rule.shadow.inherit = False

    design.textbox(slide, 4.5, 2.70, 8.0, 0.95, title, font="Arial", size=25,
                   color=design.DARK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    design.textbox(slide, 4.5, 3.62, 8.0, 1.2, subtitle, font="Arial", size=13,
                   color=design.GRAY, italic=True)
    set_speaker_notes(slide, notes)
    return slide


def references_slide(prs, title, refs, notes, slide_num=None):
    slide = base.new_slide(prs)
    design.sidebar(slide, "References", slide_num)
    design.header(slide, title, "Searchable literature aligned with the outline",
                  MARGIN, MAX_W)
    design.text_block(slide, MARGIN, CONTENT_TOP, MAX_W, FOOTER_TOP - CONTENT_TOP,
                      refs, size=10.5, bullet=False)
    set_speaker_notes(slide, notes)
    return slide


def thanks_slide(prs, notes, slide_num=None):
    slide = base.new_slide(prs)
    design.sidebar(slide, "Thanks", slide_num)
    design.textbox(slide, MARGIN, 2.7, MAX_W, 1.0, "Thank You!", font="Arial",
                   size=40, color=design.BLUE, bold=True, align=PP_ALIGN.CENTER)
    design.textbox(slide, MARGIN, 3.9, MAX_W, 0.6,
                   "Questions and comments are welcome.", font="Arial", size=18,
                   color=design.GRAY, align=PP_ALIGN.CENTER)
    set_speaker_notes(slide, notes)
    return slide


# ----------------------------------------------------------------------------- #
# Content (English, from the 17-slide outline)
# ----------------------------------------------------------------------------- #
TOC = [
    "Overview — research framework, core direction, and the four-layer system",
    "Research background and current status — burden, pathology, and the existing foundation",
    "Core problem — the key scientific problems left unsolved",
    "Implications for future research",
    "Potential innovations and feasibility analysis",
    "Main references",
]

MODULES = {
    1: dict(number="01", label="Part 1 | Overview",
            title="Overview",
            subtitle="Research framework, core direction, and the four-layer system — a quick grasp of "
                     "the whole study, from a high-AUC black box toward anatomy-interpretable, "
                     "mechanism-disentangled, dynamic, and trustworthy modelling.",
            notes="Part one gives the big picture so the committee can hold the whole study in mind: "
                  "the core direction (from black-box AUC to four auditable capabilities) and the "
                  "four-layer modular system over a shared atlas interface and multi-cohort data spine."),
    2: dict(number="02", label="Part 2 | Background",
            title="Research Background and Current Status",
            subtitle="The clinical starting point: AD burden, pathological/anatomical mechanisms, "
                     "and the current research foundation and its status.",
            notes="Part two anchors the clinical starting point — burden, mechanism, and the existing "
                  "research foundation with its honest status and limits."),
    3: dict(number="03", label="Part 3 | Core problem",
            title="Core Problem",
            subtitle="The key scientific problems that existing research leaves unsolved, and where "
                     "this work aims to break through.",
            notes="Part three distils the core scientific problems that existing work leaves unsolved, "
                  "and maps each gap to our intended breakthrough."),
    4: dict(number="04", label="Part 4 | Implications",
            title="Implications for Future Research",
            subtitle="What this auditable, trust-aware paradigm enables, and how it can extend future "
                     "AD imaging research — clinically, methodologically, and translationally.",
            notes="Part four steps back to the field level: what this paradigm enables and the "
                  "expansion space — methodological and translational — it opens up for future research."),
    5: dict(number="05", label="Part 5 | Innovations & feasibility",
            title="Potential Innovations and Feasibility Analysis",
            subtitle="The four method chapters — each with one architecture figure, one representative "
                     "formula, measured metrics, and an honest limitation — as preliminary feasibility "
                     "evidence, plus a cross-chapter overview.",
            notes="Part five is the technical core: the potential innovations and the preliminary "
                  "results that demonstrate feasibility, chapter by chapter, followed by a cross-chapter "
                  "overview and a consolidated statement of innovations and limitations."),
    6: dict(number="06", label="Part 6 | References",
            title="Main References",
            subtitle="Searchable literature and project manuscripts supporting the work.",
            notes="Part six lists the searchable anchors and project manuscripts behind the talk."),
}

CHAPTERS = [
    dict(
        key="ch1_rcspe",
        section="Innovations | Chapter 1",
        title="Chapter 1 — ARA-Net: Atlas-Guided Multimodal Staging",
        subtitle="Layer 1: an anatomy-anchored, externally validated diagnostic anchor",
        arch_img="slide9_ARANet_architecture.png",
        arch_cap="T1 MRI → 21-region atlas → multimodal features → RC-SPE → CN/MCI/AD",
        framework=[
            "Input: T1 MRI → FreeSurfer/FastSurfer 21-region atlas + clinical vars (MMSE, CDR-SB, APOE4)",
            "Core head: RC-SPE — 6 base probability streams (atlas-biomarker / atlas-clinical HGB, clinical RF, ...)",
            "Weighted log-pooling + class offsets + temperature + risk-constrained model selection",
            "Subject-level probability averaging → CN/MCI/AD staging",
            "Interpretability: atlas structural-neurodegeneration consistency (not attention=biomarker)",
        ],
        formula_tex=[
            r"$z_{i,k} = \frac{1}{T}\sum_m w_m \log(\max(p_{m,k},\ \epsilon)) + b_k$",
            r"$\tilde{p}_{i,k} = \frac{e^{\,z_{i,k}}}{\sum_c e^{\,z_{i,c}}}$",
            r"$\bar{p}_{s,k} = \frac{1}{n_s}\sum_{i \in s}\tilde{p}_{i,k}, \quad \hat{y}_s = \mathrm{argmax}_k\ \bar{p}_{s,k}$",
        ],
        formula_meaning="each base model's class log-probabilities are weighted (w_m), temperature-scaled (T) and offset (b_k), re-normalised to a calibrated distribution, then averaged within a subject to give the final stage.",
        oral="Not a simple average — a risk-constrained, multi-stream subject-level probability ensemble.",
        metrics=[
            "AIBL held-out (subject-level, n=216): Acc 0.903 | BAcc 0.833 | Macro AUC 0.937",
            "Recall CN/MCI/AD 0.961 / 0.686 / 0.852 | AD→CN = 0 | IXI CN retention 1.000",
            "Atlas AD-key consistency 0.510 vs null 0.286 (permutation p = 0.026)",
        ],
        limitation="MCI recall 0.686 is the main residual weakness under risk constraints (boundary cases pushed toward AD).",
        bridge="Provides auditable atlas evidence reused by later chapters.",
        notes="Chapter 1 answers: can we stage CN/MCI/AD with anatomy-linked evidence and true external validation? RC-SPE fuses six probability streams under a risk constraint. On AIBL held-out data we reach strong overall metrics, with the honest weakness being MCI recall 0.686. I will not claim attention weights are biomarkers; instead I report atlas structural-neurodegeneration consistency.",
    ),
    dict(
        key="ch2_disent",
        section="Innovations | Chapter 2",
        title="Chapter 2 — PathwayPro: Atrophy vs Vascular Disentanglement",
        subtitle="Layer 2: mechanism separation where >60% of AD carries vascular comorbidity",
        arch_img="slide10_PathwayPro_architecture.png",
        arch_cap="Dual encoders: T1→z_a (atrophy), FLAIR→z_v (vascular); HSIC/CLUB disentanglement",
        framework=[
            "Input: paired T1+FLAIR (ADNI Phase-0 benchmark, n=63 test)",
            "Dual encoders: 3D ResNet trunk + cross-attention (K=8 pathology queries)",
            "z_a = atrophy/neurodegeneration; z_v = vascular/WMH pathway",
            "Disentanglement: HSIC + β-TCVAE + CLUB + iVAE (cond. on age/APOE/amyloid)",
            "Zero-mask training: one checkpoint serves T1+FLAIR and T1-only deployment",
        ],
        formula_tex=[
            r"$\mathcal{L} = \mathcal{L}_{dx} + \mathcal{L}_{SUVR} + \mathcal{L}_{WMH} + \mathcal{L}_{sub} + \mathcal{L}_{cf} + \alpha\,\mathcal{L}_{disent} + \mathcal{L}_{KD}$",
            r"$\mathcal{L}_{disent} = w_H\,\mathrm{HSIC}(z_a,z_v) + w_T\,\mathrm{TC}(z_a,z_v) + w_C\,I_{CLUB}(z_a;z_v) + w_i\,\mathrm{KL}_{iVAE}$",
        ],
        formula_meaning="the total loss sums the task heads (diagnosis, SUVR, WMH, subtype, counterfactual) plus distillation; the disentanglement term L_disent uses HSIC / total-correlation / CLUB / iVAE to force the atrophy code z_a and vascular code z_v to be statistically independent.",
        oral="z_a captures atrophy, z_v captures vascular/WMH; the disentanglement losses actively separate them.",
        metrics=[
            "ADNI test (n=63, 3-seed): Acc 64.6% | BAcc 59.2% (best seed1 70.3%) | Macro-F1 57.5%",
            "z_v→WMH AUC 0.923 (vascular branch works) | disentanglement score −0.014",
            "OASIS zero-shot: BAcc 49.6%, AD-vs-CN AUC 0.774 (n=121, no fine-tuning)",
        ],
        limitation="Accuracy–disentanglement trade-off; AIBL zero-shot collapses (BAcc 33.3%) — reported honestly.",
        bridge="Addresses the vascular-comorbidity gap from the background.",
        notes="PathwayPro is a benchmark for mechanism disentanglement, not a SOTA-accuracy claim. The WMH branch is informative (AUC 0.923) but the disentanglement score is near zero, so separation is still weak. The main result must be PathwayPro (ours); FactorVAE 73% is only a baseline, not our score.",
    ),
    dict(
        key="ch3_node",
        section="Innovations | Chapter 3",
        title="Chapter 3 — A2C-NODE: Continuous-Time Causal Dynamics",
        subtitle="Layer 3: trajectory inference under irregular follow-up",
        arch_img="slide11_A2C_NODE_architecture.png",
        arch_cap="Baseline T1 → 21-region graph → graph neural ODE → diagnosis/MMSE/pMCI/ATE report",
        framework=[
            "Input: longitudinal T1-MRI with irregular visit times (CN/MCI/AD)",
            "3D encoder → 21-region node features; dynamic graph A_i = α·A_prior + (1−α)·attention",
            "Graph Neural ODE evolves a subject-specific latent state in continuous time",
            "Multi-task readout: staging, pMCI risk, MMSE, regional atrophy",
            "Counterfactual: region clamp → ATE bar chart + brain projection",
        ],
        formula_tex=[
            r"$A_i = \alpha\,A_{prior} + (1-\alpha)\,\mathrm{softmax}\left(\frac{Q_i K_i^{\top}}{\sqrt{d}}\right)$",
            r"$\frac{d h_i(t)}{dt} = f_\theta(h_i(t),\,A_i,\,e(t))$",
            r"$h_i(t_q) = h_i(0) + \int_0^{t_q} f_\theta(h_i(t),\,A_i,\,e(t))\,dt$",
        ],
        formula_meaning="the brain graph A_i blends an anatomical prior with attention; a neural ODE then integrates the regional state h_i(t) forward to any visit time t_q, giving continuous-time trajectories on irregular follow-up.",
        oral="We evolve disease state along the irregular follow-up axis, not LSTM snapshots on a fixed grid.",
        metrics=[
            "ADNI test (n=121, 5-seed): BAcc 0.854 | macro-AUC 0.963 | Brier 0.087",
            "pMCI vs sMCI AUC 0.809 (AP 0.883) | MMSE MAE 2.69 (calibrated)",
            "vs LSTM baseline BAcc 0.678  →  +17.6 percentage points",
        ],
        limitation="AIBL raw Acc 0.297 (cohort-prior shift); pMCI is trajectory separation, not a fixed 3-year prospective tool.",
        bridge="Addresses the static-snapshot gap from the background.",
        notes="Chapter 3 moves from classification to dynamics. The ODE respects irregular visits, which matters in real ADNI-style follow-up. We gain 17.6 points of BAcc over LSTM internally, but external AIBL accuracy drops sharply — I present both because external failure modes are part of the scientific story.",
    ),
    dict(
        key="ch4_conflict",
        section="Innovations | Chapter 4",
        title="Chapter 4 — CUED-AD: Trust-Aware Clinical Decision Support",
        subtitle="Layer 4: external validation wrapped with conflict & uncertainty awareness",
        arch_img="slide12_CUED_AD_architecture.png",
        arch_cap="MRI branch (SCM residuals) ∥ clinical branch → D_JS conflict → U_total → review tier",
        framework=[
            "Inputs: MRI branch (FreeSurfer ROI → SCM residuals, 24-d) ∥ clinical (MMSE/CDR-SB/ADAS + APOE)",
            "Dual Bayesian MC-Dropout classifiers; alignment keeps p_MRI and p_Clin",
            "Conflict: D_JS (Jensen–Shannon divergence) between branches",
            "Uncertainty: U_total = u_MRI + u_Clin + β·D_JS",
            "Output: AD prob + conflict/uncertainty + 'refer for human review' tier (Streamlit prototype)",
        ],
        formula_tex=[
            r"$D_{JS}(p_{MRI}, p_{Clin}) = \frac{1}{2}\mathrm{KL}(p_{MRI}\,\parallel\,m) + \frac{1}{2}\mathrm{KL}(p_{Clin}\,\parallel\,m)$",
            r"$m = \frac{p_{MRI}+p_{Clin}}{2}, \quad U_{total} = u_{MRI} + u_{Clin} + \beta\,D_{JS}$",
        ],
        formula_meaning="D_JS is the Jensen–Shannon divergence between the MRI and clinical probability vectors — how strongly the two modalities disagree; total uncertainty adds each branch's own uncertainty plus β·D_JS, and high values trigger human review.",
        oral="High D_JS or U_total is not a diagnosis — it is a 'refer for human review' signal.",
        metrics=[
            "AIBL external (AD vs NC): Acc 0.984 | AUC 0.997 | F1 0.921",
            "Conflict detection AUC 0.753 (U_total) / 0.747 (D_JS)",
            "Natural-conflict U_total 18.01 vs non-conflict 7.50 (2.4×)",
        ],
        limitation="Task is AD vs NC (not 3-class); conflict AUC ~0.75; not PACS-deployed, not FDA/NMPA-cleared.",
        bridge="Closes the trust gap: from scores to human–AI teaming.",
        notes="Chapter 4 is about communication under disagreement. When MRI and clinical evidence conflict, total uncertainty rises, shown by the U_total separation. This is a research prototype for human–AI teaming — escalate, review, or defer — not autonomous diagnosis. Its 0.984 accuracy is a binary task and must not be compared to Chapter 1's three-class accuracy.",
    ),
]

TABLE_HEADERS = ["Chapter", "Task", "Main cohort", "Core metrics", "Main limitation"]
TABLE_ROWS = [
    ["ARA-Net", "CN/MCI/AD staging", "AIBL external", "BAcc 0.833; MCI recall 0.686", "MCI boundary weak"],
    ["PathwayPro", "Atrophy/vascular disentangle", "ADNI + OASIS/AIBL", "BAcc 59.2% (best 70.3%); OASIS AUC 0.774", "Disent–acc trade-off; AIBL zero-shot fails"],
    ["A2C-NODE", "Longitudinal progression", "ADNI internal", "BAcc 0.854; pMCI AUC 0.809", "AIBL transfer poor (cohort shift)"],
    ["CUED-AD", "AD vs NC + trust", "AIBL external", "Acc 0.984; conflict AUC 0.75", "Binary task; research prototype"],
]


def build() -> Path:
    base.ensure_images()
    tmp = OUTPUT.with_suffix(".tmp.pptx")
    shutil.copy2(TEMPLATE, tmp)
    prs = Presentation(str(tmp))
    base.trim_template(prs)
    base.set_widescreen(prs)

    page = 0

    def divider(n):
        nonlocal page
        page += 1
        layout_section_divider(prs, slide_num=page, **MODULES[n])

    # 1. cover
    page += 1
    en.build_cover(prs, "Good morning, distinguished committee. I am Yuanqin Zhao, a 2025 PhD student in Precision Medicine and Public Health at Tsinghua. Today I present a four-layer progressive research program for AD imaging: from high-AUC black-box prediction toward anatomy-interpretable, mechanism-disentangled, trajectory-inferable, and trust-aware decision support. All four modules are research prototypes to be refined after this qualification exam.")

    # 2. contents (six parts)
    page += 1
    en.build_contents(prs, TOC, "My talk has six parts: first, an overview of the research framework and core direction; second, the research background and current status — the clinical starting point; third, the core problem left unsolved by existing research; fourth, the implications for future research; fifth, the potential innovations and a feasibility analysis based on my current results; and sixth, the main references.")

    # ===================== Part 1 — Overview ===================== #
    divider(1)

    page += 1
    en.layout_narrative(
        prs,
        "Overview | Direction",
        "Core direction: a four-dimension progressive framework",
        "Where → which mechanism → how it evolves → can we trust it?",
        [{"file": "slide8_clinical_vision.png", "caption": "Four-quadrant research vision"}],
        [
            "Anatomy-interpretable diagnosis: tell the clinician which regions/structures support the decision (ARA-Net).",
            "Multi-mechanism disentanglement: show how much atrophy vs vascular signal each contributes (PathwayPro z_a/z_v).",
            "Continuous-time causal dynamics: ask how a region's state shapes future trajectory and risk (A2C-NODE ODE + ATE).",
            "Trust-aware decision: answer when to trust the model and when to escalate to human review (CUED-AD U_total/D_JS).",
        ],
        conclusion="From a lab-AUC competition to auditable, transferable, risk-aware decision support.",
        bridge="And here is how the four layers fit together →",
        notes="Part one opens with the core direction — the conceptual map of the whole study. The four dimensions answer where, which mechanism, how it evolves, and whether to trust, and they map one-to-one onto the four chapters.",
        slide_num=page,
    )

    page += 1
    en.layout_narrative(
        prs,
        "Overview | System",
        "The four-layer modular system at a glance",
        "One progressive system over a shared atlas interface and multi-cohort data spine",
        [{"file": "slide4_four_dimension_framework.png", "caption": "Four-layer progressive research system"}],
        [
            "Layer 1 — ARA-Net → anatomy-interpretable, externally validated staging.",
            "Layer 2 — PathwayPro → mixed-pathology subtyping (atrophy z_a vs vascular z_v).",
            "Layer 3 — A2C-NODE → progression inference and pMCI risk stratification.",
            "Layer 4 — CUED-AD → clinical trust, conflict alerting, and rejection/review.",
            "Data spine: ADNI (development/internal) + AIBL (external) + IXI (healthy control) + OASIS (stress test).",
        ],
        conclusion="A modular four-layer research system — not a single deployed end-to-end product.",
        bridge="What is the clinical starting point that motivates it?",
        notes="The second overview slide gives the whole picture: four independently trained and validated layers over a shared atlas interface and a four-cohort data spine. I stress it is modular, not an end-to-end black box.",
        slide_num=page,
    )

    # ===================== Part 2 — Background ===================== #
    divider(2)

    page += 1
    en.layout_narrative(
        prs,
        "Background | Burden",
        "The dilemma of AD imaging analysis",
        "Real clinical need meets technically feasible but not-yet-trustworthy AI",
        [
            {"file": "slide2_AD_burden_infographic.png", "caption": "China AD burden (Jia 2022; China AD Report 2023)"},
            {"file": "slide2_ATN_framework.png", "caption": "ATN biomarker era (Frisoni, Lancet 2025)"},
            {"file": "slide2_Desikan_Killiany_atlas.png", "caption": "Anatomical priors give auditable region interface"},
        ],
        [
            "Burden: ~9.83M AD and ~15.07M dementia (60+) in China; AD is the 5th leading cause of death.",
            "Feasibility: AI + multimodal imaging advancing AD research; MaM-DiT lifts MRI-only AUC by up to 13.8%.",
            "Pathology/anatomy: hippocampus–MTL–posteromedial / DMN networks decline early and track cognition.",
            "Policy: 2026 integrated PET/MRI guideline and ATN framework standardize acquisition and reporting.",
        ],
        conclusion="AD urgently needs early, scalable, anatomy-interpretable imaging tools — not one-off black-box scores.",
        bridge="What does the existing research foundation actually deliver?",
        notes="I open Part two with the clinical dilemma. The burden is large and growing; AI plus multimodal imaging is technically feasible; anatomy and network evidence give us an auditable interface; and policy is moving toward standardized ATN reporting.",
        slide_num=page,
    )

    page += 1
    en.layout_narrative(
        prs,
        "Background | Foundation (1/2)",
        "Existing research foundation and its status",
        "Technology evolution, generalization, and interpretability reality",
        [
            {"file": "slide2_technology_timeline.png", "caption": "ML → CNN/3D-CNN → multimodal → dynamics/uncertainty"},
            {"file": "bg_XAI_brain_connectivity.webp", "caption": "Post-hoc saliency ≠ biologically valid explanation"},
        ],
        [
            "Most AD imaging AI is internally cross-validated; memory-clinic external performance drops (Mårtensson 2020, κw 0.34–0.66).",
            "≥83% of studies rely on public cohorts (e.g. ADNI); cross-site/protocol/vendor transfer is the main bottleneck.",
            "Systematic reviews: post-hoc saliency/attention is not automatically a valid biological explanation (Wen 2024).",
        ],
        conclusion="Lab AUC does not equal external clinical usability.",
        bridge="Deployment, dynamics, population and cost →",
        notes="The field has progressed along a clear technology curve, but two problems persist: external generalization to memory-clinic data is weak, and post-hoc explanations are not necessarily biologically valid. I cite Mårtensson 2020 and Wen 2024 as searchable anchors.",
        slide_num=page,
    )

    page += 1
    en.layout_narrative(
        prs,
        "Background | Foundation (2/2)",
        "Existing research foundation (continued)",
        "Deployment, dynamic prediction, population and cost",
        [
            {"file": "slide2_MaM_DiT_overview.jpg", "caption": "MRI-first / synthetic-PET route lowers cost (MaM-DiT 2026)"},
            {"file": "bg_ADNI4_design.png", "caption": "Longitudinal, multimodal cohort design (ADNI4)"},
        ],
        [
            "Most work is offline batch evaluation; prospective workflow validation and PACS/HIS integration remain future work.",
            "Discrete-time models (LSTM) resample to fixed grids and are sensitive to irregular follow-up.",
            "Western-cohort models may degrade on Asian populations / different protocols → external validation and domain adaptation needed.",
            "Multi-tracer PET is costly and low-access, motivating MRI-first or MRI+synthetic-PET routes.",
        ],
        conclusion="From 'risk scores' toward 'actionable reports' with explicit boundaries.",
        bridge="So what core scientific problems remain unsolved?",
        notes="On deployment, most work is offline; real-time, PACS-integrated, locally calibrated systems are future work — I avoid fabricated deployment numbers. On dynamics, discrete-time models struggle with irregular visits. On population and cost, external validation and MRI-first routes matter.",
        slide_num=page,
    )

    # ===================== Part 3 — Scientific problem ===================== #
    divider(3)

    page += 1
    en.layout_narrative(
        prs,
        "Problem | Bottlenecks",
        "Four core scientific problems left unsolved",
        "And the regulatory direction that reinforces them",
        [
            {"file": "bg_explainable_AI_AD_SECNN.webp", "caption": "Explanations often not anatomy/pathology-aligned"},
            {"file": "bg_DL_pipeline_AD_detection.jpg", "caption": "High-AUC CNN/ViT pipelines, weak external transfer"},
            {"file": "slide3_EMA_FDA_2026.jpg", "caption": "FDA/EMA 2025-26 good-AI-practice principles"},
        ],
        [
            "Interpretability gap: many models output labels only, without mechanism-aligned explanation.",
            "Multi-mechanism gap: >60% AD has vascular comorbidity, often collapsed into one representation.",
            "Dynamics gap: mostly static snapshots; hard to infer trajectories under irregular follow-up.",
            "Trust gap: high lab AUC ≠ clinical usability; uncertainty, conflict, external validation missing.",
        ],
        conclusion="The bottleneck is trust and generalization, not accuracy alone.",
        bridge="Each gap defines a concrete breakthrough direction →",
        notes="Part three distils the four recurring scientific problems: interpretability, multi-mechanism modeling, dynamics, and trust. Regulators (FDA 2025, FDA-EMA 2026) emphasize transparency, data governance, lifecycle management and understandable outputs — I frame this as a direction, not a claim that all models must already submit interpretability reports.",
        slide_num=page,
    )

    page += 1
    en.layout_narrative(
        prs,
        "Problem | Breakthrough",
        "Breakthrough direction and positioning",
        "Each gap mapped to a concrete response",
        [{"file": "slide3_gap_solution_mapping.png", "caption": "Gap → response mapping"}],
        [
            "Interpretability gap → ARA-Net: 21-region atlas features + structural-consistency analysis.",
            "Multi-mechanism gap → PathwayPro: T1+FLAIR dual encoders + HSIC/TC/CLUB/iVAE disentanglement.",
            "Static-snapshot gap → A2C-NODE: graph neural ODE trajectories + regional counterfactual ATE.",
            "Trust gap → CUED-AD: D_JS conflict + U_total uncertainty + tiered rejection/review.",
        ],
        conclusion="One progressive system: interpretable → disentangled → dynamic → trustworthy.",
        bridge="What does this direction imply for the field?",
        notes="This slide maps each unsolved problem to a concrete breakthrough. I deliberately avoid 'world-first' claims. Each chapter is independently trained and validated but shares a multi-cohort evidence chain, aligned with the 2026 integrated PET/MRI guideline, the ATN framework, and AI-transparency trends.",
        slide_num=page,
    )

    # ===================== Part 4 — Implications & expansion ===================== #
    divider(4)

    page += 1
    en.layout_narrative(
        prs,
        "Implications | Scenarios",
        "Clinical application scenarios (research-prototype boundaries)",
        "Prevention–diagnosis–alerting now; treatment–monitoring as research direction",
        [{"file": "slide8_clinical_vision.png", "caption": "Decision-support scenarios across the four layers"}],
        [
            "Implemented / in-progress: atlas-constrained staging; atrophy–vascular disentanglement; longitudinal risk report; conflict/uncertainty alert.",
            "Future (not yet clinically validated here): treatment planning, therapy monitoring, full PACS embedding.",
            "Wording principle: prevention/diagnosis/alerting are claimable; treatment/monitoring are stated as research directions.",
        ],
        conclusion="All four modules are research prototypes — not approved medical devices.",
        bridge="Why does this paradigm matter versus conventional pipelines?",
        notes="Part four steps back to the field level. I keep the clinical scope honest: staging, disentanglement, longitudinal reporting and conflict alerting are demonstrated; treatment planning and PACS embedding are explicitly future work.",
        slide_num=page,
    )

    page += 1
    en.layout_narrative(
        prs,
        "Implications | Advantages",
        "Advantages over conventional methods",
        "We pursue clinical trustworthiness, not just higher lab AUC",
        [],
        [
            "vs end-to-end CNN  →  modular atlas audit trails instead of an opaque embedding.",
            "vs post-hoc Grad-CAM  →  named mechanism pathways (z_a / z_v).",
            "vs fixed-grid LSTM  →  continuous-time neural ODE for irregular visits (+17.6 pp BAcc).",
            "vs point prediction  →  CUED-AD exposes conflict (D_JS) and uncertainty (U_total).",
        ],
        conclusion="One coherent, auditable stack — not four isolated high-score models.",
        bridge="What significance and expansion space does this open?",
        notes="The comparison is qualitative and structural: at each layer we add an auditable capability conventional pipelines lack. The goal is trustworthiness, not a marginally higher internal AUC.",
        slide_num=page,
    )

    page += 1
    en.layout_narrative(
        prs,
        "Implications | Significance",
        "Significance and expansion space for the field",
        "What this enables, and how far it can be extended",
        [{"file": "slide7_research_positioning.png", "caption": "From high-AUC black box to auditable evidence"}],
        [
            "Clinical significance: give clinicians information that is understandable, contestable, and reviewable — human–AI teaming, not replacement.",
            "Explicit uncertainty and conflict signals (U_total, D_JS) support safe deferral and human review.",
            "Societal value: targets China's ~10M AD patients with an MRI-first, transferable, regulation-aware route.",
            "Expansion (methods): a unified subject-level benchmark and loosely-coupled integration of the four layers.",
            "Expansion (translation): prospective multi-center pilots, local calibration, regulator dialogue, and a clinical reader study.",
        ],
        conclusion="From a lab-AUC competition toward an auditable, transferable, regulation-ready decision-support paradigm.",
        bridge="Now the concrete innovations and current results →",
        notes="I frame Part four forward-looking: clinical and societal significance, then the expansion space the paradigm opens — methodologically (unified benchmark, loose integration) and translationally (prospective pilots, calibration, regulatory dialogue, reader study).",
        slide_num=page,
    )

    # ===================== Part 5 — Innovations & results ===================== #
    divider(5)

    # four technical chapters (innovation + measured results, one figure + one formula each)
    for ch in CHAPTERS:
        page += 1
        layout_chapter_formula(prs, slide_num=page, **ch)

    # cross-chapter results overview table
    page += 1
    layout_table(
        prs,
        section="Feasibility | Overview",
        title="Cross-chapter results overview",
        subtitle="Four modular layers over a shared atlas interface and ADNI→AIBL/IXI/OASIS validation spine",
        headers=TABLE_HEADERS,
        rows=TABLE_ROWS,
        note="Do not cross-compare absolute Acc across chapters — tasks and cohorts differ.",
        notes="This overview table is the answer to 'what is done across the program.' I read it task-by-task: each chapter has its own task, cohort, headline metric, and honest boundary. The key discipline is that I never cross-compare absolute accuracy because the tasks and cohorts differ.",
        slide_num=page,
    )

    # consolidated innovations + honest limitations
    page += 1
    en.layout_narrative(
        prs,
        "Feasibility | Summary",
        "Innovation highlights and honest limitations",
        "Four auditable capabilities — and where each is still weak",
        [],
        [
            "Innovation — ARA-Net: leakage-free, subject-level externally validated atlas-guided multimodal staging (RC-SPE).",
            "Innovation — PathwayPro: a comparable T1+FLAIR disentanglement benchmark with an accuracy–disentanglement analysis.",
            "Innovation — A2C-NODE: an anatomy-anchored graph neural ODE longitudinal decision report.",
            "Innovation — CUED-AD: an MRI–clinical conflict-aware uncertainty decision-support framework.",
            "Limitation — ARA-Net MCI recall 0.686; PathwayPro disent −0.014; A2C-NODE AIBL raw Acc 0.297; CUED-AD conflict AUC ~0.75.",
        ],
        conclusion="Four auditable capabilities forming one coherent program — all current results are at research-prototype stage.",
        bridge="References supporting the work →",
        notes="I consolidate the innovations and then state every weakness plainly: the MCI boundary, the weak disentanglement score, the AIBL transfer drop, and the modest conflict-detection AUC. Honesty here is part of the scientific contribution.",
        slide_num=page,
    )

    # ===================== Part 6 — References ===================== #
    divider(6)

    page += 1
    references_slide(
        prs,
        "Main references",
        [
            "[1] Jia L, et al. Prevalence, risk factors, and management of dementia and MCI in China. Gen Psychiatr. 2022;35(1):e100751.",
            "[2] China Alzheimer Disease Data and Prevention Strategy Report. 2023.",
            "[3] Andrieu S, et al. Harnessing artificial intelligence to transform Alzheimer's disease research. Nat Med. 2025.",
            "[4] MaM-DiT. Manifold diffusion transformer for multi-tracer PET synthesis from MRI. Innovation Informatics. 2026;2(2):100048.",
            "[5] Frisoni GB, et al. The new landscape of the diagnosis of Alzheimer's disease. Lancet. 2025;406:1389-1407.",
            "[6] Berron S, et al. Longitudinal hippocampal network connectivity in AD. Ann Neurol. 2022. PMC9291910.",
            "[7] Guo T, et al. Multiplex CSF proteomics identifies biomarkers for AD. Nat Hum Behav. 2024.",
            "[8] Mohanty S, et al. Arteriolosclerosis and CAA as cerebrovascular pathways. Acta Neuropathol. 2025.",
            "[9] Wen J, et al. Explainable AI in medical imaging. Nat Mach Intell. 2024.",
            "[10] Mårtensson G, et al. Reliability of AD classification across cohorts. Alz Res Therapy. 2020.",
            "[11] FDA. Considerations for the use of AI to support regulatory decision-making. Draft Guidance. 2025.",
            "[12] FDA & EMA. Guiding principles of good AI practice in drug development. 2026.",
        ],
        "Key anchors: Jia 2022 for burden; Andrieu 2025 and Frisoni 2025 for the AI/ATN context; Mårtensson 2020 for generalization; FDA/EMA for the regulatory direction. Full chapter manuscripts are in the project repository.",
        slide_num=page,
    )

    # 20. thanks
    page += 1
    thanks_slide(prs, "Thank you for your attention. I welcome questions on the MCI boundary in ARA-Net, the disentanglement trade-off in PathwayPro, AIBL transfer in A2C-NODE, or conflict/uncertainty communication in CUED-AD.", slide_num=page)

    prs.save(str(tmp))
    tmp.replace(OUTPUT)
    print(f"Saved: {OUTPUT}")
    print(f"Slides: {len(prs.slides)} (all with speaker notes)")
    return OUTPUT


if __name__ == "__main__":
    build()
