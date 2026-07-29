#!/usr/bin/env python3
"""Embed journal brain figures into the MedIA manuscript package with full captions.

Actions:
  1) Copy B1–B7 PNGs into latex/figures and mia 06/07 folders
  2) Replace main Figure 7 artwork with redesigned B5 (Papez)
  3) Patch latex/main.tex: update Fig.7 caption + add Supplementary Figures S1–S7
  4) Write caption text/docx files
  5) Build a Word supplementary-figures document for easy paste into the MS
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]


def tex_escape(s: str) -> str:
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        "→": "$\\rightarrow$",
        "−": "--",
        "–": "--",
        "—": "---",
        "ρ": "$\\rho$",
        "Δ": "$\\Delta$",
        "≈": "$\\approx$",
        "×": "$\\times$",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)

BRAIN = ROOT / "reports/brain_figures_fcstyle/figures"
PKG = ROOT / "reports/mia_submission_package"
LATEX = PKG / "latex"
LATEX_FIG = LATEX / "figures"
SUPP = PKG / "07_supplementary"
MAIN_FIG = PKG / "06_figures"

# Map suite IDs → files
SUITE = {
    "B1": "figure_B1_anatomy_underlay_montage.png",
    "B1b": "figure_B1b_contrast_montage.png",
    "B2": "figure_B2_roi_method_heatmap_panels.png",
    "B3": "figure_B3_top_roi_dots.png",
    "B4": "figure_B4_glassbrain_summary.png",
    "B4b": "figure_B4b_dkt_surface.png",
    "B5": "figure_B5_papez_network.png",
    "B6": "figure_B6_case_montages.png",
    "B7": "figure_B7_oasis_stress.png",
}

# Supplementary numbering used in the paper
SUPP_MAP = [
    ("S1", "B1", "figS1.png"),
    ("S2", "B2", "figS2.png"),
    ("S3", "B3", "figS3.png"),
    ("S4", "B4b", "figS4.png"),
    ("S5", "B4", "figS5.png"),
    ("S6", "B6", "figS6.png"),
    ("S7", "B7", "figS7.png"),
]

CAPTION_FIG7 = (
    "Figure 7. Papez-circuit consistency on AIBL DKT atrophy (supporting evidence). "
    "(a) Circuit-node staging: CN-referenced DKT atrophy z for entorhinal cortex, hippocampus, "
    "parahippocampal cortex, cingulate, and thalamus, comparing MCI−CN (gold) versus AD−CN (rust). "
    "Higher values indicate greater tissue loss relative to cognitively normal AIBL scans. "
    "(b) Cortical DKT parcels (left/right × MCI/AD): cell values are atrophy z-scores; warm colours "
    "mark stronger AD-like loss, strongest along the entorhinal axis. "
    "(c) Anatomical Papez-proxy connectome for AD−CN: node colour/size scale with atrophy z; edges "
    "are a priori limbic links (not tractography). "
    "Coverage: CN=683, MCI=53, AD=41 matched DKT scans. "
    "Interpretation boundary: supporting MRI-proxy anatomical–cognitive consistency only — "
    "not histological Papez proof, not Braak staging, not attention-map biomarkers, and not device validation."
)

SUPP_CAPTIONS = {
    "S1": (
        "Supplementary Figure S1. Staging gradients and V6 spatial association advantage (native atlas space). "
        "Dense axial montages without anatomical/MNI underlay. "
        "(a) Group means of the MRF affected posterior q for CN, MCI, and AD (inferno scale, 0–1). "
        "(b) Group contrasts MCI−CN and AD−CN for MRF q, plus atrophy AD−CN (black-centred diverging scale). "
        "(c) Model–ROI association maps: equal-pool baseline, locked V6 RC-SPE, and the advantage map "
        "Δ(V6 − equal). Maps are atlas-filled association/effect images for interpretability support; "
        "they are not voxel-wise FDR discovery maps and do not claim clinical attention biomarkers."
    ),
    "S2": (
        "Supplementary Figure S2. ROI × stage / method heatmaps. "
        "(a) MRF q group means by ROI for CN, MCI, and AD, showing the staging gradient across the "
        "21-region atlas. "
        "(b) Signed group effects (MCI−CN, AD−CN) and model–ROI associations for equal-pool, locked "
        "V6 RC-SPE (highlighted), and V7+MRF. Positive values indicate stronger AD-linked association "
        "or greater affected posterior relative to CN."
    ),
    "S3": (
        "Supplementary Figure S3. Top-20 ROIs ranked by |MRF q AD−CN|. "
        "(a) Lollipop plot of MRF q group contrasts (MCI−CN vs AD−CN). "
        "(b) Atrophy AD−CN for the same ROIs. "
        "(c) Grouped bars comparing model–ROI associations for equal-pool, V6 RC-SPE, and V7+MRF. "
        "This panel visualises where spatial associations concentrate; V6 advantage should be read "
        "together with the locked classification metrics, not as universally higher per-ROI correlation."
    ),
    "S4": (
        "Supplementary Figure S4. AIBL DKT cortical atrophy (AD−CN) projected onto fsaverage / Destrieux. "
        "Left/right hemispheres in lateral and medial views. Warm colours indicate greater CN-referenced "
        "cortical tissue loss in AD. Supporting surface visualisation complementary to the volumetric "
        "Papez/DKT analysis in Figure 7."
    ),
    "S5": (
        "Supplementary Figure S5. Dense axial summary of key spatial maps. "
        "Rows contrast CN vs AD MRF q means, AD−CN MRF q, equal-pool association, V6 RC-SPE association, "
        "and Δ(V6 − equal). Same native atlas space and claim boundary as Supplementary Figure S1."
    ),
    "S6": (
        "Supplementary Figure S6. Subject-level MRF-q case montages on locked AIBL heldout. "
        "Top: dense axial MRF affected-posterior maps for representative correct CN/MCI/AD cases and "
        "boundary errors (MCI→AD, AD→MCI, MCI→CN). Bottom: subject-level class probabilities. "
        "Illustrates that residual errors concentrate near adjacent stages rather than AD→CN collapse."
    ),
    "S7": (
        "Supplementary Figure S7. OASIS stress-test spatial cases (limitation panel). "
        "Example OASIS subjects with MRF-q axial strips and model predictions. "
        "OASIS transfer remains unresolved in the locked protocol (BAcc ≈ 0.334) and is excluded from "
        "the primary success claim; this figure documents the spatial limitation boundary only."
    ),
}

LATEX_SUPP_BLOCK = None  # built in patch_latex()


def build_latex_supp_block() -> str:
    parts = [
        r"\clearpage",
        r"\section*{Supplementary Figures}",
        r"\setcounter{figure}{0}",
        r"\renewcommand{\thefigure}{S\arabic{figure}}",
        "",
        (
            "Supporting atlas-spatial figures accompany the locked ARA-Net evidence chain. "
            "All maps are shown in the native $96\\times 112\\times 96$ analysis crop "
            "(approximate MNI-like affine for display only). Unless stated otherwise, panels use "
            "no anatomical/MNI underlay to avoid fake-template misalignment. These figures are "
            "supporting spatial consistency / limitation panels---not voxel-wise FDR discovery "
            "and not clinical attention biomarkers."
        ),
        "",
    ]
    for sid, _bid, latex_name in SUPP_MAP:
        cap = tex_escape(SUPP_CAPTIONS[sid].split(". ", 1)[1] if ". " in SUPP_CAPTIONS[sid] else SUPP_CAPTIONS[sid])
        parts.extend(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{figures/{latex_name}}}",
                rf"\caption{{{cap}}}",
                rf"\label{{fig:{sid}}}",
                r"\end{figure}",
                "",
            ]
        )
    return "\n".join(parts)


def copy_suite() -> None:
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    MAIN_FIG.mkdir(parents=True, exist_ok=True)

    # Main Fig 7 = B5
    b5 = BRAIN / SUITE["B5"]
    shutil.copy2(b5, LATEX_FIG / "fig7.png")
    shutil.copy2(b5, MAIN_FIG / "Figure_7.png")

    for sid, bid, latex_name in SUPP_MAP:
        src = BRAIN / SUITE[bid]
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, LATEX_FIG / latex_name)
        shutil.copy2(src, SUPP / f"Figure_{sid}.png")
        # also keep B1b as extra asset
    # Extra multiplane panel for package
    shutil.copy2(BRAIN / SUITE["B1b"], SUPP / "Figure_S1b_multiplane.png")
    print(f"[copy] Fig7 + S1–S7 -> {LATEX_FIG} and {SUPP}")


def write_captions() -> None:
    # Main captions file (Fig 1–6 kept short headers; Fig7 full)
    main_caps_path = PKG / "05_figure_captions.txt"
    old = main_caps_path.read_text(encoding="utf-8") if main_caps_path.exists() else ""
    lines = []
    for n in range(1, 7):
        # preserve prior short lines if present
        found = None
        for ln in old.splitlines():
            if ln.startswith(f"Figure {n}."):
                found = ln
                break
        lines.append(found or f"Figure {n}.")
    lines.append(CAPTION_FIG7)
    main_caps_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    # Supplementary captions
    supp_txt = SUPP / "Supplementary_figure_captions.txt"
    supp_txt.write_text("\n\n".join(SUPP_CAPTIONS[s] for s, _, _ in SUPP_MAP) + "\n", encoding="utf-8")

    # Docx captions
    doc = Document()
    doc.add_heading("Figure captions", level=1)
    doc.add_paragraph(
        "Main Figures 1–6 retain manuscript captions. Figure 7 and Supplementary Figures S1–S7 "
        "use the journal brain-figure suite (native atlas space)."
    )
    for ln in lines:
        doc.add_paragraph(ln)
    doc.add_heading("Supplementary figure captions", level=1)
    for s, _, _ in SUPP_MAP:
        doc.add_paragraph(SUPP_CAPTIONS[s])
    cap_docx = PKG / "05_figure_captions.docx"
    doc.save(str(cap_docx))
    shutil.copy2(cap_docx, SUPP / "Supplementary_figure_captions.docx")
    print(f"[caps] {main_caps_path}")
    print(f"[caps] {supp_txt}")


def patch_latex() -> None:
    tex = LATEX / "main.tex"
    text = tex.read_text(encoding="utf-8")

    # Replace Figure 7 caption block
    start = text.find("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/fig7.png}")
    if start < 0:
        raise RuntimeError("Could not find fig7 block in main.tex")
    end = text.find("\\label{fig:papez}\n\\end{figure}", start)
    if end < 0:
        raise RuntimeError("Could not find fig:papez end")
    end = text.find("\\end{figure}", end) + len("\\end{figure}")
    new_fig7 = (
        "\\begin{figure}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=\\linewidth]{figures/fig7.png}\n"
        "\\caption{"
        + tex_escape(CAPTION_FIG7.replace("Figure 7. ", ""))
        + "}\n"
        "\\label{fig:papez}\n"
        "\\end{figure}"
    )
    text = text[:start] + new_fig7 + text[end:]

    # Insert pointer sentence after Papez paragraph if missing
    pointer = (
        " Extended atlas-spatial staging and model-association montages are provided as "
        "Supplementary Figures~\\ref{fig:S1}--\\ref{fig:S7}."
    )
    anchor = "Hippocampus-only DKT atrophy showed similar associations"
    if "fig:S1" not in text and anchor in text:
        # insert after the sentence ending that paragraph's first block
        idx = text.find("Amyloid PET and CSF biomarkers were unavailable on AIBL")
        if idx > 0:
            text = text[:idx] + pointer.strip() + " " + text[idx:]

    # Insert supplementary block before References if not present
    if "\\section*{Supplementary Figures}" not in text:
        marker = "\\section*{References}"
        if marker not in text:
            raise RuntimeError("References marker not found")
        text = text.replace(marker, build_latex_supp_block() + "\n" + marker, 1)

    tex.write_text(text, encoding="utf-8")
    # Also sync latex_mia copy if present
    alt = PKG / "latex_mia" / "main.tex"
    if alt.exists():
        shutil.copy2(tex, alt)
        for _, _, name in SUPP_MAP:
            shutil.copy2(LATEX_FIG / name, PKG / "latex_mia" / "figures" / name)
        shutil.copy2(LATEX_FIG / "fig7.png", PKG / "latex_mia" / "figures" / "fig7.png")
    print(f"[tex] patched {tex}")


def build_word_supplement() -> Path:
    """Standalone Word file with all supplementary figures + captions (easy to paste)."""
    out = PKG / "07_supplementary" / "ARA-Net_Supplementary_Brain_Figures.docx"
    # also copy to ARA-Net root for convenience
    out2 = ROOT / "ARA-Net_Supplementary_Brain_Figures.docx"

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    doc.add_heading("ARA-Net — Supplementary brain figures", level=1)
    p = doc.add_paragraph(
        "These panels accompany the MedIA manuscript. Main-text Figure 7 uses the redesigned "
        "Papez/DKT panel. Supplementary Figures S1–S7 provide staging gradients, model–ROI "
        "associations, surface atrophy, case montages, and the OASIS limitation panel. "
        "Claim boundary: supporting spatial consistency only — not voxel FDR, Braak staging, "
        "or clinical attention biomarkers."
    )
    p.runs[0].font.size = Pt(10)

    # Main Fig 7 preview
    doc.add_heading("Main text — Figure 7", level=2)
    doc.add_picture(str(BRAIN / SUITE["B5"]), width=Inches(6.3))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(CAPTION_FIG7)
    for run in cap.runs:
        run.font.size = Pt(9)
        run.italic = True

    for sid, bid, _ in SUPP_MAP:
        doc.add_heading(f"Supplementary Figure {sid}", level=2)
        img = BRAIN / SUITE[bid]
        width = 6.3 if sid != "S4" else 5.8
        doc.add_picture(str(img), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph(SUPP_CAPTIONS[sid])
        for run in c.runs:
            run.font.size = Pt(9)
            run.italic = True

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    shutil.copy2(out, out2)
    # Also refresh manuscript package main docx figure 7 if present
    print(f"[docx] {out}")
    print(f"[docx] {out2}")
    return out2


def main() -> None:
    copy_suite()
    write_captions()
    patch_latex()
    build_word_supplement()
    print("[done] brain figures embedded with explanatory captions")


if __name__ == "__main__":
    main()
