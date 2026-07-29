#!/usr/bin/env python3
"""Build English qualification exam PPT with speaker notes (presentation script)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

import build_qual_exam_ppt as base
import ppt_design as design

ROOT = base.ROOT
TEMPLATE = base.TEMPLATE
OUTPUT = ROOT / "Atlas-Guided-AD-Qualification-Exam-PPT-English-Speaker-Notes.pptx"

MARGIN = base.MARGIN
MAX_W = base.MAX_W
MAX_H = base.MAX_H
CONTENT_TOP = base.CONTENT_TOP
FOOTER_TOP = base.FOOTER_TOP
TITLE_TOP = base.TITLE_TOP
SUBTITLE_TOP = base.SUBTITLE_TOP
BLUE = base.BLUE
DARK = base.DARK
GRAY = base.GRAY
ACCENT = base.ACCENT
GREEN = base.GREEN


def add_textbox(slide, left, top, width, height, lines, size=11, bold_first=False, align_center=False):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.name = "Arial"
        p.font.color.rgb = DARK
        if align_center:
            p.alignment = PP_ALIGN.CENTER
        if line.startswith("Key conclusion:"):
            p.font.bold = True
            p.font.color.rgb = BLUE
        elif line.startswith("Transition:"):
            p.font.bold = True
            p.font.color.rgb = GREEN
        elif line.startswith("Figure:"):
            p.font.italic = True
            p.font.color.rgb = GRAY
            p.font.size = Pt(max(9, size - 1))
        elif line.startswith("Limitation:") or line.startswith("Main limitation:"):
            p.font.color.rgb = ACCENT
        if i == 0 and bold_first:
            p.font.bold = True


def set_speaker_notes(slide, text: str) -> None:
    if not text.strip():
        return
    tf = slide.notes_slide.notes_text_frame
    tf.clear()
    tf.text = text.strip()


def fill_shape_lines(shape, lines, size=12, bold_first=False, align_center=False):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.name = "Arial"
        p.font.color.rgb = DARK
        if align_center:
            p.alignment = PP_ALIGN.CENTER
        if i == 0 and bold_first:
            p.font.bold = True


def _clear_nonpicture_shapes(slide) -> None:
    """Drop template text/auto shapes (keep pictures such as logos) so we can
    redraw clean content over the template background."""
    for sh in list(slide.shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
            continue
        sh._element.getparent().remove(sh._element)


def build_cover(prs, notes: str) -> None:
    slide = prs.slides[0]
    _clear_nonpicture_shapes(slide)

    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.25),
                                  Inches(base.SLIDE_W), Inches(1.85))
    band.fill.solid()
    band.fill.fore_color.rgb = design.BLUE
    band.line.fill.background()

    design.textbox(
        slide, 0.8, 2.45, base.SLIDE_W - 1.6, 1.5,
        "Atlas-Guided Interpretable Multi-Mechanism Modeling of Alzheimer's "
        "Disease and Trustworthy Clinical Decision Support",
        font="Arial", size=23, color=design.WHITE, bold=True,
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
    )
    add_textbox(
        slide, 0.8, 4.6, base.SLIDE_W - 1.6, 1.4,
        [
            "Discipline: Precision Medicine and Healthcare",
            "Candidate: Yuanqin Zhao (PhD, Class of 2025)",
            "Supervisors: Prof. Shaohua Ma, Prof. Tengfei Guo",
            "iBHE, Tsinghua SIGS   |   Doctoral Qualification Exam   |   2026-06-09",
        ],
        size=13, align_center=True,
    )
    set_speaker_notes(slide, notes)


def build_contents(prs, toc: list[str], notes: str) -> None:
    slide = prs.slides[1]
    _clear_nonpicture_shapes(slide)
    design.sidebar(slide, "Contents", None)
    design.header(slide, "Contents", "Outline of this qualification exam", MARGIN, MAX_W)

    top = CONTENT_TOP + 0.25
    row_h = (FOOTER_TOP - top) / max(1, len(toc))
    for i, item in enumerate(toc):
        y = top + i * row_h
        num = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(MARGIN), Inches(y + (row_h - 0.46) / 2),
                                     Inches(0.46), Inches(0.46))
        num.fill.solid()
        num.fill.fore_color.rgb = design.BLUE
        num.line.fill.background()
        ntf = num.text_frame
        ntf.margin_left = ntf.margin_right = ntf.margin_top = ntf.margin_bottom = Inches(0)
        np = ntf.paragraphs[0]
        np.alignment = PP_ALIGN.CENTER
        nr = np.add_run()
        nr.text = str(i + 1)
        nr.font.size = Pt(16)
        nr.font.bold = True
        nr.font.name = "Arial"
        nr.font.color.rgb = design.WHITE
        design.textbox(slide, MARGIN + 0.7, y, MAX_W - 0.7, row_h, item,
                       font="Arial", size=15, color=design.DARK, bold=True,
                       anchor=MSO_ANCHOR.MIDDLE)
    set_speaker_notes(slide, notes)


def layout_narrative(prs, section, title, subtitle, panels, bullets, conclusion="", bridge="", notes="", slide_num=None):
    slide = base.new_slide(prs)
    design.sidebar(slide, section, slide_num)
    design.header(slide, title, subtitle, MARGIN, MAX_W)

    lines = list(bullets)
    if conclusion:
        lines += ["", f"Key conclusion: {conclusion}"]
    if bridge:
        lines.append(f"Transition: {bridge}")

    area_top = CONTENT_TOP
    area_h = FOOTER_TOP - CONTENT_TOP
    has_fig = any(base.img(p["file"]) for p in panels)

    if not has_fig:
        design.text_block(slide, MARGIN, area_top, MAX_W, area_h, lines, size=13)
    else:
        n_lines = len([l for l in lines if l.strip()])
        text_h = min(area_h * 0.40, max(0.85, n_lines * 0.24 + 0.12)) if lines else 0.0
        sep = 0.20 if text_h else 0.0
        fig_h = area_h - text_h - sep
        design.figure_grid(
            base.add_picture_fitted, base.fit_image, base.img, slide, panels,
            MARGIN, area_top, MAX_W, fig_h,
        )
        if lines:
            design.text_block(slide, MARGIN, area_top + fig_h + sep, MAX_W, text_h, lines, size=11)
    set_speaker_notes(slide, notes)
    return slide


def layout_chapter(prs, section, title, main_img, side_img, main_cap, side_cap, blocks, metrics, limitation, bridge, notes, slide_num=None):
    slide = base.new_slide(prs)
    design.sidebar(slide, section, slide_num)
    design.header(slide, title, "", MARGIN, MAX_W)

    area_top = CONTENT_TOP
    area_h = FOOTER_TOP - CONTENT_TOP
    text_w = 4.15
    lines = list(blocks) + ["", "Key results:"] + list(metrics) + ["", f"Main limitation: {limitation}", f"Transition: {bridge}"]
    design.text_block(slide, MARGIN, area_top, text_w, area_h, lines, size=10)

    fig_left = MARGIN + text_w + 0.30
    fig_w = MAX_W - text_w - 0.30
    panels = []
    if main_img:
        panels.append({"file": main_img, "caption": main_cap})
    if side_img:
        panels.append({"file": side_img, "caption": side_cap})
    design.figure_grid(
        base.add_picture_fitted, base.fit_image, base.img, slide, panels,
        fig_left, area_top, fig_w, area_h, force_cols=1,
    )
    set_speaker_notes(slide, notes)


def slide_spec(**kwargs) -> dict[str, Any]:
    return kwargs


SLIDES: list[dict[str, Any]] = [
    slide_spec(
        kind="cover",
        notes="Good morning, distinguished committee members. I am Yuanqin Zhao, a 2025 PhD student in Precision Medicine and Healthcare at Tsinghua SIGS iBHE, supervised by Professors Shaohua Ma and Tengfei Guo. Today I present my doctoral qualification exam on atlas-guided, interpretable, multi-mechanism modeling of Alzheimer's disease and trustworthy clinical decision support. This is a four-layer research program—not a single black-box product—linking interpretable staging, mechanism disentanglement, longitudinal dynamics, and trust-aware decision support.",
    ),
    slide_spec(
        kind="contents",
        toc=[
            "I. Background (A–J): burden → pathology → imaging → AI gaps",
            "II. Research gap and four-layer framework",
            "III. Chapters 1–4: ARA-Net, PathwayPro, A2C-NODE, CUED-AD",
            "IV. Architecture, contributions, and outlook",
            "V. References",
        ],
        notes="My talk follows a single narrative thread. First, ten background sections explain why AD matters clinically and why current AI is insufficient. Then I state the research gap and our four-layer framework. The core of the exam is four chapter-specific methods with locked metrics and explicit limitations. I close with architecture, contributions, and future work.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | Overview",
        title="Background logic: from AD burden to trustworthy AI",
        subtitle="Ten sections leading to four dissertation chapters",
        panels=[{"file": "slide2_technology_timeline.png", "caption": "2012–2026 technology evolution and four bottlenecks"}],
        bullets=[
            "Logic: (1) rising AD burden; (2) multi-mechanism, dynamic disease;",
            "(3) multimodal imaging + anatomical priors; (4) AI gaps in generalization, mechanisms, dynamics, trust;",
            "(5) response: ARA-Net → PathwayPro → A2C-NODE → CUED-AD.",
        ],
        conclusion="Anatomy-anchored evidence links interpretable diagnosis, disentanglement, longitudinal inference, and trust.",
        notes="Before diving into details, let me state the backbone of this dissertation. We start from a major public-health problem, recognize that AD is biologically complex and temporally dynamic, and acknowledge that multimodal MRI and atlas priors give us an auditable interface between AI and medicine. Existing models often score well internally but fail on external sites, mechanism separation, irregular follow-up, and clinical communication. Our answer is a progressive four-chapter stack, each adding one auditable capability.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | A. Disease burden",
        title="A1. Global and China AD disease burden",
        subtitle="Why is AD a major research priority?",
        panels=[
            {"file": "bg_global_AD_incidence_projection.png", "caption": "Global AD incidence projections (JOGH)"},
            {"file": "bg_AD_global_epidemic.jpg", "caption": "Global AD epidemic warning (UCLA)"},
            {"file": "slide2_AD_burden_infographic.png", "caption": "AD burden infographic"},
        ],
        bullets=[
            "Aging populations drive rapid growth in dementia and AD prevalence.",
            "China: ~9.83M AD cases (60+); ~15.07M dementia (Jia et al., 2022).",
            "AD is a public-health and socioeconomic challenge, not only a neurology topic.",
        ],
        conclusion="AD urgently needs early detection, staging, progression modeling, and long-term management tools.",
        bridge="Next: biological mechanisms (Aβ, Tau, neurodegeneration).",
        notes="Let me begin with disease burden. The left and center figures show that AD incidence is rising globally and regionally. In China, the numbers are already in the millions and projected to grow substantially by 2030. This creates enormous caregiver, clinical, and economic pressure. That is the first reason this topic deserves a rigorous dissertation: the clinical need is real, large, and sustained.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | A. Disease burden",
        title="A2. China aging, economic cost, and early-detection need",
        subtitle="From treatment to early screening and longitudinal management",
        panels=[
            {"file": "bg_China_aging_population_trend.png", "caption": "China aging trend 2000–2050"},
            {"file": "bg_China_aging_report.png", "caption": "China aging report data"},
            {"file": "bg_China_aging_RAND.png", "caption": "RAND elderly population projections"},
            {"file": "slide2_China_AD_report_2024.png", "caption": "China AD Report 2024"},
            {"file": "slide2_AD_cost_projection.jpg", "caption": "Global AD economic cost projection"},
        ],
        bullets=[
            "Projected ~19.11M AD cases (60+) and ~973.8B CNY direct cost by 2030.",
            "Under-diagnosis and long-term care needs motivate scalable imaging AI.",
            "Clinical need: non-invasive, repeatable, decision-support tools.",
        ],
        conclusion="Imaging AI can support early risk stratification if anchored to clinical questions.",
        bridge="Next: AD pathology—Aβ, Tau, and cascade mechanisms.",
        notes="These figures quantify the China-specific aging curve and economic burden. The key message is that we are moving from late-stage treatment toward early screening, staging, and longitudinal management. Neuroimaging AI is attractive because MRI is widely available and repeatable—but only if outputs are clinically meaningful, not just high AUC on one cohort.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | B. Pathology",
        title="B1. Aβ plaques, Tau tangles, and cascade hypothesis",
        subtitle="What biological mechanisms underlie AD?",
        panels=[
            {"file": "bg_AD_pathology_plaques_tangles.png", "caption": "Aβ plaques vs Tau tangles"},
            {"file": "bg_amyloid_cascade_hypothesis.jpg", "caption": "Amyloid cascade hypothesis"},
            {"file": "bg_amyloid_tau_synergy.png", "caption": "Aβ–Tau synergy"},
        ],
        bullets=[
            "Classic markers: amyloid plaques and neurofibrillary tangles.",
            "Cascade view: Aβ dysregulation may precede Tau, synaptic injury, and cognitive decline.",
            "Modern view: Aβ and Tau interact; Tau spread tracks symptoms closely.",
        ],
        conclusion="AD is a continuous multi-mechanism process—not a single diagnostic label.",
        bridge="Motivates multi-mechanism modeling (PathwayPro).",
        notes="Biologically, AD is not explained by one number. These panels show the two hallmark pathologies and the classical cascade, but also the more modern synergy perspective. This matters for AI because a single latent 'AD score' can hide distinct mechanisms. Chapter 2 directly responds to this by separating atrophy and vascular pathways.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | B. Pathology",
        title="B2. From molecular pathology to neurodegeneration",
        subtitle="Heterogeneity beyond a single label",
        panels=[
            {"file": "bg_tau_spreading_brain.png", "caption": "Tau spreading across the brain"},
            {"file": "bg_AD_challenges_mechanisms.webp", "caption": "Multi-mechanism AD challenges"},
        ],
        bullets=[
            "Molecular events propagate through synapses and networks to system-level decline.",
            "Patients differ in amyloid-dominant, tau-dominant, or mixed pathology.",
        ],
        conclusion="Models should separate mechanism pathways rather than one global abnormality score.",
        bridge="Next: disease progression and staging over time.",
        notes="Here I emphasize heterogeneity. Two patients both labeled AD may reflect different mixtures of amyloid, tau, neurodegeneration, inflammation, and vascular injury. That is why mechanism disentanglement is scientifically meaningful, not just an engineering trick.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | C. Progression",
        title="C. Braak staging, AD continuum, and longitudinal Tau",
        subtitle="How does AD evolve over time?",
        panels=[
            {"file": "bg_Braak_staging_tau_PET.jpg", "caption": "Braak stages and Tau PET"},
            {"file": "bg_AD_progression_stages.jpeg", "caption": "Preclinical AD → MCI → dementia curve"},
            {"file": "bg_tau_staging_longitudinal.png", "caption": "Longitudinal Tau staging"},
        ],
        bullets=[
            "Tau often spreads from medial temporal lobe outward (Braak I–VI).",
            "Continuum: CN → preclinical AD → MCI due to AD → AD dementia.",
            "Longitudinal data show AD as a dynamic process, not a static snapshot.",
        ],
        conclusion="AD requires continuous-time modeling, not one-time static classification alone.",
        bridge="Leads to Chapter 3 A2C-NODE.",
        notes="This section bridges biology to Chapter 3. Braak staging and the clinical continuum tell us AD evolves. A single baseline MRI classifier ignores when visits occur and how pathology propagates. That motivates continuous-time graph neural ODE modeling with irregular follow-up.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | D. MRI evidence",
        title="D1. Normal vs AD structural MRI",
        subtitle="What does AD look like on MRI?",
        panels=[
            {"file": "bg_healthy_vs_AD_brain_MRI.jpg", "caption": "Healthy vs AD brain MRI"},
            {"file": "bg_brain_shape_atrophy_aging_AD.webp", "caption": "Aging vs AD atrophy patterns"},
            {"file": "bg_neuroimaging_AD_advances.png", "caption": "Neuroimaging advances in AD"},
        ],
        bullets=[
            "Typical signs: hippocampal/MTL atrophy, cortical thinning, ventricular enlargement.",
            "MRI gives non-invasive neurodegeneration evidence, but changes are subtle and regional.",
        ],
        conclusion="MRI is valuable when mapped to anatomical regions—not black-box whole-brain scores.",
        bridge="Leads to Chapter 1 ARA-Net (atlas-guided staging).",
        notes="Clinicians think in regions: hippocampus, entorhinal cortex, ventricles. These images make that concrete. The dissertation's first chapter therefore uses atlas-derived regional features rather than an opaque end-to-end CNN embedding.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | D. MRI evidence",
        title="D2. HC/MCI/AD comparison and the MCI boundary",
        subtitle="MCI sits between normal aging and dementia",
        panels=[
            {"file": "slide9_brain_atrophy_AD.png", "caption": "HC / MCI / AD structural comparison"},
            {"file": "slide9_AD_subtypes_atrophy.webp", "caption": "AD subtype atrophy patterns"},
        ],
        bullets=[
            "Structural change intensifies along CN→MCI→AD, but MCI is heterogeneous and error-prone.",
            "Whole-image black-box models rarely say which region drives the decision.",
        ],
        conclusion="We need subject-level, atlas-auditable staging (ARA-Net).",
        bridge="Next: why multimodal imaging matters.",
        notes="MCI is the hardest clinical boundary—and exactly where our Chapter 1 still has weakness, which I will report honestly. These figures show why region-level evidence matters: the pattern is distributed, subtle, and subtype-dependent.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | E. Multimodal imaging",
        title="E. PET/MRI fusion, FLAIR, and MRI-to-PET synthesis",
        subtitle="Why is T1 MRI alone insufficient?",
        panels=[
            {"file": "bg_PET_MRI_AD_multimodal.webp", "caption": "Multimodal PET/MR frontier"},
            {"file": "bg_PET_MRI_dementia.gif", "caption": "PET-MRI in dementia workup"},
            {"file": "bg_FLAIR_WMH_classification.jpg", "caption": "FLAIR white-matter hyperintensities"},
            {"file": "slide2_MaM_DiT_overview.jpg", "caption": "MaM-DiT MRI-to-PET synthesis"},
        ],
        bullets=[
            "PET: molecular pathology; MRI: structure; FLAIR: WMH/vascular burden.",
            "Field is moving from single-modality structure to multimodal biology.",
        ],
        conclusion="Multimodal fusion should be mechanism-aware, not naive concatenation.",
        bridge="Leads to Chapter 2 PathwayPro (T1 atrophy + FLAIR vascular).",
        notes="Different modalities answer different questions. T1 captures atrophy; FLAIR captures white-matter and vascular burden; PET captures amyloid or tau biology. Chapter 2 explicitly encodes this split with dual encoders rather than mixing everything into one latent vector.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | F. Anatomical priors",
        title="F1. Brain atlases, ATN framework, NIA-AA definition",
        subtitle="Anatomical priors bridge medicine and AI",
        panels=[
            {"file": "slide2_FreeSurfer_atlas.png", "caption": "FreeSurfer parcellation"},
            {"file": "slide2_Desikan_Killiany_atlas.png", "caption": "Desikan-Killiany atlas"},
            {"file": "slide2_ATN_framework.png", "caption": "ATN biomarker framework"},
            {"file": "slide2_NIA_AA_biological_definition.webp", "caption": "NIA-AA 2024 biological definition"},
        ],
        bullets=[
            "Atlases convert complex MRI into interpretable regional evidence.",
            "ATN standardizes amyloid, tau, and neurodegeneration language.",
        ],
        conclusion="Anatomical priors turn black-box probabilities into region-level audit trails.",
        bridge="Supports Chapters 1 and 3 (21-region interface).",
        notes="This is the theoretical backbone of the entire thesis. FreeSurfer and Desikan-Killiany give us a shared anatomical vocabulary. ATN gives us a biological staging language. Our models are organized around these interfaces so committee members can ask 'which region?' and we can answer with evidence.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | F. Anatomical priors",
        title="F2. DMN, MTL networks, and priors in AI",
        subtitle="Memory-related networks are vulnerable in AD",
        panels=[
            {"file": "slide2_DMN_network.jpeg", "caption": "Default mode network"},
            {"file": "slide2_MTL_network_AD.jpg", "caption": "Medial temporal lobe network in AD"},
        ],
        bullets=[
            "DMN connectivity declines early in AD (precuneus, angular gyrus, PCC).",
            "Hippocampus–MTL–posteromedial networks link memory decline to pathology spread.",
            "We use 21-region atlas features instead of unconstrained deep black boxes.",
        ],
        conclusion="Network and atlas priors are shared language across all four chapters.",
        bridge="Next: cerebrovascular comorbidity.",
        notes="Networks explain why certain regions appear repeatedly in AD progression literature. Our atlas interface is not arbitrary—it aligns with known vulnerable systems such as MTL and DMN-related anatomy.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | G. Cerebrovascular comorbidity",
        title="G. WMH, cerebrovascular disease, and mixed pathology",
        subtitle="Why neurodegeneration alone is insufficient",
        panels=[
            {"file": "bg_WMH_cognitive_impairment.jpg", "caption": "WMH and cognitive impairment"},
            {"file": "bg_cerebrovascular_disease_AD.jpg", "caption": "Cerebrovascular disease and AD"},
            {"file": "bg_WMH_progression_AD.jpg", "caption": "Longitudinal WMH progression"},
        ],
        bullets=[
            "Older AD patients often have mixed vascular pathology, not 'pure AD'.",
            "WMH burden accelerates decline and affects MCI-to-AD conversion.",
            "A single latent feature can confound atrophy with vascular injury.",
        ],
        conclusion="AI should separate neurodegenerative atrophy from vascular/WMH burden.",
        bridge="Core motivation for PathwayPro (z_a vs z_v).",
        notes="This is the clinical motivation for Chapter 2. In real memory clinics, FLAIR abnormalities matter. If we only optimize accuracy on T1, vascular contributions remain hidden. PathwayPro names two pathways—atrophy and vascular—and measures whether they are disentangled.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | H. AI applications",
        title="H. What AI achieves—and what is still missing",
        subtitle="From high AUC to generalizable, trustworthy models",
        panels=[
            {"file": "bg_DL_pipeline_AD_detection.jpg", "caption": "CNN/3D-CNN AD pipeline"},
            {"file": "bg_XAI_brain_connectivity.webp", "caption": "XAI brain connectivity"},
            {"file": "bg_explainable_AI_AD_SECNN.webp", "caption": "Explainable AD diagnosis"},
        ],
        bullets=[
            "CNN/ViT/GNN can reach high AUC on CN/MCI/AD classification.",
            "Issues: leakage, weak external generalization, post-hoc explanations, no uncertainty communication.",
        ],
        conclusion="The bottleneck is trust, not accuracy alone.",
        bridge="Motivates all four dissertation chapters.",
        notes="I want to be precise: the field has progress. But many papers still rely on internal validation, slice-level splits, or saliency maps that are not anatomy-constrained. Our four chapters each target one gap: external interpretable staging, mechanism disentanglement, irregular longitudinal modeling, and conflict-aware uncertainty.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | I. Policy & regulation",
        title="I. FDA/EMA trends and trustworthy AI",
        subtitle="Why trust is also a regulatory direction",
        panels=[
            {"file": "slide3_FDA_AI_timeline.jpg", "caption": "FDA AI/ML regulatory timeline"},
            {"file": "slide3_EMA_FDA_2026.jpg", "caption": "EMA-FDA 2026 AI guidance"},
            {"file": "slide3_FDA_7step_framework.jpg", "caption": "FDA seven-step AI framework"},
        ],
        bullets=[
            "Regulators emphasize context of use, data quality, transparency, lifecycle management.",
            "Medical AI is moving from lab models toward governed clinical support tools.",
        ],
        conclusion="High AUC is insufficient; external validation, calibration, and uncertainty matter.",
        bridge="Leads to Chapter 4 CUED-AD.",
        notes="I am careful not to overclaim regulatory clearance. The point is directional: transparency, traceability, and lifecycle thinking are increasingly expected. Chapter 4 therefore reports conflict and uncertainty explicitly rather than pretending the model should always auto-decide.",
    ),
    slide_spec(
        kind="narrative",
        section="Background | J. Datasets",
        title="J. ADNI, external validation, and longitudinal follow-up",
        subtitle="Why dataset roles matter",
        panels=[
            {"file": "bg_ADNI_history_timeline.png", "caption": "ADNI history and modalities"},
            {"file": "bg_ADNI4_design.png", "caption": "ADNI4 design and follow-up"},
            {"file": "bg_AD_subtypes_longitudinal.jpg", "caption": "AD subtype trajectories"},
        ],
        bullets=[
            "ADNI: development and internal longitudinal analysis.",
            "AIBL: external validation; IXI: healthy controls; OASIS: stress tests.",
            "Subject-level splits are mandatory to avoid leakage.",
        ],
        conclusion="Reliable AD AI needs longitudinal follow-up and explicit external roles—not ADNI-only scores.",
        bridge="Background closes; next: gap mapping and framework.",
        notes="Our validation spine is deliberate. ADNI is for development; AIBL tests transport; IXI checks healthy-control behavior; OASIS stress-tests zero-shot settings. I will never compare Acc across chapters with different tasks without stating that boundary.",
    ),
    slide_spec(
        kind="narrative",
        section="Gap mapping",
        title="Four gaps mapped to four chapters",
        subtitle="Gap-to-solution overview",
        panels=[{"file": "slide3_gap_solution_mapping.png", "caption": "Gap-to-chapter mapping diagram"}],
        bullets=[
            "Gap 1 interpretable staging + external validation → ARA-Net",
            "Gap 2 mechanism disentanglement → PathwayPro",
            "Gap 3 continuous-time longitudinal inference → A2C-NODE",
            "Gap 4 uncertainty/conflict/trust → CUED-AD",
        ],
        conclusion="Four chapters form one progressive research system.",
        notes="This slide is the pivot from background to contribution. Each red gap on the figure corresponds to one blue module in our dissertation. The committee should see one coherent program, not four unrelated papers.",
    ),
    slide_spec(
        kind="narrative",
        section="Core problem | Positioning",
        title="From high-AUC black boxes to auditable clinical evidence",
        subtitle="Current status vs target",
        panels=[{"file": "slide7_research_positioning.png", "caption": "Status vs goal comparison"}],
        bullets=[
            "Status: internal validation, post-hoc XAI, naive fusion, point predictions.",
            "Target: subject-level external validation, atlas evidence, ODE trajectories, U_total support.",
            "Scope: modular research prototypes—not a deployed clinical product.",
        ],
        conclusion="We address interpretability, mechanisms, dynamics, and trust gaps.",
        notes="Here I explicitly state what this dissertation is and is not. It is a stack of auditable research prototypes with locked metrics and declared limitations. It is not an FDA-cleared device and not a single end-to-end commercial pipeline.",
    ),
    slide_spec(
        kind="narrative",
        section="Core problem | Framework",
        title="Four-layer progressive framework",
        subtitle="Where → What → How it evolves → Can we trust it?",
        panels=[
            {"file": "slide4_four_dimension_framework.png", "caption": "Four-layer funnel"},
            {"file": "slide8_clinical_vision.png", "caption": "Four-quadrant clinical scenarios"},
        ],
        bullets=[
            "Layer 1 ARA-Net: atlas-guided CN/MCI/AD staging.",
            "Layer 2 PathwayPro: T1+FLAIR atrophy/vascular disentanglement.",
            "Layer 3 A2C-NODE: 21-region graph neural ODE reports.",
            "Layer 4 CUED-AD: MRI–clinical conflict and uncertainty.",
        ],
        conclusion="Shared atlas interface and ADNI→AIBL/IXI/OASIS validation spine.",
        notes="The funnel figure is the dissertation's conceptual map. Layer 1 tells us where changes occur; Layer 2 what mechanisms are involved; Layer 3 how the patient may evolve; Layer 4 whether we should trust or escalate to human review.",
    ),
    slide_spec(
        kind="chapter",
        section="Innovations | Chapter 1",
        title="Chapter 1 — ARA-Net: Atlas-Guided Multimodal Staging",
        main_img="slide9_ARANet_architecture.png",
        side_img="figure_real_render_assets/fig2_feature_system/brain_surface_6view.png",
        main_cap="ARA-Net: six base streams + RC-SPE fusion",
        side_cap="Rendered 21-region atlas six-view (measured)",
        blocks=[
            "Input: T1 MRI → FreeSurfer 21-region atlas + clinical variables",
            "Method: six probability streams + weighted log-pooling + risk constraints",
            "Validation: AIBL held-out subject-level external test (n=216)",
        ],
        metrics=["Acc 0.903 | BAcc 0.833 | Macro AUC 0.937", "Recall 0.961/0.686/0.852 | AD→CN=0"],
        limitation="MCI recall 0.686 remains the main weakness under risk constraints.",
        bridge="Provides auditable atlas evidence for later chapters.",
        notes="Chapter 1 answers: can we stage CN/MCI/AD with anatomy-linked evidence and true external validation? The architecture figure shows six base probability flows fused by RC-SPE. On AIBL held-out data we obtain strong overall metrics, but I must highlight the honest weakness—MCI recall is 0.686. That boundary case is clinically important and remains open work.",
    ),
    slide_spec(
        kind="chapter",
        section="Innovations | Chapter 2",
        title="Chapter 2 — PathwayPro: Atrophy vs Vascular Disentanglement",
        main_img="slide10_PathwayPro_architecture.png",
        side_img="chapter2_disentangle/manuscript_results/3d_render_examples/true3d_021_S_4718_gradcam_roi_overlay.png",
        main_cap="Dual-pathway T1 atrophy + FLAIR vascular architecture",
        side_cap="PathwayPro 3D GradCAM + ROI overlay (ADNI case)",
        blocks=[
            "Input: paired T1+FLAIR (ADNI Phase-0, n=63 test)",
            "Method: z_a atrophy branch + z_v WMH/vascular branch + disentanglement loss",
            "Controls: FactorVAE baselines; OASIS/AIBL zero-shot tests",
        ],
        metrics=["ADNI 3-seed mean Acc 64.6% | BAcc 59.2%", "z_v→WMH AUC 0.923 | disent −0.014"],
        limitation="Accuracy–disentanglement trade-off: higher disent variants lower BAcc.",
        bridge="Addresses Background G: separating atrophy from vascular comorbidity.",
        notes="PathwayPro is a benchmark for mechanism disentanglement, not a claim of SOTA accuracy. The WMH branch is informative—AUC 0.923—but disentanglement score is near zero, meaning separation is still weak. I report that trade-off openly because scientific credibility matters more than cherry-picking one seed.",
    ),
    slide_spec(
        kind="chapter",
        section="Innovations | Chapter 3",
        title="Chapter 3 — A2C-NODE: Continuous-Time Longitudinal Dynamics",
        main_img="slide11_A2C_NODE_architecture.png",
        side_img="chapter3_neural_ode/A2C-NODE/a2c_remote_results/runs/v4_mm_analysis_npj_5seed/trajectory_brain_first.png",
        main_cap="21-region graph neural ODE architecture",
        side_cap="A2C-NODE longitudinal brain-region trajectory (ADNI)",
        blocks=[
            "Input: longitudinal T1-MRI with irregular visit times",
            "Method: anatomy-anchored graph neural ODE + pMCI/sMCI + MMSE reporting",
            "Baseline: LSTM on fixed time grids",
        ],
        metrics=["ADNI BAcc 0.854 | macro-AUC 0.963 | MMSE MAE 2.69", "vs LSTM BAcc 0.678 (+17.6 pp)"],
        limitation="AIBL raw Acc 0.297 reflects cohort shift—report with sensitivity analyses.",
        bridge="Addresses Background C: static snapshots → continuous trajectories.",
        notes="Chapter 3 is where we move from classification to dynamics. The ODE respects irregular visits—critical in real ADNI-style follow-up. We gain 17.6 percentage points in BAcc over LSTM on ADNI, but external AIBL accuracy drops sharply. I present both because external failure modes are part of the scientific story.",
    ),
    slide_spec(
        kind="chapter",
        section="Innovations | Chapter 4",
        title="Chapter 4 — CUED-AD: Trust-Aware Clinical Decision Support",
        main_img="slide12_CUED_AD_architecture.png",
        side_img="chapter4_generalization/artifacts/streamlit_case_images/case_02_123_S_0072_annotated.png",
        main_cap="Dual-branch fusion with conflict detection",
        side_cap="CUED-AD Streamlit case: MRI + conflict/UQ labels",
        blocks=[
            "Input: MRI branch + clinical/demographic branch",
            "Method: D_JS conflict detection + U_total uncertainty",
            "Task: AIBL AD vs NC (binary—not comparable to Ch1 three-class Acc)",
        ],
        metrics=["Acc 0.984 | AUC 0.997", "Conflict AUC 0.753 | U_total 18.01 vs 7.50 (conflict vs non-conflict)"],
        limitation="Not FDA/NMPA cleared; conflict AUC ~0.75 needs improvement.",
        bridge="Addresses Background H/I: from scores to trust-aware support.",
        notes="Chapter 4 is about communication under disagreement. When MRI and clinical evidence conflict, total uncertainty rises—shown by the separation in U_total. This module is a research prototype for human–AI teaming: escalate, review, or defer—not autonomous diagnosis.",
    ),
    slide_spec(
        kind="narrative",
        section="Preliminary Results | Brain Renders",
        title="Rendered brain evidence across four chapters",
        subtitle="Real pipeline outputs — not stock illustrations",
        panels=[
            {"file": "figure_real_render_assets/fig2_feature_system/fig2_real_brain_panels_abc.png", "caption": "Ch1: MRI → 21-region atlas feature system"},
            {"file": "figure_real_render_assets/fig5_error_structure/fig5g_brain_profile.png", "caption": "Ch1: error-mode regional brain profile"},
            {"file": "chapter2_disentangle/manuscript_results/3d_render_examples/multicase_gradcam_gallery_12_risk_columns_rendered_ppro.png", "caption": "Ch2: multi-case 3D GradCAM risk columns"},
            {"file": "chapter3_neural_ode/A2C-NODE/a2c_remote_results/runs/v4_mm_analysis_npj_5seed/brain/ate_glass_brain.png", "caption": "Ch3: ATE glass-brain effect map"},
            {"file": "chapter3_neural_ode/A2C-NODE/a2c_remote_results/runs/v4_mm_analysis_npj_5seed/trajectory_brain_first.png", "caption": "Ch3: longitudinal regional trajectory"},
            {"file": "chapter4_generalization/demo_assets/images/cued_ad_conflict_uncertainty.png", "caption": "Ch4: conflict vs uncertainty distribution"},
        ],
        bullets=[
            "All panels are rendered from this project's locked pipelines on real cohorts.",
            "Ch1–Ch3 share the atlas interface; Ch4 targets clinical communication.",
        ],
        conclusion="A visual evidence chain from atlas interpretability to trust-aware support.",
        notes="This slide is the visual punchline of the dissertation prototypes. I walk left to right: atlas feature system, error profiles, 3D GradCAM disentanglement, glass-brain ATE effects, longitudinal trajectories, and finally conflict-aware uncertainty. These are not clipart—they are outputs from our code on ADNI/AIBL-style data.",
    ),
    slide_spec(
        kind="narrative",
        section="Preliminary Results | Locked Metrics",
        title="Key quantitative figures per chapter",
        subtitle="Complement brain renders with locked performance panels",
        panels=[
            {"file": "figure_real_render_assets/fig4_external/fig4e_ad_vs_cn_roc.png", "caption": "Ch1: AIBL external ROC"},
            {"file": "chapter2_disentangle/paper/figures/fig4_phase0_benchmark.png", "caption": "Ch2: Phase-0 benchmark"},
            {"file": "chapter3_neural_ode/A2C-NODE/a2c_remote_results/paper/figures/composed/Fig2_internal_results.png", "caption": "Ch3: internal longitudinal results"},
            {"file": "chapter4_generalization/examples/real_ood_outputs/real_fig10_comprehensive_dashboard.png", "caption": "Ch4: uncertainty dashboard"},
        ],
        bullets=["Pair numbers with anatomy: metrics answer how well; brain renders answer why."],
        conclusion="Interpretability and verification go together in this dissertation.",
        notes="After the brain gallery, I briefly show the locked metric panels—external ROC, Phase-0 benchmark, longitudinal internal results, and the CUED-AD uncertainty dashboard. The message is that we report both performance and anatomical evidence, not one without the other.",
    ),
    slide_spec(
        kind="narrative",
        section="Innovations | Supplementary",
        title="Chapters 2–4 supplementary reference figures",
        subtitle="Literature support for disentanglement, ODE, and trust",
        panels=[
            {"file": "slide10_disentangled_imaging.jpg", "caption": "Imaging disentanglement review"},
            {"file": "slide11_latent_ODE.png", "caption": "Latent ODE architecture"},
            {"file": "slide11_ODE_RNN_comparison.jpg", "caption": "ODE-RNN vs RNN comparison"},
            {"file": "slide12_trustworthy_AI.jpg", "caption": "Trustworthy AI framework"},
            {"file": "slide12_UQ_review.jpg", "caption": "Clinical UQ review"},
        ],
        bullets=["Backup slide for Q&A; primary claims remain locked chapter metrics."],
        notes="This backup slide is for questions on related literature. If asked about ODE baselines or uncertainty surveys, I can point here quickly without cluttering the main chapter slides.",
    ),
    slide_spec(
        kind="narrative",
        section="Implications | Architecture",
        title="Technical architecture overview",
        subtitle="Shared atlas interface, dataset roles, task boundaries",
        panels=[{"file": "slide4_four_dimension_framework.png", "caption": "Four-layer technical stack"}],
        bullets=[
            "Shared 21-region atlas interface across all chapters.",
            "Data roles: ADNI dev | AIBL external | IXI healthy | OASIS stress test.",
            "Do not cross-compare Acc across chapters with different tasks.",
        ],
        conclusion="Prototype boundaries are explicit; prospective validation remains future work.",
        notes="Before closing, I restate the engineering architecture. All chapters speak through the same anatomical interface and validation protocol, but each chapter has its own task definition. That discipline prevents misleading cross-chapter accuracy comparisons.",
    ),
    slide_spec(
        kind="narrative",
        section="Implications | Contributions",
        title="Advantages and contributions",
        subtitle="Progressive gains over conventional pipelines",
        panels=[],
        bullets=[
            "vs end-to-end CNN: modular atlas audit trails",
            "vs post-hoc Grad-CAM: named mechanism pathways (z_a / z_v)",
            "vs fixed-grid LSTM: neural ODE for irregular visits (+17.6 pp BAcc)",
            "vs point prediction: CUED-AD exposes conflict and U_total",
            "Contribution: one coherent stack, not four isolated high-score models",
        ],
        conclusion="From major clinical burden to interpretable, disentangled, dynamic, and trust-aware prototypes.",
        notes="In summary, the dissertation's value is structural. We do not claim one model solves AD. We claim a progressive, anatomy-anchored research program with honest metrics and explicit limitations—ready for post-qualification prospective validation.",
    ),
    slide_spec(
        kind="references",
        notes="Key references include Jia 2022 for China burden, Andrieu 2025 and Frisoni 2025 for the AI and ATN context, Guo 2024 for biomarker relevance, and Mårtensson 2020 for generalization concerns. Full manuscripts for all four chapters are in the project repository.",
    ),
    slide_spec(
        kind="thanks",
        notes="Thank you for your attention. I welcome questions on external validation, the MCI boundary in ARA-Net, the disentanglement trade-off in PathwayPro, AIBL transfer in A2C-NODE, or uncertainty communication in CUED-AD.",
    ),
]


def build() -> Path:
    base.ensure_images()
    tmp = OUTPUT.with_suffix(".tmp.pptx")
    shutil.copy2(TEMPLATE, tmp)
    prs = Presentation(str(tmp))
    base.trim_template(prs)
    base.set_widescreen(prs)

    page = 0
    for spec in SLIDES:
        kind = spec["kind"]
        page += 1
        if kind == "cover":
            build_cover(prs, spec["notes"])
        elif kind == "contents":
            build_contents(prs, spec["toc"], spec["notes"])
        elif kind == "narrative":
            layout_narrative(
                prs,
                spec["section"],
                spec["title"],
                spec.get("subtitle", ""),
                spec.get("panels", []),
                spec.get("bullets", []),
                spec.get("conclusion", ""),
                spec.get("bridge", ""),
                spec.get("notes", ""),
                slide_num=page,
            )
        elif kind == "chapter":
            layout_chapter(
                prs,
                spec["section"],
                spec["title"],
                spec["main_img"],
                spec["side_img"],
                spec["main_cap"],
                spec["side_cap"],
                spec["blocks"],
                spec["metrics"],
                spec["limitation"],
                spec["bridge"],
                spec["notes"],
                slide_num=page,
            )
        elif kind == "references":
            for ri, chunk in enumerate([design.REFERENCES_VANCOUVER_EN[:10], design.REFERENCES_VANCOUVER_EN[10:]]):
                if ri:
                    page += 1
                slide = base.new_slide(prs)
                design.sidebar(slide, "References", page)
                sub = "Main references (1/2)" if ri == 0 else "Main references (2/2)"
                design.header(slide, sub, "Vancouver style, aligned with chapter manuscripts", MARGIN, MAX_W)
                design.text_block(slide, MARGIN, CONTENT_TOP, MAX_W, FOOTER_TOP - CONTENT_TOP, chunk, size=10.5, bullet=False)
                if ri == 0:
                    set_speaker_notes(slide, spec["notes"])
        elif kind == "thanks":
            slide = base.new_slide(prs)
            design.sidebar(slide, "Thanks", page)
            design.textbox(slide, MARGIN, 2.7, MAX_W, 1.0, "Thank You!",
                           size=40, color=design.BLUE, bold=True, align=PP_ALIGN.CENTER)
            design.textbox(slide, MARGIN, 3.9, MAX_W, 0.6, "Questions and comments are welcome.",
                           size=18, color=design.GRAY, align=PP_ALIGN.CENTER)
            set_speaker_notes(slide, spec["notes"])

    prs.save(str(tmp))
    tmp.replace(OUTPUT)
    print(f"Saved: {OUTPUT}")
    print(f"Slides: {len(prs.slides)} (all with speaker notes)")
    return OUTPUT


if __name__ == "__main__":
    build()
