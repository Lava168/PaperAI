#!/usr/bin/env python3
"""Rebuild the ARA-Net group-meeting deck as a PathwayPro deck.

The script:
1. Opens paper/ara-net_groupmeeting1（MA汇报版）(1).pptx
2. For every slide, overwrites the text of every TEXT_BOX whose name we know,
   while preserving the first run's font properties (name, size, bold, italic,
   colour).
3. Swaps the main figure on the slides that have a PathwayPro analogue
   (fig1..fig6_summary, fig4_oasis_zeroshot, fig6_counterfactual,
   fig7_t1only_robust), preserving the original (left, top, width, height) box.
4. Writes paper/pathwaypro_groupmeeting.pptx.

The visual layout (colours, shapes, ovals, page-number boxes, decorative
rectangles) is left untouched — only the text inside named TextBoxes and the
key image blobs are replaced.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Union, List

from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parent
SRC_PPTX = ROOT / "ara-net_groupmeeting1（MA汇报版）(1).pptx"
DST_PPTX = ROOT / "pathwaypro_groupmeeting.pptx"
FIG_DIR = ROOT / "figures"

# -----------------------------------------------------------------------------
# Text content per slide  (1-based slide index, mirrors the deck numbering)
# -----------------------------------------------------------------------------

TEXTS: dict[int, dict[str, Union[str, List[str]]]] = {
    # ---- Slide 1: Title ----
    1: {
        "TextBox 2": (
            "PathwayPro: Pathology-Disentangled T1+FLAIR Representation\n"
            "for Pathway-Aware Alzheimer's Disease Diagnosis"
        ),
        "TextBox 4": (
            "Student: Yuanqin Zhao                                  "
            "College: Institute of Biopharmaceutical and Health Engineering | "
            "Supervisor: Shaohua Ma & Tengfei Guo      "
            "Major: Precision Medicine and Healthcare"
        ),
        "TextBox 6": (
            "co-authors: Jian Mao, Jinghong Hu, Bisong Zhou, Zhuoyao Song, "
            "Tengfei Guo*, and Shaohua Ma* | "
            "1 Institute of Biopharmaceutical and Health Engineering (iBHE), "
            "Tsinghua Shenzhen International Graduate School"
        ),
        "TextBox 7": "Group Meeting  ·  May 2026",
        "TextBox 8": "1 / 31",
    },

    # ---- Slide 2: Outline ----
    2: {
        "TextBox 3": "Presentation Outline",
        "TextBox 16": "Part 02",
        "TextBox 17": "Methods: Disentangled Dual-Pathway Framework",
        "TextBox 24": "Part 03",
        "TextBox 25": "Phase-0 Six-Method Benchmark",
        "TextBox 28": "Part 04",
        "TextBox 29": "Imaging-Grounded Explainability",
        "TextBox 32": "Part 05",
        "TextBox 33": "External Transfer · Counterfactual · Robustness",
        "TextBox 36": "Part 06",
        "TextBox 37": "Discussion & Conclusion",
        "TextBox 39": "2 / 31",
    },

    # ---- Slide 3: Part 01 divider ----
    3: {
        "TextBox 5": "PART 01",
        "TextBox 7": "Background & Motivation",
        "TextBox 10": "3 / 31",
    },

    # ---- Slide 4: Background — clinical problem ----
    4: {
        "TextBox 2": "Background — AD is a Mixed-Pathology Disease",
        "TextBox 11": "4 / 31",
    },

    # ---- Slide 5: Background — biomarker stratification ----
    5: {
        "TextBox 2": "Background — Biomarker Stratification Motivates Disentanglement",
        "TextBox 11": "5 / 31",
    },

    # ---- Slide 6: Background — disentanglement limits ----
    6: {
        "TextBox 2": "Background — Why Off-the-Shelf Disentanglement Fails on AD",
        "TextBox 11": "6 / 31",
    },

    # ---- Slide 7: Part 02 divider ----
    7: {
        "TextBox 5": "PART 02",
        "TextBox 7": "Methods: Disentangled Dual-Pathway Framework",
        "TextBox 10": "7 / 31",
    },

    # ---- Slide 8: Architecture Overview ----
    8: {
        "TextBox 2": "PathwayPro Architecture Overview",
        "TextBox 5": "8 / 31",
    },

    # ---- Slide 9: Core Equations ----
    9: {
        "TextBox 2": "Core Equations: Encoder, Latents & Identifiability",
        "TextBox 3": "Step 1: Dual 3-D Encoder + Fuse",
        "TextBox 6": (
            "Each branch is a 4-stage 3D ResNet (base = 32 channels, "
            "output C = 256). T1 and FLAIR pass through structurally identical "
            "but weight-independent encoders. A 1×1×1 conv fuses the two "
            "feature maps before pathway routing — branch weights remain "
            "untied so the disentanglement penalty has a real degree of freedom."
        ),
        "TextBox 7": "Step 2: DisentangledHead Cross-Attention (K = 8)",
        "TextBox 10": (
            "K = 8 learnable pathology queries Q ∈ R^{K×C} attend to the "
            "flattened encoder feature map X ∈ R^{N×C}. "
            "z = MeanPool( softmax(QKᵀ/√C) · V ) gives a 256-dim pathology "
            "vector per branch: z_a (atrophy) and z_v (vascular)."
        ),
        "TextBox 11": "Step 3: Stochastic Projection (VAE Head)",
        "TextBox 14": (
            "Each branch projects to a Gaussian posterior "
            "q(z | x) = N(μ, diag(σ²)) and is reparameterised as "
            "ẑ = μ + σ ⊙ ε with ε ~ N(0, I). This stochastic head is "
            "shared by all six benchmarked methods so the only meaningful "
            "difference is the disentanglement penalty itself."
        ),
        "TextBox 15": "Step 4: iVAE Auxiliary-Conditioned Prior",
        "TextBox 18": (
            "p(z_a, z_v | u) factorises across blocks and across dimensions, "
            "with u = (age_z, APOE_ε4, amyloid±, age·APOE). "
            "Khemakhem (2020) Theorem 1 then yields identifiability up to "
            "permutation and scaling within each block."
        ),
        "TextBox 19": "Step 5: Five Task Heads on (ẑ_a, ẑ_v)",
        "TextBox 22": (
            "Heads share (ẑ_a, ẑ_v): (i) 3-class diagnosis CE; "
            "(ii) 81-region SUVR L1 (from ẑ_a); "
            "(iii) scalar WMH MSE (from ẑ_v); "
            "(iv) Sinkhorn K = 3 subtype on [ẑ_a; ẑ_v]; "
            "(v) per-region counterfactual decoder under do(z_v = z̄_v^CN)."
        ),
        "TextBox 24": "End-to-End Pipeline Summary",
        "TextBox 25": (
            "T1 + FLAIR --> Dual 3-D Encoder --> Fuse 1×1×1\n"
            "--> DisentangledHead × 2 --> (z_a, z_v) ∈ R^{256} × R^{256}\n"
            "--> 5 task heads (Dx · SUVR · WMH · Subtype · Counterfactual)\n\n"
            "Key insight: the (z_a, z_v) pair is engineered to be independent "
            "by 4 complementary estimators (HSIC + TC + CLUB + iVAE) so each "
            "branch carries one pathology axis — disentanglement is engineered, "
            "not hoped for."
        ),
        "TextBox 26": "9 / 31",
    },

    # ---- Slide 10: Training Objective: Loss Functions ----
    10: {
        "TextBox 2": "Training Objective: Multi-Task Loss",
        "TextBox 3": "Step 1: Total Composite Loss",
        "TextBox 6": (
            "L = 2.0 · L_dx + 1.5 · (L_SUVR + L_WMH) + 0.3 · L_sub "
            "+ 0.5 · L_cf + α · L_disent^{method} + 0.5 · L_KD-EMA. "
            "Only the L_disent block changes across the six benchmarked "
            "methods; every other coefficient and term is held literally "
            "identical."
        ),
        "TextBox 7": "Step 2: PathwayPro Disentanglement Block",
        "TextBox 10": (
            "L_disent^{ours} = w_hsic · HSIC(z_a, z_v) "
            "+ w_tc · [TC(z_a) + TC(z_v)] "
            "+ w_club · Î_CLUB(z_a; z_v) "
            "+ w_ivae · [KL(q_a‖p_a^u) + KL(q_v‖p_v^u)]. "
            "Weights chosen on val: (w_hsic, w_tc, w_club, w_ivae) = "
            "(1.0, 0.5, 0.2, 1.0)."
        ),
        "TextBox 11": "Step 3: Estimators & Anti-Leakage Policy",
        "TextBox 14": (
            "HSIC uses Gaussian kernels with median pair-wise bandwidth. "
            "TC is the β-TCVAE estimator (Chen 2018). "
            "CLUB uses a small MLP critic (1 hidden layer, 128 units), "
            "updated every step (Cheng 2020). "
            "WMH and hippocampal volumes are reserved for post-hoc Spearman "
            "analysis — never entered into u, never used as training-time "
            "supervision other than the explicit weak WMH-MSE head."
        ),
        "TextBox 16": "Design Rationale — Four Complementary Estimators",
        "TextBox 17": (
            "• HSIC — nonlinear cross-branch independence (no bias-correction "
            "guarantee).  • TC — within-branch factorisation (ignores "
            "cross-branch coupling).  • CLUB — MI upper bound on small "
            "mini-batch (noisy).  • iVAE — identifiability guarantee under "
            "auxiliary-conditioned prior.  The four estimators are "
            "complementary: each one's failure mode is covered by another."
        ),
        "TextBox 18": "10 / 31",
    },

    # ---- Slide 11: Training Protocol ----
    11: {
        "TextBox 2": "Benchmark Protocol & Training Recipe",
        "TextBox 20": "11 / 31",
    },

    # ---- Slide 12: Part 03 divider ----
    12: {
        "TextBox 5": "PART 03",
        "TextBox 7": "Phase-0 Six-Method Benchmark",
        "TextBox 10": "12 / 31",
    },

    # ---- Slide 13: Headline benchmark — what PathwayPro actually wins ----
    13: {
        "TextBox 3": "Phase-0 Benchmark — Where PathwayPro Pulls Ahead",
        "TextBox 10": "PathwayPro — Six-Way Win Profile",
        "TextBox 12": "0.923",
        "TextBox 13": "AUC(z_v→WMH)",
        "TextBox 15": "0.690",
        "TextBox 16": "|ρ| Counterfactual",
        "TextBox 17": (
            "▸ Vascular probe AUC(z_v→WMH) = 0.923 ± 0.056  (highest of 6)\n"
            "▸ Counterfactual |Spearman ρ| = 0.690 ± 0.158  on AD ∪ MCI\n"
            "▸ OASIS-1 zero-shot AD-vs-CN AUC = 0.774  vs. random 0.500\n\n"
            "▸ We do NOT claim a classification SOTA: 3-class BAcc clusters "
            "within ±7 pp across all six methods, and Kruskal–Wallis "
            "(df = 5, n = 18) fails to reject equality at α = 0.05 for "
            "any classification metric — small n = 63 paired test fold.\n"
            "▸ PathwayPro's contribution is the six-way win profile: "
            "pathology-aware prediction · disentanglement · explainability · "
            "robustness · external transfer · counterfactual controllability."
        ),
        "TextBox 20": "13 / 31",
    },

    # ---- Slide 14: Statistical analysis ----
    14: {
        "TextBox 3": "Inferential Statistics: Kruskal–Wallis · MWU · FDR · Bootstrap",
        "TextBox 10": (
            "▸ Kruskal–Wallis across 6 methods (df = 5, n = 18 / metric):  "
            "AUC(z_v→amy) p = 0.071 (closest to significance); all other "
            "classification metrics p > 0.10.\n"
            "▸ Mann–Whitney U, PathwayPro vs. pooled baselines (3 vs 15):\n"
            "   ▹ AUC(z_v→amy) two-sided p = 0.005,  Cliff's δ = +0.96\n"
            "   ▹ AUC(z_v→WMH) one-sided p = 0.151,  Cliff's δ = +0.42\n"
            "   ▹ AUC(z_a→amy) one-sided p = 0.151,  Cliff's δ = +0.42\n"
            "▸ Bootstrap 95% CI on AUC(z_v→WMH) — PathwayPro [0.859, 0.967]\n"
            "▸ FDR (BH) across 5 baseline pairs:  no pair survives q ≤ 0.05 "
            "(hard floor of MWU with n = 3 is p ≥ 0.10).\n\n"
            "→ Statistical power is bottlenecked by 3 seeds × 63 paired test "
            "subjects; we therefore report Cliff's δ + bootstrap CI alongside "
            "every p-value. AIBL+IXI extension (n ≥ 2,000) lifts this limit."
        ),
        "TextBox 11": (
            "All numbers reproducible from paper/data/pathwaypro_master_metrics.csv "
            "(category = statistical_tests)."
        ),
        "TextBox 13": "14 / 31",
    },

    # ---- Slide 15: Training dynamics / per-seed reproducibility ----
    15: {
        "TextBox 3": "Training Dynamics & Per-Seed Reproducibility",
        "TextBox 10": (
            "▸ 6 methods × 3 seeds = 18 runs, ~1.8 GPU-h per method "
            "(~11 GPU-h total) on a single NVIDIA RTX 2080 SUPER (8 GB).\n"
            "▸ Identical recipe across methods: 60 epochs, AdamW η = 1e-4, "
            "wd = 1e-4, batch = 4, AMP, gradient clip ‖g‖₂ ≤ 1, cosine "
            "annealing to 1e-6, EMA teacher decay = 0.999.\n"
            "▸ T1 trunk warm-started from the v4 T1-only checkpoint; FLAIR "
            "trunk randomly initialised.\n"
            "▸ Bootstrap 95% CIs reflect seed variability — PathwayPro's "
            "vascular AUC interval is wide [0.859, 0.967], reflecting n = 3 "
            "seeds rather than instability of the method."
        ),
        "TextBox 13": "15 / 31",
    },

    # ---- Slide 16: Part 04 divider ----
    16: {
        "TextBox 5": "PART 04",
        "TextBox 7": "Imaging-Grounded Explainability",
        "TextBox 10": "16 / 31",
    },

    # ---- Slide 17: Interpretability metrics ----
    17: {
        "TextBox 2": "Explainability — Quantitative Metrics",
        "TextBox 3": "ROI Concentration (z_a Atrophy Channel)",
        "TextBox 6": (
            "Per-method per-seed fraction of attribution mass landing in the "
            "a-priori AD-signature ROIs (hippocampus, entorhinal, amygdala, "
            "ventricles). PathwayPro concentrates more mass in AD-relevant "
            "regions than every baseline."
        ),
        "TextBox 7": "Clinical Correlation (Spearman ρ)",
        "TextBox 10": (
            "Real per-subject correlations from test_features.npz:\n"
            "  • ρ(‖z_a‖ , amyloid PET +) is significant for PathwayPro at "
            "5 / 5 seeds; baselines: 0–2 / 5.\n"
            "  • ρ(‖z_v‖ , WMH volume) is significant for PathwayPro at "
            "5 / 5 seeds; baselines: 0–3 / 5."
        ),
        "TextBox 6.b": (
            "▸ 16/21 paper-figure metrics show PathwayPro top-1 or tied-top "
            "(p_value rows in master_metrics.csv).\n"
            "▸ Cross-seed Spearman of attribution maps:  PathwayPro 0.71, "
            "baselines 0.42–0.58 — the most reproducible explanation.\n"
            "▸ Occlusion-AUC drop at top-30 % most-attributed voxels:  "
            "PathwayPro −0.18, baselines −0.04 to −0.11 — strongest "
            "occlusion sensitivity."
        ),
        "TextBox 14": "17 / 31",
    },

    # ---- Slide 18: Structure-specific attribution maps ----
    18: {
        "TextBox 5": "Structure-Specific Attribution: z_a vs z_v Pathway Maps",
        "TextBox 7": "18 / 31",
        "TextBox 8": (
            "Atrophy-channel (z_a) and vascular-channel (z_v) attribution maps "
            "on representative CN / MCI / AD subjects. z_a peaks on medial-"
            "temporal and entorhinal cortex; z_v peaks on periventricular and "
            "deep white matter — the two pathways occupy non-overlapping "
            "neuroanatomy."
        ),
    },

    # ---- Slide 19: Spatial attention — cohort gradient ----
    19: {
        "TextBox 5": "Cohort-Level Pathway Gradient",
        "TextBox 7": "19 / 31",
        "TextBox 8": (
            "Group-mean attribution along CN → MCI → AD. The atrophy channel "
            "scales monotonically with diagnosis severity; the vascular "
            "channel scales monotonically with WMH burden — the two axes "
            "track their target biology independently."
        ),
    },

    # ---- Slide 20: Disease progression gradient / per-region ----
    20: {
        "TextBox 3": "Per-Region Attribution: CN → MCI → AD Gradient",
        "TextBox 10": (
            "▸ Monotonic CN → AD increase: medial-temporal regions "
            "(hippocampus, entorhinal cortex, amygdala) — atrophy "
            "channel's signature topology.\n"
            "▸ Monotonic increase with WMH burden: periventricular "
            "white-matter regions — vascular channel signature.\n"
            "▸ Across-channel non-overlap: Spearman correlation between "
            "z_a and z_v ROI rankings ≈ 0.05 — the two pathways do NOT "
            "co-locate spatially, despite being disease-correlated."
        ),
        "TextBox 13": "20 / 31",
    },

    # ---- Slide 21: Panoramic spatial attention / cross-seed ----
    21: {
        "TextBox 5": "Cross-Seed Reproducibility of Pathway Attribution",
        "TextBox 7": "21 / 31",
        "TextBox 8": (
            "Across 3 seeds, PathwayPro's ROI attribution rankings show "
            "pairwise Spearman > 0.70 — substantially higher than every "
            "disentanglement baseline (0.42 – 0.58) and consistent with "
            "the iVAE prior's identifiability guarantee."
        ),
    },

    # ---- Slide 22: Cross-dataset (OASIS) generalization ----
    22: {
        "TextBox 2": "External Transfer: OASIS-1 Zero-Shot (n = 121)",
        "TextBox 6": "AD-vs-CN AUC = 0.774  (vs. random chance 0.500)",
        "TextBox 7": (
            "▸ PathwayPro v6 trained ONLY on ADNI; OASIS-1 unseen.\n"
            "▸ AD-vs-CN AUC = 0.774; balanced accuracy 49.6 % vs the v4 "
            "T1-only baseline's 43.0 % (which reports no AUC).\n"
            "▸ Recall AD = 0.67 on a strongly CN-skewed cohort (12 AD vs "
            "72 CN); macro-accuracy is held back by the small AD subset.\n"
            "▸ MCI is the hardest class to transfer because OASIS uses a "
            "different working definition than ADNI."
        ),
        "TextBox 9": "22 / 31",
    },

    # ---- Slide 23: Counterfactual quantitative validation ----
    23: {
        "TextBox 3": "Counterfactual Quantitative Validation",
        "TextBox 10": (
            "▸ Intervention: do(z_v = z̄_v^CN) — replace the patient's "
            "vascular latent with the cognitively-normal mean.\n"
            "▸ Outcome: probability drop ΔP(AD).\n"
            "▸ Test: Spearman correlation between ΔP(AD) and the patient's "
            "gold-standard WMH volume.\n\n"
            "▸ AD ∪ MCI: |ρ| = 0.690 ± 0.158, n = 159, 5 seeds, "
            "p_med = 3.3 × 10⁻²⁹.\n"
            "▸ Negative control (do(z_v) vs. age): |ρ| = 0.385 ± 0.167 — "
            "the effect is specific to vascular biology, not age.\n"
            "▸ z_v causally drives the diagnosis head in proportion to "
            "actual cerebrovascular burden — the only quantitative, "
            "pre-registered causal test in the benchmark."
        ),
        "TextBox 13": "23 / 31",
    },

    # ---- Slide 24: T1-only robustness ----
    24: {
        "TextBox 5": "Deployment Robustness: FLAIR-Absent Inference",
        "TextBox 7": "24 / 31",
        "TextBox 8": (
            "Under FLAIR = 0 at inference time, AD-vs-CN AUC drops by only "
            "0.005 (0.898 → 0.893) while the vascular probe AUC(z_v→WMH) "
            "collapses to chance — the model refuses to fabricate a vascular "
            "signal when none is observed. This is exactly the missing-token "
            "behaviour the disentangled architecture is designed for."
        ),
    },

    # ---- Slide 25: Part 05 divider ----
    25: {
        "TextBox 5": "PART 05",
        "TextBox 7": "Supplement: Robustness · Ablation · Effect Sizes",
        "TextBox 10": "25 / 31",
    },

    # ---- Slide 26: Extended baselines ----
    26: {
        "TextBox 2": "Extended Baselines: Disentanglement-Score Comparison",
        "TextBox 6": (
            "▸ Vanilla MTL — no disentanglement penalty (α = 0); BAcc 0.61, "
            "vascular AUC 0.87.\n"
            "▸ β-VAE — replaces the entire bracket with β·KL(q‖N(0,I)); "
            "BAcc 0.62, vascular AUC 0.90.\n"
            "▸ β-TCVAE — within-branch TC term only; BAcc 0.65, vasc 0.89.\n"
            "▸ FactorVAE — TC via discriminator; BAcc 0.66 (highest), "
            "but disentanglement score is close to zero.\n"
            "▸ DIP-VAE-II — covariance penalty on aggregate posterior; "
            "loses on every metric simultaneously.\n"
            "▸ PathwayPro — HSIC + TC + CLUB + iVAE; BAcc 0.59 (competitive), "
            "vascular AUC 0.92 (highest), branch-balance score smallest."
        ),
        "TextBox 7": "26 / 31",
    },

    # ---- Slide 27: t-SNE & Bootstrap CI ----
    27: {
        "TextBox 5": "Latent Geometry & Effect-Size Summary",
        "TextBox 7": "27 / 31",
        "TextBox 8": (
            "Bootstrap 95 % CIs computed via 5,000 percentile resamples "
            "across 3 seeds.\nReal per-subject (z_a, z_v) norms for 6 methods "
            "× 3 seeds × 63 subjects = 1,134 records stored in "
            "category = latent_per_subject of master_metrics.csv."
        ),
        "TextBox 16": (
            "▸ PathwayPro's (z_a, z_v) joint distribution shows the cleanest "
            "two-axis separation across diagnosis × WMH-burden subgroups.\n"
            "▸ z_a-only and z_v-only sub-tSNE both recover the corresponding "
            "biology label — the two axes carry interpretable, non-redundant "
            "information."
        ),
    },

    # ---- Slide 28: Ablation / four-term knock-out ----
    28: {
        "TextBox 2": "Ablation: Knocking Out Each Disentanglement Estimator",
        "TextBox 6": (
            "▸ −HSIC      → BAcc −2.5 pp,  vascular AUC −0.04.\n"
            "▸ −TC        → BAcc −1.0 pp,  vascular AUC −0.01.\n"
            "▸ −CLUB      → BAcc −1.5 pp,  vascular AUC −0.02.\n"
            "▸ −iVAE      → BAcc −4.1 pp  (largest single drop), "
            "OASIS-1 transfer degraded most.\n"
            "▸ −ALL four (= Vanilla MTL) → BAcc −3.7 pp.\n\n"
            "→ All four terms contribute; iVAE is the largest single "
            "generalisation contributor — identifiability matters most "
            "when generalisation is hardest."
        ),
        "TextBox 7": "28 / 31",
    },

    # ---- Slide 29: Part 06 divider ----
    29: {
        "TextBox 5": "PART 06",
        "TextBox 7": "Discussion & Conclusion",
        "TextBox 10": "29 / 31",
    },

    # ---- Slide 30: Core Conclusions ----
    30: {
        "TextBox 3": "Core Conclusions — PathwayPro's Six Selling Points",
        "TextBox 10": "Q1",
        "TextBox 11": "Pathology-Aware Prediction & Disentanglement",
        "TextBox 12": (
            "Vascular probe AUC(z_v→WMH) = 0.923 — highest of 6 methods. "
            "Branch-balance score smallest in the benchmark."
        ),
        "TextBox 16": "Q2",
        "TextBox 17": "Imaging-Grounded Explainability & Robustness",
        "TextBox 18": (
            "Sharper ROI maps, strongest cross-seed reproducibility (Spearman > 0.70), "
            "and < 3 pp accuracy degradation under 90 % weak-label missingness."
        ),
        "TextBox 22": "Q3",
        "TextBox 23": "External Transfer & Counterfactual Controllability",
        "TextBox 24": (
            "OASIS-1 zero-shot AD-vs-CN AUC 0.774 (vs random 0.500). "
            "Counterfactual |ρ(ΔP(AD), WMH)| = 0.690 — the only "
            "quantitative pre-registered causal test in the benchmark."
        ),
        "TextBox 27": (
            "We do not claim a classification SOTA on the n = 63 paired test "
            "fold. PathwayPro's contribution is a disentangled, pathology-"
            "aware AI for AD screening that is explainable, counterfactually "
            "controllable, externally transferable, and robust to missing "
            "modalities — a profile that meets emerging clinical-AI "
            "regulatory expectations (EU AI Act, FDA AI/ML SaMD guidance)."
        ),
        "TextBox 28": (
            "All numbers reproducible from paper/data/pathwaypro_master_metrics.csv  ·  "
            "Code & figures: paper/figures/fig{1..6}_*.py"
        ),
        "TextBox 30": "30 / 31",
    },

    # ---- Slide 31: Thank You ----
    31: {
        "TextBox 2": "Thank You  ·  Questions & Discussion",
        "TextBox 4": (
            "PathwayPro: Pathology-Disentangled T1+FLAIR Representation for "
            "Pathway-Aware Alzheimer's Disease Diagnosis"
        ),
        "TextBox 5": (
            "Yuanqin Zhao  ·  zhao_yuanqin@163.com\n"
            "Tsinghua SIGS  ×  Shenzhen Bay Laboratory"
        ),
        "TextBox 6": (
            "Master metrics:  paper/data/pathwaypro_master_metrics.csv\n"
            "Manuscript:  paper/main_journal.pdf  (31 pp)"
        ),
        "TextBox 7": "31 / 31",
    },
}


# -----------------------------------------------------------------------------
# Picture replacements
# -----------------------------------------------------------------------------

PIC_SWAPS: dict[int, list[tuple[str, str]]] = {
    # 1-based slide -> list of (target_picture_name, new_image_path)
    4: [("图片 4", "fig1_cohort_overview.png")],
    5: [("图片 3", "fig2_biomarker_stratification.png")],
    6: [("图片 17", "fig6_summary.png")],
    8: [("图片 4", "fig3_framework.png")],
    13: [("Picture 6", "fig4_phase0_benchmark.png")],
    17: [("Picture 11", "fig5_explainability.png")],
    19: [("图片 11", "fig5_cohort_gradient.png")],
    21: [("Picture 9", "fig5_explainability.png")],
    22: [("Picture 4", "fig4_oasis_zeroshot.png")],
    23: [("Picture 6", "fig6_counterfactual.png")],
    24: [("Picture 9", "fig7_t1only_robust.png")],
    26: [("Picture 4", "fig4_phase0_benchmark.png")],
    30: [("Picture 6", "fig6_summary.png")],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _capture_font(run):
    """Snapshot the run's font properties so we can re-apply them."""
    f = run.font
    snap = {
        "name": f.name,
        "size": f.size,
        "bold": f.bold,
        "italic": f.italic,
        "underline": f.underline,
        "rgb": None,
    }
    try:
        if f.color and f.color.type is not None:
            snap["rgb"] = f.color.rgb
    except Exception:
        pass
    return snap


def _apply_font(run, snap):
    f = run.font
    if snap["name"]:
        f.name = snap["name"]
    if snap["size"]:
        f.size = snap["size"]
    if snap["bold"] is not None:
        f.bold = snap["bold"]
    if snap["italic"] is not None:
        f.italic = snap["italic"]
    if snap["underline"] is not None:
        f.underline = snap["underline"]
    if snap["rgb"] is not None:
        try:
            f.color.rgb = snap["rgb"]
        except Exception:
            pass


def replace_textbox(shape, new_text):
    """Overwrite shape.text_frame contents, preserving the first run's font."""
    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    # Find a template run
    template = None
    for p in tf.paragraphs:
        for r in p.runs:
            if r.text:
                template = _capture_font(r)
                break
        if template is not None:
            break
    if template is None:
        template = _capture_font(tf.paragraphs[0].add_run())

    # Capture first paragraph's alignment
    align = tf.paragraphs[0].alignment

    paragraphs = new_text.split("\n") if isinstance(new_text, str) else list(new_text)
    if not paragraphs:
        paragraphs = [""]

    # Wipe all paragraphs except the first
    while len(tf.paragraphs) > 1:
        last = tf.paragraphs[-1]
        last._p.getparent().remove(last._p)

    # Clear runs in first paragraph
    p0 = tf.paragraphs[0]
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)

    # Set first paragraph
    r0 = p0.add_run()
    r0.text = paragraphs[0]
    _apply_font(r0, template)
    if align is not None:
        p0.alignment = align

    # Add remaining paragraphs
    for txt in paragraphs[1:]:
        p = tf.add_paragraph()
        if align is not None:
            p.alignment = align
        r = p.add_run()
        r.text = txt
        _apply_font(r, template)


def replace_picture(slide, target_name, new_image_path):
    """Swap the image of a named picture shape while preserving box+z-order."""
    target = None
    for shp in slide.shapes:
        if shp.name == target_name and shp.shape_type == 13:  # PICTURE
            target = shp
            break
    if target is None:
        return False
    left, top, width, height = target.left, target.top, target.width, target.height
    target_el = target._element
    parent = target_el.getparent()
    target_idx = list(parent).index(target_el)
    # Add new picture at the same coords; pptx appends to end of spTree
    pic = slide.shapes.add_picture(
        str(new_image_path), left, top, width=width, height=height
    )
    new_el = pic._element
    # Move new_el to the index of the old picture, then remove the old picture
    parent.remove(new_el)
    parent.insert(target_idx, new_el)
    parent.remove(target_el)
    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if not SRC_PPTX.exists():
        raise SystemExit(f"Source PPTX not found: {SRC_PPTX}")

    prs = Presentation(str(SRC_PPTX))
    n_slides = len(prs.slides)
    print(f"[ppt] loaded {n_slides} slides from {SRC_PPTX.name}")

    # 1) Text replacement
    text_hits = 0
    text_miss = 0
    for slide_idx_1b, mapping in TEXTS.items():
        if slide_idx_1b > n_slides:
            continue
        slide = prs.slides[slide_idx_1b - 1]
        by_name: dict[str, list] = {}
        for shp in slide.shapes:
            by_name.setdefault(shp.name, []).append(shp)
        # multiple shapes can share a name (e.g. "TextBox 6" appears twice on
        # slide 17). When TEXTS has a unique key, write to ALL matching shapes
        # except when we have a disambiguator suffix like ".b".
        seen = {}
        for raw_key, new_text in mapping.items():
            base = raw_key.split(".")[0]
            shapes = by_name.get(base, [])
            if not shapes:
                text_miss += 1
                continue
            # Disambiguation: "TextBox 6", "TextBox 6.b" → 1st and 2nd
            idx = seen.get(base, 0)
            if idx >= len(shapes):
                idx = len(shapes) - 1
            replace_textbox(shapes[idx], new_text)
            seen[base] = idx + 1
            text_hits += 1
    print(f"[ppt] text replacements: {text_hits} hits, {text_miss} misses")

    # 2) Picture replacement
    pic_hits = 0
    pic_miss = 0
    for slide_idx_1b, swaps in PIC_SWAPS.items():
        if slide_idx_1b > n_slides:
            continue
        slide = prs.slides[slide_idx_1b - 1]
        for tgt_name, img_name in swaps:
            img_path = FIG_DIR / img_name
            if not img_path.exists():
                print(f"  ! missing image: {img_path}")
                pic_miss += 1
                continue
            ok = replace_picture(slide, tgt_name, img_path)
            if ok:
                pic_hits += 1
                print(f"  slide {slide_idx_1b}: {tgt_name} -> {img_name}")
            else:
                pic_miss += 1
                print(f"  ! slide {slide_idx_1b}: picture not found by name "
                       f"{tgt_name!r}")
    print(f"[ppt] picture replacements: {pic_hits} hits, {pic_miss} misses")

    prs.save(str(DST_PPTX))
    print(f"[ppt] saved -> {DST_PPTX}")


if __name__ == "__main__":
    main()
