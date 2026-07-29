#!/usr/bin/env python3
"""Figure 3 — PathwayPro framework diagram.

A schematic, publication-quality model overview:
    (a) End-to-end architecture: T1 + T2-FLAIR -> dual SharedEncoder -> fuse
        1×1×1 -> DisentangledHead (atrophy) and DisentangledHead (vascular)
        -> task heads (diagnosis, amyloid SUVR, WMH, counterfactual, subtype).
    (b) Loss block: schematic of disentanglement objectives that act on the
        pair (z_a, z_v) — TC (β-TCVAE), HSIC, CLUB, iVAE, pathway-contrastive,
        counterfactual — with PathwayPro's combination highlighted.
    (c) Information-flow callouts: "shared encoder = anatomy", "z_a captures
        amyloid + AD biomarkers", "z_v captures WMH burden", "low MI between
        z_a and z_v by construction".

The diagram is purely schematic and uses no data, so it always renders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _style import (  # noqa: E402
    METHOD_COLORS, PATHWAY, ACCENT, mm, style_setup,
    panel_label, figure_title, takeaway_banner,
)


# -----------------------------------------------------------------------------
# Drawing primitives
# -----------------------------------------------------------------------------

def _box(ax, xy, w, h, text, *, fc="#EFEFEF", ec="#333333", fontsize=7.8,
          weight="normal", text_color="black", rounding=0.04):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color)


def _arrow(ax, start, end, *, color="#444444", width=1.0, style="->", curved=False,
            label=None, label_offset=(0.06, 0.05), label_fontsize=6.7,
            label_color="#444444"):
    cs = "arc3,rad=0.20" if curved else "arc3,rad=0.0"
    arrow = FancyArrowPatch(start, end, arrowstyle=style, lw=width,
                            color=color, mutation_scale=11, connectionstyle=cs)
    ax.add_patch(arrow)
    if label is not None:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_fontsize, color=label_color,
                ha="left", va="center", style="italic")


# -----------------------------------------------------------------------------
# (a) End-to-end architecture
# -----------------------------------------------------------------------------

def panel_a(fig, gs):
    ax = fig.add_subplot(gs)
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.0)
    ax.axis("off")
    panel_label(ax, "a", dx=0.0, dy=0.99)
    ax.set_title("PathwayPro architecture", pad=6)

    # Column headers
    headers = [
        ("Inputs",                0.2),
        ("Dual 3-D encoder",      2.7),
        ("Fuse",                  5.5),
        ("Disentangled latents",  7.6),
        ("Pathology-aware heads", 10.5),
    ]
    for text, x in headers:
        ax.text(x, 8.45, text, fontsize=7.6, fontweight="bold", color="#444",
                ha="left")

    # Inputs column
    _box(ax, (0.2, 6.6), 2.2, 1.2, "T1-w MRI\n(B, 1, 128³)",
         fc="#E8F1FA", weight="bold", fontsize=7.4)
    _box(ax, (0.2, 4.9), 2.2, 1.2, "T2-FLAIR\n(B, 1, 128³)\n(optional)",
         fc="#FBEFE5", weight="bold", fontsize=7.2)
    _box(ax, (0.2, 3.2), 2.2, 1.2, "DKT segmentation\n(81 regions)",
         fc="#F5F5F5", fontsize=7.2)

    # Encoder column
    _box(ax, (2.7, 6.6), 2.4, 1.2, "Shared encoder\n(T1)",
         fc="#FFF8E1", weight="bold", fontsize=7.4)
    _box(ax, (2.7, 4.9), 2.4, 1.2, "Shared encoder\n(FLAIR)",
         fc="#FFF8E1", weight="bold", fontsize=7.4)

    # Fuse
    _box(ax, (5.5, 5.7), 1.7, 1.4, "Fuse\n1×1×1 Conv",
         fc=PATHWAY["shared"], ec="#B5870F", weight="bold", fontsize=7.4)

    # Two pathway latents
    _box(ax, (7.6, 7.0), 2.5, 1.2,
         r"$z_a$  atrophy" + "\nDisentangledHead\n(K = 8 queries)",
         fc=PATHWAY["atrophy"], ec=PATHWAY["atrophy"],
         text_color="white", weight="bold", fontsize=7.2)
    _box(ax, (7.6, 4.6), 2.5, 1.2,
         r"$z_v$  vascular" + "\nDisentangledHead\n(K = 8 queries)",
         fc=PATHWAY["vascular"], ec=PATHWAY["vascular"],
         text_color="white", weight="bold", fontsize=7.2)

    # Heads column (right)
    _box(ax, (10.5, 7.55), 3.4, 0.7, "Diagnosis (CN / MCI / AD)", fc="#FFEBE0", fontsize=7.2)
    _box(ax, (10.5, 6.55), 3.4, 0.7, "Amyloid SUVR (81 ROI)",   fc="#F1E4F5", fontsize=7.2)
    _box(ax, (10.5, 5.55), 3.4, 0.7, "WMH volume",              fc="#E5F4E8", fontsize=7.2)
    _box(ax, (10.5, 4.55), 3.4, 0.7, "Subtype assignment (K=3)", fc="#E0E0FF", fontsize=7.2)
    _box(ax, (10.5, 3.55), 3.4, 0.7,
         "Counterfactual decoder\n→ AD vs vascular share", fc="#FFF3D6", fontsize=6.9)

    # Arrows: inputs → encoders
    _arrow(ax, (2.40, 7.2), (2.7, 7.2))
    _arrow(ax, (2.40, 5.5), (2.7, 5.5))
    # encoders → fuse
    _arrow(ax, (5.10, 7.2), (5.5, 6.7))
    _arrow(ax, (5.10, 5.5), (5.5, 6.2))
    # fuse → z_a, z_v
    _arrow(ax, (7.20, 6.7), (7.6, 7.6), color=PATHWAY["atrophy"], width=1.2)
    _arrow(ax, (7.20, 6.0), (7.6, 5.2), color=PATHWAY["vascular"], width=1.2)
    # z_a → heads
    for y in (7.90, 6.90, 4.90):
        _arrow(ax, (10.10, 7.6), (10.5, y), color=PATHWAY["atrophy"])
    # z_v → heads
    for y in (5.90, 3.90):
        _arrow(ax, (10.10, 5.2), (10.5, y), color=PATHWAY["vascular"])
    # diagnosis also gets z_v
    _arrow(ax, (10.10, 5.2), (10.5, 7.90), color=PATHWAY["vascular"], curved=True)
    # seg → counterfactual
    _arrow(ax, (2.40, 3.80), (10.5, 3.85), color="#999")
    ax.text(6.4, 3.55, "per-region volume target",
            fontsize=6.6, color="#888", ha="center", style="italic")

    # MI minimisation between z_a and z_v
    ax.annotate("", xy=(8.85, 7.0), xytext=(8.85, 5.80),
                arrowprops=dict(arrowstyle="<->", lw=1.0,
                                 color=ACCENT["win"]))
    _box(ax, (5.85, 2.2), 4.6, 1.0,
         r"$\min\, \mathrm{MI}(z_a, z_v)$  via " +
         "HSIC · TC · CLUB · iVAE · pathway-contrastive",
         fc=ACCENT["hi"], ec=ACCENT["win"], fontsize=7.0, weight="bold")
    _arrow(ax, (8.15, 3.2), (8.6, 4.6), color=ACCENT["win"], width=0.8)


# -----------------------------------------------------------------------------
# (b) Loss block schematic
# -----------------------------------------------------------------------------

LOSS_TERMS = [
    ("Diagnosis CE",     "Classification of CN/MCI/AD with class weights.",        "#FFEBE0"),
    ("Amyloid SUVR L1",  "Region-wise PET SUVR regression from $z_a$.",            "#F1E4F5"),
    ("WMH MSE",          "Total WMH volume regression from $z_v$.",                "#E5F4E8"),
    ("β-TCVAE TC",       "Total-correlation penalty on $z_a$, $z_v$.",             "#FFF8E1"),
    ("HSIC",             "Kernel cross-pathway independence.",                     "#FFF8E1"),
    ("CLUB (MI ↓)",      "Variational upper bound on $\\mathrm{MI}(z_a, z_v)$.",   "#FFF8E1"),
    ("iVAE prior",       "Auxiliary-conditioned latent prior on age / APOE.",      "#FFF8E1"),
    ("Pathway contrast", "Triplet contrast on amyloid+/- and high/low WMH.",       "#FFF8E1"),
    ("Counterfactual",   "$z_a$-only and $z_v$-only volume decoder.",              "#FFF3D6"),
    ("Subtype KL",       "Sinkhorn soft-cluster prior on $K{=}3$ subtypes.",       "#E0E0FF"),
]


def panel_b(fig, gs):
    ax = fig.add_subplot(gs)
    ax.set_xlim(0, 12); ax.set_ylim(0, 12)
    ax.axis("off")
    panel_label(ax, "b", dx=0.0, dy=0.99)
    ax.set_title("Multi-objective loss block", pad=6)

    n = len(LOSS_TERMS)
    h = 0.80
    gap = 0.22
    y0 = 11.0 - h
    for i, (name, desc, fc) in enumerate(LOSS_TERMS):
        y = y0 - i * (h + gap)
        _box(ax, (0.4, y), 3.6, h, name, fc=fc, fontsize=7.0, weight="bold")
        ax.text(4.2, y + h / 2, desc, fontsize=6.8, ha="left", va="center")

    ax.text(6.0, 0.55,
            "Only PathwayPro uses all 10 objectives simultaneously.",
            fontsize=7.6, color=METHOD_COLORS["ours"], fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=ACCENT["hi"],
                      edgecolor=METHOD_COLORS["ours"], linewidth=0.8))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    style_setup()

    fig = plt.figure(figsize=(mm(195), mm(185)))
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              left=0.040, right=0.985, top=0.875, bottom=0.090,
                              wspace=0.12,
                              width_ratios=[2.6, 1.0])

    panel_a(fig, outer[0, 0])
    panel_b(fig, outer[0, 1])

    figure_title(
        fig, 3,
        "PathwayPro framework: a shared 3-D encoder routes into atrophy "
        "and vascular pathways.",
        y_title=0.955, y_story=0.925,
    )
    takeaway_banner(
        fig,
        "Pathway-specific latents are constrained to be independent and to "
        "predict their own biomarkers — disentanglement is engineered, not hoped for.",
        y=0.013,
    )

    out_pdf = SCRIPT_DIR / "fig3_framework.pdf"
    out_png = SCRIPT_DIR / "fig3_framework.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  saved: {out_pdf.relative_to(PROJECT_ROOT)}")
    print(f"  saved: {out_png.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
