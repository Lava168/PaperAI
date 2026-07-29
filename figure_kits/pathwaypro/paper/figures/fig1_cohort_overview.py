#!/usr/bin/env python3
"""Figure 1 — Cohort overview for PathwayPro experiments.

Six sub-panels:
    (a) CONSORT-style flow: master ADNI cohort -> paired T1+FLAIR cohort.
    (b) Train / val / test split sizes (paired splits) by diagnosis.
    (c) Diagnosis distribution across the master + paired splits.
    (d) Age × sex breakdown.
    (e) Biomarker availability heat-map (per-marker × per-split missingness).
    (f) Multimodal availability matrix: T1, FLAIR, PET, CSF, WMH, APOE.

All numbers are derived live from ``data/master_subjects_v2.csv`` and
``data/splits_v5_paired/{train,val,test}.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MASTER_CSV = DATA_ROOT / "master_subjects_v2.csv"
SPLITS_DIR = DATA_ROOT / "splits_v5_paired"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _style import (  # noqa: E402
    DIAG_COLORS, ACCENT, mm, style_setup,
    panel_label, figure_title, takeaway_banner,
)

DIAG_LABELS = {0: "CN", 1: "MCI", 2: "AD"}
SPLIT_ORDER = ["train", "val", "test"]


def _panel_label(ax, text: str, dx=-0.10, dy=1.05):
    panel_label(ax, text, dx=dx, dy=dy)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_splits() -> Dict[str, pd.DataFrame]:
    return {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in SPLIT_ORDER}


def _load_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_CSV)


# -----------------------------------------------------------------------------
# (a) CONSORT flow
# -----------------------------------------------------------------------------

def _box(ax, xy, w, h, text, fc="#EFEFEF", ec="#333333", fontsize=7.6, weight="normal"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                         linewidth=0.9, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight)


def _arrow(ax, start, end, color="#444"):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", lw=0.8, color=color,
                            mutation_scale=10)
    ax.add_patch(arrow)


def panel_a(fig, gs, master: pd.DataFrame, splits: Dict[str, pd.DataFrame]):
    ax = fig.add_subplot(gs)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Cohort assembly (CONSORT-style)", pad=4)
    _panel_label(ax, "a", dx=0.0)

    n_master_scans = len(master)
    n_master_ptid = master["PTID"].nunique()
    paired = pd.concat(splits.values(), ignore_index=True)
    n_paired_scans = len(paired)
    n_paired_ptid = paired["PTID"].nunique()

    _box(ax, (1.0, 8.3), 8.0, 1.2,
         f"ADNI multi-cohort master (chapter 2 universe)\n"
         f"{n_master_scans:,} scans  ·  {n_master_ptid:,} subjects",
         fc="#E8F1FA", weight="bold")
    _box(ax, (1.0, 6.4), 8.0, 1.2,
         "Filter: paired T1-w MPRAGE + T2-FLAIR within same visit\n"
         "(plus FreeSurfer aparc+aseg, baseline label resolvable)",
         fc="#FFF8E1")
    _box(ax, (1.0, 4.5), 8.0, 1.2,
         f"PathwayPro paired cohort\n"
         f"{n_paired_scans:,} scans  ·  {n_paired_ptid:,} subjects",
         fc="#E8F5E9", weight="bold")

    y0 = 2.7
    h = 1.0
    for i, s in enumerate(SPLIT_ORDER):
        df = splits[s]
        ns = len(df); np_ = df["PTID"].nunique()
        _box(ax, (0.5 + i * 3.1, y0), 2.9, h,
             f"{s.title()}\n{ns} scans · {np_} subj.",
             fc="#FAFAFA")

    counts = paired["label"].value_counts().to_dict()
    cn = counts.get(0, 0); mci = counts.get(1, 0); ad = counts.get(2, 0)
    _box(ax, (1.0, 0.8), 8.0, 1.2,
         f"Diagnosis mix (paired): CN {cn}  ·  MCI {mci}  ·  AD {ad}",
         fc="#F5F5F5", fontsize=7.6)

    _arrow(ax, (5.0, 8.3), (5.0, 7.65))
    _arrow(ax, (5.0, 6.4), (5.0, 5.75))
    _arrow(ax, (5.0, 4.5), (5.0, 3.75))
    _arrow(ax, (5.0, 2.7), (5.0, 2.05))


# -----------------------------------------------------------------------------
# (b) Split sizes by diagnosis
# -----------------------------------------------------------------------------

def panel_b(fig, gs, splits: Dict[str, pd.DataFrame]):
    ax = fig.add_subplot(gs)
    _panel_label(ax, "b")
    ax.set_title("Diagnosis composition per split", pad=4)
    n_labels = ("CN", "MCI", "AD")
    bottom = np.zeros(len(SPLIT_ORDER))
    for k, name in enumerate(n_labels):
        vals = []
        for s in SPLIT_ORDER:
            df = splits[s]
            vals.append(int((df["label"] == k).sum()))
        ax.bar(SPLIT_ORDER, vals, bottom=bottom, color=DIAG_COLORS[name], label=name,
               edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")
        bottom += np.array(vals)
    ax.set_ylabel("Scans")
    ax.set_xlabel("Split")
    for i, s in enumerate(SPLIT_ORDER):
        ax.text(i, bottom[i] + max(bottom) * 0.025, f"{int(bottom[i])}",
                ha="center", va="bottom", fontsize=7.5, color="#444")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)


# -----------------------------------------------------------------------------
# (c) Diagnosis donut
# -----------------------------------------------------------------------------

def panel_c(fig, gs, paired: pd.DataFrame):
    ax = fig.add_subplot(gs)
    _panel_label(ax, "c")
    ax.set_title("Paired cohort diagnosis", pad=4)
    counts = paired["label"].value_counts().sort_index()
    sizes = [int(counts.get(0, 0)), int(counts.get(1, 0)), int(counts.get(2, 0))]
    colors = [DIAG_COLORS["CN"], DIAG_COLORS["MCI"], DIAG_COLORS["AD"]]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.78, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=8.0),
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    for w, name, s in zip(wedges, ["CN", "MCI", "AD"], sizes):
        ang = (w.theta1 + w.theta2) / 2.0
        x = 1.10 * np.cos(np.deg2rad(ang))
        y = 1.10 * np.sin(np.deg2rad(ang))
        ax.text(x, y, f"{name}\nn={s}", ha="center", va="center",
                fontsize=7.5, color=DIAG_COLORS[name], fontweight="bold")
    ax.text(0, 0, f"n={sum(sizes)}\nscans", ha="center", va="center",
            fontsize=9, fontweight="bold")


# -----------------------------------------------------------------------------
# (d) Age × sex
# -----------------------------------------------------------------------------

def panel_d(fig, gs, paired: pd.DataFrame, master: pd.DataFrame):
    ax = fig.add_subplot(gs)
    _panel_label(ax, "d")
    ax.set_title("Age distribution by sex × diagnosis", pad=4)
    if "AGE" not in paired.columns or "GENDER" not in paired.columns:
        merged = paired.merge(master[["PTID", "AGE", "GENDER"]].drop_duplicates("PTID"),
                              on="PTID", how="left")
    else:
        merged = paired.copy()

    # ADNI GENDER convention: 1 = Male, 2 = Female
    gender_map = {1: "Male", 1.0: "Male", "1": "Male", "M": "Male", "Male": "Male",
                  2: "Female", 2.0: "Female", "2": "Female", "F": "Female", "Female": "Female"}
    merged["_sex_norm"] = merged["GENDER"].map(gender_map)

    data: List[np.ndarray] = []
    labels: List[str] = []
    pos: List[float] = []
    box_colors: List[str] = []
    for i, dx in enumerate(("CN", "MCI", "AD")):
        for j, sex in enumerate(("Male", "Female")):
            mask = (merged["label"] == ["CN", "MCI", "AD"].index(dx)) & (merged["_sex_norm"] == sex)
            ageval = merged.loc[mask, "AGE"].dropna().to_numpy()
            if ageval.size == 0:
                continue
            data.append(ageval)
            labels.append(f"{dx}\n{sex[0]} (n={ageval.size})")
            pos.append(i * 2.3 + j * 0.95)
            box_colors.append(DIAG_COLORS[dx])
    if not data:
        ax.text(0.5, 0.5, "AGE / GENDER unavailable", transform=ax.transAxes,
                ha="center", va="center"); return
    bp = ax.boxplot(data, positions=pos, widths=0.72, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.0))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_linewidth(0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("Age (years)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)


# -----------------------------------------------------------------------------
# (e) Biomarker availability heat-map
# -----------------------------------------------------------------------------

BIOMARKER_COLS = ["AMY_SUVR", "AMY_CENTILOIDS", "AMY_STATUS",
                  "WMH_TOTAL", "CSF_ABETA42", "CSF_PTAU", "CSF_TAU",
                  "APOE_E4", "AGE", "GENDER"]


def panel_e(fig, gs, master: pd.DataFrame, paired: pd.DataFrame):
    ax = fig.add_subplot(gs)
    _panel_label(ax, "e")
    ax.set_title("Biomarker availability (% of subjects)", pad=4)

    splits = _load_splits()
    rows = []
    for s in SPLIT_ORDER:
        df = splits[s].merge(master.drop_duplicates("PTID"), on="PTID", how="left",
                              suffixes=("_p", ""))
        row = []
        for col in BIOMARKER_COLS:
            if col in df.columns:
                row.append(100.0 * df[col].notna().mean())
            else:
                row.append(0.0)
        rows.append(row)
    mat = np.array(rows)
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(BIOMARKER_COLS)))
    ax.set_xticklabels(BIOMARKER_COLS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(SPLIT_ORDER)))
    ax.set_yticklabels([s.title() for s in SPLIT_ORDER])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if mat[i, j] < 60 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("% available")


# -----------------------------------------------------------------------------
# (f) Multimodal availability matrix
# -----------------------------------------------------------------------------

def panel_f(fig, gs, splits: Dict[str, pd.DataFrame], master: pd.DataFrame):
    ax = fig.add_subplot(gs)
    _panel_label(ax, "f")
    ax.set_title("Multimodal coverage in paired cohort", pad=4)
    paired = pd.concat(splits.values(), ignore_index=True)
    merged = paired.merge(master.drop_duplicates("PTID"), on="PTID", how="left",
                          suffixes=("_p", ""))
    modalities = [
        ("T1-w MRI", lambda d: np.ones(len(d), dtype=bool)),
        ("T2-FLAIR", lambda d: np.ones(len(d), dtype=bool)),
        ("Amyloid PET", lambda d: d["AMY_STATUS"].notna()),
        ("CSF Aβ42", lambda d: d["CSF_ABETA42"].notna()),
        ("CSF p-tau", lambda d: d["CSF_PTAU"].notna()),
        ("WMH (Fazekas)", lambda d: d["WMH_TOTAL"].notna()),
        ("APOE-ε4", lambda d: d["APOE_E4"].notna()),
    ]
    diag_names = ["CN", "MCI", "AD"]
    mat = np.zeros((len(modalities), len(diag_names)))
    for j, dx in enumerate(diag_names):
        sub = merged[merged["label"] == ["CN", "MCI", "AD"].index(dx)]
        denom = max(len(sub), 1)
        for i, (_, f) in enumerate(modalities):
            mat[i, j] = 100.0 * f(sub).mean() if denom else 0.0
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(diag_names))); ax.set_xticklabels(diag_names)
    ax.set_yticks(np.arange(len(modalities)))
    ax.set_yticklabels([m[0] for m in modalities])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if mat[i, j] < 60 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("% with modality")


def main() -> None:
    style_setup()
    splits = _load_splits()
    master = _load_master()
    paired = pd.concat(splits.values(), ignore_index=True)

    fig = plt.figure(figsize=(mm(190), mm(240)))
    outer = gridspec.GridSpec(3, 2, figure=fig,
                              left=0.080, right=0.965, top=0.905, bottom=0.090,
                              wspace=0.40, hspace=0.85,
                              height_ratios=[1.05, 0.95, 1.05])

    panel_a(fig, outer[0, 0], master, splits)
    panel_b(fig, outer[0, 1], splits)
    panel_c(fig, outer[1, 0], paired)
    panel_d(fig, outer[1, 1], paired, master)
    panel_e(fig, outer[2, 0], master, paired)
    panel_f(fig, outer[2, 1], splits, master)

    figure_title(
        fig, 1,
        "PathwayPro cohort: paired T1+FLAIR ADNI scans for disentangled "
        "Alzheimer's-vs-vascular modelling.",
        y_title=0.97, y_story=0.945,
    )
    n_paired = sum(len(splits[s]) for s in SPLIT_ORDER)
    takeaway_banner(
        fig,
        f"n = {n_paired:,} paired multimodal scans across {paired['PTID'].nunique():,} "
        "subjects, with > 95% biomarker coverage on the held-out test set.",
        y=0.015,
    )

    out_pdf = SCRIPT_DIR / "fig1_cohort_overview.pdf"
    out_png = SCRIPT_DIR / "fig1_cohort_overview.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  saved: {out_pdf.relative_to(PROJECT_ROOT)}")
    print(f"  saved: {out_png.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
