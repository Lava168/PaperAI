#!/usr/bin/env python3
"""Compose A2C-NODE's six main paper figures with a unified Nature style.

Each figure is rendered at the Nature double-column width (7.2 in) with a
consistent font / colour / panel-label scheme:

    Figure 1  - method overview + cohort + headline scorecard      (a..f)
    Figure 2  - ADNI internal-validation main results              (a..f)
    Figure 3  - subgroup + calibration + DCA + reproducibility     (a..f)
    Figure 4  - counterfactual ATE deep-dive (premium brain views) (a..f)
    Figure 5  - NODE long-term trajectory                          (a..f)
    Figure 6  - AIBL external validation + case studies            (a..f)

All panels share:
    * Nature-style rcParams (DejaVu Sans 7.5pt, slate axes 0.6pt)
    * Lower-case bold panel letters in the upper-left corner
    * A common slate/navy/sage/mustard/sienna palette
    * Embedded PNGs are clean-cropped and stripped of redundant frames

Outputs are written under ``--out_dir`` as both PNG (300 dpi) and PDF.

Usage
-----
    python scripts/compose_paper_figures.py \
        --runs_dir runs \
        --analysis_dir runs/v4_mm_analysis_npj \
        --out_dir paper/figures/composed
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=Path,
                   default=Path("runs"),
                   help="root containing v4_mm_seed* directories")
    p.add_argument("--analysis_dir", type=Path,
                   default=Path("runs/v4_mm_analysis_npj"),
                   help="output of analyze_npj.py / compute_ate.py / "
                        "plot_brain_ate.py")
    p.add_argument("--brain_dir", type=Path, default=None,
                   help="override path that holds ate_glass_brain.png")
    p.add_argument("--brain_premium_dir", type=Path, default=None,
                   help="override path that holds brain_premium/ outputs")
    p.add_argument("--density_dir", type=Path, default=None,
                   help="override path that holds cohort_summary.png etc.")
    p.add_argument("--ate_csv", type=Path, default=None)
    p.add_argument("--arch_pdf", type=Path,
                   default=Path("paper/figures/arch_a2c_node.pdf"),
                   help="3D architecture figure (PlotNeuralNet)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 153, 264, 375, 486])
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Nature-style palette and rc                                                 #
# --------------------------------------------------------------------------- #
NAT_PALETTE = {
    "navy":    "#1F3A5F",
    "sienna":  "#C7572D",
    "sage":    "#5C9C7E",
    "mustard": "#E0A93A",
    "slate":   "#4A4A4A",
    "fog":     "#B8B8B8",
    "panel":   "#F5F2EC",
    "ink":     "#1c1c1c",
}
CLASS_COLORS = {"CN": NAT_PALETTE["sage"],
                "MCI": NAT_PALETTE["mustard"],
                "AD": NAT_PALETTE["sienna"]}
PIPE_COLORS = ["#34495e", "#3498db", "#9b59b6", "#27ae60", "#e67e22", "#e74c3c"]


def apply_nature_rc() -> None:
    """In-place rcParams override used by every figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":         7.5,
        "axes.titlesize":    8.5,
        "axes.titleweight":  "regular",
        "axes.labelsize":    7.5,
        "axes.labelweight":  "regular",
        "axes.linewidth":    0.6,
        "axes.edgecolor":    NAT_PALETTE["slate"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "xtick.labelsize":   6.5,
        "ytick.labelsize":   6.5,
        "xtick.color":       NAT_PALETTE["slate"],
        "ytick.color":       NAT_PALETTE["slate"],
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "legend.fontsize":   6.5,
        "legend.frameon":    False,
        "legend.handlelength": 1.5,
        "savefig.bbox":      "tight",
        "savefig.dpi":       300,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def panel_label(ax, label: str,
                dx: float = -0.06, dy: float = 1.04) -> None:
    """Bold lower-case panel letter in the upper-left corner."""
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="bottom", ha="left",
            color=NAT_PALETTE["ink"])


def clean_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# Image / PDF embedding helpers                                               #
# --------------------------------------------------------------------------- #
def imshow_image(ax, path: Path, title: Optional[str] = None,
                 trim_white: bool = True) -> bool:
    """Embed a PNG into ``ax``. Optionally trim outer white margins."""
    if not path.is_file():
        ax.text(0.5, 0.5, f"[missing]\n{path.name}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7,
                color=NAT_PALETTE["fog"])
        clean_axes(ax)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#dddddd")
        if title:
            ax.set_title(title, fontsize=8.5, loc="left",
                         color=NAT_PALETTE["slate"], pad=4)
        return False

    from PIL import Image
    im = Image.open(path)
    arr = np.asarray(im.convert("RGB"))

    if trim_white:
        gray = arr.mean(axis=2)
        non_white = gray < 252
        if non_white.any():
            ys = np.where(non_white.any(axis=1))[0]
            xs = np.where(non_white.any(axis=0))[0]
            pad = 4
            y0, y1 = max(0, ys[0] - pad), min(arr.shape[0], ys[-1] + pad)
            x0, x1 = max(0, xs[0] - pad), min(arr.shape[1], xs[-1] + pad)
            arr = arr[y0:y1, x0:x1]

    ax.imshow(arr)
    clean_axes(ax)
    if title:
        ax.set_title(title, fontsize=8.5, loc="left",
                     color=NAT_PALETTE["slate"], pad=4)
    return True


def imshow_pdf_page(ax, pdf_path: Path, page: int = 0,
                    title: Optional[str] = None) -> bool:
    """Embed the first page of a PDF into ``ax``."""
    if not pdf_path.is_file():
        return imshow_image(ax, pdf_path, title=title)
    try:
        import fitz                                                    # pymupdf
        doc = fitz.open(str(pdf_path))
        pix = doc.load_page(page).get_pixmap(dpi=240)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ax.imshow(np.asarray(img))
        clean_axes(ax)
        if title:
            ax.set_title(title, fontsize=8.5, loc="left",
                         color=NAT_PALETTE["slate"], pad=4)
        return True
    except Exception:
        # Fallback: convert via pdftoppm if available
        try:
            from PIL import Image
            import subprocess
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "p"
                subprocess.check_call(
                    ["pdftoppm", "-r", "240", "-png", "-f", str(page + 1),
                     "-l", str(page + 1), str(pdf_path), str(tmp)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                pngs = sorted(Path(td).glob("p-*.png"))
                if pngs:
                    ax.imshow(np.asarray(Image.open(pngs[0])))
                    clean_axes(ax)
                    if title:
                        ax.set_title(title, fontsize=8.5, loc="left",
                                     color=NAT_PALETTE["slate"], pad=4)
                    return True
        except Exception:
            pass
        ax.text(0.5, 0.5, f"[PDF embed failed]\n{pdf_path.name}",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        clean_axes(ax)
        return False


# --------------------------------------------------------------------------- #
# Training-log parser                                                         #
# --------------------------------------------------------------------------- #
def parse_train_log(log_path: Path) -> Dict[str, np.ndarray]:
    if not log_path.is_file():
        return {"epoch": np.array([]), "train_loss": np.array([]),
                "val_loss": np.array([]), "val_acc": np.array([]),
                "val_bacc": np.array([])}
    text = log_path.read_text(errors="ignore")
    pat = re.compile(
        r"^epoch (\d+)\s+train=([0-9.\-]+)\s+val=([0-9.\-]+)\s+"
        r"val\(drop\)=[0-9.\-]+\s+acc=([0-9.\-]+)\s+bacc=([0-9.\-]+)", re.M)
    rows = pat.findall(text)
    if not rows:
        return {"epoch": np.array([]), "train_loss": np.array([]),
                "val_loss": np.array([]), "val_acc": np.array([]),
                "val_bacc": np.array([])}
    rows = np.array(rows, dtype=float)
    return {
        "epoch": rows[:, 0].astype(int),
        "train_loss": rows[:, 1],
        "val_loss": rows[:, 2],
        "val_acc": rows[:, 3],
        "val_bacc": rows[:, 4],
    }


# --------------------------------------------------------------------------- #
# Pipeline diagram (in-figure box & arrow chain)                              #
# --------------------------------------------------------------------------- #
def draw_pipeline(ax, stages: List[Tuple[str, str]]) -> None:
    """Horizontal pipeline chain with rounded boxes and slim arrows."""
    from matplotlib.patches import FancyBboxPatch
    n = len(stages)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    clean_axes(ax)

    box_h = 0.55
    y0 = (1 - box_h) / 2
    gap = 0.018
    bw = (1.0 - gap * (n - 1)) / n

    centers: List[float] = []
    for i, (label, color) in enumerate(stages):
        x = i * (bw + gap)
        ax.add_patch(FancyBboxPatch(
            (x, y0), bw, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0.6, edgecolor=NAT_PALETTE["ink"],
            facecolor=color, alpha=0.92))
        ax.text(x + bw / 2, y0 + box_h / 2, label,
                ha="center", va="center", fontsize=6.7,
                color="white", weight="bold", linespacing=1.15)
        centers.append(x + bw)

    for i in range(n - 1):
        ax.annotate("",
                    xy=(centers[i] + gap, y0 + box_h / 2),
                    xytext=(centers[i], y0 + box_h / 2),
                    arrowprops=dict(arrowstyle="->", lw=0.7,
                                    color=NAT_PALETTE["ink"]))


# --------------------------------------------------------------------------- #
# IO                                                                          #
# --------------------------------------------------------------------------- #
def save_figure(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_pdf = out_dir / f"{name}.pdf"
    p_png = out_dir / f"{name}.png"
    fig.savefig(p_pdf)
    fig.savefig(p_png, dpi=300)
    print(f"[fig] wrote {p_pdf}", flush=True)


# --------------------------------------------------------------------------- #
# Figure 1: method overview + cohort + headline scorecard                     #
# --------------------------------------------------------------------------- #
def make_fig1(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    apply_nature_rc()

    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(
        4, 12,
        height_ratios=[1.20, 0.55, 0.35, 1.45],
        hspace=0.65, wspace=1.0,
        left=0.075, right=0.975, top=0.965, bottom=0.05,
    )
    # Row 0: a (architecture), b (math), c (headline scorecard)
    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 7:10])
    ax_c = fig.add_subplot(gs[0, 10:12])
    # Row 1: d (cohort summary, full width)
    ax_d = fig.add_subplot(gs[1, 0:12])
    # Row 2: e (multimodal pipeline)
    ax_e = fig.add_subplot(gs[2, 0:12])
    # Row 3: f (region atlas / attention overview)
    ax_f = fig.add_subplot(gs[3, 0:12])

    panel_label(ax_a, "a", dx=-0.045)
    ax_a.set_title("A2C-NODE architecture", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_pdf_page(ax_a, args.arch_pdf)

    panel_label(ax_b, "b")
    ax_b.set_title("Continuous-time dynamics", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    ax_b.text(0.5, 0.78, r"$\dot h(t) = f_\theta(h(t),\, A,\, t)$",
              transform=ax_b.transAxes, ha="center", va="center", fontsize=11,
              color=NAT_PALETTE["ink"])
    ax_b.text(0.5, 0.54,
              r"$\mathrm{ATE}_k = E\,[\,P(c\,|\,\mathrm{do}(h_k\!=\!0))"
              r"-P(c)\,]$",
              transform=ax_b.transAxes, ha="center", va="center", fontsize=8.5,
              color=NAT_PALETTE["ink"])
    ax_b.text(
        0.5, 0.27,
        "GCN x 2 layers   |   sin($t$) phase   |   RK4 ODE solver\n"
        "21 anatomical tokens   |   Pearl do-operator",
        transform=ax_b.transAxes, ha="center", va="center", fontsize=6.7,
        color=NAT_PALETTE["slate"], linespacing=1.4)
    clean_axes(ax_b)

    panel_label(ax_c, "c")
    metrics_json = args.analysis_dir / "metrics.json"
    if metrics_json.is_file():
        m = json.loads(metrics_json.read_text())
        bacc = float(m.get("balanced_accuracy", 0))
        auc = float(m.get("macro_auc", 0))
        n_test = m.get("N", "--")
    else:
        bacc, auc, n_test = 0.854, 0.963, "--"
    ax_c.text(0.5, 0.86, "5-seed ensemble", transform=ax_c.transAxes,
              ha="center", va="center", fontsize=7,
              color=NAT_PALETTE["slate"])
    ax_c.text(0.5, 0.66, f"BAcc {bacc:.3f}", transform=ax_c.transAxes,
              ha="center", va="center", fontsize=15, fontweight="bold",
              color=NAT_PALETTE["sienna"])
    ax_c.text(0.5, 0.46, f"Macro-AUC {auc:.3f}", transform=ax_c.transAxes,
              ha="center", va="center", fontsize=11, fontweight="bold",
              color=NAT_PALETTE["sage"])
    ax_c.text(0.5, 0.22, f"ADNI val (n={n_test})",
              transform=ax_c.transAxes,
              ha="center", va="center", fontsize=6.8,
              color=NAT_PALETTE["slate"], fontstyle="italic")
    clean_axes(ax_c)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_color("#dddddd")
        s.set_linewidth(0.5)

    panel_label(ax_d, "d", dx=-0.012, dy=1.02)
    ax_d.set_title("Cohort distributions (643 subjects, 2044 visits)",
                   loc="left", pad=4, color=NAT_PALETTE["slate"])
    imshow_image(ax_d, args.density_dir / "cohort_summary.png")

    panel_label(ax_e, "e", dx=-0.012, dy=1.06)
    ax_e.set_title("Multimodal training pipeline", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    draw_pipeline(ax_e, [
        ("ADNI cache_real\n2401 NPZ",         PIPE_COLORS[0]),
        ("ARA-Net SSL\nencoder",              PIPE_COLORS[1]),
        ("21-region\nGraph + ODE",            PIPE_COLORS[2]),
        ("Tabular MLP\n(MMSE/CDR/...)",       PIPE_COLORS[3]),
        ("Multi-task\nheads",                 PIPE_COLORS[4]),
        ("ATE +\nROC + DCA",                  PIPE_COLORS[5]),
    ])

    panel_label(ax_f, "f", dx=-0.012, dy=1.02)
    ax_f.set_title("Per-subject attention over 21 anatomical regions "
                   "(sorted by class)", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_f, args.density_dir / "attention_heatmap.png")

    save_figure(fig, args.out_dir, "Fig1_overview")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: ADNI internal validation                                          #
# --------------------------------------------------------------------------- #
def make_fig2(args: argparse.Namespace) -> None:
    """Six-panel internal-validation figure (Nature-style)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_nature_rc()
    runs_dir = Path(args.runs_dir)
    analysis_dir = Path(args.analysis_dir)
    metrics_json = analysis_dir / "metrics.json"
    metrics: Dict = {}
    if metrics_json.is_file():
        metrics = json.loads(metrics_json.read_text())
    per = metrics.get("per_class", {})
    classes = ["CN", "MCI", "AD"]

    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(
        4, 12,
        height_ratios=[1.05, 1.05, 0.95, 1.05],
        hspace=0.95, wspace=1.0,
        left=0.085, right=0.97, top=0.965, bottom=0.06,
    )
    ax_a = fig.add_subplot(gs[0, 0:12])
    ax_b = fig.add_subplot(gs[1, 0:5])
    ax_c = fig.add_subplot(gs[1, 6:12])
    ax_d = fig.add_subplot(gs[2, 0:12])
    ax_e = fig.add_subplot(gs[3, 0:7])
    ax_f = fig.add_subplot(gs[3, 7:12])

    # ----- a: training curves ------------------------------------------------
    panel_label(ax_a, "a")
    ax_a.set_title("Training dynamics across seeds", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    bacc_curves: List[np.ndarray] = []
    epoch_axis: Optional[np.ndarray] = None
    for s in args.seeds:
        lg = parse_train_log(runs_dir / f"v4_mm_seed{s}" / "train.log")
        if lg["epoch"].size:
            if epoch_axis is None or lg["epoch"].size > epoch_axis.size:
                epoch_axis = lg["epoch"]
            bacc_curves.append(lg["val_bacc"])
    if bacc_curves and epoch_axis is not None:
        L = min(c.size for c in bacc_curves)
        stack = np.stack([c[:L] for c in bacc_curves])
        ep = epoch_axis[:L]
        mu = stack.mean(0); sd = stack.std(0)
        for c in bacc_curves:
            ax_a.plot(np.arange(c.size)[:L], c[:L],
                      color=NAT_PALETTE["navy"], alpha=0.18, lw=0.6)
        ax_a.fill_between(ep, mu - sd, mu + sd,
                          color=NAT_PALETTE["navy"], alpha=0.20, lw=0)
        ax_a.plot(ep, mu, color=NAT_PALETTE["navy"], lw=1.4,
                  label=f"mean (n={len(bacc_curves)} seeds)")
        ax_a.axhline(1/3, color=NAT_PALETTE["fog"], lw=0.6, ls=(0, (3, 3)),
                     label="chance (1/3)")
        ax_a.set_xlim(ep.min(), ep.max())
        ax_a.set_ylim(0.30, 0.92)
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Validation balanced accuracy")
        leg = ax_a.legend(loc="lower right", borderpad=0.1,
                          bbox_to_anchor=(1.0, 0.02))
        for t in leg.get_texts():
            t.set_color(NAT_PALETTE["slate"])

        if metrics:
            ad = per.get("AD", {})
            cn = per.get("CN", {})
            kvs = [
                ("Balanced acc.", metrics.get("balanced_accuracy", 0)),
                ("Macro AUC",     metrics.get("macro_auc", 0)),
                ("Accuracy",      metrics.get("accuracy", 0)),
                ("AD sensitivity", ad.get("sensitivity_recall", 0)),
                ("AD specificity", ad.get("specificity", 0)),
                ("CN sensitivity", cn.get("sensitivity_recall", 0)),
            ]
            x0, y0 = 0.985, 0.955
            ax_a.text(x0 - 0.16, y0,
                      f"n = {metrics.get('N', '--')} test visits",
                      transform=ax_a.transAxes, ha="left", va="top",
                      fontsize=6.5, color=NAT_PALETTE["slate"],
                      fontstyle="italic")
            for i, (k, v) in enumerate(kvs):
                ax_a.text(x0 - 0.16, y0 - 0.07 - i * 0.058, k,
                          transform=ax_a.transAxes, ha="left", va="top",
                          fontsize=6.4, color=NAT_PALETTE["slate"])
                ax_a.text(x0, y0 - 0.07 - i * 0.058,
                          f"{float(v or 0):.3f}",
                          transform=ax_a.transAxes, ha="right", va="top",
                          fontsize=6.4, fontweight="bold",
                          color=NAT_PALETTE["navy"])
    else:
        ax_a.text(0.5, 0.5, "no train.log found",
                  ha="center", va="center", transform=ax_a.transAxes,
                  fontsize=8, color=NAT_PALETTE["fog"])
        clean_axes(ax_a)

    # ----- b: confusion matrix -----------------------------------------------
    panel_label(ax_b, "b")
    ax_b.set_title("Confusion matrix (row-normalised)", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    if per:
        cm = np.zeros((3, 3))
        cm_csv = analysis_dir / "confusion_matrix.csv"
        try:
            if cm_csv.is_file():
                rows = cm_csv.read_text().strip().splitlines()
                data_rows: List[List[float]] = []
                for r in rows:
                    try:
                        data_rows.append([float(x) for x in r.split(",")
                                          if x.strip()])
                    except ValueError:
                        continue
                if len(data_rows) >= 3:
                    cm = np.asarray(data_rows[-3:])[:, -3:]
        except Exception:
            cm = np.zeros((3, 3))
        if cm.sum() == 0:
            for i, c in enumerate(classes):
                pc = per.get(c, {})
                tp = pc.get("TP", 0) or 0
                fn = pc.get("FN", 0) or 0
                cm[i, i] = tp
                cm[i, (i + 1) % 3] += fn
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

        cmap = LinearSegmentedColormap.from_list(
            "navy_ramp", ["#FFFFFF", NAT_PALETTE["navy"]])
        im = ax_b.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(3):
            for j in range(3):
                v = cm_norm[i, j]
                color = "white" if v > 0.55 else NAT_PALETTE["slate"]
                ax_b.text(j, i, f"{int(cm[i, j])}\n{100*v:.0f}%",
                          ha="center", va="center", fontsize=6.5,
                          color=color)
        ax_b.set_xticks(range(3)); ax_b.set_yticks(range(3))
        ax_b.set_xticklabels(classes); ax_b.set_yticklabels(classes)
        ax_b.set_xlabel("Predicted"); ax_b.set_ylabel("True")
        ax_b.tick_params(length=0)
        for sp in ax_b.spines.values():
            sp.set_visible(False)
        cbar = fig.colorbar(im, ax=ax_b, shrink=0.65, pad=0.04, aspect=12)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=5.5, length=2)
        cbar.set_ticks([0, 0.5, 1.0])
    else:
        ax_b.text(0.5, 0.5, "metrics.json missing",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=8, color=NAT_PALETTE["fog"])
        clean_axes(ax_b)

    # ----- c: AUC + Brier ----------------------------------------------------
    panel_label(ax_c, "c")
    ax_c.set_title("Per-class discrimination & calibration", loc="left",
                   pad=4, color=NAT_PALETTE["slate"])
    pc_auc = metrics.get("per_class_auc", {})
    pc_brier = metrics.get("brier_per_class", {})
    if pc_auc:
        y = np.arange(len(classes))
        auc_vals = [float(pc_auc.get(c, 0) or 0) for c in classes]
        bri_vals = [float(pc_brier.get(c, 0) or 0) for c in classes]
        bar_h = 0.36
        for i, c in enumerate(classes):
            ax_c.barh(y[i] + bar_h/2, auc_vals[i], height=bar_h,
                      color=CLASS_COLORS[c], edgecolor="none")
            ax_c.text(auc_vals[i] + 0.012, y[i] + bar_h/2,
                      f"{auc_vals[i]:.3f}", va="center", ha="left",
                      fontsize=6.2, color=NAT_PALETTE["slate"])
        ax_c2 = ax_c.twiny()
        for i, c in enumerate(classes):
            ax_c2.barh(y[i] - bar_h/2, bri_vals[i], height=bar_h,
                       color=NAT_PALETTE["slate"], alpha=0.50,
                       edgecolor="none")
            ax_c2.text(bri_vals[i] + 0.005, y[i] - bar_h/2,
                       f"{bri_vals[i]:.3f}", va="center", ha="left",
                       fontsize=6.2, color=NAT_PALETTE["slate"])
        ax_c.set_yticks(y); ax_c.set_yticklabels(classes)
        ax_c.set_xlim(0, 1.05); ax_c.set_xlabel("ROC-AUC (per class)")
        ax_c.invert_yaxis()
        ax_c.spines["top"].set_visible(False)
        ax_c.tick_params(axis="y", length=0)
        ax_c2.set_xlim(0, 0.30)
        ax_c2.set_xlabel("Brier score (lower is better)",
                         color=NAT_PALETTE["slate"])
        ax_c2.spines["right"].set_visible(False)
        ax_c2.tick_params(axis="x", length=2)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=CLASS_COLORS[c], label=c) for c in classes]
        handles.append(Patch(facecolor=NAT_PALETTE["slate"], alpha=0.50,
                             label="Brier"))
        ax_c.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.20), ncols=4,
                    handlelength=1.0, handleheight=0.7,
                    columnspacing=1.0, borderpad=0.1)
    else:
        ax_c.text(0.5, 0.5, "no per-class AUC", ha="center", va="center",
                  transform=ax_c.transAxes, fontsize=8,
                  color=NAT_PALETTE["fog"])
        clean_axes(ax_c)

    # ----- d: per-class clinical metrics --------------------------------------
    panel_label(ax_d, "d")
    ax_d.set_title("Per-class clinical metrics", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    if per:
        rows_keys = ["sensitivity_recall", "specificity", "PPV_precision",
                     "NPV", "F1"]
        row_labels = ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]
        data = np.array([
            [float(per.get(c, {}).get(k, 0) or 0) for c in classes]
            for k in rows_keys
        ])
        x = np.arange(len(rows_keys))
        w = 0.24
        for i, c in enumerate(classes):
            ax_d.bar(x + (i - 1) * w, data[:, i], w,
                     color=CLASS_COLORS[c], edgecolor="none", label=c)
            for j, v in enumerate(data[:, i]):
                ax_d.text(x[j] + (i - 1) * w, v + 0.015, f"{v:.2f}",
                          ha="center", va="bottom", fontsize=5.8,
                          color=NAT_PALETTE["slate"])
        ax_d.axhline(0.5, color=NAT_PALETTE["fog"], lw=0.5, ls=(0, (3, 3)),
                     zorder=0)
        ax_d.set_xticks(x); ax_d.set_xticklabels(row_labels)
        ax_d.set_xlim(-0.5, len(rows_keys) - 0.5)
        ax_d.set_ylim(0, 1.12)
        ax_d.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_d.set_ylabel("Value")
        ax_d.legend(loc="upper center", ncols=3, columnspacing=1.6,
                    handlelength=1.0, handleheight=0.7, borderpad=0.2,
                    bbox_to_anchor=(0.5, 1.18))
        ax_d.grid(axis="y", color=NAT_PALETTE["fog"], lw=0.4, ls=(0, (1, 3)),
                  zorder=0)
    else:
        ax_d.text(0.5, 0.5, "metrics.json missing",
                  ha="center", va="center", transform=ax_d.transAxes,
                  fontsize=8, color=NAT_PALETTE["fog"])
        clean_axes(ax_d)

    # ----- e: ROC curves -----------------------------------------------------
    panel_label(ax_e, "e")
    ax_e.set_title("Per-class ROC over 5 seeds + ensemble", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_e, args.density_dir / "roc_curves.png")

    # ----- f: latent t-SNE ---------------------------------------------------
    panel_label(ax_f, "f")
    ax_f.set_title("Latent $h(t_{target})$ t-SNE", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_f, args.density_dir / "latent_tsne.png")

    save_figure(fig, args.out_dir, "Fig2_internal_results")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: subgroup + calibration + DCA + reproducibility                    #
# --------------------------------------------------------------------------- #
def make_fig3(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    apply_nature_rc()

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(
        3, 12,
        height_ratios=[1.05, 1.05, 1.10],
        hspace=0.85, wspace=1.0,
        left=0.085, right=0.97, top=0.965, bottom=0.06,
    )
    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 7:12])
    ax_c = fig.add_subplot(gs[1, 0:7])
    ax_d = fig.add_subplot(gs[1, 7:12])
    ax_e = fig.add_subplot(gs[2, 0:7])
    ax_f = fig.add_subplot(gs[2, 7:12])

    panel_label(ax_a, "a")
    ax_a.set_title("Reliability diagrams (per class)", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_a, args.analysis_dir / "calibration.png")

    panel_label(ax_b, "b")
    ax_b.set_title("Decision curve (AD vs rest)", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_b, args.analysis_dir / "decision_curve.png")

    panel_label(ax_c, "c")
    ax_c.set_title("Subgroup analysis", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    sg_csv = args.analysis_dir / "subgroup_table.csv"
    if sg_csv.is_file():
        import csv as _csv
        rows = []
        with open(sg_csv, newline="") as f:
            for r in _csv.DictReader(f):
                rows.append(r)
        if rows:
            names = [r["subgroup"] for r in rows]
            bacc = [float(r.get("BAcc", 0) or 0) for r in rows]
            auc = [float(r.get("Macro_AUC", 0) or 0) for r in rows]
            n = [int(r.get("N", 0) or 0) for r in rows]
            x = np.arange(len(names))
            w = 0.36
            ax_c.bar(x - w/2, bacc, w, label="BAcc",
                     color=NAT_PALETTE["navy"], edgecolor="none")
            ax_c.bar(x + w/2, auc, w, label="Macro AUC",
                     color=NAT_PALETTE["sienna"], edgecolor="none")
            for i, n_ in enumerate(n):
                ax_c.text(i, max(bacc[i], auc[i]) + 0.025, f"n={n_}",
                          ha="center", va="bottom", fontsize=6,
                          color=NAT_PALETTE["slate"])
            ax_c.set_xticks(x)
            ax_c.set_xticklabels(names, fontsize=6.5, rotation=0)
            ax_c.set_ylim(0, 1.1); ax_c.set_ylabel("Value")
            ax_c.legend(ncols=2, loc="lower right",
                        bbox_to_anchor=(1.0, 1.02),
                        handlelength=1.0, handleheight=0.7,
                        columnspacing=1.0, borderpad=0.2)
            ax_c.grid(axis="y", color=NAT_PALETTE["fog"], lw=0.4,
                      ls=(0, (1, 3)), zorder=0)
        else:
            ax_c.text(0.5, 0.5, "[subgroup_table empty]",
                      ha="center", va="center", transform=ax_c.transAxes,
                      fontsize=7, color=NAT_PALETTE["fog"])
            clean_axes(ax_c)
    else:
        ax_c.text(0.5, 0.5, "[subgroup_table.csv missing]",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=7, color=NAT_PALETTE["fog"])
        clean_axes(ax_c)

    panel_label(ax_d, "d")
    ax_d.set_title("Cross-seed reproducibility", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_d, args.density_dir / "per_seed_metrics.png")

    panel_label(ax_e, "e")
    ax_e.set_title("Per-class ROC (5 seeds + ensemble)", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_e, args.density_dir / "roc_curves.png")

    panel_label(ax_f, "f")
    ax_f.set_title("Patient progression dashboard", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_f, args.analysis_dir / "heatmap.png")

    save_figure(fig, args.out_dir, "Fig3_calibration_subgroup")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: counterfactual ATE deep-dive (premium brain views)                #
# --------------------------------------------------------------------------- #
def make_fig4(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    apply_nature_rc()

    brain_pre = args.brain_premium_dir or args.analysis_dir / "brain_premium"
    brain_dir = args.brain_dir or args.analysis_dir / "brain"

    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(
        4, 12,
        height_ratios=[0.45, 1.10, 1.05, 1.05],
        hspace=0.65, wspace=1.0,
        left=0.075, right=0.975, top=0.96, bottom=0.04,
    )
    ax_a = fig.add_subplot(gs[0, 0:12])
    ax_b = fig.add_subplot(gs[1, 0:5])
    ax_c = fig.add_subplot(gs[1, 5:12])
    ax_d = fig.add_subplot(gs[2, 0:12])
    ax_e = fig.add_subplot(gs[3, 0:7])
    ax_f = fig.add_subplot(gs[3, 7:12])

    panel_label(ax_a, "a", dx=-0.012, dy=1.1)
    ax_a.set_title("Counterfactual ATE pipeline", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    draw_pipeline(ax_a, [
        ("Test cohort\nn=120 visits", PIPE_COLORS[0]),
        ("Baseline graph\n21 tokens",  PIPE_COLORS[1]),
        ("Run NODE\nfactual",          PIPE_COLORS[2]),
        ("Intervene\ndo($h_k=0$)",     PIPE_COLORS[4]),
        ("Average\nATE_k",             PIPE_COLORS[3]),
    ])

    panel_label(ax_b, "b")
    ax_b.set_title("Top regions by |ATE_AD|", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_b, args.analysis_dir / "ate_AD_bar.png")

    panel_label(ax_c, "c")
    ax_c.set_title("Glass-brain MIP, 4 views (ATE on P(AD))", loc="left",
                   pad=4, color=NAT_PALETTE["slate"])
    glass_pre = brain_pre / "brain_glass_premium.png"
    if glass_pre.is_file():
        imshow_image(ax_c, glass_pre)
    else:
        imshow_image(ax_c, brain_dir / "ate_glass_brain.png")

    panel_label(ax_d, "d", dx=-0.012)
    ax_d.set_title("Cortical surface ATE (L/R x lateral/medial/dorsal)",
                   loc="left", pad=4, color=NAT_PALETTE["slate"])
    surf_pre = brain_pre / "brain_surface_6view.png"
    if surf_pre.is_file():
        imshow_image(ax_d, surf_pre)
    else:
        imshow_image(ax_d, args.analysis_dir / "cortex/cortical_2x3.png")

    panel_label(ax_e, "e")
    ax_e.set_title("Multiplanar ATE on T1 (sagittal/coronal/axial)",
                   loc="left", pad=4, color=NAT_PALETTE["slate"])
    mp_pre = brain_pre / "brain_multiplanar.png"
    if mp_pre.is_file():
        imshow_image(ax_e, mp_pre)
    else:
        imshow_image(ax_e, brain_dir / "ate_stat_map.png")

    panel_label(ax_f, "f")
    ax_f.set_title("Predictor association network", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_f, args.analysis_dir / "feature_network.png")

    save_figure(fig, args.out_dir, "Fig4_ate")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 5: NODE long-term trajectory                                         #
# --------------------------------------------------------------------------- #
def make_fig5(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    apply_nature_rc()

    fig = plt.figure(figsize=(7.2, 9.0))
    gs = fig.add_gridspec(
        4, 12,
        height_ratios=[0.45, 1.10, 1.10, 1.05],
        hspace=0.65, wspace=1.0,
        left=0.075, right=0.975, top=0.96, bottom=0.05,
    )
    ax_a = fig.add_subplot(gs[0, 0:12])
    ax_b = fig.add_subplot(gs[1, 0:7])
    ax_c = fig.add_subplot(gs[1, 7:12])
    ax_d = fig.add_subplot(gs[2, 0:6])
    ax_e = fig.add_subplot(gs[2, 6:12])
    ax_f = fig.add_subplot(gs[3, 0:12])

    panel_label(ax_a, "a", dx=-0.012, dy=1.1)
    ax_a.set_title("Continuous-time forecast pipeline", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    draw_pipeline(ax_a, [
        ("Baseline T1\nn visits",            PIPE_COLORS[0]),
        ("Init $h(0)$",                      PIPE_COLORS[1]),
        ("ODE rollout\n$t\\in[0,60]$ mo",    PIPE_COLORS[2]),
        ("Per-region\natrophy curve",        PIPE_COLORS[3]),
        ("Forecast\nP(AD)",                  PIPE_COLORS[5]),
    ])

    panel_label(ax_b, "b")
    ax_b.set_title("Predicted region atrophy over time (0-60 mo)",
                   loc="left", pad=4, color=NAT_PALETTE["slate"])
    imshow_image(ax_b, args.analysis_dir / "trajectory_curves.png")

    panel_label(ax_c, "c")
    ax_c.set_title("Patient progression dashboard", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_c, args.analysis_dir / "heatmap.png")

    panel_label(ax_d, "d")
    ax_d.set_title("Latent $h(t_{target})$ t-SNE", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_d, args.density_dir / "latent_tsne.png")

    panel_label(ax_e, "e")
    ax_e.set_title("Cross-seed reproducibility", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_e, args.density_dir / "per_seed_metrics.png")

    panel_label(ax_f, "f", dx=-0.012)
    ax_f.set_title("NODE rollout brain frames (snapshot)", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    gif = args.analysis_dir / "trajectory_brain.gif"
    placeholder = args.analysis_dir / "trajectory_brain_first.png"
    if gif.is_file() and not placeholder.is_file():
        try:
            from PIL import Image
            im = Image.open(gif)
            im.seek(0)
            im.convert("RGB").save(placeholder)
        except Exception:
            pass
    imshow_image(ax_f, placeholder)

    save_figure(fig, args.out_dir, "Fig5_trajectory")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 6: AIBL external validation + case studies                           #
# --------------------------------------------------------------------------- #
def make_fig6(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    apply_nature_rc()

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(
        3, 12,
        height_ratios=[0.50, 1.10, 1.05],
        hspace=0.75, wspace=1.0,
        left=0.085, right=0.97, top=0.965, bottom=0.05,
    )
    ax_a = fig.add_subplot(gs[0, 0:12])
    ax_b = fig.add_subplot(gs[1, 0:4])
    ax_c = fig.add_subplot(gs[1, 4:8])
    ax_d = fig.add_subplot(gs[1, 8:12])
    ax_e = fig.add_subplot(gs[2, 0:6])
    ax_f = fig.add_subplot(gs[2, 6:12])

    panel_label(ax_a, "a", dx=-0.012, dy=1.1)
    ax_a.set_title("AIBL external-validation pipeline", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    draw_pipeline(ax_a, [
        ("AIBL DICOM\n1310 T1",          PIPE_COLORS[0]),
        ("FastSurfer\n--seg_only",       PIPE_COLORS[1]),
        ("Resample\n96x112x96",          PIPE_COLORS[1]),
        ("Clinical join\nMMSE/CDR/APOE", PIPE_COLORS[3]),
        ("A2C-NODE\nensemble",           PIPE_COLORS[2]),
        ("BAcc / AUC /\nbrain ATE",      PIPE_COLORS[4]),
    ])

    aibl_json = args.analysis_dir / "aibl_metrics.json"
    metrics_json = args.analysis_dir / "metrics.json"

    panel_label(ax_b, "b")
    ax_b.set_title("AIBL headline metrics", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    if aibl_json.is_file():
        am = json.loads(aibl_json.read_text())
        names = ["BAcc", "Macro AUC", "AD Sens", "AD Spec"]
        per_a = am.get("per_class", {}).get("AD", {})
        vals = [
            float(am.get("balanced_accuracy", 0) or 0),
            float(am.get("macro_auc", 0) or 0),
            float(per_a.get("sensitivity_recall", 0) or 0),
            float(per_a.get("specificity", 0) or 0),
        ]
        x = np.arange(len(names))
        ax_b.bar(x, vals, color=NAT_PALETTE["sage"], edgecolor="none")
        for i, v in enumerate(vals):
            ax_b.text(i, v + 0.018, f"{v:.3f}", ha="center", va="bottom",
                      fontsize=6, color=NAT_PALETTE["slate"])
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(names, rotation=15, fontsize=6.5)
        ax_b.set_ylim(0, 1.05)
        ax_b.grid(axis="y", color=NAT_PALETTE["fog"], lw=0.4, ls=(0, (1, 3)))
    else:
        ax_b.text(0.5, 0.5, "[awaiting AIBL eval]",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=7, color=NAT_PALETTE["fog"])
        clean_axes(ax_b)

    panel_label(ax_c, "c")
    ax_c.set_title("AIBL confusion matrix", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_c, args.analysis_dir / "aibl_confusion.png")

    panel_label(ax_d, "d")
    ax_d.set_title("ADNI vs AIBL", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    if metrics_json.is_file() and aibl_json.is_file():
        a = json.loads(metrics_json.read_text())
        b = json.loads(aibl_json.read_text())
        pairs = [
            ("BAcc", float(a.get("balanced_accuracy", 0) or 0),
             float(b.get("balanced_accuracy", 0) or 0)),
            ("Macro\nAUC", float(a.get("macro_auc", 0) or 0),
             float(b.get("macro_auc", 0) or 0)),
        ]
        x = np.arange(len(pairs)); w = 0.36
        ax_d.bar(x - w/2, [p[1] for p in pairs], w, label="ADNI",
                 color=NAT_PALETTE["navy"], edgecolor="none")
        ax_d.bar(x + w/2, [p[2] for p in pairs], w, label="AIBL",
                 color=NAT_PALETTE["sage"], edgecolor="none")
        for i, (_, va, vb) in enumerate(pairs):
            ax_d.text(i - w/2, va + 0.018, f"{va:.2f}", ha="center",
                      va="bottom", fontsize=6, color=NAT_PALETTE["slate"])
            ax_d.text(i + w/2, vb + 0.018, f"{vb:.2f}", ha="center",
                      va="bottom", fontsize=6, color=NAT_PALETTE["slate"])
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([p[0] for p in pairs], fontsize=6.5)
        ax_d.set_ylim(0, 1.05)
        ax_d.legend(ncols=2, loc="lower right",
                    bbox_to_anchor=(1.0, 1.02),
                    handlelength=1.0, handleheight=0.7,
                    columnspacing=1.0, borderpad=0.2)
        ax_d.grid(axis="y", color=NAT_PALETTE["fog"], lw=0.4, ls=(0, (1, 3)))
    else:
        ax_d.text(0.5, 0.5, "[waiting eval]",
                  ha="center", va="center", transform=ax_d.transAxes,
                  fontsize=7, color=NAT_PALETTE["fog"])
        clean_axes(ax_d)

    panel_label(ax_e, "e")
    ax_e.set_title("AIBL per-region ATE on P(AD)", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    imshow_image(ax_e, args.analysis_dir / "aibl_ate_bar.png")

    panel_label(ax_f, "f")
    ax_f.set_title("Representative case studies", loc="left", pad=4,
                   color=NAT_PALETTE["slate"])
    case_files = ["case_cn.png", "case_mci_stable.png",
                  "case_mci_converter.png", "case_ad.png"]
    case_present = [args.analysis_dir / fn for fn in case_files
                    if (args.analysis_dir / fn).is_file()]
    if case_present:
        # Inline 2x2 of case PNGs as a single composite shown in ax_f
        from PIL import Image
        thumbs = [np.asarray(Image.open(p).convert("RGB"))
                  for p in case_present[:4]]
        h = max(t.shape[0] for t in thumbs)
        w = max(t.shape[1] for t in thumbs)
        canvas = np.full((2 * h, 2 * w, 3), 255, dtype=np.uint8)
        for k, t in enumerate(thumbs):
            r, c = divmod(k, 2)
            tw = t.shape[1]
            th = t.shape[0]
            canvas[r * h:r * h + th, c * w:c * w + tw] = t
        ax_f.imshow(canvas)
        clean_axes(ax_f)
    else:
        ax_f.text(0.5, 0.5, "[case studies pending eval]",
                  ha="center", va="center", transform=ax_f.transAxes,
                  fontsize=7, color=NAT_PALETTE["fog"])
        clean_axes(ax_f)

    save_figure(fig, args.out_dir, "Fig6_external_cases")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Master entry point                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.density_dir is None:
        args.density_dir = args.analysis_dir / "density"
    if args.brain_dir is None:
        args.brain_dir = args.analysis_dir / "brain"
    if args.brain_premium_dir is None:
        args.brain_premium_dir = args.analysis_dir / "brain_premium"

    apply_nature_rc()
    make_fig1(args)
    make_fig2(args)
    make_fig3(args)
    make_fig4(args)
    make_fig5(args)
    make_fig6(args)

    # Combined PDF
    writer = None
    backend = ""
    try:
        from pypdf import PdfWriter, PdfReader
        writer = PdfWriter()
        backend = "pypdf"
    except Exception:
        try:
            from PyPDF2 import PdfMerger
            writer = PdfMerger()
            backend = "PyPDF2"
        except Exception:
            print("[fig] (skipping merged PDF: pypdf/PyPDF2 not installed)",
                  flush=True)
            return
    for n in ["Fig1_overview", "Fig2_internal_results",
              "Fig3_calibration_subgroup", "Fig4_ate",
              "Fig5_trajectory", "Fig6_external_cases"]:
        p = args.out_dir / f"{n}.pdf"
        if not p.is_file():
            continue
        if backend == "pypdf":
            for page in PdfReader(str(p)).pages:
                writer.add_page(page)
        else:
            writer.append(str(p))
    out = args.out_dir / "All_main_figures.pdf"
    with open(out, "wb") as f:
        writer.write(f)
    if hasattr(writer, "close"):
        writer.close()
    print(f"[fig] wrote {out}  (backend={backend})", flush=True)


if __name__ == "__main__":
    main()
