#!/usr/bin/env python3
"""Generate manuscript-positioning figures for CUED-AD.

The figures are intentionally built with Pillow rather than matplotlib so the
script can run in the lightweight Codex runtime available in this workspace.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "manuscript_packaging_figures"


COLORS = {
    "ink": (28, 38, 48),
    "muted": (94, 108, 121),
    "line": (201, 210, 219),
    "panel": (248, 250, 252),
    "blue": (38, 111, 219),
    "sky": (154, 201, 255),
    "green": (38, 154, 112),
    "mint": (170, 224, 206),
    "orange": (230, 127, 57),
    "red": (204, 67, 67),
    "purple": (108, 91, 190),
    "white": (255, 255, 255),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/Users/mac/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


F = {
    "title": font(42, True),
    "h1": font(30, True),
    "h2": font(24, True),
    "body": font(20),
    "small": font(16),
    "tiny": font(14),
    "label": font(18, True),
}


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font_obj: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textbbox((0, 0), test, font=font_obj)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] = COLORS["line"],
    radius: int = 16,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color=COLORS["muted"], width: int = 4) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    p1 = (end[0] - size * math.cos(angle - math.pi / 6), end[1] - size * math.sin(angle - math.pi / 6))
    p2 = (end[0] - size * math.cos(angle + math.pi / 6), end[1] - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=color)


def poly_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color=COLORS["muted"], width: int = 4) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    start, end = points[-2], points[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    p1 = (end[0] - size * math.cos(angle - math.pi / 6), end[1] - size * math.sin(angle - math.pi / 6))
    p2 = (end[0] - size * math.cos(angle + math.pi / 6), end[1] - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=color)


def text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    font_obj: ImageFont.ImageFont,
    fill=COLORS["ink"],
    line_gap: int = 6,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, max_width, font_obj):
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((70, 48), title, font=F["title"], fill=COLORS["ink"])
    draw.text((72, 105), subtitle, font=F["body"], fill=COLORS["muted"])


def figure1_system() -> None:
    img = Image.new("RGB", (1800, 1100), COLORS["white"])
    draw = ImageDraw.Draw(img)
    draw_title(
        draw,
        "Figure 1. CUED-AD: conflict-aware causal Bayesian decision support",
        "The system predicts AD risk, detects MRI-clinical disagreement, and explains uncertainty with causal ROI counterfactuals.",
    )

    # Left cohort/data panel
    rounded_box(draw, (70, 180, 430, 410), COLORS["panel"])
    draw.text((95, 205), "A. Leakage-free cohorts", font=F["h2"], fill=COLORS["ink"])
    text_block(draw, (95, 250), "ADNI subject-level train/val/test split; AIBL external validation; AD-vs-NC primary task.", 300, F["body"], COLORS["muted"])
    draw.text((95, 355), "MCI retained for traceability", font=F["small"], fill=COLORS["purple"])

    # MRI branch
    rounded_box(draw, (520, 170, 880, 350), (238, 246, 255), outline=(162, 196, 236))
    draw.text((545, 195), "B. MRI branch", font=F["h2"], fill=COLORS["ink"])
    text_block(draw, (545, 240), "T1 segmentation -> 24 AD-relevant ROI residuals -> prior-constrained SCM structural residuals.", 300, F["body"], COLORS["muted"])

    rounded_box(draw, (950, 170, 1285, 350), (239, 250, 246), outline=(156, 211, 190))
    draw.text((975, 195), "C. Clinical branch", font=F["h2"], fill=COLORS["ink"])
    text_block(draw, (975, 240), "MMSE, CDR-SB, ADAS11/13 plus missingness masks -> Bayesian clinical classifier.", 285, F["body"], COLORS["muted"])

    rounded_box(draw, (565, 470, 815, 610), COLORS["white"], outline=COLORS["blue"])
    draw.text((590, 495), "pMRI(AD)", font=F["label"], fill=COLORS["blue"])
    draw.text((590, 535), "uMRI", font=F["label"], fill=COLORS["blue"])
    draw.text((590, 570), "MC dropout", font=F["small"], fill=COLORS["muted"])

    rounded_box(draw, (995, 470, 1245, 610), COLORS["white"], outline=COLORS["green"])
    draw.text((1020, 495), "pClin(AD)", font=F["label"], fill=COLORS["green"])
    draw.text((1020, 535), "uClin", font=F["label"], fill=COLORS["green"])
    draw.text((1020, 570), "MC dropout", font=F["small"], fill=COLORS["muted"])

    # Center disagreement/fusion
    rounded_box(draw, (720, 710, 1100, 900), (255, 248, 240), outline=(235, 173, 115))
    draw.text((755, 735), "D. Cross-modal reliability", font=F["h2"], fill=COLORS["ink"])
    text_block(draw, (755, 782), "JS divergence measures MRI-clinical disagreement. Total uncertainty = uMRI + uClin + beta*DJS.", 310, F["body"], COLORS["muted"])

    # Output
    rounded_box(draw, (1280, 625, 1710, 925), (251, 244, 250), outline=(209, 170, 219))
    draw.text((1310, 655), "E. Clinician-facing output", font=F["h2"], fill=COLORS["ink"])
    outputs = [
        ("AD risk probability", COLORS["purple"]),
        ("Total uncertainty", COLORS["orange"]),
        ("Conflict warning light", COLORS["red"]),
        ("ROI counterfactual sliders", COLORS["blue"]),
    ]
    yy = 710
    for label, c in outputs:
        draw.ellipse((1312, yy + 5, 1330, yy + 23), fill=c)
        draw.text((1342, yy), label, font=F["body"], fill=COLORS["ink"])
        yy += 42

    # Mini risk card
    rounded_box(draw, (1320, 878, 1665, 910), COLORS["white"], outline=COLORS["line"], radius=12)
    draw.rectangle((1345, 890, 1580, 900), fill=(235, 239, 244))
    draw.rectangle((1345, 890, 1515, 900), fill=COLORS["orange"])
    draw.text((1595, 883), "Review", font=F["small"], fill=COLORS["red"])

    # Arrows
    arrow(draw, (430, 295), (520, 260))
    poly_arrow(draw, [(430, 350), (475, 430), (1120, 430), (1120, 350)])
    arrow(draw, (700, 350), (690, 470), COLORS["blue"])
    arrow(draw, (1120, 350), (1120, 470), COLORS["green"])
    arrow(draw, (815, 540), (900, 710), COLORS["muted"])
    arrow(draw, (995, 540), (920, 710), COLORS["muted"])
    arrow(draw, (1100, 805), (1280, 770), COLORS["purple"])

    # Footer claim
    rounded_box(draw, (70, 965, 1710, 1040), COLORS["panel"], outline=(225, 231, 238))
    draw.text((95, 987), "Manuscript claim:", font=F["label"], fill=COLORS["ink"])
    draw.text((270, 987), "when evidence agrees, CUED-AD predicts; when evidence conflicts, CUED-AD warns and explains.", font=F["body"], fill=COLORS["muted"])

    img.save(OUT / "figure1_cued_ad_system_overview.png")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def metric_rows() -> list[dict[str, str]]:
    rows = read_csv(ROOT / "artifacts" / "evaluation_cued_ad_perf" / "cued_ad_perf_metrics.csv")
    wanted = {
        "CUED_AD_Perf_ClinicalPerformanceEnsemble": "CUED-AD-Perf",
        "Clinical_Logistic": "Clinical Logistic",
        "Clinical_XGBoost": "Clinical XGBoost",
        "Clinical_LightGBM": "Clinical LightGBM",
        "Clinical_CatBoost": "Clinical CatBoost",
        "MRIplusClinical_HGB": "MRI+Clinical HGB",
        "Clinical_only_BNN": "Clinical BNN",
        "MRI_only_BNN": "MRI BNN",
    }
    out = []
    for row in rows:
        if row["cohort"] == "AIBL_external" and row["model"] in wanted:
            item = dict(row)
            item["display"] = wanted[row["model"]]
            out.append(item)
    order = list(wanted.values())
    out.sort(key=lambda r: order.index(r["display"]))
    return out


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y_label: str, max_y: float = 1.0) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=COLORS["line"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["line"], width=2)
    for t in np.linspace(0, max_y, 5):
        y = y1 - int((t / max_y) * (y1 - y0))
        draw.line((x0 - 5, y, x1, y), fill=(236, 240, 244), width=1)
        draw.text((x0 - 55, y - 10), f"{t:.2f}", font=F["tiny"], fill=COLORS["muted"])
    draw.text((x0 - 65, y0 - 35), y_label, font=F["small"], fill=COLORS["muted"])


def figure2_performance() -> None:
    rows = metric_rows()
    img = Image.new("RGB", (1800, 1080), COLORS["white"])
    draw = ImageDraw.Draw(img)
    draw_title(
        draw,
        "Figure 2. External AD-vs-NC performance under strong baselines",
        "CUED-AD-Perf remains competitive with clinical XGBoost/LightGBM/CatBoost while preserving reliability outputs.",
    )

    metrics = [("acc", "ACC", COLORS["blue"]), ("auc", "AUC", COLORS["green"]), ("f1", "F1", COLORS["orange"])]
    chart = (150, 230, 1670, 780)
    draw_axes(draw, chart, "AIBL external metric")
    x0, y0, x1, y1 = chart
    group_w = (x1 - x0) / len(rows)
    bar_w = group_w / 5
    for i, row in enumerate(rows):
        cx = x0 + i * group_w + group_w / 2
        for j, (key, label, color) in enumerate(metrics):
            val = float(row[key])
            bh = int(val * (y1 - y0))
            bx = int(cx - 1.5 * bar_w + j * bar_w)
            draw.rounded_rectangle((bx, y1 - bh, bx + int(bar_w * 0.8), y1), radius=6, fill=color)
        label = row["display"]
        lines = wrap_text(draw, label, int(group_w * 0.85), F["tiny"])
        yy = y1 + 18
        for line in lines:
            tw = draw.textbbox((0, 0), line, font=F["tiny"])[2]
            draw.text((int(cx - tw / 2), yy), line, font=F["tiny"], fill=COLORS["ink"])
            yy += 17

    # Legend
    lx = 1380
    for i, (_, label, color) in enumerate(metrics):
        draw.rectangle((lx, 178 + i * 30, lx + 22, 198 + i * 30), fill=color)
        draw.text((lx + 32, 174 + i * 30), label, font=F["small"], fill=COLORS["ink"])

    # Key callout
    rounded_box(draw, (170, 850, 1660, 990), COLORS["panel"])
    callout = (
        "Key positioning: CUED-AD-Perf reaches ACC 0.984, AUC 0.997 and F1 0.920 on AIBL. "
        "Clinical CatBoost has slightly higher AUC, but CUED-AD-Perf provides the paper's reliability layer: "
        "uncertainty, disagreement, and counterfactual explanation."
    )
    text_block(draw, (205, 885), callout, 1420, F["body"], COLORS["ink"])

    img.save(OUT / "figure2_external_performance_baselines.png")


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve_manual(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.quantile(score, np.linspace(1, 0, 120))
    points = []
    p = max(1, int(y_true.sum()))
    n = max(1, int((1 - y_true).sum()))
    for thr in thresholds:
        pred = score >= thr
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        points.append((fp / n, tp / p))
    points = sorted(set(points))
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    return hist / max(1, hist.max())


def draw_hist_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    conflict: np.ndarray,
    nonconflict: np.ndarray,
    title: str,
    x_label: str,
    color_conflict=COLORS["red"],
    color_non=COLORS["blue"],
) -> None:
    x0, y0, x1, y1 = box
    rounded_box(draw, (x0 - 25, y0 - 55, x1 + 25, y1 + 80), COLORS["panel"], outline=(230, 236, 242))
    draw.text((x0, y0 - 42), title, font=F["h2"], fill=COLORS["ink"])
    lo = float(min(np.nanmin(conflict), np.nanmin(nonconflict)))
    hi = float(max(np.nanmax(conflict), np.nanmax(nonconflict)))
    bins = np.linspace(lo, hi, 22)
    hc = histogram(conflict, bins)
    hn = histogram(nonconflict, bins)
    w = (x1 - x0) / len(hc)
    for i, (a, b) in enumerate(zip(hn, hc)):
        bx = x0 + i * w
        draw.rectangle((int(bx), int(y1 - a * (y1 - y0)), int(bx + w * 0.86), y1), fill=(*color_non[:3],) if len(color_non) == 3 else color_non)
        draw.rectangle((int(bx + w * 0.12), int(y1 - b * (y1 - y0)), int(bx + w * 0.98), y1), fill=color_conflict)
    draw.line((x0, y1, x1, y1), fill=COLORS["line"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["line"], width=2)
    draw.text((x0, y1 + 18), x_label, font=F["small"], fill=COLORS["muted"])
    draw.text((x0, y1 + 45), f"non-conflict mean {mean(nonconflict):.2f}", font=F["tiny"], fill=color_non)
    draw.text((x0 + 250, y1 + 45), f"conflict mean {mean(conflict):.2f}", font=F["tiny"], fill=color_conflict)


def draw_roc_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y: np.ndarray, u: np.ndarray, djs: np.ndarray) -> None:
    x0, y0, x1, y1 = box
    rounded_box(draw, (x0 - 25, y0 - 55, x1 + 25, y1 + 80), COLORS["panel"], outline=(230, 236, 242))
    draw.text((x0, y0 - 42), "C. Conflict detection ROC", font=F["h2"], fill=COLORS["ink"])
    draw.line((x0, y1, x1, y1), fill=COLORS["line"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["line"], width=2)
    draw.line((x0, y1, x1, y0), fill=(205, 212, 220), width=2)

    def plot(score: np.ndarray, color, label: str) -> None:
        fpr, tpr = roc_curve_manual(y, score)
        pts = [(int(x0 + f * (x1 - x0)), int(y1 - t * (y1 - y0))) for f, t in zip(fpr, tpr)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=5, joint="curve")
        draw.text((x0 + 25, y0 + 25 + (0 if label.startswith("U") else 34)), label, font=F["small"], fill=color)

    plot(u, COLORS["orange"], f"U_total AUC {auc_rank(y, u):.3f}")
    plot(djs, COLORS["purple"], f"D_JS AUC {auc_rank(y, djs):.3f}")
    draw.text((x0, y1 + 18), "False positive rate", font=F["small"], fill=COLORS["muted"])
    draw.text((x0 - 10, y0 - 28), "True positive rate", font=F["small"], fill=COLORS["muted"])


def draw_rejection_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y_label: np.ndarray, pred: np.ndarray, u: np.ndarray) -> None:
    x0, y0, x1, y1 = box
    rounded_box(draw, (x0 - 25, y0 - 55, x1 + 25, y1 + 80), COLORS["panel"], outline=(230, 236, 242))
    draw.text((x0, y0 - 42), "D. Uncertainty rejection curve", font=F["h2"], fill=COLORS["ink"])
    draw.line((x0, y1, x1, y1), fill=COLORS["line"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["line"], width=2)
    order = np.argsort(-u)
    xs, ys = [], []
    for frac in np.linspace(0, 0.5, 51):
        reject = int(frac * len(u))
        keep = order[reject:]
        acc = float((pred[keep] == y_label[keep]).mean())
        xs.append(frac)
        ys.append(acc)
    ymin, ymax = 0.72, 0.9
    pts = [
        (int(x0 + x * 2 * (x1 - x0)), int(y1 - (min(max(y, ymin), ymax) - ymin) / (ymax - ymin) * (y1 - y0)))
        for x, y in zip(xs, ys)
    ]
    draw.line(pts, fill=COLORS["green"], width=5)
    for frac, acc in zip([0, 0.1, 0.25, 0.5], [ys[0], ys[10], ys[25], ys[50]]):
        px = int(x0 + frac * 2 * (x1 - x0))
        py = int(y1 - (min(max(acc, ymin), ymax) - ymin) / (ymax - ymin) * (y1 - y0))
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=COLORS["green"])
    draw.text((x0, y1 + 18), "Rejected highest-uncertainty fraction", font=F["small"], fill=COLORS["muted"])
    draw.text((x0 - 4, y0 - 28), "Retained accuracy", font=F["small"], fill=COLORS["muted"])


def figure3_conflict() -> None:
    rows = read_csv(ROOT / "artifacts" / "evaluation_external_ablation" / "predictions_AIBL_conflict_pool_CUED_AD_full.csv")
    is_conflict = np.array([int(float(r["is_conflict"])) for r in rows], dtype=int)
    y_label = np.array([int(float(r["binary_label"])) for r in rows], dtype=int)
    pred = np.array([int(float(r["pred_binary"])) for r in rows], dtype=int)
    u = np.array([float(r["U_total"]) for r in rows], dtype=float)
    djs = np.array([float(r["D_JS"]) for r in rows], dtype=float)

    img = Image.new("RGB", (1800, 1200), COLORS["white"])
    draw = ImageDraw.Draw(img)
    draw_title(
        draw,
        "Figure 3. Cross-modal conflict is surfaced as uncertainty",
        "AIBL natural conflict samples show higher total uncertainty and MRI-clinical JS disagreement.",
    )
    draw_hist_panel(draw, (110, 245, 790, 510), u[is_conflict == 1], u[is_conflict == 0], "A. Total uncertainty", "U_total")
    draw_hist_panel(draw, (970, 245, 1660, 510), djs[is_conflict == 1], djs[is_conflict == 0], "B. MRI-clinical disagreement", "D_JS")
    draw_roc_panel(draw, (110, 725, 790, 990), is_conflict, u, djs)
    draw_rejection_panel(draw, (970, 725, 1660, 990), y_label, pred, u)

    rounded_box(draw, (115, 1060, 1660, 1140), COLORS["panel"], outline=(225, 231, 238))
    text = (
        f"AIBL conflict pool: {int(is_conflict.sum())} natural conflicts and {len(is_conflict)-int(is_conflict.sum())} non-conflicts. "
        f"Mean U_total {mean(u[is_conflict==1]):.2f} vs {mean(u[is_conflict==0]):.2f}; "
        f"mean D_JS {mean(djs[is_conflict==1]):.3f} vs {mean(djs[is_conflict==0]):.3f}."
    )
    text_block(draw, (145, 1084), text, 1490, F["body"], COLORS["ink"])
    img.save(OUT / "figure3_conflict_uncertainty_aibl.png")


def manuscript_skeleton() -> None:
    path = ROOT / "artifacts" / "manuscript_draft" / "CUED_AD_manuscript_skeleton.md"
    path.write_text(
        """# Detecting when MRI and cognitive scores disagree: a causal Bayesian framework for reliable Alzheimer's disease assessment

## Abstract

Structural MRI and cognitive scales provide complementary evidence for Alzheimer's disease assessment, but they may disagree in clinically important cases such as preserved cognition despite medial temporal atrophy, atypical clinical impairment, or noisy scale recording. Most diagnostic AI systems optimize classification accuracy without explicitly identifying these cross-modal contradictions or communicating when predictions should be reviewed. We developed CUED-AD, a causal Bayesian multimodal framework for AD-vs-normal cognition assessment. CUED-AD learns a prior-constrained structural causal model over Alzheimer's disease-related brain regions, uses structural residuals as MRI features, trains Bayesian MRI and clinical branches, and quantifies cross-modal disagreement using Jensen-Shannon divergence.

In ADNI, subject-level splits were used to prevent leakage across longitudinal visits. AIBL was used as an external validation cohort with the same final feature space of 24 shared ROI residuals and clinical scale features with missingness masks. The performance-oriented CUED-AD variant achieved strong external AD-vs-NC classification on AIBL (ACC 0.984, AUC 0.997, F1 0.920). Natural MRI-clinical conflict in AIBL was associated with a marked increase in total uncertainty (18.01 vs 7.50), and uncertainty-based conflict detection reached an AUC of approximately 0.753. CUED-AD therefore reframes Alzheimer's disease AI from pure classification toward reliability-aware decision support: when evidence agrees, it predicts; when evidence conflicts, it warns and explains.

## Introduction

### Clinical Motivation

Alzheimer's disease assessment often combines structural MRI and cognitive scales. These modalities are complementary, but they do not always agree. Some patients show medial temporal atrophy while retaining near-normal cognitive scores, whereas others show abnormal cognitive scores with less clear structural evidence. These cases are clinically important because they may represent cognitive reserve, atypical disease, comorbidity, or data-quality issues.

### Gap

Most AI systems report a single class probability. This is insufficient for clinical review because a highly confident prediction may still be unreliable when evidence sources disagree.

### Contribution

CUED-AD addresses this reliability gap by combining causal ROI modeling, Bayesian modality-specific classifiers, cross-modal disagreement detection, and counterfactual explanation.

## Methods

### Cohorts and Leakage-Free Splitting

Describe ADNI subject-level train/validation/test splitting and AIBL external validation. Report ADNI AD/NC sample and subject counts, AIBL external sample and subject counts, MCI retention but exclusion from the primary AD-vs-NC task, and final 24-ROI common feature space.

### MRI ROI Feature Construction

Describe FreeSurfer/FastSurfer or cache-seg-derived ROI features, ICV/age/sex residualization, final 24 AD-relevant ROI residuals, and current FastSurfer recovery status as a limitation or updated analysis if complete.

### Stage 1: Structural Causal Model

Describe prior-constrained SCM over AD-relevant brain regions, structural residual extraction, and counterfactual intervention support.

### Stage 2: Bayesian Single-Modality Branches

Describe MRI branch using SCM residuals and clinical branch using MMSE, CDR-SB, ADAS11, ADAS13 plus missingness masks. Report MC Dropout and uncertainty estimates.

### Stage 3: Cross-Modal Disagreement and Fusion

Define MRI-clinical Jensen-Shannon divergence, total uncertainty, conflict regularization, and final fusion.

### Conflict Sample Definition

Define natural conflict and synthetic conflict. State that ADNI conflict analysis is exploratory due to small AD/NC conflict count, whereas AIBL natural conflict is the primary external conflict validation.

## Results

### Leakage-Free Cohort Construction

Report subject-level split, no PTID overlap, diagnosis distribution, missingness, and final feature alignment.

### External Classification Performance

Use Figure 2 and report CUED-AD-Perf vs Clinical Logistic, XGBoost, LightGBM, CatBoost, MRI+Clinical HGB, MRI-only BNN, and Clinical-only BNN.

### Cross-Modal Conflict Detection

Use Figure 3. Report AIBL natural conflict uncertainty and JS divergence differences, ROC AUC, and rejection curve.

### Ablation and Sensitivity Analyses

Report No clinical, No SCM, No conflict regularization, missingness mask ablation, MMSE+CDRSB-only, complete-case, and synthetic conflict sensitivity.

### Counterfactual Explanation

Use Figure 4 once generated. Report hippocampal and entorhinal interventions and uncertainty trajectories.

### Clinician-Facing Prototype

Use Figure 5 once generated. Emphasize retrospective decision-support workflow, not autonomous diagnosis.

## Discussion

This study reframes Alzheimer's disease AI from a single-output classification problem into a reliability-aware multimodal decision-support problem. While cognitive scales can be highly discriminative for AD-vs-NC classification, clinical deployment requires more than high accuracy: models must recognize when imaging and cognitive evidence are inconsistent, quantify the uncertainty induced by this inconsistency, and provide interpretable signals that support human review.

### Strengths

- Subject-level leakage-free ADNI splitting.
- AIBL external validation.
- Strong clinical baselines.
- Explicit conflict detection.
- Causal counterfactual explanation.
- Clinician-facing prototype.

### Limitations

- Primary task is AD vs NC, not full MCI staging or differential diagnosis.
- ADNI MRI features currently include cache-seg fallback until full FastSurfer recovery is incorporated.
- ADNI internal conflict sample count is small; AIBL is the stronger conflict evidence.
- The system has not been prospectively validated.

## Figures

- Figure 1: `artifacts/manuscript_packaging_figures/figure1_cued_ad_system_overview.png`
- Figure 2: `artifacts/manuscript_packaging_figures/figure2_external_performance_baselines.png`
- Figure 3: `artifacts/manuscript_packaging_figures/figure3_conflict_uncertainty_aibl.png`
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure1_system()
    figure2_performance()
    figure3_conflict()
    manuscript_skeleton()
    print(f"generated assets in {OUT}")


if __name__ == "__main__":
    main()
