#!/usr/bin/env python3
"""npj Digital Medicine-style analysis on a per-subject prediction NPZ.

Outputs into ``--out_dir``:
    metrics.json          headline numbers (BAcc, macro-AUC, per-class
                          sens/spec/PPV/NPV at argmax threshold, Brier).
    confusion_matrix.csv  3x3 raw counts.
    confusion_matrix.png  heatmap with annotations.
    calibration.png       reliability diagram per class + multiclass Brier.
    decision_curve.png    net-benefit curves for "AD" detection across
                          probability thresholds.
    subgroup_table.csv    BAcc / macro-AUC stratified by age band & sex.
    error_audit.csv       top-K mis-classified subjects (sorted by
                          confidence in the wrong class).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CLASS_NAMES = ("CN", "MCI", "AD")


# --------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="Per-subject prediction NPZ produced by predict_to_npz.py")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--top_k_errors", type=int, default=20)
    return p.parse_args()


# --------------------------------------------------------------------- core
def per_class_confusion(pred: np.ndarray, lab: np.ndarray, c: int):
    pos = lab == c
    pp = pred == c
    tp = int((pos & pp).sum())
    fn = int((pos & ~pp).sum())
    fp = int((~pos & pp).sum())
    tn = int((~pos & ~pp).sum())
    return tp, fp, tn, fn


def headline_metrics(prob: np.ndarray, lab: np.ndarray) -> Dict:
    pred = prob.argmax(-1)
    out: Dict[str, object] = {"N": int(lab.size)}

    # Balanced accuracy + per-class recall/precision/f1
    recalls, specs, ppvs, npvs, supports = [], [], [], [], []
    per_class = {}
    for c in range(3):
        tp, fp, tn, fn = per_class_confusion(pred, lab, c)
        sup = int(tp + fn)
        rec = tp / sup if sup else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        f1 = 2 * ppv * rec / (ppv + rec) if (ppv and rec and not np.isnan(ppv) and not np.isnan(rec)) else float("nan")
        recalls.append(rec); specs.append(spec); ppvs.append(ppv); npvs.append(npv); supports.append(sup)
        per_class[CLASS_NAMES[c]] = {
            "support": sup, "sensitivity_recall": rec, "specificity": spec,
            "PPV_precision": ppv, "NPV": npv, "F1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        }

    out["accuracy"] = float((pred == lab).mean())
    out["balanced_accuracy"] = float(np.nanmean(recalls))
    out["per_class"] = per_class

    # Multiclass Brier (Sec 1 of Brier 1950 / sklearn impl): mean sq error
    onehot = np.zeros_like(prob)
    onehot[np.arange(lab.size), lab] = 1.0
    out["brier_multiclass"] = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))

    # Macro-AUC (one-vs-rest) implemented manually so we don't need sklearn
    auc_per_class = {}
    for c in range(3):
        y = (lab == c).astype(np.int64)
        s = prob[:, c]
        auc_per_class[CLASS_NAMES[c]] = roc_auc_score_simple(y, s)
    out["macro_auc"] = float(np.nanmean(list(auc_per_class.values())))
    out["per_class_auc"] = auc_per_class
    return out


def roc_auc_score_simple(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney U based AUC, robust to ties."""
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1)
    # average ranks over ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sum_per_group = np.bincount(inv, ranks)
    avg = sum_per_group / counts
    ranks = avg[inv]
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def confusion_matrix(pred: np.ndarray, lab: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(lab, pred):
        cm[t, p] += 1
    return cm


# ------------------------------------------------------------------ plotting
def save_confusion(cm: np.ndarray, out_png: Path) -> None:
    from _plot_style import apply as _style
    _style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def calibration_curve(prob_c: np.ndarray, y_c: np.ndarray, n_bins: int = 10):
    """Reliability bins: returns (bin_centers, observed_freq, support)."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, freqs, sup = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob_c >= lo) & (prob_c < hi if i < n_bins - 1 else prob_c <= hi)
        if mask.sum() == 0:
            continue
        centers.append(float(prob_c[mask].mean()))
        freqs.append(float(y_c[mask].mean()))
        sup.append(int(mask.sum()))
    return np.array(centers), np.array(freqs), np.array(sup)


def save_calibration(prob: np.ndarray, lab: np.ndarray, out_png: Path) -> Dict:
    from _plot_style import apply as _style
    _style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    out_brier = {}
    for c in range(3):
        ax = axes[c]
        prob_c = prob[:, c]
        y_c = (lab == c).astype(np.float64)
        cx, cy, _ = calibration_curve(prob_c, y_c, n_bins=10)
        if cx.size:
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.plot(cx, cy, "o-", color="C0")
        b = float(np.mean((prob_c - y_c) ** 2))
        out_brier[CLASS_NAMES[c]] = b
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{CLASS_NAMES[c]}  Brier={b:.3f}", fontsize=10)
        ax.set_xlabel("Predicted prob")
        if c == 0:
            ax.set_ylabel("Observed freq")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_brier


def save_decision_curve(prob_ad: np.ndarray, y_ad: np.ndarray, out_png: Path) -> None:
    """Decision curve analysis for the AD-vs-rest task."""
    from _plot_style import apply as _style
    _style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = y_ad.size
    pi = float(y_ad.mean())
    thresholds = np.linspace(0.01, 0.99, 99)
    nb_model, nb_all, nb_none = [], [], []
    for t in thresholds:
        pp = prob_ad >= t
        tp = int((pp & (y_ad == 1)).sum())
        fp = int((pp & (y_ad == 0)).sum())
        odds = t / (1 - t) if (1 - t) > 0 else float("inf")
        nb_model.append(tp / n - (fp / n) * odds)
        nb_all.append(pi - (1 - pi) * odds)
        nb_none.append(0.0)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(thresholds, nb_model, label="A2C-NODE", color="C0")
    ax.plot(thresholds, nb_all, label="treat all", linestyle="--", color="gray")
    ax.plot(thresholds, nb_none, label="treat none", linestyle=":", color="black")
    ax.set_xlabel("Threshold prob"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve (AD vs rest)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(min(nb_model + nb_all + nb_none) - 0.02, max(nb_model + nb_all) + 0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ subgroup
def subgroup_table(prob: np.ndarray, lab: np.ndarray, age: np.ndarray,
                   sex: np.ndarray) -> List[Dict]:
    rows = []
    def block(name: str, mask: np.ndarray):
        if mask.sum() < 5:
            return
        m = headline_metrics(prob[mask], lab[mask])
        rows.append({
            "subgroup": name,
            "N": int(mask.sum()),
            "BAcc": round(m["balanced_accuracy"], 3),
            "Macro_AUC": round(m["macro_auc"], 3),
            "Brier": round(m["brier_multiclass"], 3),
            "AD_sens": round(m["per_class"]["AD"]["sensitivity_recall"], 3),
            "AD_spec": round(m["per_class"]["AD"]["specificity"], 3),
        })
    block("Overall", np.ones_like(lab, dtype=bool))
    if not np.all(np.isnan(age)):
        block("Age <= 70", (age <= 70) & ~np.isnan(age))
        block("Age 71-79", (age > 70) & (age <= 79) & ~np.isnan(age))
        block("Age >= 80", (age >= 80) & ~np.isnan(age))
    if not np.all(np.isnan(sex)):
        block("Male",   sex == 1)
        block("Female", sex == 0)
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    import csv as _csv
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ main
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.input, allow_pickle=True)
    prob = d["prob"].astype(np.float64)
    lab = d["label"].astype(np.int64)
    ptid = d["ptid"]
    age = d["age"].astype(np.float64)
    sex = d["sex"].astype(np.float64)

    # 1) Headline metrics
    metrics = headline_metrics(prob, lab)
    metrics["input"] = str(args.input)

    # 2) Confusion matrix
    cm = confusion_matrix(prob.argmax(-1), lab)
    np.savetxt(args.out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")
    save_confusion(cm, args.out_dir / "confusion_matrix.png")

    # 3) Calibration
    brier_per_class = save_calibration(prob, lab, args.out_dir / "calibration.png")
    metrics["brier_per_class"] = brier_per_class

    # 4) Decision curve (AD vs rest)
    save_decision_curve(prob[:, 2], (lab == 2).astype(np.int64),
                        args.out_dir / "decision_curve.png")

    # 5) Subgroup table
    sg = subgroup_table(prob, lab, age, sex)
    write_csv(args.out_dir / "subgroup_table.csv", sg)

    # 6) Error audit
    pred = prob.argmax(-1)
    err = np.where(pred != lab)[0]
    err = err[np.argsort(-prob[err, pred[err]])]                        # most confident wrong
    err = err[: args.top_k_errors]
    rows: List[Dict] = []
    for i in err:
        rows.append({
            "ptid": str(ptid[i]),
            "true": CLASS_NAMES[int(lab[i])],
            "pred": CLASS_NAMES[int(pred[i])],
            "p_CN": round(float(prob[i, 0]), 3),
            "p_MCI": round(float(prob[i, 1]), 3),
            "p_AD": round(float(prob[i, 2]), 3),
            "age": round(float(age[i]), 1) if not np.isnan(age[i]) else "",
            "sex": ("M" if sex[i] == 1 else "F") if not np.isnan(sex[i]) else "",
        })
    write_csv(args.out_dir / "error_audit.csv", rows)

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[analyze] BAcc={metrics['balanced_accuracy']:.3f}  "
          f"Macro-AUC={metrics['macro_auc']:.3f}  "
          f"Brier={metrics['brier_multiclass']:.3f}  "
          f"-> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
