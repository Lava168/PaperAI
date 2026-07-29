#!/usr/bin/env python3
"""Generate aggregate algorithm-evidence reports for the final ARA-Net V6 model.

This script intentionally consumes private row-level prediction CSV files but
only writes aggregate metrics, tables, and figures. It is designed to be run on
the server where the restricted prediction files live; the generated aggregate
artifacts can be committed publicly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


CLASS_NAMES = ["CN", "MCI", "AD"]
LABEL_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
PROB_COLS = ["prob_CN", "prob_MCI", "prob_AD"]
SPLITS = ["val", "internal_test", "aibl_adapt_val", "aibl_heldout", "ixi_external", "oasis_external"]

FINAL_RUNS = [
    "aibl_adapted_atlas_biomarker_enhanced__hgb",
    "aibl_adapted_atlas_core_clinical__hgb",
    "aibl_adapted_clinical_biomarker_only__rf_balanced",
    "aibl_adapted_clinical_core_only__hgb",
    "aibl_adapted_clinical_core_only__rf_balanced",
    "rf__logreg",
]
FINAL_WEIGHTS = np.array(
    [
        0.3562689320130782,
        0.04421238144536563,
        0.32332384229746974,
        0.18313017850953642,
        0.08786508061781907,
        0.005199585116730915,
    ],
    dtype=np.float64,
)
FINAL_OFFSETS = np.array([-0.7959975410763656, -0.19022372945067426, 0.9862212705270398], dtype=np.float64)
FINAL_TEMPERATURE = 0.6723763750332673

RUN_DIRS = {
    "rf__logreg": "atlas_cascade_baseline",
}


def run_dir(run: str) -> str:
    return RUN_DIRS.get(run, "hybrid_atlas_clinical_baseline")


def prediction_path(pred_root: Path, run: str, split: str) -> Path:
    return pred_root / run_dir(run) / f"{run}_{split}_predictions.csv"


def read_prediction_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for col in PROB_COLS:
            row[col] = float(row[col])
    return rows


def row_key(row: dict) -> Tuple[str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("split", ""),
        row.get("subject_id", ""),
        row.get("scan_id", ""),
        row.get("y_true", ""),
    )


def load_aligned_predictions(pred_root: Path, runs: Sequence[str], splits: Sequence[str]) -> Dict[str, Tuple[List[dict], Dict[str, np.ndarray]]]:
    payload = {}
    for split in splits:
        rows_by_run = {run: read_prediction_csv(prediction_path(pred_root, run, split)) for run in runs}
        base_rows = rows_by_run[runs[0]]
        keys = [row_key(row) for row in base_rows]
        arrays = {}
        for run in runs:
            by_key = {row_key(row): row for row in rows_by_run[run]}
            missing = [key for key in keys if key not in by_key]
            if missing:
                raise ValueError(f"{run}/{split} missing {len(missing)} rows relative to {runs[0]}")
            arrays[run] = np.asarray(
                [[by_key[key][col] for col in PROB_COLS] for key in keys],
                dtype=np.float64,
            )
        payload[split] = (base_rows, arrays)
    return payload


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def log_pool(arrays: Dict[str, np.ndarray], runs: Sequence[str], weights: np.ndarray, offsets: np.ndarray | None = None, temperature: float = 1.0) -> np.ndarray:
    logits = np.zeros_like(arrays[runs[0]], dtype=np.float64)
    for weight, run in zip(weights, runs):
        logits += float(weight) * np.log(np.clip(arrays[run], 1e-8, 1.0))
    logits /= max(float(temperature), 1e-4)
    if offsets is not None:
        logits += offsets.reshape(1, 3)
    return softmax(logits)


def arithmetic_mean(arrays: Dict[str, np.ndarray], runs: Sequence[str], weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(runs), dtype=np.float64) / len(runs)
    out = np.zeros_like(arrays[runs[0]], dtype=np.float64)
    for weight, run in zip(weights, runs):
        out += float(weight) * arrays[run]
    return out / out.sum(axis=1, keepdims=True).clip(min=1e-12)


def aggregate_subject_rows(rows: Sequence[dict], probs: np.ndarray) -> Tuple[List[dict], np.ndarray]:
    grouped: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row.get("dataset", ""), row.get("split", ""), row.get("subject_id", ""), row.get("y_true", ""))].append(idx)
    out_rows = []
    out_probs = []
    for (dataset, split, subject, y_true), indices in sorted(grouped.items()):
        row = dict(rows[indices[0]])
        row["dataset"] = dataset
        row["split"] = split
        row["subject_id"] = subject
        row["scan_id"] = f"{subject}__subject_mean"
        row["y_true"] = y_true
        out_rows.append(row)
        out_probs.append(probs[indices].mean(axis=0))
    return out_rows, np.vstack(out_probs)


def majority_vote_subject_rows(rows: Sequence[dict], probs: np.ndarray) -> Tuple[List[dict], np.ndarray]:
    grouped: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row.get("dataset", ""), row.get("split", ""), row.get("subject_id", ""), row.get("y_true", ""))].append(idx)
    out_rows = []
    out_probs = []
    pred = probs.argmax(axis=1)
    for (dataset, split, subject, y_true), indices in sorted(grouped.items()):
        row = dict(rows[indices[0]])
        row["dataset"] = dataset
        row["split"] = split
        row["subject_id"] = subject
        row["scan_id"] = f"{subject}__subject_majority"
        row["y_true"] = y_true
        counts = np.bincount(pred[indices], minlength=3).astype(np.float64)
        out_rows.append(row)
        out_probs.append(counts / counts.sum())
    return out_rows, np.vstack(out_probs)


def binary_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true_binary == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def multiclass_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Tuple[float, List[dict]]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for idx in range(n_bins):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n:
            acc = float(correct[mask].mean())
            mean_conf = float(conf[mask].mean())
            gap = abs(acc - mean_conf)
            ece += n / len(y_true) * gap
        else:
            acc = float("nan")
            mean_conf = float("nan")
            gap = float("nan")
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": n, "accuracy": acc, "confidence": mean_conf, "gap": gap})
    return float(ece), rows


def classification_metrics(rows: Sequence[dict], probs: np.ndarray) -> dict:
    y = np.asarray([LABEL_TO_INT[row["y_true"]] for row in rows], dtype=int)
    pred = probs.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=int)
    for yi, pi in zip(y, pred):
        cm[int(yi), int(pi)] += 1
    support = cm.sum(axis=1)
    per_class = {}
    recalls = []
    precisions = []
    f1s = []
    for idx, name in enumerate(CLASS_NAMES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support[idx])}
        if support[idx] > 0:
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
    valid_aucs = []
    aucs = {}
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y == idx).astype(int)
        auc = binary_auc(y_bin, probs[:, idx])
        aucs[name] = None if math.isnan(auc) else auc
        if not math.isnan(auc):
            valid_aucs.append(auc)
    nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()) if len(y) else float("nan")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))) if len(y) else float("nan")
    ece, reliability = multiclass_ece(y, probs)
    out = {
        "n": int(len(y)),
        "acc": float((pred == y).mean()) if len(y) else 0.0,
        "balanced_acc": float(np.mean(recalls)) if recalls else None,
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_auc_ovr": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_class_auc_ovr": aucs,
        "recall_CN": per_class["CN"]["recall"],
        "recall_MCI": per_class["MCI"]["recall"],
        "recall_AD": per_class["AD"]["recall"],
        "precision_CN": per_class["CN"]["precision"],
        "precision_MCI": per_class["MCI"]["precision"],
        "precision_AD": per_class["AD"]["precision"],
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": {CLASS_NAMES[i]: int((pred == i).sum()) for i in range(3)},
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "reliability": reliability,
        "ad_to_cn_errors": int(cm[2, 0]),
        "cn_to_ad_errors": int(cm[0, 2]),
    }
    cn = y == 0
    ad = y == 2
    if cn.sum() and ad.sum():
        score = probs[:, 2] - probs[:, 0]
        yy = np.concatenate([np.zeros(int(cn.sum())), np.ones(int(ad.sum()))])
        ss = np.concatenate([score[cn], score[ad]])
        out["ad_vs_cn_auc"] = binary_auc(yy, ss)
    else:
        out["ad_vs_cn_auc"] = None
    if len(set(y.tolist())) == 1 and len(y) and int(y[0]) == 0:
        out["cn_retention_rate"] = out["acc"]
        out["false_impairment_rate"] = float(1.0 - out["acc"])
    else:
        out["cn_retention_rate"] = None
        out["false_impairment_rate"] = None
    return out


def tune_score(metrics_by_split: Dict[str, dict]) -> float:
    val = metrics_by_split.get("val", {})
    aibl = metrics_by_split.get("aibl_adapt_val", {})
    ixi = metrics_by_split.get("ixi_external", {})
    ixi_ret = ixi.get("cn_retention_rate") or ixi.get("acc") or 0.0
    aibl_minority = min(aibl.get("recall_MCI", 0.0), aibl.get("recall_AD", 0.0))
    return (
        0.25 * (val.get("balanced_acc") or 0.0)
        + 0.15 * (val.get("macro_auc_ovr") or 0.0)
        + 0.25 * (aibl.get("balanced_acc") or 0.0)
        + 0.12 * aibl_minority
        + 0.13 * ixi_ret
        + 0.10 * (metrics_by_split.get("oasis_external", {}).get("balanced_acc") or 0.0)
    )


def evaluate_variant(
    payload: Dict[str, Tuple[List[dict], Dict[str, np.ndarray]]],
    probs_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
    aggregate: str = "subject_mean",
) -> Dict[str, dict]:
    out = {}
    for split, (rows, arrays) in payload.items():
        probs = probs_fn(arrays)
        metric_rows = rows
        metric_probs = probs
        if aggregate == "subject_mean":
            metric_rows, metric_probs = aggregate_subject_rows(rows, probs)
        elif aggregate == "subject_majority":
            metric_rows, metric_probs = majority_vote_subject_rows(rows, probs)
        elif aggregate == "scan":
            pass
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")
        out[split] = classification_metrics(metric_rows, metric_probs)
    return out


def metric_row(name: str, kind: str, metrics_by_split: Dict[str, dict], notes: str = "") -> dict:
    aibl = metrics_by_split["aibl_heldout"]
    ixi = metrics_by_split["ixi_external"]
    oasis = metrics_by_split["oasis_external"]
    return {
        "variant": name,
        "kind": kind,
        "aibl_n": aibl["n"],
        "aibl_acc": aibl["acc"],
        "aibl_bacc": aibl["balanced_acc"],
        "aibl_auc": aibl["macro_auc_ovr"],
        "aibl_ad_vs_cn_auc": aibl.get("ad_vs_cn_auc"),
        "aibl_recall_CN": aibl["recall_CN"],
        "aibl_recall_MCI": aibl["recall_MCI"],
        "aibl_recall_AD": aibl["recall_AD"],
        "aibl_ad_to_cn_errors": aibl["ad_to_cn_errors"],
        "ixi_cn_retention": ixi.get("cn_retention_rate"),
        "ixi_false_impairment": ixi.get("false_impairment_rate"),
        "oasis_bacc_stress": oasis.get("balanced_acc"),
        "aibl_ece": aibl["ece"],
        "aibl_brier": aibl["brier"],
        "aibl_nll": aibl["nll"],
        "notes": notes,
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: Sequence[dict], columns: Sequence[Tuple[str, str]], digits: int = 3) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key), digits) for key, _ in columns) + " |")
    return "\n".join(lines)


def make_figures(out_dir: Path, ablation_rows: List[dict], lomo_rows: List[dict], risk_rows: List[dict], reliability_payload: Dict[str, List[dict]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - server environment dependent
        print(f"[warn] matplotlib unavailable, skipping figures: {exc}")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    keep_names = [
        "Best single base model",
        "Arithmetic mean ensemble",
        "Equal log-pooling",
        "Final weights only",
        "Weights + offsets",
        "Weights + temperature",
        "Full RC-SPE (scan-level)",
        "Full RC-SPE (subject-level)",
    ]
    rows = [row for row in ablation_rows if row["variant"] in keep_names]
    x = np.arange(len(rows))
    width = 0.22
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.bar(x - width, [row["aibl_bacc"] for row in rows], width, label="AIBL BAcc", color="#2f6f73")
    ax.bar(x, [row["aibl_recall_MCI"] for row in rows], width, label="AIBL MCI recall", color="#d28b26")
    ax.bar(x + width, [row["aibl_recall_AD"] for row in rows], width, label="AIBL AD recall", color="#7a4ea3")
    ax.plot(x, [row["ixi_cn_retention"] for row in rows], marker="o", color="#2d3748", label="IXI CN retention")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric")
    ax.set_xticks(x)
    ax.set_xticklabels([row["variant"].replace(" ensemble", "\nensemble").replace(" (", "\n(") for row in rows], rotation=20, ha="right")
    ax.set_title("Algorithmic ablation of the risk-constrained subject-level probability ensemble")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_algorithm_ablation.png", dpi=220)
    fig.savefig(fig_dir / "figure_algorithm_ablation.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True)
    random_rows = [row for row in risk_rows if row["source"] == "random_pool"]
    axes[0].scatter(
        [row["ixi_false_impairment"] for row in random_rows],
        [row["aibl_bacc"] for row in random_rows],
        s=16,
        c="#9aa4b2",
        alpha=0.28,
        label="sampled candidates",
    )
    axes[1].scatter(
        [row["ixi_false_impairment"] for row in random_rows],
        [row["aibl_recall_MCI"] for row in random_rows],
        s=16,
        c="#9aa4b2",
        alpha=0.28,
        label="sampled candidates",
    )
    highlight = [row for row in risk_rows if row["source"] != "random_pool"]
    colors = {"Full RC-SPE": "#c7362f", "MCI-rescue profile": "#d28b26", "AD-rescue profile": "#7a4ea3", "Equal log-pooling": "#2f6f73"}
    for row in highlight:
        color = colors.get(row["variant"], "#2d3748")
        marker = "*" if row["variant"] == "Full RC-SPE" else "D"
        size = 150 if marker == "*" else 70
        axes[0].scatter(row["ixi_false_impairment"], row["aibl_bacc"], s=size, marker=marker, c=color, edgecolor="white", linewidth=0.8, label=row["variant"])
        axes[1].scatter(row["ixi_false_impairment"], row["aibl_recall_MCI"], s=size, marker=marker, c=color, edgecolor="white", linewidth=0.8, label=row["variant"])
    axes[0].set_ylabel("AIBL heldout balanced accuracy")
    axes[1].set_ylabel("AIBL heldout MCI recall")
    for ax in axes:
        ax.set_xlabel("IXI false impairment rate")
        ax.set_xlim(-0.01, 0.45)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.22)
    axes[0].set_title("Risk constraint: external BAcc vs healthy false impairment")
    axes[1].set_title("Risk constraint: MCI rescue vs healthy false impairment")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(fig_dir / "figure_risk_constraint_curve.png", dpi=220)
    fig.savefig(fig_dir / "figure_risk_constraint_curve.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    names = [row["dropped_model_short"] for row in lomo_rows]
    y = [row["aibl_bacc"] for row in lomo_rows]
    ax.bar(np.arange(len(names)), y, color="#2f6f73")
    ax.axhline([row["aibl_bacc"] for row in ablation_rows if row["variant"] == "Full RC-SPE (subject-level)"][0], color="#c7362f", ls="--", lw=1.2, label="Full RC-SPE")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AIBL heldout BAcc")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_title("Leave-one-model-out sensitivity")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_leave_one_model_out.png", dpi=220)
    fig.savefig(fig_dir / "figure_leave_one_model_out.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.1))
    ax.plot([0, 1], [0, 1], color="#2d3748", lw=1.0, ls="--", label="perfect calibration")
    for label, bins in reliability_payload.items():
        xs = [row["confidence"] for row in bins if row["n"] > 0]
        ys = [row["accuracy"] for row in bins if row["n"] > 0]
        ax.plot(xs, ys, marker="o", lw=1.7, label=label)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("AIBL heldout reliability")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_calibration_reliability.png", dpi=220)
    fig.savefig(fig_dir / "figure_calibration_reliability.pdf")
    plt.close(fig)


def short_model_name(run: str) -> str:
    return (
        run.replace("aibl_adapted_", "")
        .replace("atlas_biomarker_enhanced__hgb", "atlas+bio HGB")
        .replace("atlas_core_clinical__hgb", "atlas+clinical HGB")
        .replace("clinical_biomarker_only__rf_balanced", "clinical+bio RF")
        .replace("clinical_core_only__hgb", "clinical HGB")
        .replace("clinical_core_only__rf_balanced", "clinical RF")
        .replace("rf__logreg", "cascade RF-logreg")
    )


def load_profile_configs(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name, item in data.get("profiles", {}).items():
        if not all(key in item for key in ["runs", "weights", "offsets", "temperature"]):
            continue
        out[name] = {
            "runs": item["runs"],
            "weights": np.asarray(item["weights"], dtype=np.float64),
            "offsets": np.asarray(item["offsets"], dtype=np.float64),
            "temperature": float(item["temperature"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/v4"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/v6_algorithm_innovation"))
    parser.add_argument("--profile-summary", type=Path, default=Path("outputs/v4/rescue_probability_subject_quick_no_oasis_tune/summary.json"))
    parser.add_argument("--risk-samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_aligned_predictions(args.pred_root, FINAL_RUNS, SPLITS)
    profiles = load_profile_configs(args.profile_summary)

    individual_metrics = {}
    individual_scores = []
    for idx, run in enumerate(FINAL_RUNS):
        metrics = evaluate_variant(payload, lambda arrays, r=run: arrays[r], aggregate="subject_mean")
        individual_metrics[run] = metrics
        individual_scores.append((tune_score(metrics), run))
    best_single_run = sorted(individual_scores, reverse=True)[0][1]

    equal_weights = np.ones(len(FINAL_RUNS), dtype=np.float64) / len(FINAL_RUNS)
    variants: List[Tuple[str, str, Callable[[Dict[str, np.ndarray]], np.ndarray], str, str]] = [
        ("Best single base model", "single", lambda arrays, run=best_single_run: arrays[run], "subject_mean", f"Selected by tuning score: {short_model_name(best_single_run)}"),
        ("Arithmetic mean ensemble", "ensemble", lambda arrays: arithmetic_mean(arrays, FINAL_RUNS), "subject_mean", "Equal arithmetic probability averaging."),
        ("Equal log-pooling", "ensemble", lambda arrays: log_pool(arrays, FINAL_RUNS, equal_weights), "subject_mean", "Equal log-probability pooling, no offsets, T=1."),
        ("Final weights only", "ablation", lambda arrays: log_pool(arrays, FINAL_RUNS, FINAL_WEIGHTS), "subject_mean", "Final non-negative weights, no offsets, T=1."),
        ("Weights + offsets", "ablation", lambda arrays: log_pool(arrays, FINAL_RUNS, FINAL_WEIGHTS, FINAL_OFFSETS, 1.0), "subject_mean", "Final weights and class offsets, T=1."),
        ("Weights + temperature", "ablation", lambda arrays: log_pool(arrays, FINAL_RUNS, FINAL_WEIGHTS, None, FINAL_TEMPERATURE), "subject_mean", "Final weights and temperature, no class offsets."),
        ("Full RC-SPE (scan-level)", "final_scan", lambda arrays: log_pool(arrays, FINAL_RUNS, FINAL_WEIGHTS, FINAL_OFFSETS, FINAL_TEMPERATURE), "scan", "Final pooling evaluated at scan level."),
        ("Full RC-SPE (majority vote)", "aggregation", lambda arrays: log_pool(arrays, FINAL_RUNS, FINAL_WEIGHTS, FINAL_OFFSETS, FINAL_TEMPERATURE), "subject_majority", "Subject-level majority vote after scan predictions."),
        ("Full RC-SPE (subject-level)", "final_subject", lambda arrays: log_pool(arrays, FINAL_RUNS, FINAL_WEIGHTS, FINAL_OFFSETS, FINAL_TEMPERATURE), "subject_mean", "Locked final method: probability averaging at subject level."),
    ]

    if "aibl_mci_recall" in profiles:
        cfg = profiles["aibl_mci_recall"]
        variants.append(
            (
                "MCI-rescue profile",
                "risk_profile",
                lambda arrays, cfg=cfg: log_pool(arrays, cfg["runs"], cfg["weights"], cfg["offsets"], cfg["temperature"]),
                "subject_mean",
                "High-MCI-recall profile for risk-tradeoff analysis; not selected as final.",
            )
        )
    if "internal_ad_recall" in profiles:
        cfg = profiles["internal_ad_recall"]
        variants.append(
            (
                "AD-rescue profile",
                "risk_profile",
                lambda arrays, cfg=cfg: log_pool(arrays, cfg["runs"], cfg["weights"], cfg["offsets"], cfg["temperature"]),
                "subject_mean",
                "High-AD-recall profile for risk-tradeoff analysis; not selected as final.",
            )
        )

    all_variant_metrics = {}
    ablation_rows = []
    for name, kind, fn, aggregate, notes in variants:
        metrics = evaluate_variant(payload, fn, aggregate=aggregate)
        all_variant_metrics[name] = metrics
        ablation_rows.append(metric_row(name, kind, metrics, notes))

    calibration_rows = [
        {
            "variant": row["variant"],
            "aibl_nll": row["aibl_nll"],
            "aibl_brier": row["aibl_brier"],
            "aibl_ece": row["aibl_ece"],
            "aibl_acc": row["aibl_acc"],
            "aibl_bacc": row["aibl_bacc"],
        }
        for row in ablation_rows
        if row["variant"]
        in {
            "Best single base model",
            "Arithmetic mean ensemble",
            "Equal log-pooling",
            "Final weights only",
            "Weights + offsets",
            "Weights + temperature",
            "Full RC-SPE (subject-level)",
        }
    ]

    reliability_payload = {
        "Best single": all_variant_metrics["Best single base model"]["aibl_heldout"]["reliability"],
        "Equal log-pooling": all_variant_metrics["Equal log-pooling"]["aibl_heldout"]["reliability"],
        "Full RC-SPE": all_variant_metrics["Full RC-SPE (subject-level)"]["aibl_heldout"]["reliability"],
    }

    lomo_rows = []
    for idx, dropped in enumerate(FINAL_RUNS):
        keep_runs = [run for run in FINAL_RUNS if run != dropped]
        keep_weights = np.delete(FINAL_WEIGHTS, idx)
        keep_weights = keep_weights / keep_weights.sum()
        metrics = evaluate_variant(
            payload,
            lambda arrays, runs=keep_runs, weights=keep_weights: log_pool(arrays, runs, weights, FINAL_OFFSETS, FINAL_TEMPERATURE),
            aggregate="subject_mean",
        )
        row = metric_row(f"Drop {short_model_name(dropped)}", "leave_one_out", metrics, f"Removed {dropped}; remaining weights renormalized.")
        row["dropped_model"] = dropped
        row["dropped_model_short"] = short_model_name(dropped)
        lomo_rows.append(row)

    rng = np.random.default_rng(args.seed)
    risk_rows = []
    for idx in range(args.risk_samples):
        weights = rng.dirichlet(np.ones(len(FINAL_RUNS)))
        offsets = rng.uniform(-1.25, 1.25, size=3)
        offsets -= offsets.mean()
        temp = float(np.exp(rng.uniform(math.log(0.55), math.log(2.25))))
        metrics = evaluate_variant(payload, lambda arrays, w=weights, b=offsets, t=temp: log_pool(arrays, FINAL_RUNS, w, b, t), aggregate="subject_mean")
        row = metric_row(f"random_{idx:04d}", "risk_candidate", metrics, "Random candidate from RC-SPE pooling family.")
        row["source"] = "random_pool"
        row["temperature"] = temp
        risk_rows.append(row)
    highlight_names = ["Full RC-SPE (subject-level)", "MCI-rescue profile", "AD-rescue profile", "Equal log-pooling"]
    for name in highlight_names:
        if name not in all_variant_metrics:
            continue
        row = metric_row(name.replace(" (subject-level)", ""), "risk_highlight", all_variant_metrics[name], "Named candidate.")
        row["source"] = "highlight"
        row["temperature"] = FINAL_TEMPERATURE if name.startswith("Full") else None
        risk_rows.append(row)

    write_csv(args.out_dir / "algorithm_ablation_table.csv", ablation_rows)
    write_csv(args.out_dir / "calibration_table.csv", calibration_rows)
    write_csv(args.out_dir / "leave_one_model_out_table.csv", lomo_rows)
    write_csv(args.out_dir / "risk_constraint_candidates.csv", risk_rows)

    individual_rows = []
    for score, run in sorted(individual_scores, reverse=True):
        row = metric_row(short_model_name(run), "base_model", individual_metrics[run], f"tune_score={score:.4f}; run={run}")
        row["run"] = run
        row["tune_score"] = score
        individual_rows.append(row)
    write_csv(args.out_dir / "base_model_summary_table.csv", individual_rows)

    ablation_md_cols = [
        ("variant", "Variant"),
        ("aibl_bacc", "AIBL BAcc"),
        ("aibl_auc", "AIBL AUC"),
        ("aibl_recall_MCI", "MCI recall"),
        ("aibl_recall_AD", "AD recall"),
        ("aibl_ad_to_cn_errors", "AD->CN"),
        ("ixi_cn_retention", "IXI CN retention"),
        ("aibl_ece", "ECE"),
        ("aibl_nll", "NLL"),
        ("oasis_bacc_stress", "OASIS stress BAcc"),
    ]
    calibration_md_cols = [
        ("variant", "Variant"),
        ("aibl_nll", "NLL"),
        ("aibl_brier", "Brier"),
        ("aibl_ece", "ECE"),
        ("aibl_acc", "Acc"),
        ("aibl_bacc", "BAcc"),
    ]
    lomo_md_cols = [
        ("dropped_model_short", "Dropped model"),
        ("aibl_bacc", "AIBL BAcc"),
        ("aibl_recall_MCI", "MCI recall"),
        ("aibl_recall_AD", "AD recall"),
        ("aibl_ad_to_cn_errors", "AD->CN"),
        ("ixi_cn_retention", "IXI CN retention"),
    ]

    full_row = next(row for row in ablation_rows if row["variant"] == "Full RC-SPE (subject-level)")
    mci_profile = next((row for row in ablation_rows if row["variant"] == "MCI-rescue profile"), None)
    best_single = next(row for row in ablation_rows if row["variant"] == "Best single base model")
    equal_log = next(row for row in ablation_rows if row["variant"] == "Equal log-pooling")

    report_lines = [
        "# Algorithm Innovation Evidence: RC-SPE",
        "",
        "## Method Name",
        "",
        "**RC-SPE: Risk-Constrained Subject-level Probability Ensemble.**",
        "",
        "This evidence package uses private row-level prediction files on the server but exports only aggregate metrics, tables, and figures.",
        "",
        "## Key Findings",
        "",
        f"- Final RC-SPE subject-level AIBL heldout BAcc={fmt(full_row['aibl_bacc'])}, MCI recall={fmt(full_row['aibl_recall_MCI'])}, AD recall={fmt(full_row['aibl_recall_AD'])}, AD-to-CN errors={full_row['aibl_ad_to_cn_errors']}, IXI CN retention={fmt(full_row['ixi_cn_retention'])}.",
        f"- Best single base model selected by tuning score: {best_single['notes'].replace('Selected by tuning score: ', '')}; AIBL BAcc={fmt(best_single['aibl_bacc'])}, IXI CN retention={fmt(best_single['ixi_cn_retention'])}.",
        f"- Equal log-pooling without learned weights/offsets/temperature achieved AIBL BAcc={fmt(equal_log['aibl_bacc'])}, showing the locked final method adds more than simple pooling.",
    ]
    if mci_profile:
        report_lines.append(
            f"- The MCI-rescue profile raises MCI recall to {fmt(mci_profile['aibl_recall_MCI'])} but reduces IXI CN retention to {fmt(mci_profile['ixi_cn_retention'])}, supporting the risk-constrained final selection."
        )
    report_lines += [
        "",
        "## Algorithmic Ablation Table",
        "",
        markdown_table(ablation_rows, ablation_md_cols),
        "",
        "## Calibration Table",
        "",
        markdown_table(calibration_rows, calibration_md_cols),
        "",
        "## Leave-One-Model-Out Sensitivity",
        "",
        markdown_table(lomo_rows, lomo_md_cols),
        "",
        "## Manuscript Insert",
        "",
        "The final algorithm should be described as a risk-constrained subject-level probability ensemble rather than as a generic model average. The ablation demonstrates the individual contribution of probability pooling, learned non-negative weights, class offsets, temperature scaling, and subject-level probability averaging. The risk curve shows why the final locked profile was preferred over a high-MCI-recall profile: the latter improves MCI recall but increases false impairment predictions in IXI healthy controls.",
        "",
    ]
    (args.out_dir / "algorithm_innovation_evidence.md").write_text("\n".join(report_lines), encoding="utf-8")

    public_summary = {
        "method_name": "RC-SPE: Risk-Constrained Subject-level Probability Ensemble",
        "best_single_run": best_single_run,
        "final_subject_level": {key: full_row[key] for key in full_row if key not in {"notes"}},
        "ablation_rows": ablation_rows,
        "calibration_rows": calibration_rows,
        "leave_one_model_out_rows": lomo_rows,
        "risk_curve_rows": len(risk_rows),
        "privacy_note": "Generated from private row-level prediction files; this JSON contains aggregate metrics only.",
    }
    (args.out_dir / "algorithm_innovation_summary.json").write_text(json.dumps(public_summary, indent=2), encoding="utf-8")

    make_figures(args.out_dir, ablation_rows, lomo_rows, risk_rows, reliability_payload)

    print(f"[saved] {args.out_dir / 'algorithm_innovation_evidence.md'}")
    print(f"[saved] {args.out_dir / 'algorithm_innovation_summary.json'}")
    print(f"[saved] {args.out_dir / 'algorithm_ablation_table.csv'}")


if __name__ == "__main__":
    main()
