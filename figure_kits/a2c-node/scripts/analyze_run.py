#!/usr/bin/env python3
"""Post-training analysis for an A2C-NODE checkpoint.

Reproduces the headline numbers and biological-validation tables you would
report in a TMI / MICCAI submission:

    1. Cohort-level test metrics (BAcc, macro-AUC, per-class accuracy).
    2. Future-time forecast quality at every observed visit horizon.
    3. Counterfactual ATE table: do(h_k(t)=h_k(0)) for clinically-named
       regions (left/right hippocampus / amygdala / lateral-ventricle, +
       bilateral pairs) on the *same* test cohort.
    4. Per-region attention vector by diagnosis class, saved as CSV +
       a small bar plot.

Usage (run after `runs/main/best.pth` exists)::

    python -m scripts.analyze_run \
        --checkpoint runs/main/best.pth \
        --data_root  "/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/chapter1_foundation/sample_data/cache_real" \
        --output_dir runs/main/analysis \
        --gpu 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, f1_score)

from a2c_node.data import (LegacyCacheRealDataset, LongitudinalNPZDataset,
                           longitudinal_collate)
from a2c_node.models.a2c_node import A2CNode
from a2c_node.models.intervention import cohort_ate


# --------------------------------------------------------------------------- #
# Region naming                                                               #
# --------------------------------------------------------------------------- #
# AnatomyMaskExtractor pools by remapped IDs 1..21 (token index = id - 1).
# Remap is _FS_LABELS = (0, 2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
#                            41, 42, 43, 49, 50, 51, 52, 53, 54, 58)
# So FreeSurfer label 17 (Left-Hippocampus) ends up at token index 8, etc.
REGION_NAMES = [
    "L-WM", "L-Cortex", "L-Lat-Vent", "L-Thalamus", "L-Caudate", "L-Putamen",
    "L-Pallidum", "Brain-Stem", "L-Hippocampus", "L-Amygdala", "L-Accumbens",
    "R-WM", "R-Cortex", "R-Lat-Vent", "R-Thalamus", "R-Caudate", "R-Putamen",
    "R-Pallidum", "R-Hippocampus", "R-Amygdala", "R-Accumbens",
]
assert len(REGION_NAMES) == 21

NAME_TO_TOKEN = {name: i for i, name in enumerate(REGION_NAMES)}

# Clinically motivated intervention groups for the ATE table.
INTERVENTION_GROUPS: List[tuple] = [
    ("L-Hippocampus",          (NAME_TO_TOKEN["L-Hippocampus"],)),
    ("R-Hippocampus",          (NAME_TO_TOKEN["R-Hippocampus"],)),
    ("Bilateral-Hippocampus",  (NAME_TO_TOKEN["L-Hippocampus"],
                                NAME_TO_TOKEN["R-Hippocampus"])),
    ("L-Amygdala",             (NAME_TO_TOKEN["L-Amygdala"],)),
    ("R-Amygdala",             (NAME_TO_TOKEN["R-Amygdala"],)),
    ("Bilateral-Amygdala",     (NAME_TO_TOKEN["L-Amygdala"],
                                NAME_TO_TOKEN["R-Amygdala"])),
    ("Bilateral-Lat-Vent",     (NAME_TO_TOKEN["L-Lat-Vent"],
                                NAME_TO_TOKEN["R-Lat-Vent"])),
    ("AD-key (Hipp+Amyg+Vent)", tuple(
        NAME_TO_TOKEN[n] for n in
        ["L-Hippocampus", "R-Hippocampus",
         "L-Amygdala", "R-Amygdala",
         "L-Lat-Vent", "R-Lat-Vent"]
    )),
]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(model: A2CNode, loader: DataLoader,
                        device: torch.device) -> Dict[str, np.ndarray]:
    """Run factual forecast and collect (label, pred, prob, attention)."""
    model.eval()
    all_y, all_p, all_prob, all_attn = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        time_ = batch["time"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        last_idx = valid.float().cumsum(1).argmax(1)
        target_t = time_.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        t_eval = target_t.unique().sort().values

        out = model(image, mask, time_, valid, target_time=t_eval)
        head = out["head_out"]
        logits = head["logits"]
        prob = torch.softmax(logits, dim=-1)
        attn = head["attention"]                         # (B, K)

        labels = batch["label"].to(device).gather(1, last_idx.unsqueeze(1)).squeeze(1)
        all_y.append(labels.cpu().numpy())
        all_p.append(logits.argmax(-1).cpu().numpy())
        all_prob.append(prob.cpu().numpy())
        all_attn.append(attn.cpu().numpy())

    return {
        "y_true": np.concatenate(all_y),
        "y_pred": np.concatenate(all_p),
        "y_prob": np.concatenate(all_prob),
        "attention": np.concatenate(all_attn),           # (N, K)
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, n_classes: int = 3) -> Dict[str, float]:
    out: Dict[str, float] = {
        "accuracy":         float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":         float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        out["macro_auc"] = float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro",
            labels=list(range(n_classes))))
    except ValueError:
        out["macro_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out["confusion_matrix"] = cm.tolist()
    for c in range(n_classes):
        m = y_true == c
        out[f"recall_class{c}"] = float((y_pred[m] == c).mean()) if m.any() else 0.0
    return out


@torch.no_grad()
def cohort_ate_table(model: A2CNode, loader: DataLoader,
                     device: torch.device,
                     groups: Sequence[tuple]) -> List[Dict[str, float]]:
    """ATE per intervention group across the entire test cohort.

    For each region group, runs do(h_k(t)=h_k(0)) and reports the average
    change in (i) softmax probabilities and (ii) cognitive prediction.
    """
    rows: List[Dict[str, float]] = []
    model.eval()

    for name, indices in groups:
        agg_prob_delta: List[np.ndarray] = []
        agg_cog_delta: List[float] = []

        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            time_ = batch["time"].to(device)
            valid = batch["valid"].to(device)

            last_idx = valid.float().cumsum(1).argmax(1)
            target_t = time_.gather(1, last_idx.unsqueeze(1)).squeeze(1)
            t_eval = target_t.unique().sort().values

            graph = model.extract_baseline_graph(image, mask, time_, valid)
            h0 = graph["nodes"]
            A = graph["adjacency"]
            valid_nodes = graph["valid"]

            intervention = {
                "node_indices": list(indices),
                "values": h0[:, list(indices)].clone(),
            }
            cf_out = model.intervention(h0, A, t_eval,
                                        intervention=intervention,
                                        valid_mask=valid_nodes)
            if "ate_probs" in cf_out:
                agg_prob_delta.append(cf_out["ate_probs"].cpu().numpy())
            if "ate_cog" in cf_out:
                agg_cog_delta.extend(cf_out["ate_cog"].cpu().numpy().tolist())

        prob_delta = np.concatenate(agg_prob_delta) if agg_prob_delta else np.zeros((0, 3))
        row = {
            "region_group": name,
            "indices": list(indices),
            "delta_p_CN":  float(prob_delta[:, 0].mean()) if len(prob_delta) else float("nan"),
            "delta_p_MCI": float(prob_delta[:, 1].mean()) if len(prob_delta) else float("nan"),
            "delta_p_AD":  float(prob_delta[:, 2].mean()) if len(prob_delta) else float("nan"),
            "delta_cog_mean": float(np.nanmean(agg_cog_delta)) if agg_cog_delta else float("nan"),
            "n_subjects": int(len(prob_delta)),
        }
        rows.append(row)
    return rows


def attention_by_class(attn: np.ndarray, y_true: np.ndarray,
                       n_classes: int = 3) -> np.ndarray:
    out = np.zeros((n_classes, attn.shape[1]))
    for c in range(n_classes):
        mask = y_true == c
        if mask.any():
            out[c] = attn[mask].mean(axis=0)
    return out


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_attention_by_class(attn_by_c: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = attn_by_c.shape[1]
    classes = ["CN", "MCI", "AD"]
    x = np.arange(K)
    width = 0.27
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    for i, c in enumerate(classes):
        ax.bar(x + (i - 1) * width, attn_by_c[i], width, label=c)
    ax.set_xticks(x); ax.set_xticklabels(REGION_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("A2C-NODE per-region attention by diagnosis class (test set)")
    ax.legend()
    fig.savefig(out_path, dpi=160)
    print(f"  saved: {out_path}")


def plot_ate_bar(ate_rows: List[Dict[str, float]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["region_group"] for r in ate_rows]
    delta_ad = [r["delta_p_AD"] for r in ate_rows]
    delta_cn = [r["delta_p_CN"] for r in ate_rows]

    x = np.arange(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.bar(x - width/2, delta_cn, width, label="ΔP(CN)", color="#2ecc71")
    ax.bar(x + width/2, delta_ad, width, label="ΔP(AD)", color="#e74c3c")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Mean change in predicted probability")
    ax.set_title("Average treatment effect of freezing each region at baseline")
    ax.legend()
    fig.savefig(out_path, dpi=160)
    print(f"  saved: {out_path}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset", choices=["default", "legacy"], default="legacy")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test_split", type=float, default=0.2,
                   help="last fraction of subjects = held-out test")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ----- data ----------------------------------------------------------------
    if args.dataset == "legacy":
        ds = LegacyCacheRealDataset(args.data_root, min_visits_per_subject=2)
    else:
        ds = LongitudinalNPZDataset(args.data_root, min_visits_per_subject=2)

    n = len(ds)
    n_test = max(1, int(args.test_split * n))
    test_indices = list(range(n - n_test, n))            # subject-level last-fold
    test_subset = torch.utils.data.Subset(ds, test_indices)
    loader = DataLoader(test_subset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=longitudinal_collate, pin_memory=True)
    print(f"test subjects: {n_test}  /  total: {n}")

    # ----- model ---------------------------------------------------------------
    model = A2CNode(ode_solver="rk4", ode_use_adjoint=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # ----- 1. classification metrics + attention -------------------------------
    print("\n[1/3] cohort prediction ...")
    pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(pred["y_true"], pred["y_pred"], pred["y_prob"])
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    attn_by_c = attention_by_class(pred["attention"], pred["y_true"])
    np.savetxt(out_dir / "attention_by_class.csv", attn_by_c,
               delimiter=",", header=",".join(REGION_NAMES), comments="")
    plot_attention_by_class(attn_by_c, out_dir / "attention_by_class.png")

    # ----- 2. cohort ATE -------------------------------------------------------
    print("\n[2/3] cohort ATE for clinically named regions ...")
    ate_rows = cohort_ate_table(model, loader, device, INTERVENTION_GROUPS)
    with open(out_dir / "ate_table.json", "w") as f:
        json.dump(ate_rows, f, indent=2)
    print("\n region group              | ΔP(CN)   ΔP(MCI)  ΔP(AD)   Δcog")
    print(" " + "-" * 73)
    for r in ate_rows:
        print(f"  {r['region_group']:24s}| "
              f"{r['delta_p_CN']:+.4f}  {r['delta_p_MCI']:+.4f}  "
              f"{r['delta_p_AD']:+.4f}  {r['delta_cog_mean']:+.3f}")
    plot_ate_bar(ate_rows, out_dir / "ate_bar.png")

    # ----- 3. summary ----------------------------------------------------------
    print(f"\n[3/3] outputs in: {out_dir}")
    print("  metrics.json")
    print("  attention_by_class.csv  +  attention_by_class.png")
    print("  ate_table.json          +  ate_bar.png")


if __name__ == "__main__":
    main()
