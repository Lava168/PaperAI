#!/usr/bin/env python3
"""Generate the high-density auxiliary panels that make A2C-NODE's six main
figures *visually data-rich*:

    1. cohort_summary.png      Age / sex / edu / dx / MMSE / APOE4 distributions
                               of the cache_real cohort.
    2. per_seed_metrics.png    Per-seed BAcc / acc / Macro-AUC box-plot strip
                               (shows 5 seeds + ensemble + TTA dot).
    3. roc_curves.png          5-seed ROC overlay + ensemble bold + AUC inset.
    4. latent_tsne.png         t-SNE of h(t_target) embeddings, coloured by
                               true class, with predicted-class marker shape.
    5. attention_heatmap.png   subjects × 21 regions attention weights, with
                               class side-bar (CN / MCI / AD) and region bars
                               on top showing class-mean attention.

These panels are then plugged into the existing compose_paper_figures.py so
each main figure's "data density" goes up an order of magnitude.

All scripts are CPU-only and can run on a remote server; the optional
``--ckpt`` path is only needed for ``latent_tsne`` + ``attention_heatmap``
because those need to actually run a forward pass to extract latents.
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REGION_NAMES = (
    "L-WM", "L-Cortex", "L-LV",
    "L-Thal", "L-Caud", "L-Put", "L-Pall",
    "BrStem", "L-Hippo", "L-Amyg", "L-Acc",
    "R-WM", "R-Cortex", "R-LV",
    "R-Thal", "R-Caud", "R-Put", "R-Pall",
    "R-Hippo", "R-Amyg", "R-Acc",
)
CLASS_NAMES = ("CN", "MCI", "AD")
CLASS_COLORS = ("#2980b9", "#f39c12", "#c0392b")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--cache_real", type=Path,
                   default=Path("/home/lry/atlas_guided_attention Alzheimer's "
                                "Disease Dynamics/chapter1_foundation/sample_data/cache_real"))
    p.add_argument("--clinical_csv", default="/home/lry/adni_clinical/ADNIMERGE_May15.2014.csv")
    p.add_argument("--csf_csv",
                   default="/home/lry/adni_clinical/UPENN_CSF Biomarkers_baseline_May15.2014.csv")
    p.add_argument("--seed_dirs", nargs="+", type=Path,
                   default=[Path(f"runs/v4_mm_seed{s}") for s in (42, 153, 264, 375, 486)])
    p.add_argument("--ensemble_npz", type=Path,
                   default=Path("runs/v4_mm_ensemble_3seed.npz"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("runs/v4_mm_seed42/best.pth"),
                   help="checkpoint to extract h_target latents + attentions")
    p.add_argument("--gpu", default="0")
    p.add_argument("--max_subjects", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
def setup_style():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_style import apply
    apply(extra=("nature",))


def panel_label(ax, label, dx=-0.05, dy=1.05):
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="bottom", ha="left")


# ============================================================== 1. cohort
def make_cohort_summary(args) -> None:
    """Build a 6-panel cohort summary using ADNIMERGE + cache_real filenames."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from a2c_node.data.clinical_table import ClinicalTable, FEATURE_NAMES_NO_CSF

    fname_re = re.compile(r"^ADNI_(\d{3}_S_\d{4})_(sc|bl|m\d{2,3})_I\d+\.npz$")
    visits = {}
    pt_age, pt_sex, pt_edu, pt_apoe, pt_mmse, pt_dx = ([] for _ in range(6))
    visit_counts = {}
    for fp in args.cache_real.glob("ADNI_*.npz"):
        m = fname_re.match(fp.name)
        if not m: continue
        ptid, vc = m.group(1), m.group(2)
        visit_counts.setdefault(ptid, set()).add(vc)
    if not visit_counts:
        print("[cohort] no NPZ files matched"); return

    clin = ClinicalTable.from_adnimerge(
        args.clinical_csv, csf_csv=args.csf_csv,
        include_csf=True)
    for ptid in visit_counts:
        v, m_ = clin.get(ptid, "bl")
        idx = lambda n: FEATURE_NAMES_NO_CSF.index(n)
        if m_[idx("AGE")]:      pt_age.append(float(v[idx("AGE")]))
        if m_[idx("PTGENDER")]: pt_sex.append(int(v[idx("PTGENDER")]))
        if m_[idx("PTEDUCAT")]: pt_edu.append(float(v[idx("PTEDUCAT")]))
        if m_[idx("APOE4")]:    pt_apoe.append(int(v[idx("APOE4")]))
        if m_[idx("MMSE")]:     pt_mmse.append(float(v[idx("MMSE")]))

    # Per-PTID label = the most frequent label across that subject's NPZ files
    pt_label_count = {}
    for fp in args.cache_real.glob("ADNI_*.npz"):
        m = fname_re.match(fp.name)
        if not m: continue
        with np.load(fp, allow_pickle=True) as z:
            lab = int(z["label"]) if "label" in z.files else -1
        pt_label_count.setdefault(m.group(1), []).append(lab)
    for ptid, labs in pt_label_count.items():
        if labs:
            pt_dx.append(max(set(labs), key=labs.count))

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    ax = axes[0, 0]
    ax.hist(pt_age, bins=20, color="#3498db", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("# subjects"); ax.set_title("Age")
    ax.text(0.97, 0.95, f"$\\mu$={np.mean(pt_age):.1f}\n$\\sigma$={np.std(pt_age):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    panel_label(ax, "a")

    ax = axes[0, 1]
    sex_counts = [pt_sex.count(0), pt_sex.count(1)]
    ax.pie(sex_counts, labels=["F", "M"], colors=["#e74c3c", "#3498db"],
           autopct="%1.0f%%", textprops={"fontsize": 8}, wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
    ax.set_title("Sex")
    panel_label(ax, "b")

    ax = axes[0, 2]
    ax.hist(pt_edu, bins=15, color="#9b59b6", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Years of education"); ax.set_ylabel("# subjects"); ax.set_title("Education")
    panel_label(ax, "c")

    ax = axes[1, 0]
    apoe_counts = [pt_apoe.count(0), pt_apoe.count(1), pt_apoe.count(2)]
    ax.bar(["0", "1", "2"], apoe_counts, color="#27ae60", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("APOE4 alleles"); ax.set_ylabel("# subjects"); ax.set_title("APOE4 dose")
    for i, c in enumerate(apoe_counts):
        ax.text(i, c + 1, f"{c}", ha="center", fontsize=7)
    panel_label(ax, "d")

    ax = axes[1, 1]
    ax.hist(pt_mmse, bins=20, color="#e67e22", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("MMSE"); ax.set_ylabel("# subjects"); ax.set_title("MMSE")
    ax.axvline(24, color="red", lw=0.8, ls="--"); ax.text(24.3, ax.get_ylim()[1]*0.85,
            "MCI cutoff", fontsize=7, color="red")
    panel_label(ax, "e")

    ax = axes[1, 2]
    dx_counts = [pt_dx.count(i) for i in range(3)]
    bars = ax.bar(CLASS_NAMES, dx_counts, color=CLASS_COLORS,
                  edgecolor="black", linewidth=0.4)
    ax.set_ylabel("# subjects"); ax.set_title("Diagnosis (per subject)")
    for b, c in zip(bars, dx_counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{c}",
                ha="center", fontsize=8, fontweight="bold")
    # visit count subhistogram
    visit_n = [len(s) for s in visit_counts.values()]
    ax.text(0.95, 0.95,
            f"N={sum(dx_counts)} subj\n{sum(visit_n)} visits\n{np.mean(visit_n):.1f} v/subj",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    panel_label(ax, "f")

    fig.suptitle("Cohort summary (ADNI cache_real)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = args.out_dir / "cohort_summary"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[cohort] wrote {out}.png", flush=True)


# ============================================================== 2. per-seed metrics
def parse_train_log(log: Path):
    if not log.is_file(): return None
    text = log.read_text(errors="ignore")
    pat = re.compile(r"^epoch (\d+)\s+train=([0-9.\-]+)\s+val=([0-9.\-]+)\s+val\(drop\)=[0-9.\-]+\s+"
                     r"acc=([0-9.\-]+)\s+bacc=([0-9.\-]+)", re.M)
    rows = pat.findall(text)
    if not rows: return None
    rows = np.array(rows, dtype=float)
    return {"ep": rows[:, 0].astype(int), "tr": rows[:, 1],
            "vl": rows[:, 2], "acc": rows[:, 3], "bacc": rows[:, 4]}


def make_per_seed_metrics(args) -> None:
    import matplotlib.pyplot as plt
    rows = []
    for d in args.seed_dirs:
        lg = parse_train_log(d / "train.log")
        if lg is None: continue
        rows.append({
            "seed": d.name.replace("v4_mm_seed", ""),
            "best_bacc": float(np.max(lg["bacc"])),
            "last_bacc": float(lg["bacc"][-1]),
            "best_acc": float(np.max(lg["acc"])),
            "last_acc": float(lg["acc"][-1]),
        })
    if not rows:
        print("[per-seed] no train.log found"); return

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    seeds = [r["seed"] for r in rows]

    ax = axes[0]
    ax.bar(seeds, [r["best_bacc"] for r in rows], color="#3498db",
           edgecolor="black", linewidth=0.4)
    for i, r in enumerate(rows):
        ax.text(i, r["best_bacc"] + 0.005, f"{r['best_bacc']:.3f}",
                ha="center", fontsize=7)
    ax.axhline(0.78, color="green", ls="--", lw=0.7, label="npj DM 0.78")
    ax.set_ylim(0.5, 0.9); ax.set_ylabel("Validation BAcc (best)")
    ax.set_title("Per-seed best BAcc"); ax.legend(fontsize=7, frameon=False)
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.35
    x = np.arange(len(seeds))
    ax.bar(x - width/2, [r["best_bacc"] for r in rows], width,
           label="best", color="#27ae60", edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, [r["last_bacc"] for r in rows], width,
           label="last", color="#95a5a6", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_ylim(0.5, 0.9); ax.set_ylabel("BAcc")
    ax.set_title("best vs last (early-stopping helps)")
    ax.legend(fontsize=7, frameon=False)
    panel_label(ax, "b")

    ax = axes[2]
    bacc_arr = np.array([r["best_bacc"] for r in rows])
    ax.boxplot([bacc_arr], vert=True, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#3498db", alpha=0.6, edgecolor="black"),
               medianprops=dict(color="black"))
    for v in bacc_arr:
        ax.scatter(1 + np.random.uniform(-0.05, 0.05), v, s=30, color="#c0392b",
                   edgecolor="black", linewidth=0.4, zorder=5)
    ax.set_xticks([1]); ax.set_xticklabels(["5 seeds"])
    ax.set_ylim(0.5, 0.9); ax.set_ylabel("Val BAcc")
    ax.set_title(f"Cross-seed mean = {bacc_arr.mean():.3f} ± {bacc_arr.std():.3f}")
    panel_label(ax, "c")

    fig.suptitle("Cross-seed reproducibility", y=1.04, fontsize=11)
    fig.tight_layout()
    out = args.out_dir / "per_seed_metrics"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[per-seed] wrote {out}.png", flush=True)


# ============================================================== 3. ROC curves
def roc_curve_simple(y_true: np.ndarray, score: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-score)
    y_sorted = y_true[order]
    P = (y_true == 1).sum(); N = (y_true == 0).sum()
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0], tps / P])
    fpr = np.concatenate([[0], fps / N])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def make_roc_curves(args) -> None:
    import matplotlib.pyplot as plt

    seed_npz = []
    for d in args.seed_dirs:
        p = d / "preds_tta_seed0.npz"
        if p.is_file():
            seed_npz.append(p)
    if not seed_npz:
        print("[roc] no per-seed predictions found"); return

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    cmap = plt.get_cmap("tab10")

    for c, (cls, ax) in enumerate(zip(CLASS_NAMES, axes)):
        for i, p in enumerate(seed_npz):
            d = np.load(p, allow_pickle=True)
            prob = d["prob"][:, c]
            y = (d["label"] == c).astype(np.int64)
            fpr, tpr, auc = roc_curve_simple(y, prob)
            ax.plot(fpr, tpr, lw=0.9, color=cmap(i % 10), alpha=0.7,
                    label=f"seed {p.parent.name.split('seed')[-1]}  AUC={auc:.3f}")
        # ensemble (if available)
        if args.ensemble_npz.is_file():
            d = np.load(args.ensemble_npz, allow_pickle=True)
            prob = d["prob"][:, c]
            y = (d["label"] == c).astype(np.int64)
            fpr, tpr, auc = roc_curve_simple(y, prob)
            ax.plot(fpr, tpr, lw=2.0, color="black",
                    label=f"ensemble  AUC={auc:.3f}")

        ax.plot([0, 1], [0, 1], "--", lw=0.5, color="gray")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC ({cls} vs rest)")
        ax.legend(fontsize=6, loc="lower right", frameon=False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        panel_label(ax, "abc"[c])

    fig.suptitle("Per-class ROC over 5 seeds + ensemble", y=1.04, fontsize=11)
    fig.tight_layout()
    out = args.out_dir / "roc_curves"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[roc] wrote {out}.png", flush=True)


# ============================================================== 4. latent t-SNE
def make_latent_tsne(args) -> None:
    import matplotlib.pyplot as plt
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from a2c_node.data.legacy_cache_dataset import LegacyCacheRealDataset
    from a2c_node.data.npz_dataset import longitudinal_collate
    from a2c_node.data.clinical_table import ClinicalTable
    from a2c_node.models.a2c_node import A2CNode

    if not args.ckpt.is_file():
        print(f"[tsne] ckpt missing: {args.ckpt}"); return
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = LegacyCacheRealDataset(str(args.cache_real), min_visits_per_subject=2,
                                    augment=False)
    n = len(full); n_train = int(0.8*n); n_val = max(1, n-n_train)
    g = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = torch.utils.data.random_split(full, [n_train, n_val], generator=g)
    train_ptids = [full.subjects[i].ptid for i in train_idx.indices]

    clin = ClinicalTable.from_adnimerge(args.clinical_csv,
                                         csf_csv=args.csf_csv, include_csf=True)
    clin.fit_stats(train_ptids)
    full = LegacyCacheRealDataset(str(args.cache_real), min_visits_per_subject=2,
                                    augment=False, clinical=clin)
    g = torch.Generator().manual_seed(args.seed)
    _, val_idx = torch.utils.data.random_split(full, [n_train, n_val], generator=g)
    eval_subset = torch.utils.data.Subset(full, list(val_idx.indices)[:args.max_subjects])
    loader = torch.utils.data.DataLoader(eval_subset, batch_size=4, shuffle=False,
                                          num_workers=2, collate_fn=longitudinal_collate)

    model = A2CNode(ode_solver="rk4", ode_use_adjoint=False,
                    pretrained_encoder=None, encoder_type="aranet",
                    tabular_num_features=clin.num_features,
                    tabular_embed_dim=128, tabular_hidden_dim=128).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)

    h_pool = []
    labs = []
    attentions = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device); mask = batch["mask"].to(device)
            time_ = batch["time"].to(device); valid = batch["valid"].to(device)
            tabular = batch["tabular"].to(device) if "tabular" in batch else None
            tab_mask = batch["tab_mask"].to(device) if "tab_mask" in batch else None
            last_idx = valid.float().cumsum(dim=1).argmax(dim=1)
            target_t = time_.gather(1, last_idx.unsqueeze(1)).squeeze(1)
            t_eval = target_t.unique().sort().values
            out = model(image, mask, time_, valid, target_time=t_eval,
                        tabular=tabular, tab_mask=tab_mask, tabular_target_idx=last_idx)
            h_target = out["h_target"]                             # (B,K,C)
            head = out["head_out"]
            attn = head.get("attention")                           # (B,K)
            B, K, C = h_target.shape
            # subject-level pooled latent = attention-weighted mean of regions
            if attn is not None:
                weights = attn.unsqueeze(-1)
                pooled = (h_target * weights).sum(dim=1)             # (B,C)
                attentions.append(attn.cpu().numpy())
            else:
                pooled = h_target.mean(dim=1)
            h_pool.append(pooled.cpu().numpy())
            label = batch["label"].to(device)
            tgt = label.gather(1, last_idx.unsqueeze(1)).squeeze(1).cpu().numpy()
            labs.append(tgt)
    if not h_pool:
        print("[tsne] no batches"); return
    H = np.concatenate(h_pool, axis=0)
    L = np.concatenate(labs, axis=0)
    print(f"[tsne] H={H.shape}  labels={L.shape}", flush=True)
    np.savez(args.out_dir / "latent_dump.npz", h=H, label=L,
             attention=(np.concatenate(attentions, axis=0) if attentions else np.zeros((0,))))

    # t-SNE -> 2D
    try:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=min(30, max(5, len(L)//5)),
                   init="pca", random_state=0).fit_transform(H)
    except Exception as e:
        print(f"[tsne] sklearn TSNE failed: {e}"); return

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for c in range(3):
        m = L == c
        ax.scatter(emb[m, 0], emb[m, 1], s=22, c=CLASS_COLORS[c],
                   edgecolor="black", linewidth=0.4,
                   label=f"{CLASS_NAMES[c]}  n={int(m.sum())}", alpha=0.8)
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
    ax.set_title("Latent $h(t_{target})$ pooled embedding (val)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = args.out_dir / "latent_tsne"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne] wrote {out}.png", flush=True)


# ============================================================== 5. attention heatmap
def make_attention_heatmap(args) -> None:
    """Heatmap of subjects × 21 regions, sorted by class, with class side-bar."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    p = args.out_dir / "latent_dump.npz"
    if not p.is_file():
        print("[attn] latent_dump.npz missing (run latent_tsne first)"); return
    d = np.load(p, allow_pickle=True)
    A = d.get("attention")
    L = d["label"]
    if A is None or A.size == 0:
        print("[attn] no attention recorded"); return
    if A.shape[0] != L.shape[0]:
        print(f"[attn] mismatched shapes A={A.shape} L={L.shape}"); return

    order = np.argsort(L)                           # CN < MCI < AD
    A = A[order]; L = L[order]

    fig = plt.figure(figsize=(11, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 1.4],
                          width_ratios=[1.0, 0.04], hspace=0.05, wspace=0.04)

    # Top: per-class mean attention bar
    ax_top = fig.add_subplot(gs[0, 0])
    width = 0.27
    x = np.arange(21)
    for c in range(3):
        m = L == c
        if m.sum():
            ax_top.bar(x + (c - 1) * width, A[m].mean(axis=0), width,
                       color=CLASS_COLORS[c], edgecolor="black", linewidth=0.3,
                       label=f"{CLASS_NAMES[c]} (n={int(m.sum())})")
    ax_top.set_xticks(x); ax_top.set_xticklabels(REGION_NAMES, fontsize=6, rotation=70)
    ax_top.set_ylabel("Mean attention")
    ax_top.legend(fontsize=7, frameon=False, ncol=3)
    ax_top.set_title("Per-class regional attention profile")
    panel_label(ax_top, "a", dx=-0.02, dy=1.1)

    # Heatmap: subjects × regions
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(A, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(x); ax.set_xticklabels(REGION_NAMES, fontsize=6, rotation=70)
    ax.set_ylabel("Subjects (sorted by class)")
    ax.set_yticks([])
    panel_label(ax, "b", dx=-0.02, dy=1.02)

    # Side band for class
    ax_band = fig.add_subplot(gs[1, 1])
    for i, l in enumerate(L):
        ax_band.add_patch(Rectangle((0, len(L) - 1 - i), 1, 1,
                                    color=CLASS_COLORS[int(l)], linewidth=0))
    ax_band.set_xlim(0, 1); ax_band.set_ylim(0, len(L))
    ax_band.axis("off")

    cbar_ax = fig.add_axes([0.93, 0.13, 0.012, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("attention", fontsize=8)

    fig.suptitle("Attention over 21 anatomical regions", y=1.02, fontsize=11)
    out = args.out_dir / "attention_heatmap"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[attn] wrote {out}.png", flush=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    make_cohort_summary(args)
    make_per_seed_metrics(args)
    make_roc_curves(args)
    make_latent_tsne(args)
    make_attention_heatmap(args)


if __name__ == "__main__":
    main()
