#!/usr/bin/env python3
"""Robustness curve: BAcc vs random visit-drop fraction.

For each subject in the ADNI val split, we keep the first and the last
visit and randomly drop a fraction ``p`` of the intermediate visits.
We re-run the multimodal A2C-NODE 5-seed ensemble for each ``p`` and
plot BAcc(mean +/- std) vs ``p``.

The script reuses ``LegacyCacheRealDataset`` + ``ClinicalTable`` and
the existing ensemble checkpoints; nothing is re-trained.

Outputs (default under ``runs/v4_mm_analysis_npj_5seed/``):
    robustness_curve.png / .pdf  -- the figure
    robustness_curve.json        -- raw numbers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from a2c_node.data.legacy_cache_dataset import LegacyCacheRealDataset
from a2c_node.data.npz_dataset import longitudinal_collate
from a2c_node.data.clinical_table import ClinicalTable
from a2c_node.models.a2c_node import A2CNode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path("/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/chapter1_foundation/sample_data/cache_real"))
    p.add_argument("--clinical_csv", type=Path,
                   default=Path("/home/lry/adni_clinical/ADNIMERGE_May15.2014.csv"))
    p.add_argument("--csf_csv", type=Path,
                   default=Path("/home/lry/adni_clinical/UPENN_CSF Biomarkers_baseline_May15.2014.csv"))
    p.add_argument("--runs_root", type=Path, default=Path("runs"))
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 153, 264, 375, 486])
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
    p.add_argument("--out_dir", type=Path,
                   default=Path("runs/v4_mm_analysis_npj_5seed"))
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for visit dropping (deterministic)")
    return p.parse_args()


def drop_visits(rec_visits, p: float, rng: np.random.Generator):
    """Keep first + last visit; randomly drop a fraction p of the rest."""
    if len(rec_visits) <= 2 or p <= 0:
        return rec_visits
    middle = rec_visits[1:-1]
    if not middle:
        return rec_visits
    keep = rng.random(len(middle)) >= p
    if not keep.any():
        # ensure at least 1 middle visit if any existed and p<1
        keep[rng.integers(0, len(middle))] = True
    kept = [middle[i] for i in range(len(middle)) if keep[i]]
    return [rec_visits[0]] + kept + [rec_visits[-1]]


@torch.no_grad()
def run_one(model, dataset, val_indices, device, drop_p: float,
            rng: np.random.Generator) -> Dict[str, float]:
    preds, labels = [], []
    for idx in val_indices:
        rec = dataset.subjects[idx]
        # monkey-patch visits temporarily
        original_visits = rec.visits
        rec.visits = drop_visits(original_visits, drop_p, rng)
        try:
            batch = longitudinal_collate([dataset[idx]])
        finally:
            rec.visits = original_visits
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        time_ = batch["time"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        tabular = batch.get("tabular")
        tab_mask = batch.get("tab_mask")
        if tabular is not None:
            tabular = tabular.to(device)
            tab_mask = tab_mask.to(device)

        last_idx = valid.float().cumsum(dim=1).argmax(dim=1)
        target_t = time_.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        t_eval = target_t.unique().sort().values

        out = model(image, mask, time_, valid, target_time=t_eval,
                    tabular=tabular, tab_mask=tab_mask,
                    tabular_target_idx=last_idx)
        logits = out["head_out"]["logits"]
        prob = logits.softmax(-1)
        # mirror flip TTA
        out_f = model(torch.flip(image, dims=[-1]), mask, time_, valid,
                      target_time=t_eval, tabular=tabular, tab_mask=tab_mask,
                      tabular_target_idx=last_idx)
        prob = 0.5 * prob + 0.5 * out_f["head_out"]["logits"].softmax(-1)
        pred = prob.argmax(-1).cpu().numpy()
        tgt = label.gather(1, last_idx.unsqueeze(1)).squeeze(1).cpu().numpy()
        preds.append(pred)
        labels.append(tgt)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    classes = [0, 1, 2]
    sens = []
    for c in classes:
        m = labels == c
        if m.any():
            sens.append((preds[m] == c).mean())
    return {"BAcc": float(np.mean(sens)),
            "acc": float((preds == labels).mean()),
            "n": int(len(labels))}


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build clinical table (same as training)
    clin = ClinicalTable.from_adnimerge(
        str(args.clinical_csv),
        csf_csv=str(args.csf_csv) if args.csf_csv.is_file() else None,
        include_csf=True,
    )
    full = LegacyCacheRealDataset(str(args.data_root),
                                  min_visits_per_subject=2, augment=False)
    n = len(full)
    n_train = int(0.8 * n)
    n_val = max(1, n - n_train)

    # Use seed 0 for split (matches predict_to_npz default)
    g = torch.Generator().manual_seed(0)
    train_idx, val_idx = torch.utils.data.random_split(
        full, [n_train, n_val], generator=g)
    train_ptids = [full.subjects[i].ptid for i in train_idx.indices]
    clin.fit_stats(train_ptids)

    full = LegacyCacheRealDataset(str(args.data_root),
                                  min_visits_per_subject=2, augment=False,
                                  clinical=clin)
    g = torch.Generator().manual_seed(0)
    train_idx, val_idx = torch.utils.data.random_split(
        full, [n_train, n_val], generator=g)
    val_indices = list(val_idx.indices)
    print(f"[robust] dataset N={n}  val={len(val_indices)}", flush=True)

    results: Dict[str, Dict[float, List[float]]] = {"BAcc": {}, "acc": {}}
    for p in args.fractions:
        results["BAcc"][p] = []
        results["acc"][p] = []

    for seed in args.seeds:
        ck = args.runs_root / f"v4_mm_seed{seed}" / "best.pth"
        if not ck.is_file():
            print(f"[robust] missing ckpt for seed {seed}: {ck}", flush=True)
            continue
        model = A2CNode(
            ode_solver="rk4", ode_use_adjoint=False, encoder_type="aranet",
            pretrained_encoder=None,
            tabular_num_features=clin.num_features,
            tabular_embed_dim=128, tabular_hidden_dim=128,
        ).to(device)
        state = torch.load(ck, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"[robust] seed {seed} loaded", flush=True)
        for p in args.fractions:
            rng = np.random.default_rng(args.seed * 1000 + int(p * 10) * 7 + seed)
            r = run_one(model, full, val_indices, device, p, rng)
            results["BAcc"][p].append(r["BAcc"])
            results["acc"][p].append(r["acc"])
            print(f"[robust]   seed={seed} p={p:.2f}  BAcc={r['BAcc']:.3f}  "
                  f"acc={r['acc']:.3f}  n={r['n']}", flush=True)
        del model
        torch.cuda.empty_cache()

    # Aggregate
    summary = {"fractions": list(args.fractions),
               "seeds": list(args.seeds),
               "BAcc_mean": [float(np.mean(results["BAcc"][p])) for p in args.fractions],
               "BAcc_std": [float(np.std(results["BAcc"][p])) for p in args.fractions],
               "acc_mean": [float(np.mean(results["acc"][p])) for p in args.fractions],
               "acc_std": [float(np.std(results["acc"][p])) for p in args.fractions],
               "raw": {str(p): {"BAcc": results["BAcc"][p], "acc": results["acc"][p]}
                       for p in args.fractions}}

    (args.out_dir / "robustness_curve.json").write_text(
        json.dumps(summary, indent=2))

    # Plot
    sys.path.insert(0, str(PROJ_ROOT / "scripts"))
    from _plot_style import apply
    apply(extra=("nature",))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.array(args.fractions)
    ys = np.array(summary["BAcc_mean"])
    yerr = np.array(summary["BAcc_std"])

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.fill_between(xs, ys - yerr, ys + yerr, color="#1F3A5F", alpha=0.2,
                    lw=0)
    ax.plot(xs, ys, marker="o", color="#1F3A5F", lw=1.6, markersize=5,
            label=f"A2C-NODE (n={len(args.seeds)} seeds)")
    ax.axhline(1/3, color="#B8B8B8", lw=0.6, ls=(0, (3, 3)),
               label="chance (1/3)")
    ax.set_xlabel("Visit-drop fraction $p$")
    ax.set_ylabel("Validation balanced accuracy")
    ax.set_ylim(0.30, 0.95)
    ax.set_xlim(-0.02, max(xs) + 0.02)
    ax.set_title("Robustness to irregular follow-up", loc="left",
                 fontsize=9, color="#4A4A4A", pad=4)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    out_png = args.out_dir / "robustness_curve.png"
    out_pdf = args.out_dir / "robustness_curve.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"[robust] wrote {out_png} {out_pdf}")
    print("\nfinal table:")
    print(f"{'p':>5} {'BAcc mean':>10} {'std':>6}")
    for p, m, s in zip(xs, ys, yerr):
        print(f"{p:>5.2f} {m:>10.3f} {s:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
