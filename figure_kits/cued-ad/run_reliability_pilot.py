#!/usr/bin/env python3
"""
Chapter 4 pilot: Bayesian reliability and OOD evaluation.

This is intentionally a pilot script: it reuses the Chapter 2 PathwayPro
checkpoint and cached cohorts, then produces uncertainty, density, OOD, and
selective-prediction summaries without retraining the backbone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset


LABEL_NAMES = ["CN", "MCI", "AD"]


@dataclass
class CohortConfig:
    name: str
    kind: str
    csv: str | None = None
    cache_dir: str | None = None
    flair_dir: str | None = None
    suvr_dir: str | None = None
    max_n: int | None = None


class CacheNPZDataset(Dataset):
    """Generic NPZ dataset with keys image/seg/label; FLAIR is optional."""

    def __init__(
        self,
        cache_dir: str,
        csv_path: str | None = None,
        flair_dir: str | None = None,
        max_n: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.flair_dir = Path(flair_dir) if flair_dir else None
        samples: list[dict] = []
        if csv_path:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                scan_id = str(row.get("scan_id", row.get("subject_id", "")))
                ptid = str(row.get("PTID", row.get("subject_id", scan_id)))
                candidates = [
                    self.cache_dir / f"{scan_id}.npz",
                    self.cache_dir / f"{ptid}.npz",
                    self.cache_dir / f"OASIS_{scan_id}.npz",
                    self.cache_dir / f"OASIS_{ptid}.npz",
                ]
                npz = next((p for p in candidates if p.exists()), None)
                if npz is None:
                    continue
                label = int(row["label"]) if "label" in row and not pd.isna(row["label"]) else None
                samples.append({
                    "id": scan_id,
                    "ptid": ptid,
                    "npz": str(npz),
                    "label": label,
                    "age": float(row.get("AGE", np.nan)),
                })
        else:
            for npz in sorted(self.cache_dir.glob("*.npz")):
                samples.append({"id": npz.stem, "ptid": npz.stem, "npz": str(npz), "label": None, "age": np.nan})
        if max_n:
            samples = samples[:max_n]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        data = np.load(s["npz"], allow_pickle=False)
        image = torch.from_numpy(data["image"].astype(np.float32)).unsqueeze(0)
        label = s["label"]
        if label is None and "label" in data:
            label = int(np.asarray(data["label"]).item())

        flair = None
        if self.flair_dir is not None:
            for cand in (self.flair_dir / f"{s['ptid']}.npz", self.flair_dir / f"{s['id']}.npz"):
                if cand.exists():
                    fd = np.load(cand, allow_pickle=False)
                    if "flair" in fd:
                        flair = torch.from_numpy(fd["flair"].astype(np.float32)).unsqueeze(0)
                    break
        if flair is None:
            flair = image.clone()

        return {
            "image": image,
            "flair": flair,
            "label": torch.tensor(int(label), dtype=torch.long),
            "sample_id": s["id"],
            "ptid": s["ptid"],
            "age": torch.tensor(float(s["age"]), dtype=torch.float),
        }


class ArtifactDataset(Dataset):
    """Deterministic corruptions applied to an existing dataset."""

    def __init__(self, base: Dataset, max_n: int | None = None) -> None:
        self.base = base
        n = len(base) if max_n is None else min(len(base), max_n)
        self.indices = list(range(n))

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _corrupt(x: torch.Tensor, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(10_000 + idx)
        y = x.clone()
        brain = (y != 0).float()
        noise = torch.randn(y.shape, generator=g, dtype=y.dtype) * 0.20
        y = y + noise * brain
        # Stripe artifact along one axis.
        stripe = torch.zeros_like(y)
        stripe[:, :, :, ::7] = 0.35
        y = y + stripe * brain
        # Central dropout block.
        _, d, h, w = y.shape
        z0, z1 = int(d * 0.35), int(d * 0.65)
        y0, y1 = int(h * 0.35), int(h * 0.65)
        x0, x1 = int(w * 0.35), int(w * 0.65)
        if idx % 2 == 0:
            y[:, z0:z1, y0:y1, x0:x1] *= 0.15
        return y

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        b = dict(self.base[idx])
        b["image"] = self._corrupt(b["image"], idx)
        b["flair"] = self._corrupt(b["flair"], idx)
        return b


def collate(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if torch.is_tensor(vals[0]):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def load_model(ckpt_path: Path, chapter2_dir: Path, device: torch.device):
    # Chapter 2 imports modules as `models.*`, while the project root also has a
    # `models/` package. Run from chapter2_dir so Python resolves the intended
    # namespace package first.
    os.chdir(chapter2_dir)
    sys.path.insert(0, str(chapter2_dir))
    for mod in list(sys.modules):
        if mod == "models" or mod.startswith("models.") or mod == "losses" or mod.startswith("losses."):
            del sys.modules[mod]
    for pkg_name in ("models", "losses"):
        pkg_dir = chapter2_dir / pkg_name
        if pkg_dir.exists():
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_dir)]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg
    from train_disentangle_v6 import DisentangleV6Model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    saved_args = ckpt.get("args", {})
    feature_dim = saved_args.get("feature_dim", 256)
    latent_dim = saved_args.get("latent_dim", 64)
    method = saved_args.get("method", "ours")
    stochastic = method in ("betavae", "tcvae", "factorvae", "ours", "dipvae2")
    model = DisentangleV6Model(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        num_subtypes=saved_args.get("num_subtypes", 3),
        stochastic=stochastic,
        use_grl=False,
    )

    ckpt_q = None
    for k, v in state.items():
        if k.endswith("pathology_query"):
            ckpt_q = int(v.shape[1])
            break
    if ckpt_q is not None:
        for branch in ("atrophy_head", "vascular_head"):
            head = getattr(model.backbone, branch, None)
            if head is not None and hasattr(head, "pathology_query") and head.pathology_query.shape[1] != ckpt_q:
                dim = head.pathology_query.shape[2]
                head.pathology_query = nn.Parameter(torch.randn(1, ckpt_q, dim) * 0.02)
                head.num_queries = ckpt_q

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def enable_mc_dropout(model: nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(probs + 1e-8)).sum(axis=-1)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int,
    use_mc: bool,
) -> dict[str, np.ndarray | list[str]]:
    all_probs, all_logits, all_za, all_zv = [], [], [], []
    labels, ids = [], []
    n_passes = mc_samples if use_mc else 1
    for pass_idx in range(n_passes):
        if use_mc:
            enable_mc_dropout(model)
        else:
            model.eval()
        pass_probs, pass_logits, pass_za, pass_zv = [], [], [], []
        pass_labels, pass_ids = [], []
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            fl = batch["flair"].to(device, non_blocking=True)
            flair_missing = batch.get("flair_missing")
            if flair_missing is not None:
                flair_missing = flair_missing.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if flair_missing is not None:
                    out = model(x, fl, flair_missing=flair_missing)
                else:
                    out = model(x, fl)
            logits = out["diagnosis_logits"].float()
            probs = F.softmax(logits, dim=1)
            pass_probs.append(probs.cpu().numpy())
            pass_logits.append(logits.cpu().numpy())
            pass_za.append(out["z_atrophy"].float().cpu().numpy())
            pass_zv.append(out["z_vascular"].float().cpu().numpy())
            pass_labels.append(batch["label"].cpu().numpy())
            if "sample_id" in batch:
                pass_ids.extend([str(x) for x in batch["sample_id"]])
            else:
                pass_ids.extend([str(i) for i in range(len(pass_ids), len(pass_ids) + len(batch["label"]))])
        all_probs.append(np.concatenate(pass_probs, axis=0))
        all_logits.append(np.concatenate(pass_logits, axis=0))
        all_za.append(np.concatenate(pass_za, axis=0))
        all_zv.append(np.concatenate(pass_zv, axis=0))
        if pass_idx == 0:
            labels = [int(x) for x in np.concatenate(pass_labels, axis=0)]
            ids = pass_ids

    probs_s = np.stack(all_probs, axis=0)
    logits_s = np.stack(all_logits, axis=0)
    za_s = np.stack(all_za, axis=0)
    zv_s = np.stack(all_zv, axis=0)
    mean_probs = probs_s.mean(axis=0)
    pred_ent = predictive_entropy(mean_probs)
    exp_ent = predictive_entropy(probs_s).mean(axis=0)
    mutual_info = pred_ent - exp_ent
    return {
        "ids": ids,
        "labels": np.asarray(labels, dtype=np.int64),
        "probs": mean_probs,
        "logits": logits_s.mean(axis=0),
        "z": np.concatenate([za_s.mean(axis=0), zv_s.mean(axis=0)], axis=1),
        "predictive_entropy": pred_ent,
        "expected_entropy": exp_ent,
        "mutual_information": mutual_info,
        "confidence": mean_probs.max(axis=1),
        "energy": -np.log(np.exp(logits_s.mean(axis=0)).sum(axis=1) + 1e-8),
    }


def ece_score(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_multiclass(y: np.ndarray, probs: np.ndarray, n_classes: int = 3) -> float:
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def class_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    out = {
        "n": int(len(y)),
        "label_dist": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "ece": ece_score(y, probs),
        "brier": brier_multiclass(y, probs),
    }
    try:
        out["nll"] = float(log_loss(y, probs, labels=[0, 1, 2]))
    except Exception:
        out["nll"] = float("nan")
    mask = np.isin(y, [0, 2])
    if mask.sum() > 2 and len(np.unique(y[mask])) == 2:
        out["auc_ad_vs_cn"] = float(roc_auc_score((y[mask] == 2).astype(int), probs[mask, 2]))
    return out


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    y = np.concatenate([np.zeros_like(id_scores, dtype=int), np.ones_like(ood_scores, dtype=int)])
    s = np.concatenate([id_scores, ood_scores])
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
    }
    # FPR at 95% TPR.
    thresholds = np.sort(np.unique(s))
    best_fpr = 1.0
    for t in thresholds:
        pred = s >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if tpr >= 0.95:
            best_fpr = min(best_fpr, fpr)
    out["fpr_at_95_tpr"] = float(best_fpr)
    return out


def selective_metrics(y: np.ndarray, probs: np.ndarray, risk: np.ndarray) -> list[dict]:
    rows = []
    order = np.argsort(risk)
    n = len(y)
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(round(n * cov)))
        idx = order[:k]
        rows.append({
            "coverage": float(k / n),
            "n_retained": int(k),
            "acc": float(accuracy_score(y[idx], probs[idx].argmax(axis=1))),
            "balanced_acc": float(balanced_accuracy_score(y[idx], probs[idx].argmax(axis=1))),
            "mean_risk": float(risk[idx].mean()),
        })
    return rows


def make_hybrid_scores(results: dict[str, dict], val_name: str = "ADNI_val") -> None:
    val = results[val_name]
    refs = {}
    for key in ["predictive_entropy", "mutual_information", "mahalanobis", "energy"]:
        x = np.asarray(val[key])
        refs[key] = (float(np.nanmean(x)), float(np.nanstd(x) + 1e-8))
    for res in results.values():
        parts = []
        for key in ["predictive_entropy", "mutual_information", "mahalanobis", "energy"]:
            mu, sd = refs[key]
            parts.append((np.asarray(res[key]) - mu) / sd)
        res["hybrid_risk"] = np.vstack(parts).mean(axis=0)


def write_figures(out_dir: Path, frame: pd.DataFrame, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for domain, g in frame.groupby("domain"):
        axes[0].hist(g["predictive_entropy"], bins=30, alpha=0.45, density=True, label=domain)
        axes[1].hist(g["mahalanobis"], bins=30, alpha=0.45, density=True, label=domain)
        axes[2].hist(g["hybrid_risk"], bins=30, alpha=0.45, density=True, label=domain)
    axes[0].set_title("Predictive entropy")
    axes[1].set_title("Latent Mahalanobis")
    axes[2].set_title("Hybrid risk")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pilot_score_distributions.png", dpi=220)
    plt.close(fig)

    rows = []
    for cohort, metrics in summary["ood_metrics"].items():
        for score, vals in metrics.items():
            rows.append((cohort, score, vals["auroc"]))
    if rows:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [f"{c}\n{s}" for c, s, _ in rows]
        ax.bar(np.arange(len(rows)), [v for _, _, v in rows], color="#4E79A7")
        ax.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("OOD AUROC")
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "pilot_ood_aurocs.png", dpi=220)
        plt.close(fig)


def build_adni_dataset(chapter2_dir: Path, split_csv: str, data_root: Path, max_n: int | None):
    os.chdir(chapter2_dir)
    sys.path.insert(0, str(chapter2_dir))
    from train_pathway_v5 import PathwayDatasetV5

    ds = PathwayDatasetV5(
        split_csv,
        str(data_root / "full_cache"),
        str(data_root / "flair_cache"),
        str(data_root / "regional_suvr_cache"),
        augment=False,
        flair_mask_prob=0.0,
        memory_cache=False,
    )
    if max_n:
        return Subset(ds, list(range(min(max_n, len(ds)))))
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--ckpt", default="chapter2_disentangle/outputs_disentangle_v6_zeromask/ours_seed0/best_model.pth")
    ap.add_argument("--out_dir", default="chapter4_generalization/pilot_outputs/reliability_pilot")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--mc_samples", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    project_root = Path(args.project_root)
    chapter2_dir = project_root / "chapter2_disentangle"
    data_root = chapter2_dir / "data"
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(chapter2_dir)
    sys.path.insert(0, str(chapter2_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pilot] device={device} gpu={args.gpu} mc={args.mc_samples} out={out_dir}")
    model = load_model(project_root / args.ckpt, chapter2_dir, device)

    max_n = 24 if args.smoke else None
    adni_train = build_adni_dataset(chapter2_dir, str(data_root / "splits_v5_paired_qc_clean/train.csv"), data_root, max_n)
    cohorts: list[tuple[str, Dataset, bool]] = [
        ("ADNI_val", build_adni_dataset(chapter2_dir, str(data_root / "splits_v5_paired_qc_clean/val.csv"), data_root, max_n), True),
        ("ADNI_test", build_adni_dataset(chapter2_dir, str(data_root / "splits_v5_paired_qc_clean/test.csv"), data_root, max_n), True),
        ("AIBL", CacheNPZDataset("/path/to/aibl/cache_real", max_n=max_n), True),
        ("OASIS", CacheNPZDataset(str(data_root / "oasis_cache_128"), str(data_root / "splits_oasis_v2/test.csv"), max_n=max_n), True),
    ]
    cohorts.append(("Artifact_ADNI", ArtifactDataset(cohorts[1][1], max_n=max_n), True))

    print("[pilot] fitting ADNI train latent density")
    train_loader = DataLoader(adni_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    train_res = run_inference(model, train_loader, device, mc_samples=1, use_mc=False)
    scaler = StandardScaler().fit(train_res["z"])
    z_train = scaler.transform(train_res["z"])
    cov = LedoitWolf().fit(z_train)

    results: dict[str, dict] = {}
    for name, ds, use_mc in cohorts:
        print(f"[pilot] cohort={name} n={len(ds)}")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
        res = run_inference(model, loader, device, mc_samples=args.mc_samples, use_mc=use_mc)
        z = scaler.transform(res["z"])
        res["mahalanobis"] = cov.mahalanobis(z)
        results[name] = res

    make_hybrid_scores(results)

    summary: dict = {
        "config": {
            "ckpt": args.ckpt,
            "mc_samples": args.mc_samples,
            "batch_size": args.batch_size,
            "smoke": args.smoke,
        },
        "cohorts": {},
        "ood_metrics": {},
        "selective_prediction": {},
    }
    frames = []
    for name, res in results.items():
        y = res["labels"]
        probs = res["probs"]
        summary["cohorts"][name] = class_metrics(y, probs)
        summary["selective_prediction"][name] = selective_metrics(y, probs, res["hybrid_risk"])
        pred = probs.argmax(axis=1)
        frames.append(pd.DataFrame({
            "domain": name,
            "sample_id": res["ids"],
            "label": y,
            "pred": pred,
            "correct": (pred == y).astype(int),
            "p_CN": probs[:, 0],
            "p_MCI": probs[:, 1],
            "p_AD": probs[:, 2],
            "confidence": res["confidence"],
            "predictive_entropy": res["predictive_entropy"],
            "expected_entropy": res["expected_entropy"],
            "mutual_information": res["mutual_information"],
            "energy": res["energy"],
            "mahalanobis": res["mahalanobis"],
            "hybrid_risk": res["hybrid_risk"],
        }))

    id_res = results["ADNI_test"]
    score_keys = ["predictive_entropy", "mutual_information", "mahalanobis", "energy", "hybrid_risk"]
    for name in [k for k in results.keys() if k not in ("ADNI_val", "ADNI_test")]:
        summary["ood_metrics"][name] = {}
        for score in score_keys:
            summary["ood_metrics"][name][score] = ood_metrics(id_res[score], results[name][score])

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out_dir / "pilot_predictions.csv", index=False)
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    write_figures(out_dir, frame, summary)
    print(json.dumps(summary, indent=2)[:6000])
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
