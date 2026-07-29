#!/usr/bin/env python3
"""Build NIfTI brain-figure assets for fcHMRF-comparable ARA-Net plots.

Creates:
  - consensus 21-ROI atlas template (from cache segs)
  - group atrophy-z and MRF-q maps
  - method association maps (ROI Spearman vs p_AD)
  - methods × ROI × stage matrices for heatmaps / top-k plots
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roi_graph_mrf import (  # noqa: E402
    MRFParams,
    atrophy_zscores,
    build_roi_graph,
    fit_cn_volume_stats,
    infer_row,
    mean_field_infer,
    volumes_from_row,
)
from train_atlas_feature_baseline import FS_LABELS, REGION_NAMES  # noqa: E402

try:
    import nibabel as nib
    from nibabel import Nifti1Image
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"nibabel required: {exc}")


CLASS_NAMES = ["CN", "MCI", "AD"]
LABEL_TO_INT = {n: i for i, n in enumerate(CLASS_NAMES)}

# Locked MNI-like affine for 96x112x96 crop (2 mm).
TEMPLATE_AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 94.0],
        [0.0, 2.0, 0.0, -114.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CUTS = {
    "axial": (-28, -12, 0, 12, 28, 40),
    "coronal": (-52, -28, -8, 12, 32),
    "sagittal": (-36, -18, 0, 18, 36),
}

METHOD_COLUMNS = [
    "atrophy_AD_mean",
    "atrophy_AD_minus_CN",
    "mrf_q_AD_mean",
    "mrf_q_AD_minus_CN",
    "mrf_q_MCI_minus_CN",
    "assoc_v6_rcspe",
    "assoc_v7_mrf",
    "assoc_equal_logpool",
]


def read_feature_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(float(row["label"]))
        if "label_name" not in row or not row["label_name"]:
            row["label_name"] = CLASS_NAMES[row["label"]] if 0 <= row["label"] <= 2 else "UNK"
    return rows


def save_nifti(data: np.ndarray, path: Path, affine: np.ndarray = TEMPLATE_AFFINE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    nib.save(img, str(path))


def load_seg(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        return data["seg"].astype(np.int16)


def build_mean_underlay(
    feature_rows: Sequence[dict],
    cache_root: Path,
    max_scans: int = 40,
    seed: int = 42,
) -> np.ndarray:
    """Mean T1 intensity over ADNI CN scans for anatomical underlay."""
    rng = np.random.default_rng(seed)
    cands = []
    for r in feature_rows:
        if str(r.get("dataset", "")).upper() != "ADNI" or int(r.get("label", -1)) != 0:
            continue
        path = resolve_npz_path(r, cache_root)
        if path is not None:
            cands.append(path)
    if not cands:
        for r in feature_rows:
            path = resolve_npz_path(r, cache_root)
            if path is not None:
                cands.append(path)
    if not cands:
        raise FileNotFoundError("No NPZ available for anatomical underlay.")
    if len(cands) > max_scans:
        cands = [cands[i] for i in rng.choice(len(cands), size=max_scans, replace=False)]
    acc = None
    for i, path in enumerate(cands):
        with np.load(path, allow_pickle=True) as data:
            img = data["image"].astype(np.float32)
        # robust normalize per scan
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        acc = img if acc is None else acc + img
        if (i + 1) % 20 == 0:
            print(f"[underlay] {i+1}/{len(cands)}", flush=True)
    assert acc is not None
    return (acc / len(cands)).astype(np.float32)


def resolve_npz_path(row: dict, cache_root: Path) -> Optional[Path]:
    """Resolve NPZ path; CSV may use a different Unicode apostrophe than the filesystem."""
    raw = Path(str(row.get("path", "")))
    if raw.exists():
        return raw
    scan_id = str(row.get("scan_id", "")).strip()
    if not scan_id:
        return None
    cand = cache_root / f"{scan_id}.npz"
    if cand.exists():
        return cand
    hits = list(cache_root.glob(f"*{scan_id}*.npz"))
    return hits[0] if hits else None


def build_consensus_atlas(
    feature_rows: Sequence[dict],
    cache_root: Path,
    max_scans: int = 120,
    seed: int = 42,
) -> Tuple[np.ndarray, dict, np.ndarray]:
    """Build atlas volume with complete 21 ROI labels.

    Prefer a single high-coverage CN reference segmentation (all FS labels
    present). Also store a soft frequency map for QC.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for r in feature_rows:
        path = resolve_npz_path(r, cache_root)
        if path is None:
            continue
        if str(r.get("dataset", "")).upper() == "ADNI" and int(r.get("label", -1)) == 0:
            candidates.append({**r, "path": str(path)})
    if len(candidates) < 20:
        candidates = []
        for r in feature_rows:
            path = resolve_npz_path(r, cache_root)
            if path is not None:
                candidates.append({**r, "path": str(path)})
    if not candidates:
        raise FileNotFoundError("No feature rows with existing NPZ paths for atlas consensus.")

    # Score scans by how many of the 21 ROI labels are present with enough voxels.
    scored = []
    want = set(int(x) for x in FS_LABELS[1:])
    for row in candidates:
        seg = load_seg(Path(row["path"]))
        present = {int(v) for v in np.unique(seg) if int(v) in want}
        # coverage score: number of labels + log total ROI voxels
        roi_vox = int(np.isin(seg, list(want)).sum())
        scored.append((len(present), roi_vox, row, seg if len(present) == len(want) else None))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_n, best_vox, best_row, best_seg = scored[0]
    if best_seg is None:
        best_seg = load_seg(Path(best_row["path"]))
    atlas = best_seg.astype(np.int16)

    # Optional frequency QC over a subsample
    pick = [t[2] for t in scored[: min(max_scans, len(scored))]]
    shape = atlas.shape
    freq = np.zeros((len(FS_LABELS),) + shape, dtype=np.float32)
    for i, row in enumerate(pick):
        seg = load_seg(Path(row["path"]))
        for li, lab in enumerate(FS_LABELS):
            freq[li][seg == int(lab)] += 1.0
        if (i + 1) % 40 == 0:
            print(f"[atlas] frequency {i+1}/{len(pick)}", flush=True)
    freq /= max(len(pick), 1)

    present_labels = sorted(int(x) for x in np.unique(atlas) if int(x) != 0)
    meta = {
        "mode": "best_coverage_reference",
        "reference_scan_id": best_row.get("scan_id"),
        "reference_path": best_row.get("path"),
        "n_labels_present": int(best_n),
        "n_roi_voxels": int(best_vox),
        "n_freq_scans": len(pick),
        "shape": list(shape),
        "labels": FS_LABELS,
        "regions": REGION_NAMES,
        "present_labels": present_labels,
        "affine": TEMPLATE_AFFINE.tolist(),
        "cache_root": str(cache_root),
    }
    _ = rng  # reserved for future stochastic QC
    return atlas, meta, freq


def roi_vector_to_volume(atlas: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Map length-21 ROI values onto atlas voxels (background=0)."""
    out = np.zeros(atlas.shape, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape[0] != len(REGION_NAMES):
        raise ValueError(f"Expected {len(REGION_NAMES)} ROI values, got {vals.shape[0]}")
    for region, lab, val in zip(REGION_NAMES, FS_LABELS[1:], vals):
        out[atlas == int(lab)] = float(val)
        _ = region
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def pooled_probs_simple(
    arrays: Sequence[np.ndarray],
    weights: np.ndarray,
    offsets: np.ndarray,
    temperature: float,
) -> np.ndarray:
    logits = np.zeros_like(arrays[0], dtype=np.float64)
    for w, a in zip(weights, arrays):
        logits += float(w) * np.log(np.clip(a, 1e-8, 1.0))
    logits = logits / max(float(temperature), 1e-4)
    logits += offsets.reshape(1, 3)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def read_pred_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["prob_CN"] = float(row["prob_CN"])
        row["prob_MCI"] = float(row["prob_MCI"])
        row["prob_AD"] = float(row["prob_AD"])
    return rows


def align_preds_to_features(
    feature_rows: Sequence[dict],
    pred_rows: Sequence[dict],
) -> Tuple[List[dict], np.ndarray]:
    by_scan = {r.get("scan_id", ""): r for r in pred_rows}
    kept = []
    probs = []
    for fr in feature_rows:
        pr = by_scan.get(fr.get("scan_id", ""))
        if pr is None:
            continue
        kept.append(fr)
        probs.append([pr["prob_CN"], pr["prob_MCI"], pr["prob_AD"]])
    if not kept:
        raise RuntimeError("No overlapping scan_ids between features and predictions.")
    return kept, np.asarray(probs, dtype=np.float64)


def compute_scan_mrf_and_atrophy(
    rows: Sequence[dict],
    cn_mean: np.ndarray,
    cn_std: np.ndarray,
    edges,
    params: MRFParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, atrophy(n,21), q(n,21)."""
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    atrophy = np.zeros((len(rows), len(REGION_NAMES)), dtype=np.float64)
    q = np.zeros_like(atrophy)
    for i, row in enumerate(rows):
        vols = volumes_from_row(row)
        az = atrophy_zscores(vols, cn_mean, cn_std)
        qi = mean_field_infer(az, edges, params)
        atrophy[i] = az
        q[i] = qi
    return y, atrophy, q


def group_mean(mat: np.ndarray, y: np.ndarray, cls: int) -> np.ndarray:
    mask = y == cls
    if not np.any(mask):
        return np.zeros(mat.shape[1], dtype=np.float64)
    return mat[mask].mean(axis=0)


def association_by_roi(mat: np.ndarray, score: np.ndarray) -> np.ndarray:
    out = np.zeros(mat.shape[1], dtype=np.float64)
    for j in range(mat.shape[1]):
        out[j] = spearman(mat[:, j], score)
    return out


def write_matrix_csv(path: Path, matrix: np.ndarray, row_names: Sequence[str], col_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", *col_names])
        for name, row in zip(row_names, matrix):
            writer.writerow([name, *[f"{v:.6f}" for v in row]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("chapter1_foundation/outputs/v4/atlas_feature_cache_v4.csv"),
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("chapter1_foundation/outputs/v4"),
    )
    parser.add_argument(
        "--v7-config",
        type=Path,
        default=Path("chapter1_foundation/ARA-Net/reports/v7_mrf_constrained/v7_aibl_priority_profile.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("chapter1_foundation/ARA-Net/reports/brain_figures_fcstyle"),
    )
    parser.add_argument("--max-atlas-scans", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("chapter1_foundation/sample_data/cache_real"),
    )
    args = parser.parse_args()

    out = args.out_dir
    assets = out / "assets"
    maps_dir = assets / "maps"
    mat_dir = assets / "matrices"
    assets.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)

    # Copy spec
    spec_src = SCRIPT_DIR.parent / "docs" / "BRAIN_FIGURE_SPEC.md"
    if spec_src.exists():
        (out / "BRAIN_FIGURE_SPEC.md").write_text(spec_src.read_text(encoding="utf-8"), encoding="utf-8")

    rows = read_feature_csv(args.feature_csv)
    print(f"[data] feature rows={len(rows)}")

    # Focus cohort for group maps: AIBL heldout + adapt val + ADNI val (stable external story)
    focus = [
        r
        for r in rows
        if str(r.get("split", "")) in {"aibl_heldout", "aibl_adapt_val", "val", "internal_test", "train"}
    ]
    print(f"[data] focus rows={len(focus)}")

    print("[atlas] building consensus template...")
    atlas, atlas_meta, freq = build_consensus_atlas(
        rows, cache_root=args.cache_root, max_scans=args.max_atlas_scans, seed=args.seed
    )
    save_nifti(atlas.astype(np.float32), assets / "atlas_template.nii.gz")
    save_nifti(freq.max(axis=0), assets / "atlas_label_frequency_max.nii.gz")
    (assets / "atlas_template_meta.json").write_text(json.dumps(atlas_meta, indent=2), encoding="utf-8")
    print(
        f"[atlas] shape={atlas.shape} labels={sorted(int(x) for x in np.unique(atlas))} "
        f"ref={atlas_meta.get('reference_scan_id')}",
        flush=True,
    )

    print("[underlay] building mean T1 anatomical background...")
    underlay = build_mean_underlay(rows, args.cache_root, max_scans=40, seed=args.seed)
    save_nifti(underlay, assets / "anat_underlay_mean_t1.nii.gz")
    print(f"[underlay] saved shape={underlay.shape}", flush=True)

    # CN stats + MRF
    train_cn = [r for r in rows if str(r.get("split")) == "train" and int(r["label"]) == 0]
    cn_mean, cn_std = fit_cn_volume_stats(train_cn if train_cn else [r for r in rows if int(r["label"]) == 0])
    params = MRFParams()
    edges = build_roi_graph(params.couple_strength, params.anti_strength)

    y, atrophy, q = compute_scan_mrf_and_atrophy(focus, cn_mean, cn_std, edges, params)
    print(f"[mrf] computed atrophy/q for n={len(y)}")

    # Group vectors
    vectors: Dict[str, np.ndarray] = {}
    for cls, name in enumerate(CLASS_NAMES):
        vectors[f"atrophy_{name}_mean"] = group_mean(atrophy, y, cls)
        vectors[f"mrf_q_{name}_mean"] = group_mean(q, y, cls)
    vectors["atrophy_AD_minus_CN"] = vectors["atrophy_AD_mean"] - vectors["atrophy_CN_mean"]
    vectors["atrophy_MCI_minus_CN"] = vectors["atrophy_MCI_mean"] - vectors["atrophy_CN_mean"]
    vectors["mrf_q_AD_minus_CN"] = vectors["mrf_q_AD_mean"] - vectors["mrf_q_CN_mean"]
    vectors["mrf_q_MCI_minus_CN"] = vectors["mrf_q_MCI_mean"] - vectors["mrf_q_CN_mean"]

    # Alias locked method names
    vectors["atrophy_AD_mean"] = vectors["atrophy_AD_mean"]
    vectors["mrf_q_AD_mean"] = vectors["mrf_q_AD_mean"]

    # Prediction associations on AIBL heldout where possible
    heldout_feat = [r for r in rows if str(r.get("split")) == "aibl_heldout"]
    pred_dirs = {
        "atlas_bio": args.pred_root / "hybrid_atlas_clinical_baseline",
        "cascade": args.pred_root / "atlas_cascade_baseline",
    }
    base_runs = [
        "aibl_adapted_atlas_biomarker_enhanced__hgb",
        "aibl_adapted_atlas_core_clinical__hgb",
        "aibl_adapted_clinical_biomarker_only__rf_balanced",
        "aibl_adapted_clinical_core_only__hgb",
        "aibl_adapted_clinical_core_only__rf_balanced",
        "rf__logreg",
    ]
    run_paths = []
    for run in base_runs:
        folder = pred_dirs["cascade"] if run == "rf__logreg" else pred_dirs["atlas_bio"]
        path = folder / f"{run}_aibl_heldout_predictions.csv"
        if path.exists():
            run_paths.append((run, path))
    print(f"[pred] found {len(run_paths)} base runs for AIBL heldout")

    arrays = []
    aligned_rows = None
    for run, path in run_paths:
        pred_rows = read_pred_csv(path)
        aligned, probs = align_preds_to_features(heldout_feat, pred_rows)
        if aligned_rows is None:
            aligned_rows = aligned
            # recompute atrophy/q on aligned heldout only
            y_h, atrophy_h, q_h = compute_scan_mrf_and_atrophy(aligned, cn_mean, cn_std, edges, params)
        arrays.append(probs)
        print(f"[pred] {run}: n={len(aligned)}")

    if aligned_rows is not None and arrays:
        # V6 locked
        from generate_algorithm_innovation_evidence import FINAL_OFFSETS, FINAL_TEMPERATURE, FINAL_WEIGHTS

        w6 = FINAL_WEIGHTS[: len(arrays)]
        w6 = w6 / w6.sum()
        p_v6 = pooled_probs_simple(arrays, w6, FINAL_OFFSETS, FINAL_TEMPERATURE)[:, 2]
        vectors["assoc_v6_rcspe"] = association_by_roi(atrophy_h, p_v6)

        # equal log-pool
        w_eq = np.ones(len(arrays)) / len(arrays)
        p_eq = pooled_probs_simple(arrays, w_eq, np.zeros(3), 1.0)[:, 2]
        vectors["assoc_equal_logpool"] = association_by_roi(atrophy_h, p_eq)

        # V7 priority profile if present
        if args.v7_config.exists():
            cfg = json.loads(args.v7_config.read_text(encoding="utf-8"))
            # base 6 weights only for association (MRF stream is structural)
            w7 = np.asarray(cfg["weights"][: len(arrays)], dtype=np.float64)
            if w7.sum() <= 0:
                w7 = w_eq
            else:
                w7 = w7 / w7.sum()
            off = cfg.get("offsets", {})
            offsets = np.array([off.get("CN", 0.0), off.get("MCI", 0.0), off.get("AD", 0.0)], dtype=np.float64)
            t = float(cfg.get("temperature", 1.0))
            p_v7 = pooled_probs_simple(arrays, w7, offsets, t)[:, 2]
            vectors["assoc_v7_mrf"] = association_by_roi(atrophy_h, p_v7)
            vectors["assoc_v7_mrf_q"] = association_by_roi(q_h, p_v7)
        else:
            vectors["assoc_v7_mrf"] = association_by_roi(q_h, p_v6)

    # Ensure all method columns exist
    for name in METHOD_COLUMNS:
        if name not in vectors:
            vectors[name] = np.zeros(len(REGION_NAMES), dtype=np.float64)
            print(f"[warn] missing vector {name}, filled zeros")

    # Save ROI vectors + NIfTI maps
    roi_table = {"regions": REGION_NAMES, "vectors": {k: v.tolist() for k, v in vectors.items()}}
    (mat_dir / "roi_vectors.json").write_text(json.dumps(roi_table, indent=2), encoding="utf-8")

    for name in METHOD_COLUMNS:
        vol = roi_vector_to_volume(atlas, vectors[name])
        save_nifti(vol, maps_dir / f"{name}.nii.gz")
        print(f"[map] {name}")

    # Extra useful maps
    for name in [
        "atrophy_CN_mean",
        "atrophy_MCI_mean",
        "mrf_q_CN_mean",
        "mrf_q_MCI_mean",
        "atrophy_MCI_minus_CN",
        "assoc_v7_mrf_q",
    ]:
        if name in vectors:
            save_nifti(roi_vector_to_volume(atlas, vectors[name]), maps_dir / f"{name}.nii.gz")

    # Matrix for heatmap: methods × ROIs for three stages (using group means / effects)
    stage_maps = {
        "CN": "mrf_q_CN_mean",
        "MCI": "mrf_q_MCI_mean",
        "AD": "mrf_q_AD_mean",
        "AD_minus_CN": "mrf_q_AD_minus_CN",
        "MCI_minus_CN": "mrf_q_MCI_minus_CN",
        "atrophy_AD_minus_CN": "atrophy_AD_minus_CN",
        "assoc_v6": "assoc_v6_rcspe",
        "assoc_v7": "assoc_v7_mrf",
        "assoc_equal": "assoc_equal_logpool",
    }
    row_names = list(stage_maps.keys())
    mat = np.vstack([vectors[stage_maps[k]] for k in row_names])
    write_matrix_csv(mat_dir / "methods_roi_matrix.csv", mat, row_names, REGION_NAMES)

    # Top-20 by |AD-CN MRF effect|
    effect = np.abs(vectors["mrf_q_AD_minus_CN"])
    order = np.argsort(-effect)
    top_idx = order[:20].tolist()
    top_payload = {
        "rank_by": "abs(mrf_q_AD_minus_CN)",
        "regions": [REGION_NAMES[i] for i in top_idx],
        "values": {
            key: [float(vectors[key][i]) for i in top_idx]
            for key in [
                "mrf_q_AD_minus_CN",
                "mrf_q_MCI_minus_CN",
                "atrophy_AD_minus_CN",
                "assoc_v6_rcspe",
                "assoc_v7_mrf",
                "assoc_equal_logpool",
            ]
            if key in vectors
        },
    }
    (mat_dir / "top20_rois.json").write_text(json.dumps(top_payload, indent=2), encoding="utf-8")

    manifest = {
        "feature_csv": str(args.feature_csv),
        "n_focus": int(len(focus)),
        "n_heldout_aligned": int(len(aligned_rows) if aligned_rows else 0),
        "method_columns": METHOD_COLUMNS,
        "cuts": CUTS,
        "maps_dir": str(maps_dir),
        "affine": TEMPLATE_AFFINE.tolist(),
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] assets -> {out}")


if __name__ == "__main__":
    main()
