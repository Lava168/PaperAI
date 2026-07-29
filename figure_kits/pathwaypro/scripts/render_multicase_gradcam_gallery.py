#!/usr/bin/env python3
"""Render a risk-ordered multi-subject Grad-CAM gallery from real volumes.

Each case contains:
- a T1 axial slice with real Grad-CAM overlay
- a 3D surface rendering of high-attribution Grad-CAM voxels

Cases are sorted by the model's predicted AD probability (`p_ad`) and sampled
across the full risk range. This is intended for a manuscript/supplementary
figure, not for model training.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


LABELS = {0: "CN", 1: "MCI", 2: "AD"}
RISK_LABELS_3 = ["Low p(AD)", "Intermediate p(AD)", "High p(AD)"]


def robust01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr)
    arr = np.clip(arr, lo, hi)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    finite = cam[np.isfinite(cam)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return np.zeros_like(cam, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99.8])
    if hi <= lo:
        return np.zeros_like(cam, dtype=np.float32)
    cam = np.clip(cam, lo, hi)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)


def best_axial_slice(cam: np.ndarray, brain: np.ndarray) -> int:
    weighted = np.asarray(cam, dtype=np.float32) * np.asarray(brain, dtype=np.float32)
    z = int(np.argmax(weighted.sum(axis=(0, 1))))
    if weighted[:, :, z].max() > 0:
        return z
    return int(np.argmax(brain.sum(axis=(0, 1))))


def shaded_facecolors(base_color, normals: np.ndarray, alpha: float, light_dir=(0.3, -0.6, 0.74)):
    rgba = np.array(to_rgba(base_color), dtype=float)
    light = np.asarray(light_dir, dtype=float)
    light = light / (np.linalg.norm(light) + 1e-8)
    face_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    intensity = np.clip(face_normals @ light, 0, 1)
    intensity = 0.46 + 0.54 * intensity
    colors = np.empty((len(normals), 4), dtype=float)
    colors[:, :3] = np.clip(rgba[:3] * intensity[:, None], 0, 1)
    colors[:, 3] = alpha
    return colors


def add_surface(ax, mask: np.ndarray, color, alpha=1.0, step_size=1, linewidth=0.0, shade=False):
    mask = np.asarray(mask, dtype=np.uint8)
    if int(mask.sum()) < 30:
        return None
    verts, faces, normals, _ = measure.marching_cubes(mask, level=0.5, step_size=step_size)
    mesh = Poly3DCollection(verts[faces], linewidth=linewidth)
    if shade:
        face_normals = normals[faces].mean(axis=1)
        mesh.set_facecolor(shaded_facecolors(color, face_normals, alpha))
    else:
        mesh.set_facecolor(color)
        mesh.set_alpha(alpha)
    mesh.set_edgecolor((0, 0, 0, 0.03))
    ax.add_collection3d(mesh)
    return verts


def add_cam_surface(ax, cam: np.ndarray, brain: np.ndarray, percentile: float, alpha=0.95):
    threshold = float(np.percentile(cam[brain], percentile))
    cam_mask = (cam >= threshold) & brain
    if int(cam_mask.sum()) < 30:
        threshold = float(np.percentile(cam[brain], 98.0))
        cam_mask = (cam >= threshold) & brain
    mask = np.asarray(cam_mask, dtype=np.uint8)
    if int(mask.sum()) < 30:
        return cam_mask
    verts, faces, normals, _ = measure.marching_cubes(mask, level=0.5, step_size=1)
    coords = np.clip(np.rint(verts).astype(int), 0, np.asarray(cam.shape) - 1)
    values = cam[coords[:, 0], coords[:, 1], coords[:, 2]]
    face_values = values[faces].mean(axis=1)
    vmin, vmax = np.percentile(face_values, [5, 98]) if len(face_values) else (0, 1)
    scaled = np.clip((face_values - vmin) / (vmax - vmin + 1e-6), 0, 1)
    colors = plt.get_cmap("autumn_r")(scaled)
    face_normals = normals[faces].mean(axis=1)
    light = np.asarray((0.25, -0.55, 0.79), dtype=float)
    light = light / (np.linalg.norm(light) + 1e-8)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
    shade = 0.55 + 0.45 * np.clip(face_normals @ light, 0, 1)
    colors[:, :3] = np.clip(colors[:, :3] * shade[:, None], 0, 1)
    colors[:, 3] = alpha
    mesh = Poly3DCollection(verts[faces], linewidth=0.015)
    mesh.set_facecolor(colors)
    mesh.set_edgecolor((0, 0, 0, 0.035))
    ax.add_collection3d(mesh)
    return cam_mask


def bbox(mask: np.ndarray, pad=8):
    coords = np.argwhere(mask)
    if coords.size == 0:
        shape = np.asarray(mask.shape)
        return np.zeros(3), shape
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad, np.asarray(mask.shape))
    return lo.astype(float), hi.astype(float)


def setup_3d_axis(ax, lo, hi):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=20, azim=-52)
    ax.set_axis_off()
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))


def risk_color(value: float, vmin: float, vmax: float):
    if vmax <= vmin:
        t = 0.5
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
    # Blue -> amber -> red, intentionally readable on white backgrounds.
    if t < 0.5:
        tt = t / 0.5
        c0 = np.array([42, 110, 176], dtype=float)
        c1 = np.array([242, 183, 64], dtype=float)
    else:
        tt = (t - 0.5) / 0.5
        c0 = np.array([242, 183, 64], dtype=float)
        c1 = np.array([202, 76, 58], dtype=float)
    rgb = (c0 * (1 - tt) + c1 * tt) / 255.0
    return tuple(rgb.tolist())


def risk_title_color(value: float, vmin: float, vmax: float):
    r, g, b = risk_color(value, vmin, vmax)
    return (r, g, b, 0.22)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["subject_id", "label", "pred", "correct", "p_ad"])
    df = pd.read_csv(path)
    keep = [c for c in ["subject_id", "label", "pred", "correct", "p_ad"] if c in df.columns]
    return df[keep].copy()


def list_candidate_cases(args) -> pd.DataFrame:
    subjects = pd.read_csv(args.gradcam_subjects)
    subjects = subjects[(subjects["branch"] == args.branch) & (subjects["target"] == args.target)].copy()
    subjects["p_ad"] = pd.to_numeric(subjects["p_ad"], errors="coerce")
    preds = load_predictions(args.predictions)
    if not preds.empty:
        subjects = subjects.merge(preds.drop(columns=["p_ad"], errors="ignore"), on="subject_id", how="left")

    rows = []
    for _, row in subjects.sort_values("p_ad").iterrows():
        sid = str(row["subject_id"])
        cache = args.cache_dir / f"{sid}.npz"
        cam_path = args.project_root / str(row["cam_path"])
        if not cache.exists() or not cam_path.exists():
            continue
        try:
            cam = np.asarray(nib.load(str(cam_path)).get_fdata(), dtype=np.float32)
            cam_max = float(np.nanmax(cam))
            cam_sum = float(np.nansum(cam))
        except Exception:
            continue
        if not np.isfinite(cam_max) or cam_max <= args.min_cam_max or cam_sum <= args.min_cam_sum:
            continue
        out = row.to_dict()
        out["cache_path"] = cache
        out["cam_abs_path"] = cam_path
        out["cam_max"] = cam_max
        out["cam_sum"] = cam_sum
        rows.append(out)

    if not rows:
        raise RuntimeError("No non-empty Grad-CAM cases found.")
    return pd.DataFrame(rows).sort_values("p_ad").reset_index(drop=True)


def select_cases(candidates: pd.DataFrame, n_cases: int, mode: str) -> pd.DataFrame:
    if len(candidates) <= n_cases:
        return candidates.copy()
    if mode == "quantile":
        idx = np.linspace(0, len(candidates) - 1, n_cases).round().astype(int)
        idx = np.unique(idx)
        while len(idx) < n_cases:
            for i in range(len(candidates)):
                if i not in idx:
                    idx = np.append(idx, i)
                    break
        return candidates.iloc[np.sort(idx[:n_cases])].copy()
    if mode == "extremes":
        half = n_cases // 2
        chosen = pd.concat([candidates.head(half), candidates.tail(n_cases - half)])
        return chosen.sort_values("p_ad").copy()
    return candidates.head(n_cases).copy()


def render_slice(
    ax,
    image: np.ndarray,
    cam: np.ndarray,
    brain: np.ndarray,
    title: str,
    title_bg=(1, 1, 1, 0),
):
    z = best_axial_slice(cam, brain)
    img = robust01(image[:, :, z]).T
    overlay = cam[:, :, z].T
    ax.imshow(img, cmap="gray", origin="lower")
    rgba = plt.get_cmap("inferno")(np.clip(overlay, 0, 1))
    rgba[..., 3] = np.where(overlay > 0.06, np.clip((overlay - 0.06) / 0.5, 0, 0.76), 0)
    ax.imshow(rgba, origin="lower")
    ax.set_title(
        title,
        fontsize=8.4,
        pad=2,
        bbox={"facecolor": title_bg, "edgecolor": (0, 0, 0, 0.12), "boxstyle": "round,pad=0.2"},
    )
    ax.set_axis_off()


def render_3d(ax, cam: np.ndarray, brain: np.ndarray, percentile: float, render_style: str):
    if render_style == "enhanced":
        add_surface(ax, brain, "#C9D0D8", alpha=0.16, step_size=3, shade=True)
        cam_mask = add_cam_surface(ax, cam, brain, percentile, alpha=0.96)
    else:
        threshold = float(np.percentile(cam[brain], percentile))
        cam_mask = (cam >= threshold) & brain
        if int(cam_mask.sum()) < 30:
            threshold = float(np.percentile(cam[brain], 98.0))
            cam_mask = (cam >= threshold) & brain
        add_surface(ax, brain, "#D8DDE5", alpha=0.06, step_size=3)
        add_surface(ax, cam_mask, "#F05A28", alpha=0.90, step_size=1, linewidth=0.01)
    lo, hi = bbox(brain | cam_mask, pad=8)
    setup_3d_axis(ax, lo, hi)


def iter_case_layout(selected: pd.DataFrame, cols: int, layout: str):
    if layout == "risk_columns":
        chunks = np.array_split(selected.sort_values("p_ad").reset_index(drop=True), cols)
        rows = max(len(chunk) for chunk in chunks)
        for c, chunk in enumerate(chunks):
            for r, (_, row) in enumerate(chunk.iterrows()):
                yield r, c, row
        return rows, chunks

    rows = int(math.ceil(len(selected) / cols))
    chunks = []
    for i, (_, row) in enumerate(selected.iterrows()):
        yield i // cols, i % cols, row
    return rows, chunks


def risk_column_chunks(selected: pd.DataFrame, cols: int):
    if cols <= 1:
        return [selected.sort_values("p_ad").reset_index(drop=True)]
    return list(np.array_split(selected.sort_values("p_ad").reset_index(drop=True), cols))


def render_gallery(args, selected: pd.DataFrame):
    n = len(selected)
    chunks = risk_column_chunks(selected, args.cols) if args.layout == "risk_columns" else []
    rows = max(len(chunk) for chunk in chunks) if chunks else int(math.ceil(n / args.cols))
    fig = plt.figure(figsize=(args.cols * 3.35, rows * 3.12 + 1.05), facecolor="white")
    gs = fig.add_gridspec(rows, args.cols * 2, wspace=0.02, hspace=0.30)
    risk_min = float(selected["p_ad"].min())
    risk_max = float(selected["p_ad"].max())

    if args.layout == "risk_columns":
        for c, chunk in enumerate(chunks):
            if chunk.empty:
                continue
            label = RISK_LABELS_3[c] if args.cols == 3 and c < len(RISK_LABELS_3) else f"Risk bin {c + 1}"
            lo = float(chunk["p_ad"].min())
            hi = float(chunk["p_ad"].max())
            x = (c + 0.5) / args.cols
            fig.text(
                x,
                0.925,
                f"{label}: pAD {lo:.3f}-{hi:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                bbox={
                    "facecolor": risk_title_color((lo + hi) / 2, risk_min, risk_max),
                    "edgecolor": (0, 0, 0, 0.16),
                    "boxstyle": "round,pad=0.28",
                },
            )

    cases = []
    if args.layout == "risk_columns":
        for c, chunk in enumerate(chunks):
            for r, (_, row) in enumerate(chunk.iterrows()):
                cases.append((r, c, row))
    else:
        for i, (_, row) in enumerate(selected.iterrows()):
            cases.append((i // args.cols, i % args.cols, row))

    for r, c, row in cases:
        sid = str(row["subject_id"])
        data = np.load(row["cache_path"])
        image = data["image"].astype(np.float32)
        seg = data["seg"].astype(np.int16)
        brain = seg > 0
        cam_raw = np.asarray(nib.load(str(row["cam_abs_path"])).get_fdata(), dtype=np.float32)
        cam = normalize_cam(cam_raw) * brain
        label = LABELS.get(int(row["label"]), str(row.get("label", "?"))) if pd.notna(row.get("label", np.nan)) else "?"
        pred = LABELS.get(int(row["pred"]), str(row.get("pred", "?"))) if pd.notna(row.get("pred", np.nan)) else "?"
        correct = int(row["correct"]) if pd.notna(row.get("correct", np.nan)) else None
        mark = "OK" if correct == 1 else ("ERR" if correct == 0 else "")
        p_ad = float(row["p_ad"])
        title = f"{sid}\n{label}->{pred} {mark}; pAD={p_ad:.3f}"

        ax_slice = fig.add_subplot(gs[r, c * 2])
        render_slice(ax_slice, image, cam, brain, title, title_bg=risk_title_color(p_ad, risk_min, risk_max))
        ax_3d = fig.add_subplot(gs[r, c * 2 + 1], projection="3d")
        render_3d(ax_3d, cam, brain, args.cam_percentile, args.render_style)

    fig.suptitle(
        f"Risk-ordered multi-subject T1 Grad-CAM rendering ({args.model_name}, {args.target})",
        fontsize=15,
        y=0.995,
    )
    if args.layout == "risk_columns":
        fig.text(
            0.5,
            0.955,
            "Low predicted AD risk  \u2192  High predicted AD risk",
            ha="center",
            va="center",
            fontsize=11,
            color="black",
        )
    fig.text(
        0.5,
        0.012,
        f"Each case shows real T1+Grad-CAM slice and true 3D high-attribution voxels. Color bands and columns encode p(AD) from low to high; n={n}.",
        ha="center",
        fontsize=9,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def write_case_table(selected: pd.DataFrame, out: Path):
    cols = [
        "subject_id",
        "p_ad",
        "score",
        "label",
        "pred",
        "correct",
        "cam_max",
        "cam_sum",
        "cam_path",
    ]
    selected[[c for c in cols if c in selected.columns]].to_csv(out, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/full_cache"))
    parser.add_argument("--gradcam_subjects", type=Path, default=Path("analysis_explainability_factorvae_seed2_full/gradcam_subjects.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("analysis_eval_candidates/factorvae_seed2/predictions.csv"))
    parser.add_argument("--branch", default="t1")
    parser.add_argument("--target", default="ad_logit")
    parser.add_argument("--n_cases", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--selection", choices=["quantile", "extremes", "first"], default="quantile")
    parser.add_argument("--layout", choices=["row_major", "risk_columns"], default="row_major")
    parser.add_argument("--cam_percentile", type=float, default=99.0)
    parser.add_argument("--min_cam_max", type=float, default=1e-7)
    parser.add_argument("--min_cam_sum", type=float, default=1e-5)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render_style", choices=["standard", "enhanced"], default="standard")
    parser.add_argument("--model_name", default="PPRO")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    if not args.cache_dir.is_absolute():
        args.cache_dir = args.project_root / args.cache_dir
    if not args.gradcam_subjects.is_absolute():
        args.gradcam_subjects = args.project_root / args.gradcam_subjects
    if not args.predictions.is_absolute():
        args.predictions = args.project_root / args.predictions
    if not args.predictions.exists():
        fallback = args.project_root / "results" / "analysis_eval_candidates" / "factorvae_seed2" / "predictions.csv"
        if fallback.exists():
            args.predictions = fallback
    if not args.out.is_absolute():
        args.out = args.project_root / args.out

    candidates = list_candidate_cases(args)
    selected = select_cases(candidates, args.n_cases, args.selection)
    render_gallery(args, selected)
    write_case_table(selected, args.out.with_suffix(".cases.csv"))
    print(args.out)
    print(args.out.with_suffix(".cases.csv"))


if __name__ == "__main__":
    main()
