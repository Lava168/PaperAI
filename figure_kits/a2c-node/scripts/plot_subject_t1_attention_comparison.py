#!/usr/bin/env python3
"""Typical ADNI T1 + A2C attention vs baseline attention heatmaps (English, Nature style).

This renders a single subject's T1 MRI (3 orthogonal mid-slices), plus two
region-level heatmaps overlaid using the subject's FastSurfer aparc+aseg labels
stored in the ARA-Net ``cache_real`` NPZ.

Heatmaps:
  - A2C-NODE: AD class-mean per-region attention weights from
    ``runs/main/analysis/attention_by_class.csv`` (row order: CN, MCI, AD).
  - Baseline: ARA-Net attention adjacency from one representative CV run
    (chapter1_foundation experiment_results_ssl). We average over AD subjects
    and attention heads, then reduce to node importance by mean outgoing
    attention (row mean).

Outputs:
  paper/figures/results/Figure_subject_attention.{png,pdf}
  paper/figures/results/Figure_subject_attention_data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compose_paper_figures import NAT_PALETTE, apply_nature_rc, panel_label  # noqa: E402


FS_LABELS_21 = [
    2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
    41, 42, 43, 49, 50, 51, 52, 53, 54, 58,
]
REGION_NAMES_21 = [
    "L-WM", "L-Cortex", "L-Lat-Vent", "L-Thalamus", "L-Caudate", "L-Putamen",
    "L-Pallidum", "Brain-Stem", "L-Hippocampus", "L-Amygdala", "L-Accumbens",
    "R-WM", "R-Cortex", "R-Lat-Vent", "R-Thalamus", "R-Caudate", "R-Putamen",
    "R-Pallidum", "R-Hippocampus", "R-Amygdala", "R-Accumbens",
]
assert len(FS_LABELS_21) == 21
assert len(REGION_NAMES_21) == 21


def load_npz_subject(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    d = np.load(path, allow_pickle=True)
    img = np.asarray(d["image"], dtype=np.float32)
    seg = np.asarray(d["seg"], dtype=np.int32)
    label = int(np.asarray(d["label"]).item())
    return img, seg, label


def load_a2c_ad_attention(attn_csv: Path) -> np.ndarray:
    # CSV: header = 21 region names; rows: CN, MCI, AD
    lines = attn_csv.read_text().strip().splitlines()
    if len(lines) < 4:
        raise SystemExit(f"unexpected attention_by_class.csv rows: {len(lines)}")
    # AD row is line 4 (0-based idx 3)
    ad = np.array([float(x) for x in lines[3].split(",")], dtype=np.float32)
    if ad.shape[0] != 21:
        raise SystemExit(f"expected 21 attention weights, got {ad.shape}")
    s = float(ad.sum())
    return ad / (s if s > 0 else 1.0)


def load_baseline_ad_node_weights(all_results_json: Path) -> np.ndarray:
    obj = json.loads(all_results_json.read_text())
    # pick one ARA-Net config entry that includes attention_maps
    key = None
    for k in obj.keys():
        if k.startswith("Ours (Atlas+AnatDist)") and "fold0" in k:
            key = k
            break
    if key is None:
        key = next(iter(obj.keys()))
    v = obj[key]
    A = np.asarray(v.get("attention_maps"), dtype=np.float32)  # (N, H, 21, 21)
    y = np.asarray(v.get("attention_labels"), dtype=np.int64)  # (N,)
    if A.ndim != 4 or A.shape[-1] != 21 or y.ndim != 1:
        raise SystemExit(f"unexpected baseline attention shapes: A={A.shape}, y={y.shape}")
    m = y == 2  # AD
    if not np.any(m):
        raise SystemExit("no AD subjects found in baseline attention_labels")
    A_ad = A[m].mean(axis=(0, 1))  # (21, 21)
    node = A_ad.mean(axis=1)  # outgoing mean attention per node
    node = np.clip(node, 0, None)
    s = float(node.sum())
    return node / (s if s > 0 else 1.0)


def weights_to_volume(seg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vol = np.zeros(seg.shape, dtype=np.float32)
    for fs_id, w in zip(FS_LABELS_21, weights):
        vol[seg == fs_id] = float(w)
    return vol


def mid_slices(vol: np.ndarray) -> Tuple[int, int, int]:
    z = vol.shape[2] // 2
    y = vol.shape[1] // 2
    x = vol.shape[0] // 2
    return x, y, z


def norm_img(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return img
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subject_npz",
        type=Path,
        default=Path(
            "/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/"
            "chapter1_foundation/sample_data/cache_real/ADNI_002_S_0295_sc_I118671.npz"
        ),
    )
    ap.add_argument(
        "--a2c_attention_csv",
        type=Path,
        default=PROJ / "runs/main/analysis/attention_by_class.csv",
    )
    ap.add_argument(
        "--baseline_all_results_json",
        type=Path,
        default=Path(
            "/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/"
            "chapter1_foundation/outputs/experiments/experiment_results_ssl/"
            "seed_42/all_results.json"
        ),
    )
    ap.add_argument("--out_dir", type=Path, default=PROJ / "paper/figures/results")
    args = ap.parse_args()

    apply_nature_rc()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    img, seg, y = load_npz_subject(args.subject_npz)
    img_n = norm_img(img)

    w_a2c = load_a2c_ad_attention(args.a2c_attention_csv)
    w_bsl = load_baseline_ad_node_weights(args.baseline_all_results_json)

    hm_a2c = weights_to_volume(seg, w_a2c)
    hm_bsl = weights_to_volume(seg, w_bsl)

    sx, sy, sz = mid_slices(img)

    fig = plt.figure(figsize=(7.2, 4.2))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.06, right=0.98, bottom=0.08, top=0.90,
        wspace=0.02, hspace=0.10,
    )

    def draw_triplet(col: int, base_title: str, overlay: np.ndarray | None) -> None:
        # sagittal (x), coronal (y), axial (z)
        slc = [
            (img_n[sx, :, :].T, None if overlay is None else overlay[sx, :, :].T, "Sagittal"),
            (img_n[:, sy, :].T, None if overlay is None else overlay[:, sy, :].T, "Coronal"),
            (img_n[:, :, sz].T, None if overlay is None else overlay[:, :, sz].T, "Axial"),
        ]
        for r, (bg, ov, name) in enumerate(slc):
            ax = fig.add_subplot(gs[r, col])
            ax.imshow(bg, cmap="gray", origin="lower", interpolation="nearest")
            if ov is not None:
                ax.imshow(
                    ov,
                    cmap="magma",
                    origin="lower",
                    interpolation="nearest",
                    alpha=0.70,
                    vmin=0,
                    vmax=float(max(1e-6, np.max(overlay))),
                )
            ax.set_axis_off()
            if r == 0:
                ax.set_title(base_title, loc="left", pad=3, color=NAT_PALETTE["slate"])
            ax.text(
                0.02, 0.98, name,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=6.5,
                color="white" if ov is not None else NAT_PALETTE["panel"],
                bbox=dict(boxstyle="round,pad=0.18", facecolor=(0, 0, 0, 0.25), edgecolor="none"),
            )

    draw_triplet(0, f"Typical ADNI T1 (subject: {args.subject_npz.stem})", None)
    draw_triplet(1, "A2C-NODE attention (AD class mean)", hm_a2c)
    draw_triplet(2, "Baseline attention (AD mean; node importance)", hm_bsl)

    # Panel letters
    panel_label(fig.axes[0], "a", dx=-0.12, dy=1.05)
    panel_label(fig.axes[3], "b", dx=-0.12, dy=1.05)
    panel_label(fig.axes[6], "c", dx=-0.12, dy=1.05)

    fig.suptitle(
        "Subject-level illustration: raw T1 and region-level attention heatmaps",
        y=0.98,
        fontsize=8.5,
        color=NAT_PALETTE["slate"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "Figure_subject_attention.png"
    out_pdf = args.out_dir / "Figure_subject_attention.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    data = {
        "subject_npz": str(args.subject_npz),
        "subject_label": y,
        "fs_labels_21": FS_LABELS_21,
        "region_names_21": REGION_NAMES_21,
        "a2c_attention_ad": w_a2c.tolist(),
        "baseline_node_importance_ad": w_bsl.tolist(),
        "baseline_source": str(args.baseline_all_results_json),
        "a2c_source": str(args.a2c_attention_csv),
    }
    (args.out_dir / "Figure_subject_attention_data.json").write_text(json.dumps(data, indent=2))

    print(f"[subject-attn] wrote {out_png} / {out_pdf}")


if __name__ == "__main__":
    main()

