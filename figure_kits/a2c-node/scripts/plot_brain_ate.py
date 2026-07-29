#!/usr/bin/env python3
"""Render the per-region ATE on a 3D brain (glass-brain projection +
statistical parametric map / SPM style).

Inputs
------
* an ATE CSV produced by :mod:`scripts.compute_ate` (one row per region,
  with ``region_idx``, ``region_name``, ``ate_AD``, ...)
* a FastSurfer ``aparc.DKTatlas+aseg.deep.mgz`` from any subject — used
  purely as the *spatial* atlas; voxels carrying each anatomical FS
  label are replaced by that region's ATE on AD probability.

Outputs (under ``--out_dir``)
-----------------------------
* ``ate_stat.nii.gz``    The "statistical parametric map"
* ``ate_glass_brain.png`` Glass-brain MIP (lateral / coronal / axial)
* ``ate_stat_map.png``    Multi-slice axial stat-map overlay on T1 anat
* ``ate_top_regions.txt`` Plain-text summary of the strongest regions
"""
from __future__ import annotations

import argparse
import csv as _csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib

# Map our 21 model regions -> the FreeSurfer label IDs they correspond to.
# Bilateral cortex is mapped via the DKT-atlas cortical IDs (1000-1035 L,
# 2000-2035 R) because FastSurfer's deep merged segmentation does not use
# the lumped 3 / 42 codes for the cortex sheet itself.
REGION_TO_FS: List[Tuple[str, List[int]]] = [
    ("Left-Cerebral-WM",         [2]),
    ("Left-Cerebral-Cortex",     list(range(1000, 1036))),
    ("Left-Lateral-Ventricle",   [4]),
    ("Left-Thalamus",            [10]),
    ("Left-Caudate",             [11]),
    ("Left-Putamen",             [12]),
    ("Left-Pallidum",            [13]),
    ("Brain-Stem",               [16]),
    ("Left-Hippocampus",         [17]),
    ("Left-Amygdala",            [18]),
    ("Left-Accumbens",           [26]),
    ("Right-Cerebral-WM",        [41]),
    ("Right-Cerebral-Cortex",    list(range(2000, 2036))),
    ("Right-Lateral-Ventricle",  [43]),
    ("Right-Thalamus",           [49]),
    ("Right-Caudate",            [50]),
    ("Right-Putamen",            [51]),
    ("Right-Pallidum",           [52]),
    ("Right-Hippocampus",        [53]),
    ("Right-Amygdala",           [54]),
    ("Right-Accumbens",          [58]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ate_csv", required=True, type=Path,
                   help="output of compute_ate.py (region,ate_AD,ate_CN,ate_MCI,...)")
    p.add_argument("--aparc",
                   default="/home/lry/aibl/fastsurfer_seg/fs_subjects/AIBL_1000_I1111792/"
                           "mri/aparc.DKTatlas+aseg.deep.mgz",
                   type=Path,
                   help="any FastSurfer aparc+aseg .mgz, used as the spatial atlas")
    p.add_argument("--orig", type=Path, default=None,
                   help="optional companion T1 (orig.mgz) for stat-map overlay; "
                        "if not given we synthesise a brain mask from the atlas")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--column", choices=["AD", "MCI", "CN"], default="AD",
                   help="which ATE column to render")
    p.add_argument("--show_top_k", type=int, default=10)
    return p.parse_args()


def load_ate(csv_path: Path, column: str) -> Dict[str, float]:
    col = f"ate_{column}"
    out: Dict[str, float] = {}
    with open(csv_path, newline="") as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            try:
                out[r["region_name"]] = float(r[col])
            except Exception:
                continue
    return out


def build_stat_volume(aparc: nib.Nifti1Image, ate_by_name: Dict[str, float]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (stat_data float32, anat_mask uint8)."""
    data = np.asarray(aparc.dataobj, dtype=np.int32)
    stat = np.zeros(data.shape, dtype=np.float32)
    mask = np.zeros(data.shape, dtype=np.uint8)
    miss = []
    for name, fs_ids in REGION_TO_FS:
        v = ate_by_name.get(name)
        if v is None:
            miss.append(name)
            continue
        sel = np.isin(data, np.asarray(fs_ids, dtype=np.int32))
        stat[sel] = v
        mask[sel] = 1
    if miss:
        print("[brain] WARN: ATE missing for", miss, flush=True)
    return stat, mask


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ate = load_ate(args.ate_csv, args.column)
    if not ate:
        raise SystemExit(f"no ATE rows parsed from {args.ate_csv}")
    print(f"[brain] {len(ate)} regions with ate_{args.column}; "
          f"range [{min(ate.values()):+.3f}, {max(ate.values()):+.3f}]",
          flush=True)

    aparc = nib.load(str(args.aparc))
    stat, mask = build_stat_volume(aparc, ate)

    # Save NIfTI
    out_nii = args.out_dir / "ate_stat.nii.gz"
    nib.save(nib.Nifti1Image(stat, aparc.affine), out_nii)
    print(f"[brain] wrote {out_nii}", flush=True)

    # ---------------- shared style --------------------------------------
    from _plot_style import apply as _style
    _style(extra=("nature",))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting

    abs_max = max(1e-6, float(np.max(np.abs(stat))))
    threshold = max(0.005, 0.2 * abs_max)                              # 20% of peak

    # ---------------- (a) glass brain MIP -------------------------------
    fig = plt.figure(figsize=(7.0, 2.4))
    disp = plotting.plot_glass_brain(
        out_nii, figure=fig,
        display_mode="lyrz",                                            # L,Y,R,Z
        colorbar=True, plot_abs=False, threshold=threshold,
        symmetric_cbar=True, vmin=-abs_max, vmax=abs_max,
        cmap="RdBu_r",
        title=f"ATE on P({args.column})  do(h_k=0) - factual",
    )
    out_glass = args.out_dir / "ate_glass_brain.png"
    fig.savefig(out_glass, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[brain] wrote {out_glass}", flush=True)

    # ---------------- (b) statistical parametric map (anat overlay) ------
    bg_img = None
    if args.orig is not None and args.orig.is_file():
        bg_img = str(args.orig)
    fig = plt.figure(figsize=(8.0, 2.6))
    disp2 = plotting.plot_stat_map(
        out_nii, bg_img=bg_img, figure=fig,
        display_mode="z", cut_coords=7,
        colorbar=True, threshold=threshold,
        symmetric_cbar=True, vmin=-abs_max, vmax=abs_max,
        cmap="RdBu_r",
        title=f"Per-region ATE on P({args.column})",
    )
    out_stat = args.out_dir / "ate_stat_map.png"
    fig.savefig(out_stat, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[brain] wrote {out_stat}", flush=True)

    # ---------------- (c) top-K text summary ----------------------------
    sorted_items = sorted(ate.items(), key=lambda kv: -abs(kv[1]))
    lines = [f"Top {args.show_top_k} regions by |ATE_{args.column}|"]
    for n, v in sorted_items[: args.show_top_k]:
        lines.append(f"  {n:<24}  {v:+.4f}")
    out_txt = args.out_dir / "ate_top_regions.txt"
    out_txt.write_text("\n".join(lines))
    print(f"[brain] wrote {out_txt}", flush=True)


if __name__ == "__main__":
    main()
