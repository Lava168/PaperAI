#!/usr/bin/env python3
"""High-end brain structural figures for the A2C-NODE paper.

Style follows npj Digital Medicine / Nature Biomedical Engineering anatomical
overlays: 6-view cortical surface, premium glass brain MIP, multiplanar
T1+stat overlays, and a single composite panel PDF.

Inputs
------
* ATE CSV from ``compute_ate.py`` (region_idx, region_name, ate_AD, ate_MCI, ate_CN)
* a FastSurfer aparc+aseg .mgz used as spatial atlas
* (optional) the matching T1 (orig.mgz) as anatomical background

Outputs (under ``--out_dir``)
-----------------------------
* ``brain_surface_6view.{png,pdf}``     L/R x lateral/medial/dorsal cortical surface
* ``brain_glass_premium.{png,pdf}``     4-view glass brain MIP (l, y, r, z)
* ``brain_multiplanar.{png,pdf}``       sagittal + coronal + axial multiplanar stat overlay
* ``brain_combined_panel.pdf``          single 3-panel composite for paper figure
* ``ate_stat_volume.nii.gz``            the volumetric stat parametric map
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib

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
    p.add_argument("--ate_csv", required=True, type=Path)
    p.add_argument(
        "--aparc",
        default="/home/lry/aibl/fastsurfer_seg/fs_subjects/AIBL_1000_I1111792/"
                "mri/aparc.DKTatlas+aseg.deep.mgz",
        type=Path,
    )
    p.add_argument(
        "--orig",
        default="/home/lry/aibl/fastsurfer_seg/fs_subjects/AIBL_1000_I1111792/"
                "mri/orig.mgz",
        type=Path,
    )
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--column", choices=["AD", "MCI", "CN"], default="AD")
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--dpi", type=int, default=400)
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


def build_stat_volume(aparc_img: nib.Nifti1Image,
                      ate_by_name: Dict[str, float]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(aparc_img.dataobj, dtype=np.int32)
    stat = np.zeros(data.shape, dtype=np.float32)
    mask = np.zeros(data.shape, dtype=np.uint8)
    for name, fs_ids in REGION_TO_FS:
        v = ate_by_name.get(name)
        if v is None:
            continue
        sel = np.isin(data, np.asarray(fs_ids, dtype=np.int32))
        stat[sel] = v
        mask[sel] = 1
    return stat, mask


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_style import apply as _style                                # type: ignore
    _style(extra=("nature",))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.image import imread
    from nilearn import datasets as nl_datasets
    from nilearn import plotting, surface

    ate = load_ate(args.ate_csv, args.column)
    if not ate:
        raise SystemExit(f"no ATE rows parsed from {args.ate_csv}")
    print(f"[premium] {len(ate)} regions; ate_{args.column} range "
          f"[{min(ate.values()):+.4f}, {max(ate.values()):+.4f}]", flush=True)

    aparc = nib.load(str(args.aparc))
    stat_data, _mask = build_stat_volume(aparc, ate)
    stat_img = nib.Nifti1Image(stat_data, aparc.affine)
    stat_path = args.out_dir / "ate_stat_volume.nii.gz"
    nib.save(stat_img, str(stat_path))

    abs_max = max(1e-6, float(np.max(np.abs(stat_data))))
    threshold = max(0.0005, 0.15 * abs_max)
    print(f"[premium] abs_max={abs_max:.4f}  threshold={threshold:.4f}",
          flush=True)

    bg_img = str(args.orig) if args.orig.is_file() else None

    # -------------------------------------------------------------- surface
    fsavg = nl_datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    def _vol_to_surf(hemi_pial: str) -> np.ndarray:
        try:
            return surface.vol_to_surf(stat_img, hemi_pial,
                                       interpolation="linear",
                                       radius=4.0, n_samples=10)
        except TypeError:
            return surface.vol_to_surf(stat_img, hemi_pial,
                                       radius=4.0, n_samples=10)

    surf_lh = _vol_to_surf(fsavg["pial_left"])
    surf_rh = _vol_to_surf(fsavg["pial_right"])
    s_abs = max(1e-6, float(np.nanmax(np.abs(np.concatenate([surf_lh, surf_rh])))))
    s_thr = max(0.0005, 0.15 * s_abs)

    fig = plt.figure(figsize=(11.0, 5.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.05)

    panels = [
        ("Left lateral",  fsavg["infl_left"],  fsavg["sulc_left"],  surf_lh, "left",  "lateral"),
        ("Left medial",   fsavg["infl_left"],  fsavg["sulc_left"],  surf_lh, "left",  "medial"),
        ("Dorsal (LH)",   fsavg["infl_left"],  fsavg["sulc_left"],  surf_lh, "left",  "dorsal"),
        ("Right lateral", fsavg["infl_right"], fsavg["sulc_right"], surf_rh, "right", "lateral"),
        ("Right medial",  fsavg["infl_right"], fsavg["sulc_right"], surf_rh, "right", "medial"),
        ("Dorsal (RH)",   fsavg["infl_right"], fsavg["sulc_right"], surf_rh, "right", "dorsal"),
    ]

    for k, (title, mesh, sulc, vec, hemi, view) in enumerate(panels):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(gs[r, c], projection="3d")
        kw = dict(stat_map=vec, hemi=hemi, view=view, bg_map=sulc,
                  cmap=args.cmap, threshold=s_thr,
                  vmax=s_abs, colorbar=False, axes=ax, figure=fig,
                  bg_on_data=True, alpha=1.0)
        try:
            plotting.plot_surf_stat_map(mesh, **kw)
        except TypeError:
            kw.pop("bg_on_data", None)
            kw.pop("alpha", None)
            plotting.plot_surf_stat_map(mesh, **kw)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_axis_off()

    cax = fig.add_axes([0.32, 0.04, 0.38, 0.025])
    sm = plt.cm.ScalarMappable(cmap=args.cmap,
                               norm=plt.Normalize(vmin=-s_abs, vmax=s_abs))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(f"ATE on P({args.column})  do(h_k = 0) - factual",
                 fontsize=10)
    cb.outline.set_linewidth(0.5)

    fig.suptitle(
        f"A2C-NODE counterfactual ATE on cortical surface "
        f"(5-seed ensemble, P({args.column}))",
        fontsize=12, y=0.98)
    surf_png = args.out_dir / "brain_surface_6view.png"
    surf_pdf = args.out_dir / "brain_surface_6view.pdf"
    fig.savefig(surf_png, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    fig.savefig(surf_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[premium] wrote {surf_png}", flush=True)

    # ----------------------------------------------------------- glass brain
    fig = plt.figure(figsize=(11.0, 3.0))
    plotting.plot_glass_brain(
        str(stat_path), figure=fig, display_mode="lyrz",
        colorbar=True, plot_abs=False, threshold=threshold,
        symmetric_cbar=True, vmin=-abs_max, vmax=abs_max,
        cmap=args.cmap, alpha=0.85,
        title=f"Glass-brain MIP   ATE on P({args.column}) "
              f"(5-seed ensemble, |peak|={abs_max:.3f})",
    )
    glass_png = args.out_dir / "brain_glass_premium.png"
    glass_pdf = args.out_dir / "brain_glass_premium.pdf"
    fig.savefig(glass_png, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    fig.savefig(glass_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[premium] wrote {glass_png}", flush=True)

    # ----------------------------------------------------------- multiplanar
    fig = plt.figure(figsize=(11.5, 6.5))
    fig.suptitle(
        f"Multiplanar reconstruction: per-region ATE on P({args.column}) "
        f"overlaid on T1",
        fontsize=12, y=0.985)
    for row, (mode, label) in enumerate([
        ("x", "Sagittal"),
        ("y", "Coronal"),
        ("z", "Axial"),
    ]):
        ax = fig.add_axes([0.03, 0.69 - row * 0.32, 0.94, 0.27])
        plotting.plot_stat_map(
            str(stat_path), bg_img=bg_img, figure=fig, axes=ax,
            display_mode=mode, cut_coords=7,
            colorbar=(row == 1), threshold=threshold,
            symmetric_cbar=True, vmin=-abs_max, vmax=abs_max,
            cmap=args.cmap, annotate=True, draw_cross=False,
            title=f"{label}",
        )
    mp_png = args.out_dir / "brain_multiplanar.png"
    mp_pdf = args.out_dir / "brain_multiplanar.pdf"
    fig.savefig(mp_png, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    fig.savefig(mp_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[premium] wrote {mp_png}", flush=True)

    # ----------------------------------------------------------- composite
    fig = plt.figure(figsize=(11.0, 13.5))
    gs = gridspec.GridSpec(3, 1, figure=fig,
                           height_ratios=[1.05, 0.55, 1.45], hspace=0.05)

    def _embed(ax, png_path: Path, label: str) -> None:
        ax.imshow(imread(str(png_path)))
        ax.set_axis_off()
        ax.text(-0.005, 1.005, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")

    _embed(fig.add_subplot(gs[0]), surf_png,  "a")
    _embed(fig.add_subplot(gs[1]), glass_png, "b")
    _embed(fig.add_subplot(gs[2]), mp_png,    "c")
    comp_pdf = args.out_dir / "brain_combined_panel.pdf"
    fig.savefig(comp_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out_dir / "brain_combined_panel.png",
                dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[premium] wrote {comp_pdf}", flush=True)


if __name__ == "__main__":
    main()
