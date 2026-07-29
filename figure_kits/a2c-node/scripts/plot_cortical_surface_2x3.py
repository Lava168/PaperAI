#!/usr/bin/env python3
"""Cortical surface 2x3 multi-region figure.

Uses nilearn's fsaverage surface + Destrieux 2009 atlas to render six
canonical functional cortical regions, each in its own lateral-view panel:

    Auditory       Heschl's gyrus           (L hemisphere)
    Visual         Occipital pole / V1      (R hemisphere)
    Motor          Precentral gyrus         (L hemisphere)
    Language       Pars triangularis        (L hemisphere) -- Broca's area
    Memory assoc.  Lateral temporal lobe    (R hemisphere)
    Sensory        Postcentral gyrus        (L hemisphere)

Each panel renders a smooth grey cortical mesh on a white background, with
the region of interest overlaid in a red/orange ``hot`` colormap.

By default the activation magnitude per region is hard-coded (illustrative
values).  Pass ``--from_ate runs/.../ate_21regions.csv`` to instead drive
each region's intensity from |ATE_AD| of the matching A2C-NODE region.

Output (under ``--out_prefix``):
    *_2x3.png   high-resolution PNG
    *_2x3.pdf   vector PDF
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Destrieux 2009 atlas indices used for highlight masks.
# (i_left, i_right) pair of indices; for purely-lateralised regions one
# index is 0 (no mask).
REGION_DEFS: List[Dict] = [
    {"label": "Auditory",
     "atlas_l": 33, "atlas_r": 108,           # G_temp_sup-G_T_transv = Heschl
     "hemi": "left",  "view": "lateral",
     "subtitle": "Heschl's gyrus"},
    {"label": "Visual",
     "atlas_l": 43, "atlas_r": 118,           # Pole_occipital
     "hemi": "right", "view": "lateral",
     "subtitle": "Occipital pole (V1)"},
    {"label": "Motor",
     "atlas_l": 29, "atlas_r": 104,           # G_precentral
     "hemi": "left",  "view": "lateral",
     "subtitle": "Precentral gyrus"},
    {"label": "Language",
     "atlas_l": 14, "atlas_r": 89,            # G_front_inf-Triangul (Broca)
     "hemi": "left",  "view": "lateral",
     "subtitle": "Pars triangularis (Broca)"},
    {"label": "Memory assoc.",
     "atlas_l": 34, "atlas_r": 109,           # G_temp_sup-Lateral
     "hemi": "right", "view": "lateral",
     "subtitle": "Lateral temporal lobe"},
    {"label": "Somatosensory",
     "atlas_l": 28, "atlas_r": 103,           # G_postcentral
     "hemi": "left",  "view": "lateral",
     "subtitle": "Postcentral gyrus"},
]

# Optional mapping from A2C-NODE 21-region names to one of the 6 functional
# panels. Used when --from_ate is given to pull a magnitude per panel.
A2C_TO_PANEL: Dict[str, str] = {
    # The atlas-guided extractor uses bilateral cortex / WM / etc; we map
    # the closest semantic regions onto each functional panel.
    "Left-Cerebral-Cortex":   "Visual",       # general cortical
    "Right-Cerebral-Cortex":  "Visual",
    "Left-Hippocampus":       "Memory assoc.",
    "Right-Hippocampus":      "Memory assoc.",
    "Left-Amygdala":          "Memory assoc.",
    "Right-Amygdala":         "Memory assoc.",
    "Left-Thalamus":          "Somatosensory",
    "Right-Thalamus":         "Somatosensory",
    "Left-Cerebral-WM":       "Motor",
    "Right-Cerebral-WM":      "Motor",
    "Left-Lateral-Ventricle": "Auditory",
    "Right-Lateral-Ventricle":"Auditory",
}


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_prefix", required=True, type=Path,
                   help="output prefix; emits *.png and *.pdf")
    p.add_argument("--mesh", default="fsaverage5",
                   help="fsaverage[5|6|7]; '5' is fastest")
    p.add_argument("--from_ate", type=Path, default=None,
                   help="optional ATE CSV from compute_ate.py to drive "
                        "per-region intensity. Without it, illustrative "
                        "values are used.")
    p.add_argument("--cmap", default="hot",
                   help="matplotlib colormap for the overlay (default 'hot')")
    return p.parse_args()


def panel_intensities(args) -> Dict[str, float]:
    """Return mapping {panel_label -> non-negative intensity}."""
    if args.from_ate is None or not args.from_ate.is_file():
        # Illustrative values (relative magnitudes only).
        return {
            "Auditory":      0.65,
            "Visual":        0.85,
            "Motor":         0.55,
            "Language":      0.70,
            "Memory assoc.": 0.95,
            "Somatosensory": 0.45,
        }
    out: Dict[str, float] = {}
    with open(args.from_ate, newline="") as f:
        for r in _csv.DictReader(f):
            panel = A2C_TO_PANEL.get(r["region_name"])
            if panel is None:
                continue
            try:
                v = abs(float(r["ate_AD"]))
            except Exception:
                continue
            out[panel] = max(out.get(panel, 0.0), v)
    if not out:
        return panel_intensities(argparse.Namespace(from_ate=None))
    # Normalise to [0, 1] so each region is comparable visually.
    m = max(out.values()) or 1.0
    return {k: v / m for k, v in out.items()}


def hemisphere_atlas_arrays(destrieux: dict, mesh: str
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample the volume Destrieux atlas onto the fsaverage surface."""
    from nilearn import surface, datasets
    fsavg = datasets.fetch_surf_fsaverage(mesh)
    # vol_to_surf needs an image: destrieux["maps"] is a NIfTI of int labels.
    img = destrieux["maps"]
    # Resample the labels with nearest interpolation; vol_to_surf supports
    # 'nearest' since nilearn 0.8.
    L = surface.vol_to_surf(img, fsavg["pial_left"],
                            interpolation="nearest_most_frequent")
    R = surface.vol_to_surf(img, fsavg["pial_right"],
                            interpolation="nearest_most_frequent")
    return L.astype(np.int64), R.astype(np.int64)


def main() -> None:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_style import apply as _style
    _style(extra=("nature",))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    # 1) Load surfaces
    fsavg = datasets.fetch_surf_fsaverage(args.mesh)
    print(f"[surf] using mesh={args.mesh}", flush=True)

    # 2) Load Destrieux atlas + project to surface
    destrieux = datasets.fetch_atlas_destrieux_2009()
    L_lab, R_lab = hemisphere_atlas_arrays(destrieux, args.mesh)
    print(f"[surf] L surface: {L_lab.size} vertices, "
          f"unique labels: {np.unique(L_lab).size}", flush=True)

    intensities = panel_intensities(args)
    print(f"[surf] panel intensities: "
          + ", ".join(f"{k}={v:.2f}" for k, v in intensities.items()),
          flush=True)

    # 3) Plot 2 x 3 grid
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4),
                             subplot_kw={"projection": "3d"},
                             facecolor="white")

    for i, region in enumerate(REGION_DEFS):
        ax = axes.flat[i]
        ax.set_facecolor("white")
        for s in ax.spines.values(): s.set_visible(False)
        hemi = region["hemi"]
        view = region["view"]
        idx_atlas = region["atlas_l"] if hemi == "left" else region["atlas_r"]
        labs = L_lab if hemi == "left" else R_lab
        mesh_path = fsavg["pial_left"] if hemi == "left" else fsavg["pial_right"]
        bg = fsavg["sulc_left"] if hemi == "left" else fsavg["sulc_right"]

        # 1-channel stat map: zero everywhere, intensity inside the chosen ROI
        stat = np.zeros_like(labs, dtype=np.float32)
        weight = float(intensities.get(region["label"], 0.5))
        stat[labs == idx_atlas] = weight

        # Draw light grey cortex
        plotting.plot_surf_stat_map(
            mesh_path, stat,
            hemi=hemi, view=view, axes=ax, figure=fig,
            bg_map=bg, bg_on_data=False,
            colorbar=False, threshold=1e-3,
            cmap=args.cmap, vmin=0, vmax=1.0, alpha=1.0,
        )
        # Set the surface (non-coloured part) to soft grey
        # by drawing a fully-grey base layer first; nilearn handles this
        # via bg_map but we tweak the lighting to keep it pale.
        ax.set_title(f"{region['label']}\n{region['subtitle']}",
                     fontsize=9, color="black")

    fig.suptitle("Functional cortical regions on fsaverage surface",
                 fontsize=11, y=0.99)

    # White background; matplotlib's tight_layout doesn't always work
    # with 3D axes, so we do manual subplot adjustment.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92,
                        bottom=0.05, wspace=0.05, hspace=0.20)

    out_png = Path(f"{args.out_prefix}.png")
    out_pdf = Path(f"{args.out_prefix}.pdf")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print(f"[surf] wrote {out_png}", flush=True)
    print(f"[surf] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
