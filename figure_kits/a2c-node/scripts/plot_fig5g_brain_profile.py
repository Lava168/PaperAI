#!/usr/bin/env python3
"""Figure 5(g) - photoreal 3D glass-brain structural profile by group.

Renders a Nature-BME style 3D brain: the MNI152 cortical surface (real gyri /
sulci, soft-shaded, translucent) with the hippocampus (violet) and lateral
ventricle (blue) shown inside, in two anatomical views (sagittal + coronal),
for the four Figure-5 groups (correct MCI, MCI->AD, correct AD, AD->MCI).

To make the AD-like structural progression visible *in the brain*, the
subcortical atlas meshes are isotropically scaled to each group's mean volume
(from ``fig5_brain_profile.npz``: hippocampus shrinks, ventricle expands).
The -3..+3 rulers below report the exact group-mean z-scores (the quantitative
record); the rendering is the illustration.

Everything is offline + pure Python:
    * MNI152 template + brain mask        (ships with nilearn)
    * Harvard-Oxford subcortical atlas    (offline nilearn_data) for hippo/vent
    * scikit-image marching_cubes + matplotlib 3D  (no external 3D software)

Usage
-----
    MPLCONFIGDIR=/tmp/mpl .../python scripts/plot_fig5g_brain_profile.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection           # noqa: E402
import nibabel as nib                                             # noqa: E402
from skimage import measure                                       # noqa: E402
from scipy import ndimage as ndi                                  # noqa: E402
from scipy.ndimage import affine_transform                        # noqa: E402
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask  # noqa: E402
from nilearn.image import resample_to_img                         # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parents[1]
HO_PATH = ("/home/lry/nilearn_data/fsl/data/atlases/HarvardOxford/"
           "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz")
LBL_HIPPO = (9, 19)
LBL_VENT = (3, 14)
GROUPS = ["correct_MCI", "MCI_to_AD", "correct_AD", "AD_to_MCI"]
TITLES = {"correct_MCI": "Correct MCI", "MCI_to_AD": "MCI \u2192 AD",
          "correct_AD": "Correct AD", "AD_to_MCI": "AD \u2192 MCI"}
C_HIPPO = (0.49, 0.26, 0.64)
C_VENT = (0.16, 0.41, 0.69)
C_CORTEX = (0.80, 0.80, 0.83)
RULER = [("Hippocampal volume\n(normalized z)", "hippo_z", "#7B3FA0"),
         ("Lateral ventricular volume\n(normalized z)", "vent_z", "#2F6FB0"),
         ("AD-like atlas z-score", "ad_like_z", "#C0392B")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path,
                   default=PROJ_ROOT / "runs/aibl_eval/fig5_brain_profile.npz")
    p.add_argument("--out", type=Path,
                   default=PROJ_ROOT / "runs/aibl_eval/fig5g_brain_profile")
    p.add_argument("--cortex_alpha", type=float, default=0.30)
    p.add_argument("--cortex_step", type=int, default=3)
    return p.parse_args()


def smooth_mesh(mask, sigma, step, level=0.5):
    if np.asarray(mask).sum() < 30:
        return None
    vol = ndi.gaussian_filter(np.asarray(mask, float), sigma)
    try:
        v, f, _, _ = measure.marching_cubes(vol, level=level, step_size=step)
    except (ValueError, RuntimeError):
        return None
    return v, f


def shade(verts, faces, rgb, light=(-0.35, 0.5, 0.8), amb=0.5):
    tr = verts[faces]
    n = np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    inten = amb + (1 - amb) * np.clip(n @ L, 0, 1)
    return np.clip(np.asarray(rgb, float)[None, :] * inten[:, None], 0, 1)


def add_mesh(ax, mesh, rgb, alpha, amb=0.5):
    if mesh is None:
        return
    v, f = mesh
    fc = shade(v, f, rgb, amb=amb)
    if alpha < 1.0:
        fc = np.concatenate([fc, np.full((len(fc), 1), alpha)], 1)
    pc = Poly3DCollection(v[f]); pc.set_facecolor(fc); pc.set_edgecolor("none")
    pc.set_zsort("average"); ax.add_collection3d(pc)


def scale_about_centroid(mesh, s):
    if mesh is None:
        return None
    v, f = mesh
    c = v.mean(0)
    return c + s * (v - c), f


def scale_mask(mask, s):
    """Isotropically scale a binary mask about its centroid (order-0)."""
    if mask.sum() == 0 or abs(s - 1.0) < 1e-3:
        return mask
    c = np.array(ndi.center_of_mass(mask))
    matrix = np.eye(3) / s
    offset = c - matrix @ c
    out = affine_transform(mask.astype(np.float32), matrix, offset=offset,
                           order=0, output_shape=mask.shape)
    return out > 0.5


def _arrow_to(ax, mask_rot, color, label=None, frm=(0.62, 0.80)):
    """Small arrowhead pointing at the centroid of a (rotated) mask."""
    if mask_rot.sum() < 3:
        return
    H, W = mask_rot.shape
    yx = np.argwhere(mask_rot)
    cy, cx = yx.mean(0)
    x0, y0 = frm[0] * W, frm[1] * H
    ax.annotate("", xy=(cx, cy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                mutation_scale=11, shrinkA=0, shrinkB=1),
                zorder=8)
    if label:
        ax.text(x0, y0 - 0.03 * H, label, color=color, fontsize=7,
                ha="center", va="bottom", fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                          alpha=0.7))


def draw_slice(ax, bg2d, hip2d, ven2d, cn_hip2d, cn_ven2d, title,
               arrow_hip=False, arrow_ven=False, labels=False):
    bg = np.rot90(bg2d)
    ax.imshow(bg, cmap="gray", interpolation="bilinear")
    ov = np.zeros((*bg.shape, 4))
    ven = np.rot90(ven2d).astype(bool)
    hip = np.rot90(hip2d).astype(bool)
    ov[ven] = [C_VENT[0], C_VENT[1], C_VENT[2], 0.80]
    ov[hip] = [C_HIPPO[0], C_HIPPO[1], C_HIPPO[2], 0.88]
    ax.imshow(ov, interpolation="nearest")
    # CN baseline reference (dashed outline)
    cnh = np.rot90(cn_hip2d).astype(float)
    cnv = np.rot90(cn_ven2d).astype(float)
    if cnv.sum() > 2:
        ax.contour(cnv, levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="dashed")
    if cnh.sum() > 2:
        ax.contour(cnh, levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="dashed")
    if arrow_ven:
        _arrow_to(ax, ven, "#7FB2E5", "ventricle" if labels else None,
                  frm=(0.80, 0.82))
    if arrow_hip:
        _arrow_to(ax, hip, "#C39BE0", "hippocampus" if labels else None,
                  frm=(0.78, 0.18))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.6"); sp.set_linewidth(0.6)
    ax.set_title(title, fontsize=8, color="0.45", pad=1)


def draw_ruler(ax, value, color, label, show_label):
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-1.1, 1.4); ax.axis("off")
    v = float(np.clip(value, -3, 3))
    ax.add_patch(plt.Rectangle((-3, -0.26), 6, 0.52, facecolor="0.93",
                               edgecolor="0.7", lw=0.7, zorder=1))
    ax.axvline(0, color="0.55", lw=0.8, zorder=2)
    ax.add_patch(plt.Rectangle((min(0, v), -0.26), abs(v), 0.52,
                               facecolor=color, alpha=0.85, edgecolor="none",
                               zorder=3))
    ax.plot([v], [0.0], marker="^", ms=8, color="black", zorder=5)
    tx = float(np.clip(v, -2.4, 2.4))
    ax.text(tx, 0.55, f"{value:+.1f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=color, zorder=6)
    ax.text(-3, -0.62, "\u22123", ha="center", va="top", fontsize=6.5, color="0.55")
    ax.text(3, -0.62, "+3", ha="center", va="top", fontsize=6.5, color="0.55")
    if show_label:
        ax.text(-3.8, 0, label, ha="right", va="center", fontsize=8.5)


def main() -> int:
    args = parse_args()
    prof = np.load(args.profile, allow_pickle=True)
    grp = np.asarray([str(g) for g in prof["group"]])
    true = prof["true"]
    zvals = {key: {g: float(np.nanmean(prof[key][grp == g])) for g in GROUPS}
             for _, key, _ in RULER}
    mmse = {g: float(np.nanmean(prof["mmse"][grp == g])) for g in GROUPS}

    # group volume ratios vs true-CN -> isotropic linear scale (cube root)
    cn = (true == 0)
    cn_h = float(np.nanmean(prof["hippo_norm"][cn]))
    cn_v = float(np.nanmean(prof["vent_norm"][cn]))
    scale_h, scale_v = {}, {}
    for g in GROUPS:
        m = (grp == g)
        rh = float(np.nanmean(prof["hippo_norm"][m])) / cn_h
        rv = float(np.nanmean(prof["vent_norm"][m])) / cn_v
        scale_h[g] = float(np.clip(rh ** (1 / 3), 0.80, 1.25))
        scale_v[g] = float(np.clip(rv ** (1 / 3), 0.80, 1.60))

    # --- anatomy meshes (MNI space) ---------------------------------------- #
    print("[fig5g] loading MNI template + Harvard-Oxford ...", flush=True)
    tmpl = load_mni152_template(resolution=1)
    mask = load_mni152_brain_mask(resolution=1)
    T = np.asarray(tmpl.dataobj, float)
    M = np.asarray(mask.dataobj) > 0
    T = T * M
    ho = resample_to_img(nib.load(HO_PATH), tmpl, interpolation="nearest",
                         force_resample=True, copy_header=True)
    HO = np.asarray(ho.dataobj)
    hippo_mask = (HO == LBL_HIPPO[0]) | (HO == LBL_HIPPO[1])
    vent_mask = (HO == LBL_VENT[0]) | (HO == LBL_VENT[1])

    # crop to brain bbox (shared coords)
    nz = np.argwhere(M); lo = np.maximum(nz.min(0) - 4, 0)
    hi = np.minimum(nz.max(0) + 5, np.array(T.shape))
    sl = tuple(slice(lo[k], hi[k]) for k in range(3))
    T = T[sl]; hippo_mask = hippo_mask[sl]; vent_mask = vent_mask[sl]
    X, Y, Z = T.shape

    print("[fig5g] building meshes ...", flush=True)
    thr = np.percentile(T[T > 0], 30)
    cortex = smooth_mesh(T, sigma=1.5, step=args.cortex_step, level=thr)
    hippo0 = smooth_mesh(hippo_mask, sigma=1.0, step=2)
    vent0 = smooth_mesh(vent_mask, sigma=1.2, step=2)
    print(f"[fig5g] cortex faces={0 if cortex is None else len(cortex[1])}",
          flush=True)

    # tri-planar slice positions (cropped coords): centre on the structures
    hc = np.array(ndi.center_of_mass(hippo_mask)).round().astype(int)
    vc = np.array(ndi.center_of_mass(vent_mask)).round().astype(int)
    sx = int(hc[0])               # sagittal: parasagittal through a hippocampus
    cy = int(hc[1])               # coronal: through the hippocampus
    az = int(vc[2])               # axial: through the lateral ventricle body

    def setview(ax, view):
        ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.set_zlim(0, Z)
        ax.set_box_aspect((X, Y, Z))
        ax.view_init(elev=8, azim=180 if view == "sagittal" else 90)
        ax.axis("off")

    # --- figure ------------------------------------------------------------ #
    fig = plt.figure(figsize=(16.5, 9.6))
    outer = fig.add_gridspec(1, 4, wspace=0.12, left=0.08, right=0.99,
                             top=0.82, bottom=0.08)
    print("[fig5g] rendering panels ...", flush=True)
    for ci, g in enumerate(GROUPS):
        # 3D meshes (row 1) + scaled masks (row 2 slices)
        mh = scale_about_centroid(hippo0, scale_h[g])
        mv = scale_about_centroid(vent0, scale_v[g])
        hmask_g = scale_mask(hippo_mask, scale_h[g])
        vmask_g = scale_mask(vent_mask, scale_v[g])

        inner = outer[0, ci].subgridspec(
            3, 1, height_ratios=[2.4, 2.5, 1.9], hspace=0.05)

        # --- row 1: 3D glass brain (UNCHANGED: sagittal + coronal) --------- #
        row1 = inner[0].subgridspec(1, 2, wspace=0.0)
        for vi, view in enumerate(["sagittal", "coronal"]):
            ax = fig.add_subplot(row1[0, vi], projection="3d")
            add_mesh(ax, cortex, C_CORTEX, args.cortex_alpha, amb=0.55)
            add_mesh(ax, mv, C_VENT, 0.95)
            add_mesh(ax, mh, C_HIPPO, 1.0)
            setview(ax, view)
            ax.set_title(view, fontsize=8, color="0.45", pad=-4)

        # --- row 2: cross-sectional cutaways (3 planes) -------------------- #
        # dashed outline = CN baseline (unscaled atlas); fill = group mean
        lab = (ci == 0)
        row2 = inner[1].subgridspec(1, 3, wspace=0.08)
        ax = fig.add_subplot(row2[0, 0])
        draw_slice(ax, T[sx, :, :], hmask_g[sx, :, :], vmask_g[sx, :, :],
                   hippo_mask[sx, :, :], vent_mask[sx, :, :], "sagittal",
                   arrow_hip=True, labels=False)
        ax = fig.add_subplot(row2[0, 1])
        draw_slice(ax, T[:, cy, :], hmask_g[:, cy, :], vmask_g[:, cy, :],
                   hippo_mask[:, cy, :], vent_mask[:, cy, :], "coronal",
                   arrow_hip=True, arrow_ven=True, labels=lab)
        ax = fig.add_subplot(row2[0, 2])
        draw_slice(ax, T[:, :, az], hmask_g[:, :, az], vmask_g[:, :, az],
                   hippo_mask[:, :, az], vent_mask[:, :, az], "axial",
                   arrow_ven=True, labels=False)

        pos = outer[0, ci].get_position(fig)
        fig.text(pos.x0 + pos.width / 2, 0.835,
                 f"{TITLES[g]}\n(n={int((grp==g).sum())}, MMSE {mmse[g]:.1f})",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

        # --- rulers -------------------------------------------------------- #
        ruler_gs = inner[2].subgridspec(3, 1, hspace=0.55)
        for ri, (lab, key, col) in enumerate(RULER):
            axr = fig.add_subplot(ruler_gs[ri, 0])
            draw_ruler(axr, zvals[key][g], col, lab, show_label=(ci == 0))

    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=C_HIPPO, label="Hippocampus (L+R)"),
               Patch(facecolor=C_VENT, label="Lateral ventricle (L+R)"),
               Patch(facecolor=C_CORTEX, label="MNI152 cortical surface"),
               Line2D([0], [0], color="0.35", lw=1.2, ls="--",
                      label="CN baseline (dashed)")]
    fig.text(0.5, 0.99, "Figure 5(g)  Brain structural profile rendering",
             ha="center", fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.967))
    fig.text(0.5, 0.915, "Arrows mark the changing structures \u2014 filled = group "
             "mean, dashed = CN reference: ventricle expands & hippocampus "
             "shrinks toward AD", ha="center", fontsize=9, color="0.4",
             style="italic")
    axA = fig.add_axes([0.10, 0.015, 0.82, 0.04]); axA.axis("off")
    axA.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                 arrowprops=dict(arrowstyle="-|>", lw=2.4, color="0.25"))
    axA.text(0.0, 0.5, "less AD-like", ha="left", va="center", fontsize=11,
             color="0.25")
    axA.text(1.0, 0.5, "more AD-like", ha="right", va="center", fontsize=11,
             fontweight="bold", color="#C0392B")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png = args.out.with_suffix(".png"); pdf = args.out.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[fig5g] wrote {png}\n[fig5g] wrote {pdf}")
    for g in GROUPS:
        print(f"  {g:<12} scale_hippo={scale_h[g]:.3f} scale_vent={scale_v[g]:.3f}"
              f"  " + "  ".join(f"{k}={zvals[k][g]:+.2f}" for _, k, _ in RULER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
