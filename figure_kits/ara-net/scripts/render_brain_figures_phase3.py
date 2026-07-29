#!/usr/bin/env python3
"""Phase-3 maximal brain figures: DKT surface, Papez network, case montages.

Outputs under reports/brain_figures_fcstyle/figures/:
  figure_B4b_dkt_surface.*
  figure_B5_papez_network.*
  figure_B6_case_montages.*
  figure_B7_oasis_stress.*
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import nibabel as nib
    from nilearn import datasets, plotting
    from nilearn.plotting import plot_connectome, plot_surf_stat_map
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"nilearn required: {exc}")

from analyze_papez_circuit_biomarkers import (  # noqa: E402
    PAPEZ_CIRCUIT_NODES_DKT,
    aibl_fs_subject_id,
    clean_float,
    parse_dkt_stats,
)
from build_brain_figure_assets import (  # noqa: E402
    TEMPLATE_AFFINE,
    resolve_npz_path,
    roi_vector_to_volume,
    save_nifti,
)
from roi_graph_mrf import (  # noqa: E402
    MRFParams,
    atrophy_zscores,
    build_roi_graph,
    fit_cn_volume_stats,
    mean_field_infer,
    volumes_from_row,
)
from train_atlas_feature_baseline import REGION_NAMES  # noqa: E402


AIBL_SCAN_RE = re.compile(r"^AIBL_([0-9]+)_([^_]+)_I([0-9]+)$")

# DKT cortical structure -> Destrieux surface label names (approximate anatomical map)
DKT_TO_DESTRIEUX = {
    "ctx-lh-entorhinal": ["Pole_temporal", "G_oc-temp_med-Parahip"],
    "ctx-rh-entorhinal": ["Pole_temporal", "G_oc-temp_med-Parahip"],
    "ctx-lh-parahippocampal": ["G_oc-temp_med-Parahip"],
    "ctx-rh-parahippocampal": ["G_oc-temp_med-Parahip"],
    "ctx-lh-rostralanteriorcingulate": ["G_and_S_cingul-Ant"],
    "ctx-rh-rostralanteriorcingulate": ["G_and_S_cingul-Ant"],
    "ctx-lh-caudalanteriorcingulate": ["G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post"],
    "ctx-rh-caudalanteriorcingulate": ["G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post"],
    "ctx-lh-posteriorcingulate": ["G_cingul-Post-dorsal", "G_cingul-Post-ventral"],
    "ctx-rh-posteriorcingulate": ["G_cingul-Post-dorsal", "G_cingul-Post-ventral"],
    "ctx-lh-isthmuscingulate": ["G_cingul-Post-ventral", "S_pericallosal"],
    "ctx-rh-isthmuscingulate": ["G_cingul-Post-ventral", "S_pericallosal"],
}

# MNI coordinates for Papez-proxy nodes (L/R).
PAPEZ_COORDS = {
    "L-Hippocampus": (-26.0, -22.0, -14.0),
    "R-Hippocampus": (26.0, -22.0, -14.0),
    "L-Thalamus": (-11.0, -18.0, 7.0),
    "R-Thalamus": (11.0, -18.0, 7.0),
    "L-Entorhinal": (-25.0, -8.0, -28.0),
    "R-Entorhinal": (25.0, -8.0, -28.0),
    "L-Parahippocampal": (-24.0, -32.0, -16.0),
    "R-Parahippocampal": (24.0, -32.0, -16.0),
    "L-Cingulate": (-6.0, -18.0, 38.0),
    "R-Cingulate": (6.0, -18.0, 38.0),
}

PAPEZ_EDGES = [
    ("L-Entorhinal", "L-Hippocampus"),
    ("R-Entorhinal", "R-Hippocampus"),
    ("L-Hippocampus", "L-Parahippocampal"),
    ("R-Hippocampus", "R-Parahippocampal"),
    ("L-Parahippocampal", "L-Cingulate"),
    ("R-Parahippocampal", "R-Cingulate"),
    ("L-Cingulate", "L-Thalamus"),
    ("R-Cingulate", "R-Thalamus"),
    ("L-Thalamus", "L-Hippocampus"),
    ("R-Thalamus", "R-Hippocampus"),
    ("L-Cingulate", "R-Cingulate"),
    ("L-Hippocampus", "R-Hippocampus"),
]


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def node_vol_norm(volumes: Dict[str, float], brain_seg: float, structs: Sequence[str]) -> float:
    vals = [volumes[s] for s in structs if s in volumes and math.isfinite(volumes[s])]
    if not vals or not math.isfinite(brain_seg) or brain_seg <= 0:
        return math.nan
    return float(sum(vals) / brain_seg)


def dkt_atrophy_z(
    rows: Sequence[dict],
    dkt_root: Path,
    cn_rows: Sequence[dict],
) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    """Return AD−CN and MCI−CN node atrophy z (higher = more AD-like)."""
    def collect(subset: Sequence[dict]) -> Tuple[Dict[str, List[float]], int]:
        bucket: Dict[str, List[float]] = defaultdict(list)
        hit = 0
        for row in subset:
            fs_id = aibl_fs_subject_id(str(row.get("scan_id", "")))
            if not fs_id:
                continue
            stats = dkt_root / fs_id / "stats" / "aseg+DKT.VINN.stats"
            if not stats.exists():
                continue
            brain_seg, vols = parse_dkt_stats(stats)
            hit += 1
            for node, structs in PAPEZ_CIRCUIT_NODES_DKT.items():
                v = node_vol_norm(vols, brain_seg, structs)
                if math.isfinite(v):
                    bucket[node].append(v)
            for struct in DKT_TO_DESTRIEUX:
                if struct in vols and math.isfinite(brain_seg) and brain_seg > 0:
                    bucket[struct].append(float(vols[struct] / brain_seg))
        return bucket, hit

    cn_bucket, cn_hit = collect(cn_rows)
    ad_bucket, ad_hit = collect([r for r in rows if int(r.get("label", -1)) == 2])
    mci_bucket, mci_hit = collect([r for r in rows if int(r.get("label", -1)) == 1])

    def effect(case: Dict[str, List[float]]) -> Dict[str, float]:
        out = {}
        for key, vals in case.items():
            ref = cn_bucket.get(key, [])
            if len(vals) < 3 or len(ref) < 3:
                continue
            mu = float(np.mean(ref))
            sd = float(np.std(ref))
            sd = sd if sd > 1e-8 else 1.0
            # tissue loss: negative volume z => positive atrophy
            out[key] = float(-(np.mean(vals) - mu) / sd)
        return out

    meta = {"cn_hit": cn_hit, "ad_hit": ad_hit, "mci_hit": mci_hit}
    return effect(ad_bucket), effect(mci_bucket), meta


def fill_destrieux_surface(
    labels: Sequence[str],
    map_hemi: np.ndarray,
    struct_values: Dict[str, float],
    hemi: str,
) -> np.ndarray:
    """Broadcast DKT cortical atrophy values onto Destrieux fsaverage labels."""
    name_to_idx = {str(n): i for i, n in enumerate(labels)}
    out = np.zeros(map_hemi.shape, dtype=np.float64)
    counts = np.zeros(map_hemi.shape, dtype=np.float64)
    prefix = "ctx-lh-" if hemi == "left" else "ctx-rh-"
    for struct, val in struct_values.items():
        if not struct.startswith(prefix):
            continue
        for dest_name in DKT_TO_DESTRIEUX.get(struct, []):
            idx = name_to_idx.get(dest_name)
            if idx is None:
                continue
            mask = map_hemi == idx
            out[mask] += float(val)
            counts[mask] += 1.0
    mask = counts > 0
    out[mask] /= counts[mask]
    return out


def render_dkt_surface(ad_eff: Dict[str, float], out_path: Path) -> None:
    fsaverage = datasets.fetch_surf_fsaverage()
    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [str(x) for x in atlas["labels"]]

    fig = plt.figure(figsize=(12.5, 8.5), dpi=220)
    # 2 hemis × 2 views
    views = [
        ("left", "lateral", 0, 0),
        ("left", "medial", 0, 1),
        ("right", "lateral", 1, 0),
        ("right", "medial", 1, 1),
    ]
    vmax = float(np.nanmax([abs(v) for k, v in ad_eff.items() if k.startswith("ctx-")] or [1.0]))
    vmax = max(vmax, 0.5)
    for hemi, view, r, c in views:
        ax = fig.add_subplot(2, 2, r * 2 + c + 1, projection="3d")
        map_hemi = atlas["map_left"] if hemi == "left" else atlas["map_right"]
        data = fill_destrieux_surface(labels, np.asarray(map_hemi), ad_eff, hemi)
        surf_mesh = fsaverage[f"infl_{hemi}"]
        bg = fsaverage[f"sulc_{hemi}"]
        plot_surf_stat_map(
            surf_mesh,
            data,
            hemi=hemi,
            view=view,
            bg_map=bg,
            bg_on_data=True,
            cmap="cold_hot",
            vmax=vmax,
            vmin=-vmax,
            colorbar=(c == 1 and r == 0),
            axes=ax,
            title=f"{hemi} {view}",
        )
    fig.suptitle(
        "Figure B4b. AIBL DKT cortical atrophy (AD−CN) on fsaverage / Destrieux",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B4b -> {out_path.with_suffix('.png')}")


def render_papez_network(ad_eff: Dict[str, float], out_path: Path) -> None:
    # Map DKT nodes + coarse aliases onto coordinate nodes
    node_names = list(PAPEZ_COORDS.keys())
    coords = np.asarray([PAPEZ_COORDS[n] for n in node_names], dtype=float)
    # node strength from DKT effects
    strength = []
    for name in node_names:
        key = name.split("-", 1)[-1].lower()
        val = 0.0
        if "hippocampus" in key:
            val = float(ad_eff.get("hippocampus", 0.0))
        elif "thalamus" in key:
            val = float(ad_eff.get("thalamus", 0.0))
        elif "entorhinal" in key:
            val = float(ad_eff.get("entorhinal", 0.0))
        elif "parahippocampal" in key:
            val = float(ad_eff.get("parahippocampal", 0.0))
        elif "cingulate" in key:
            val = float(ad_eff.get("cingulate", 0.0))
        strength.append(val)
    strength = np.asarray(strength, dtype=float)
    # adjacency
    idx = {n: i for i, n in enumerate(node_names)}
    adj = np.zeros((len(node_names), len(node_names)), dtype=float)
    for a, b in PAPEZ_EDGES:
        i, j = idx[a], idx[b]
        w = 0.5 * (abs(strength[i]) + abs(strength[j])) + 0.2
        adj[i, j] = w
        adj[j, i] = w

    node_size = 40 + 120 * (strength - strength.min()) / max(float(np.ptp(strength)), 1e-6)

    # Map strengths to colors manually (nilearn 0.13 has no node_cmap)
    norm = (strength - strength.min()) / max(float(np.ptp(strength)), 1e-6)
    node_colors = [plt.cm.plasma(float(v)) for v in norm]

    fig = plt.figure(figsize=(11, 5.5), dpi=220)
    ax1 = fig.add_subplot(1, 2, 1)
    plot_connectome(
        adj,
        coords,
        node_color=node_colors,
        node_size=node_size,
        edge_threshold="30%",
        edge_cmap="Greys",
        colorbar=True,
        axes=ax1,
        title="Papez-proxy connectome (node = AD−CN atrophy)",
        display_mode="lzry",
    )
    ax2 = fig.add_subplot(1, 2, 2)
    order = np.argsort(-strength)
    ax2.barh(
        [node_names[i] for i in order][::-1],
        strength[order][::-1],
        color=[plt.cm.plasma(float(v)) for v in norm[order][::-1]],
    )
    ax2.set_xlabel("AD−CN DKT atrophy z")
    ax2.set_title("Node strengths")
    ax2.axvline(0.0, color="gray", lw=0.8)
    fig.suptitle("Figure B5. Papez-circuit network (DKT AIBL AD−CN)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B5 -> {out_path.with_suffix('.png')}")


def _scan_mrf_map(
    feature_row: dict,
    atlas: np.ndarray,
    cn_mean: np.ndarray,
    cn_std: np.ndarray,
    edges,
    params: MRFParams,
) -> np.ndarray:
    az = atrophy_zscores(volumes_from_row(feature_row), cn_mean, cn_std)
    q = mean_field_infer(az, edges, params)
    return roi_vector_to_volume(atlas, q)


def pick_cases(pred_rows: Sequence[dict], n_per: int = 1) -> Dict[str, dict]:
    """Pick representative correct/incorrect cases with confidence."""
    scored = []
    for r in pred_rows:
        probs = np.array([float(r["prob_CN"]), float(r["prob_MCI"]), float(r["prob_AD"])], dtype=float)
        pred = int(probs.argmax())
        true = {"CN": 0, "MCI": 1, "AD": 2}[r["y_true"]]
        conf = float(probs[pred])
        margin = float(probs[pred] - sorted(probs)[-2])
        scored.append({**r, "_pred": pred, "_true": true, "_conf": conf, "_margin": margin, "_correct": pred == true})

    picks: Dict[str, dict] = {}
    def take(key: str, subset):
        subset = sorted(subset, key=lambda r: (-r["_conf"], -r["_margin"]))
        if subset:
            picks[key] = subset[0]

    take("correct_CN", [r for r in scored if r["_correct"] and r["_true"] == 0])
    take("correct_MCI", [r for r in scored if r["_correct"] and r["_true"] == 1])
    take("correct_AD", [r for r in scored if r["_correct"] and r["_true"] == 2])
    take("err_MCI_to_AD", [r for r in scored if (not r["_correct"]) and r["_true"] == 1 and r["_pred"] == 2])
    take("err_AD_to_MCI", [r for r in scored if (not r["_correct"]) and r["_true"] == 2 and r["_pred"] == 1])
    take("err_MCI_to_CN", [r for r in scored if (not r["_correct"]) and r["_true"] == 1 and r["_pred"] == 0])
    return picks


def render_case_montages(
    picks: Dict[str, dict],
    feature_by_scan: Dict[str, dict],
    atlas: np.ndarray,
    underlay: Optional[object],
    cn_mean: np.ndarray,
    cn_std: np.ndarray,
    edges,
    params: MRFParams,
    out_path: Path,
) -> None:
    keys = [
        "correct_CN",
        "correct_MCI",
        "correct_AD",
        "err_MCI_to_AD",
        "err_AD_to_MCI",
        "err_MCI_to_CN",
    ]
    keys = [k for k in keys if k in picks]
    if not keys:
        print("[warn] no cases to plot")
        return
    fig = plt.figure(figsize=(2.6 * len(keys) + 0.8, 6.8), dpi=220)
    gs = GridSpec(2, len(keys), figure=fig, wspace=0.08, hspace=0.25)
    for col, key in enumerate(keys):
        row = picks[key]
        feat = feature_by_scan.get(row["scan_id"])
        ax0 = fig.add_subplot(gs[0, col])
        ax1 = fig.add_subplot(gs[1, col])
        title = f"{key}\n{row['scan_id']}\nT={row['y_true']} P={row['y_pred']}"
        if feat is None:
            ax0.set_title(title + "\n(no feat)", fontsize=8)
            ax0.axis("off")
            ax1.axis("off")
            continue
        qmap = _scan_mrf_map(feat, atlas, cn_mean, cn_std, edges, params)
        img = nib.Nifti1Image(qmap.astype(np.float32), TEMPLATE_AFFINE)
        kwargs = dict(
            display_mode="z",
            cut_coords=(-20, 0, 20),
            axes=ax0,
            colorbar=False,
            annotate=False,
            draw_cross=False,
            black_bg=True,
            bg_img=None,
            cmap="plasma",
            vmin=0,
            vmax=1,
            threshold=1e-4,
        )
        plotting.plot_stat_map(img, **kwargs)
        ax0.set_title(title, fontsize=8)
        # probability bars
        probs = [float(row["prob_CN"]), float(row["prob_MCI"]), float(row["prob_AD"])]
        ax1.bar(["CN", "MCI", "AD"], probs, color=["#4c78a8", "#f58518", "#e45756"])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("p" if col == 0 else "")
        ax1.tick_params(labelsize=8)
    fig.suptitle("Figure B6. Subject-level MRF-q cases (correct vs boundary errors)", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B6 -> {out_path.with_suffix('.png')}")


def render_oasis_stress(
    oasis_rows: Sequence[dict],
    feature_by_scan: Dict[str, dict],
    atlas: np.ndarray,
    underlay: Optional[object],
    cn_mean: np.ndarray,
    cn_std: np.ndarray,
    edges,
    params: MRFParams,
    out_path: Path,
) -> None:
    # OASIS labels are often all CN in this pipeline; show false impairment if any
    false_imp = []
    true_cn = []
    for r in oasis_rows:
        probs = np.array([float(r["prob_CN"]), float(r["prob_MCI"]), float(r["prob_AD"])])
        pred = int(probs.argmax())
        item = {**r, "_pred": pred, "_p_imp": float(probs[1] + probs[2]), "_conf": float(probs[pred])}
        if pred != 0:
            false_imp.append(item)
        else:
            true_cn.append(item)
    false_imp = sorted(false_imp, key=lambda r: -r["_p_imp"])[:3]
    true_cn = sorted(true_cn, key=lambda r: -r["_conf"])[:3]
    picks = [("false_impairment", r) for r in false_imp] + [("true_CN", r) for r in true_cn]
    if not picks:
        print("[warn] no OASIS rows")
        return

    fig = plt.figure(figsize=(2.6 * len(picks) + 0.6, 3.4), dpi=220)
    gs = GridSpec(1, len(picks), figure=fig, wspace=0.12, top=0.78, bottom=0.08)
    for col, (tag, row) in enumerate(picks):
        ax = fig.add_subplot(gs[0, col])
        feat = feature_by_scan.get(row["scan_id"])
        short_id = str(row["scan_id"])[-18:]
        title = f"{tag} | pred={['CN','MCI','AD'][row['_pred']]}\n{short_id}"
        if feat is None:
            ax.set_title(title + "\n(no feat)", fontsize=7)
            ax.axis("off")
            continue
        qmap = _scan_mrf_map(feat, atlas, cn_mean, cn_std, edges, params)
        img = nib.Nifti1Image(qmap.astype(np.float32), TEMPLATE_AFFINE)
        kwargs = dict(
            display_mode="z",
            cut_coords=(-16, 0, 16),
            axes=ax,
            colorbar=False,
            annotate=False,
            draw_cross=False,
            black_bg=True,
            bg_img=None,
            cmap="plasma",
            vmin=0,
            vmax=1,
            threshold=1e-4,
        )
        plotting.plot_stat_map(img, **kwargs)
        ax.set_title(title, fontsize=7, pad=8, color="0.15")
    fig.suptitle("Figure B7. OASIS stress-test spatial cases (limitation panel)", fontsize=12, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B7 -> {out_path.with_suffix('.png')}")


def patch_b1_dual_colorbar(assets_dir: Path, fig_dir: Path) -> None:
    """Re-render method montage with separate atrophy / signed colorbars (no anat halo)."""
    from render_brain_figures_fcstyle import CUTS, load_map

    maps_dir = assets_dir / "maps"
    bg = None  # never use anat/MNI underlay — causes white-shadow misalignment look
    panels = [
        ("atrophy_AD_minus_CN", "Atrophy AD−CN", -2.5, 2.5, "cold_hot", "atrophy"),
        ("mrf_q_MCI_minus_CN", "MRF q MCI−CN", -0.6, 0.6, "cold_hot", "signed"),
        ("mrf_q_AD_minus_CN", "MRF q AD−CN", -0.6, 0.6, "cold_hot", "signed"),
        ("assoc_v6_rcspe", "V6 assoc", -0.6, 0.6, "cold_hot", "signed"),
        ("assoc_v7_mrf", "V7 assoc", -0.6, 0.6, "cold_hot", "signed"),
        ("assoc_equal_logpool", "Equal pool", -0.6, 0.6, "cold_hot", "signed"),
    ]
    panels = [p for p in panels if (maps_dir / f"{p[0]}.nii.gz").exists()]
    n = len(panels)
    fig = plt.figure(figsize=(2.45 * n + 1.4, 8.6), dpi=240)
    gs = GridSpec(3, n, figure=fig, wspace=0.04, hspace=0.10)
    views = [("z", "Axial", CUTS["z"]), ("y", "Coronal", CUTS["y"]), ("x", "Sagittal", CUTS["x"])]
    for col, (name, title, vmin, vmax, cmap, _fam) in enumerate(panels):
        img = load_map(maps_dir, name)
        for row, (mode, label, cuts) in enumerate(views):
            ax = fig.add_subplot(gs[row, col])
            kwargs = dict(
                display_mode=mode,
                cut_coords=cuts,
                axes=ax,
                colorbar=False,
                annotate=False,
                draw_cross=False,
                black_bg=True,
                bg_img=None,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                threshold=1e-4,
            )
            plotting.plot_stat_map(img, **kwargs)
            if col == 0:
                ax.text(-0.06, 0.5, label, transform=ax.transAxes, va="center", ha="right", fontsize=11, rotation=90)
            if row == 0:
                ax.set_title(title, fontsize=10, pad=4)
    cax1 = fig.add_axes([0.915, 0.55, 0.012, 0.3])
    sm1 = plt.cm.ScalarMappable(cmap="cold_hot", norm=plt.Normalize(vmin=-2.5, vmax=2.5))
    fig.colorbar(sm1, cax=cax1, label="Atrophy z")
    cax2 = fig.add_axes([0.915, 0.15, 0.012, 0.3])
    sm2 = plt.cm.ScalarMappable(cmap="cold_hot", norm=plt.Normalize(vmin=-0.6, vmax=0.6))
    fig.colorbar(sm2, cax=cax2, label="Signed map")
    fig.suptitle("Figure B1. Method montages (native space, dual colorbars)", fontsize=13, y=0.995)
    out = fig_dir / "figure_B1_anatomy_underlay_montage"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B1 dual-cbar refreshed -> {out.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("chapter1_foundation/outputs/v4/atlas_feature_cache_v4.csv"),
    )
    parser.add_argument(
        "--dkt-root",
        type=Path,
        default=Path("data/external/aibl/fastsurfer_seg/fs_subjects"),
    )
    parser.add_argument(
        "--pred-aibl",
        type=Path,
        default=Path(
            "chapter1_foundation/outputs/v4/rescue_probability_subject_no_oasis_tune/balanced_aibl_heldout_predictions.csv"
        ),
    )
    parser.add_argument(
        "--pred-oasis",
        type=Path,
        default=Path(
            "chapter1_foundation/outputs/v4/rescue_probability_subject_no_oasis_tune/balanced_oasis_external_predictions.csv"
        ),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("chapter1_foundation/ARA-Net/reports/brain_figures_fcstyle"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("chapter1_foundation/sample_data/cache_real"),
    )
    args = parser.parse_args()

    assets = args.assets_dir / "assets"
    fig_dir = args.assets_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = read_csv(args.feature_csv)
    for r in feature_rows:
        r["label"] = int(float(r["label"]))
    feature_by_scan = {r["scan_id"]: r for r in feature_rows}

    aibl = [r for r in feature_rows if str(r.get("dataset", "")).upper() == "AIBL"]
    cn_ref = [r for r in aibl if r["split"] in {"aibl_adapt_train", "aibl_adapt_val"} and r["label"] == 0]
    heldout = [r for r in aibl if r["split"] == "aibl_heldout"]
    print(f"[dkt] computing AD/MCI effects on heldout={len(heldout)} cn_ref={len(cn_ref)}")
    ad_eff, mci_eff, meta = dkt_atrophy_z(heldout, args.dkt_root, cn_ref)
    (assets / "matrices" / "dkt_ad_minus_cn.json").write_text(
        json.dumps({"meta": meta, "ad_minus_cn": ad_eff, "mci_minus_cn": mci_eff}, indent=2),
        encoding="utf-8",
    )
    print(f"[dkt] meta={meta} n_keys={len(ad_eff)}")

    render_dkt_surface(ad_eff, fig_dir / "figure_B4b_dkt_surface")
    render_papez_network(ad_eff, fig_dir / "figure_B5_papez_network")

    # Case montages
    atlas_img = nib.load(str(assets / "atlas_template.nii.gz"))
    atlas = np.asanyarray(atlas_img.dataobj)
    underlay = None
    underlay_path = assets / "anat_underlay_mean_t1.nii.gz"
    if underlay_path.exists():
        from brain_figure_align import prepare_anat_underlay

        atlas_for_mask = assets / "atlas_template.nii.gz"
        atlas_img_mask = nib.load(str(atlas_for_mask)) if atlas_for_mask.exists() else None
        underlay = prepare_anat_underlay(underlay_path, atlas_img_mask, dim=0.40, dilate_iter=0)

    train_cn = [r for r in feature_rows if r.get("split") == "train" and r["label"] == 0]
    cn_mean, cn_std = fit_cn_volume_stats(train_cn if train_cn else [r for r in feature_rows if r["label"] == 0])
    params = MRFParams()
    edges = build_roi_graph(params.couple_strength, params.anti_strength)

    aibl_pred = read_csv(args.pred_aibl)
    picks = pick_cases(aibl_pred)
    (assets / "matrices" / "case_picks.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in picks.items()}, indent=2),
        encoding="utf-8",
    )
    render_case_montages(
        picks, feature_by_scan, atlas, underlay, cn_mean, cn_std, edges, params, fig_dir / "figure_B6_case_montages"
    )

    if args.pred_oasis.exists():
        oasis_pred = read_csv(args.pred_oasis)
        render_oasis_stress(
            oasis_pred,
            feature_by_scan,
            atlas,
            underlay,
            cn_mean,
            cn_std,
            edges,
            params,
            fig_dir / "figure_B7_oasis_stress",
        )

    patch_b1_dual_colorbar(assets, fig_dir)
    print(f"[done] phase3 figures -> {fig_dir}")


if __name__ == "__main__":
    main()
