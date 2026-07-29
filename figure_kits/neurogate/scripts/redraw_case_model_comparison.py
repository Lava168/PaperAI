#!/usr/bin/env python3
"""Formal case-level multi-model brain-map figures (Fig03–Fig06).

Layout principle
----------------
Rows / cases = clinical exemplars (CN / MCI / AD)
Columns / conditions = models (Plain CNN, ResNet-18, NeuroGate −Atlas, NeuroGate Full)

Display: skull-stripped orthogonal slices + anatomical markers (Hipp / Amyg / Vent).
Attribution: atlas region attention (NeuroGate Full) vs Grad-CAM (CNN baselines / −Atlas).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chapter1_foundation.data.foundation_loader import (  # noqa: E402
    remap_segmentation,
)
from chapter1_foundation.preprocess_brain_extract import brain_extract  # noqa: E402
from chapter1_foundation.run_experiment_v3 import build_model  # noqa: E402

CLASS_NAMES = ["CN", "MCI", "AD"]
# Nature-style palette (aligned with foundation_visualization.NATURE_PALETTE)
NATURE = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
CLASS_COLORS = {"CN": NATURE[0], "MCI": NATURE[1], "AD": NATURE[2]}
MODEL_COLORS = {
    "Plain CNN": NATURE[0],
    "3D ResNet-18": NATURE[1],
    "NeuroGate (−Atlas)": NATURE[3],
    "NeuroGate (Full)": NATURE[2],
}
# Soft warm overlay (Nature-like; avoid neon inferno)
_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "nature_heat",
    ["#F7F4EF", "#F2C14E", "#F28E2B", "#E15759", "#8C2D2D"],
)
_BOX_FACE = "#F7F4EF"
_BOX_EDGE = "#4E79A7"
_INK = "#2F2F2F"
_MUTED = "#666666"

# Contiguous 1..21 order = foundation_loader._FS_LABELS[1:]
REGION_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud", "L-Put", "L-Pall",
    "Brainstem", "L-Hipp", "L-Amyg", "L-Acc",
    "R-WM", "R-Ctx", "R-Vent", "R-Thal", "R-Caud", "R-Put", "R-Pall",
    "R-Hipp", "R-Amyg", "R-Acc",
]
MARKERS = {
    "L-Hipp": 8,
    "R-Hipp": 18,
    "L-Amyg": 9,
    "R-Amyg": 19,
    "L-Vent": 2,
    "R-Vent": 13,
}


def set_style():
    """Nature-journal style (matches foundation_visualization.set_nature_style)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": _INK,
        "axes.labelcolor": _INK,
        "text.color": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "legend.frameon": False,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
    })


def model_color_list(model_order):
    return [MODEL_COLORS.get(m, NATURE[i % len(NATURE)]) for i, m in enumerate(model_order)]


def style_bar(ax, shorts, vals, colors, ylabel="", title="", ylim=None, fmt="{:.2f}", ypad=None,
              value_bbox=False):
    bars = ax.bar(shorts, vals, color=colors, width=0.68, edgecolor=_INK, linewidth=0.45, zorder=2)
    if ylim is not None:
        lo, hi = ylim
        # extra headroom so value labels never hit the title
        ax.set_ylim(lo, hi * 1.08 if hi > 0 else 1)
    else:
        ax.set_ylim(0, max(vals) * 1.28 if max(vals) > 0 else 1)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, pad=8, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_axisbelow(True)
    pad = ypad if ypad is not None else (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    for i, v in enumerate(vals):
        kw = dict(ha="center", va="bottom", fontsize=6.5, color=_INK, clip_on=False, zorder=4)
        if value_bbox:
            kw["bbox"] = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.6)
        ax.text(i, v + pad, fmt.format(v), **kw)
    return bars


def note_box(ax, text, fontsize=7):
    ax.axis("off")
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes, fontsize=fontsize, va="top", ha="left",
        color=_INK, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=_BOX_FACE, edgecolor=_BOX_EDGE, linewidth=0.8),
    )

def load_npz_case(path: Path) -> Dict:
    z = np.load(path, allow_pickle=True)
    image = z["image"].astype(np.float32)
    seg_raw = z["seg"].astype(np.int64)
    seg = remap_segmentation(seg_raw)
    label = int(z["label"])
    # brain mask via threshold LCC (same family as brain-extract QC)
    be = brain_extract(
        image, affine=np.eye(4), model_mask=(seg > 0).astype(np.float32),
        percentile=15.0, dilate=1, peel=2, grow=3, use_model_mask=True,
        do_reorient=False, source_path=str(path),
    )
    mask = be.mask_ras.astype(bool)
    masked = image * mask
    return {
        "path": path,
        "stem": path.stem,
        "image": image,
        "masked": masked,
        "mask": mask,
        "seg": seg,
        "label": label,
    }


def pick_cases(cache_dir: Path) -> Dict[int, Path]:
    """Pick one high-quality ADNI case per class with rich atlas coverage."""
    best: Dict[int, Tuple[float, Path]] = {}
    for p in sorted(cache_dir.glob("ADNI_*.npz")):
        z = np.load(p)
        lab = int(z["label"])
        seg = remap_segmentation(z["seg"].astype(np.int64))
        # require key AD structures
        score = 0.0
        for rid in MARKERS.values():
            score += float((seg == rid).sum())
        score += 0.01 * float((seg > 0).sum())
        if lab not in best or score > best[lab][0]:
            best[lab] = (score, p)
    if len(best) < 3:
        raise RuntimeError(f"Need CN/MCI/AD cases under {cache_dir}")
    return {k: v[1] for k, v in best.items()}


def mid_slices(vol: np.ndarray):
    """Return mid-plane slices. Volume axes match foundation cache: (S, C, A).

    sagittal = vol[i, :, :], coronal = vol[:, j, :], axial = vol[:, :, k]
    """
    d, h, w = vol.shape
    return {
        "sag": (d // 2, vol[d // 2]),
        "cor": (h // 2, vol[:, h // 2]),
        "ax": (w // 2, vol[:, :, w // 2]),
    }


def normalize_img(im: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    return np.clip((im - lo) / (hi - lo + 1e-8), 0, 1)


def overlay_heat(image2d: np.ndarray, heat2d: np.ndarray, gamma=0.65, alpha=0.58):
    """Nature-style warm overlay on grayscale MRI (restrained, not neon)."""
    imn = normalize_img(image2d)
    rgb = np.stack([imn, imn, imn], axis=-1)
    ht = np.asarray(heat2d, dtype=np.float32)
    if ht.max() > ht.min():
        ht = (ht - ht.min()) / (ht.max() - ht.min())
    ht = np.power(np.clip(ht, 0, 1), gamma)
    color = _HEAT_CMAP(ht)[..., :3]
    a = np.clip((ht - 0.08) / 0.92, 0, 1) * alpha
    out = rgb * (1 - a[..., None]) + color * a[..., None]
    return np.rot90(np.clip(out, 0, 1))


def region_att_to_volume(seg: np.ndarray, region_att: np.ndarray) -> np.ndarray:
    att = np.asarray(region_att, dtype=np.float32).ravel()
    if att.max() > att.min():
        att = (att - att.min()) / (att.max() - att.min())
    heat = np.zeros(seg.shape, dtype=np.float32)
    for rid in range(1, 22):
        if rid - 1 < len(att):
            heat[seg == rid] = att[rid - 1]
    return heat


def centroid_2d(mask2d: np.ndarray) -> Optional[Tuple[float, float]]:
    if mask2d.sum() < 5:
        return None
    cy, cx = ndi.center_of_mass(mask2d.astype(float))
    h, w = mask2d.shape
    x = cy
    y = h - 1 - cx
    return float(x), float(y)


def draw_markers(ax, seg_slice: np.ndarray, which=("L-Vent", "R-Vent"), view="ax"):
    """Subtle anatomical markers; limited set to avoid overlap."""
    colors = {
        "L-Hipp": NATURE[0], "R-Hipp": NATURE[0],
        "L-Amyg": NATURE[1], "R-Amyg": NATURE[1],
        "L-Vent": NATURE[3], "R-Vent": NATURE[3],
    }
    # stagger label offsets so boxes do not stack
    offsets = {
        "L-Vent": (-18, 10), "R-Vent": (10, 10),
        "L-Hipp": (-20, -12), "R-Hipp": (10, -12),
        "L-Amyg": (-22, 2), "R-Amyg": (12, 2),
    }
    for name in which:
        rid = MARKERS.get(name)
        if rid is None:
            continue
        m = seg_slice == rid
        c = centroid_2d(m)
        if c is None:
            continue
        x, y = c
        ox, oy = offsets.get(name, (8, 6))
        ax.plot(x, y, "o", ms=2.8, color=colors[name],
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=5.5,
            color="white",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=colors[name], lw=0.55),
            bbox=dict(boxstyle="round,pad=0.10", facecolor=colors[name],
                      edgecolor="none", alpha=0.90),
            zorder=6,
        )


def load_ckpt_model(model_name: str, use_atlas: bool, ckpt: Path, device):
    model = build_model(model_name, use_atlas, device, dropout=0.3)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, missing, unexpected


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._fwd)
        self._h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out

    def _bwd(self, module, gin, gout):
        self.gradients = gout[0]

    def close(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, image, seg, class_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        image = image.detach().requires_grad_(True)
        out = self.model(image, seg, return_features=True)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        acts = self.activations[0]  # C,D,H,W
        grads = self.gradients[0]
        weights = grads.mean(dim=(1, 2, 3))
        cam = (weights[:, None, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        # upsample to image size
        cam_t = torch.from_numpy(cam)[None, None]
        cam_up = F.interpolate(cam_t, size=image.shape[2:], mode="trilinear", align_corners=False)
        cam_up = cam_up[0, 0].numpy()
        cam_up = cam_up - cam_up.min()
        cam_up = cam_up / (cam_up.max() + 1e-8)
        return cam_up, probs[0].detach().cpu().numpy()


def target_layer_for(model, model_name: str):
    if model_name == "resnet3d":
        return model.layer4[-1]
    if model_name == "plaincnn":
        return find_last_conv(model.features)
    if model_name == "ours":
        return find_last_conv(model.encoder)
    raise ValueError(model_name)


def find_last_conv(module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv3d):
            last = m
    return last


def neurogate_region_attention(model, image, seg) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model(image, seg, return_attention=True)
        probs = torch.softmax(out["logits"], dim=-1)[0].cpu().numpy()
        if "attention" not in out or out["attention"] is None:
            # fallback: zeros
            return np.zeros(21, dtype=np.float32), probs
        aw = out["attention"]  # (B,H,N,N) or (B,N,N)
        if aw.ndim == 4:
            recv = aw[0].mean(0).mean(0).cpu().numpy()  # (N,) attention received
        elif aw.ndim == 3:
            recv = aw[0].mean(0).cpu().numpy()
        else:
            recv = aw.reshape(-1)[:21].cpu().numpy()
        return recv.astype(np.float32), probs


def prepare_tensor(case, device):
    img = torch.from_numpy(case["image"]).float().unsqueeze(0).unsqueeze(0).to(device)
    seg = torch.from_numpy(case["seg"]).long().unsqueeze(0).to(device)
    return img, seg


def compute_attributions(cases: Dict[int, dict], models: Dict[str, dict], device):
    """models[name] = {model, model_name, use_atlas, kind: attn|gradcam}"""
    results = {}
    for lab, case in cases.items():
        img, seg = prepare_tensor(case, device)
        results[lab] = {}
        for mname, meta in models.items():
            model = meta["model"]
            if meta["kind"] == "attn":
                recv, probs = neurogate_region_attention(model, img, seg)
                heat = region_att_to_volume(case["seg"], recv)
                heat = heat * case["mask"]
                results[lab][mname] = {
                    "heat": heat, "probs": probs, "region_att": recv, "kind": "attn",
                }
            else:
                layer = meta.get("layer") or find_last_conv(model)
                cam = GradCAM3D(model, layer)
                try:
                    heat, probs = cam(img, seg, class_idx=lab)
                finally:
                    cam.close()
                heat = heat * case["mask"]
                # also aggregate Grad-CAM into regions for bar comparison
                reg = np.zeros(21, dtype=np.float32)
                for rid in range(1, 22):
                    m = case["seg"] == rid
                    if m.any():
                        reg[rid - 1] = float(heat[m].mean())
                results[lab][mname] = {
                    "heat": heat, "probs": probs, "region_att": reg, "kind": "gradcam",
                }
    return results


def show_panel(ax, case, heat, view="ax", markers=True, title="", ylabel=""):
    slices_img = mid_slices(case["masked"])
    slices_ht = mid_slices(heat)
    slices_seg = mid_slices(case["seg"])
    im = slices_img[view][1]
    ht = slices_ht[view][1]
    ax.imshow(overlay_heat(im, ht))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if markers and view in ("ax", "cor"):
        which = ("L-Vent", "R-Vent") if view == "ax" else ("L-Hipp", "R-Hipp", "L-Vent")
        draw_markers(ax, slices_seg[view][1], which=which, view=view)
    if title:
        ax.set_title(title, fontsize=8, pad=3)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, fontweight="bold", labelpad=3)


def method_tag(ax, kind: str):
    # top-left to avoid colliding with bottom markers / bar titles
    ax.text(
        0.03, 0.97, "Atlas attn" if kind == "attn" else "Grad-CAM",
        transform=ax.transAxes, fontsize=5.5, color="white", va="top",
        bbox=dict(facecolor="#2F2F2F", alpha=0.55, pad=1.2, edgecolor="none",
                  boxstyle="round,pad=0.18"),
    )


def _short_model(name: str) -> str:
    return {
        "Plain CNN": "Plain",
        "3D ResNet-18": "ResNet",
        "NeuroGate (−Atlas)": "−Atlas",
        "NeuroGate (Full)": "Full",
    }.get(name, name)


def _load_quant(viz_dir: Path) -> Optional[dict]:
    p = viz_dir / "attribution_quant_summary.json"
    if not p.exists():
        p = ROOT / "experiment_results/viz_baselines/attribution_quant_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _quant_by_model(quant: dict) -> Dict[str, dict]:
    return {r["model"]: r for r in quant.get("table_main", [])}


def draw_fig03(out: Path, cases, attrs, model_order, quant: Optional[dict] = None):
    set_style()
    n_m = len(model_order)
    fig = plt.figure(figsize=(10.6, 7.8))
    # outer: brain block + gap + quant block (prevents title/brain collisions)
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[3.35, 1.15],
        hspace=0.14, left=0.06, right=0.99, top=0.90, bottom=0.06,
    )
    gs = gridspec.GridSpecFromSubplotSpec(
        3, n_m + 1, subplot_spec=outer[0], hspace=0.05, wspace=0.04,
    )
    gq = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1], wspace=0.12,
    )
    qmap = _quant_by_model(quant) if quant else {}
    colors = model_color_list(model_order)

    for r, lab in enumerate([0, 1, 2]):
        case = cases[lab]
        ax = fig.add_subplot(gs[r, 0])
        im = mid_slices(case["masked"])["ax"][1]
        ax.imshow(np.rot90(normalize_img(im)), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        draw_markers(ax, mid_slices(case["seg"])["ax"][1], which=("L-Vent", "R-Vent"))
        if r == 0:
            ax.set_title("MRI + labels", fontsize=8, pad=3)
        ax.set_ylabel(f"{CLASS_NAMES[lab]}", fontsize=9, fontweight="bold", labelpad=3)
        for c, mname in enumerate(model_order):
            ax = fig.add_subplot(gs[r, c + 1])
            kind = attrs[lab][mname]["kind"]
            show_panel(ax, case, attrs[lab][mname]["heat"], view="ax",
                       markers=False,
                       title=_short_model(mname) if r == 0 else "")
            method_tag(ax, kind)

    ax = fig.add_subplot(gq[0, 0:2])
    if quant:
        style_bar(ax, [_short_model(m) for m in model_order],
                  [qmap[m]["piecewise"] for m in model_order], colors,
                  ylabel="Piecewise ↑", title="(d) Anatomical structure (n=120)",
                  ylim=(0, 1.18))
    else:
        ax.axis("off")

    ax = fig.add_subplot(gq[0, 2:4])
    if quant:
        vals = [qmap[m]["top10_mass_pct"] for m in model_order]
        style_bar(ax, [_short_model(m) for m in model_order], vals, colors,
                  ylabel="Top-10% mass (%)", title="(e) Spatial concentration (n=120)",
                  fmt="{:.0f}", ylim=(0, max(vals) * 1.22))
    else:
        ax.axis("off")

    ax = fig.add_subplot(gq[0, 4])
    if quant:
        lines = [
            f"n={quant.get('n_cases', 120)} (40×3)",
            "piece  " + " ".join(f"{qmap[m]['piecewise']:.2f}" for m in model_order),
            "top10  " + " ".join(f"{qmap[m]['top10_mass_pct']:.0f}" for m in model_order),
            "AD-ROI " + " ".join(f"{qmap[m]['ad_roi_share_pct']:.0f}" for m in model_order),
            "", "Claim: structure ↑", "(not AD-ROI mass)",
        ]
        note_box(ax, "\n".join(lines), fontsize=6.2)
    else:
        ax.axis("off")

    n = quant.get("n_cases", 120) if quant else 120
    fig.suptitle(f"Figure 3. Multi-model attribution + quantification (n={n})",
                 fontsize=11, y=0.97, color=_INK)
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("wrote", out)


def draw_fig04(out: Path, cases, attrs, model_order, quant: Optional[dict] = None):
    set_style()
    lab = 2
    case = cases[lab]
    views = ["ax", "cor", "sag"]
    qmap = _quant_by_model(quant) if quant else {}
    colors = model_color_list(model_order)
    fig = plt.figure(figsize=(9.6, 7.6))
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[3.2, 1.05],
        hspace=0.14, left=0.07, right=0.99, top=0.90, bottom=0.06,
    )
    gs = gridspec.GridSpecFromSubplotSpec(
        3, len(model_order), subplot_spec=outer[0], hspace=0.05, wspace=0.04,
    )
    gq = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.14,
                                          width_ratios=[1.1, 1.0])
    for r, view in enumerate(views):
        for c, mname in enumerate(model_order):
            ax = fig.add_subplot(gs[r, c])
            show_panel(ax, case, attrs[lab][mname]["heat"], view=view,
                       markers=False,
                       title=_short_model(mname) if r == 0 else "",
                       ylabel=["Axial", "Coronal", "Sagittal"][r] if c == 0 else "")
            # markers only on Full coronal (clearest anatomy), avoid axial clutter
            if "Full" in mname and view == "cor":
                draw_markers(ax, mid_slices(case["seg"])["cor"][1],
                             which=("L-Hipp", "L-Vent"), view="cor")
    ax = fig.add_subplot(gq[0, 0])
    if quant:
        style_bar(ax, [_short_model(m) for m in model_order],
                  [qmap[m]["piecewise"] for m in model_order], colors,
                  ylabel="Piecewise ↑",
                  title=f"(d) Structure score (n={quant.get('n_cases', 120)})",
                  ylim=(0, 1.18))
    else:
        ax.axis("off")

    ax = fig.add_subplot(gq[0, 1])
    if quant:
        lines = ["Paired Wilcoxon (Full > baseline, piecewise)", ""]
        for m, tstat in quant.get("piecewise_tests_vs_full", {}).items():
            lines.append(f"vs {_short_model(m)}: Δ=+{tstat['mean_diff']:.2f}  "
                         f"d={tstat['cohens_d']:.1f}  p={tstat['p']:.1e}")
        lines += ["", "Full = region-bounded; CNN Grad-CAM = diffuse."]
        note_box(ax, "\n".join(lines), fontsize=7)
    else:
        ax.axis("off")

    fig.suptitle(f"Figure 4. Multi-view + stats — AD ({case['stem'][:26]})",
                 fontsize=11, y=0.97)
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("wrote", out)


def draw_fig05(out: Path, cases, attrs, model_order, quant: Optional[dict] = None):
    set_style()
    fig = plt.figure(figsize=(11.4, 8.2))
    gs = gridspec.GridSpec(
        3, 4, figure=fig, height_ratios=[0.95, 0.95, 1.2],
        hspace=0.28, wspace=0.16, left=0.07, right=0.98, top=0.90, bottom=0.14,
    )
    full_name = [m for m in model_order if "Full" in m][0]
    others = [m for m in model_order if m != full_name][:3]
    qmap = _quant_by_model(quant) if quant else {}
    colors = model_color_list(model_order)
    shorts = [_short_model(m) for m in model_order]

    ax = fig.add_subplot(gs[0, 0])
    if quant:
        style_bar(ax, shorts, [qmap[m]["piecewise"] for m in model_order], colors,
                  ylabel="Score", title="(a) Piecewise ↑", ylim=(0, 1.2))
    ax = fig.add_subplot(gs[0, 1])
    if quant:
        vals = [qmap[m]["top10_mass_pct"] for m in model_order]
        style_bar(ax, shorts, vals, colors, title="(b) Top-10% mass (%)",
                  fmt="{:.0f}", ylim=(0, max(vals) * 1.22))
    ax = fig.add_subplot(gs[0, 2])
    if quant:
        vals = [qmap[m]["ad_roi_share_pct"] for m in model_order]
        null_v = quant.get("null_ad_roi_share_pct", 28.57)
        # keep null line clear of value labels: annotate on left only
        ymax = max(vals + [null_v, 30]) * 1.32
        style_bar(ax, shorts, vals, colors, ylabel="%", title="(c) AD-ROI share ≈ null",
                  fmt="{:.0f}", ylim=(0, ymax), value_bbox=True)
        ax.plot([-0.45, 0.35], [null_v, null_v], color=_MUTED, ls="--", lw=0.9, zorder=1, clip_on=False)
        ax.text(-0.45, null_v + ymax * 0.03, f"null {null_v:.1f}%",
                ha="left", va="bottom", fontsize=6, color=_MUTED)
    ax = fig.add_subplot(gs[0, 3])
    if quant:
        cas = quant.get("population_full_cas", {}).get("cas_abs", float("nan"))
        msg = (f"Paired n={quant.get('n_cases', 120)}\n\nSupported\n"
               "• Structure (piecewise)\n• Concentration vs ResNet\n\n"
               f"Not supported\n• Higher Hipp/Amyg mass\n• CAS={cas:.3f}\n\n")
        for m, t in quant.get("piecewise_tests_vs_full", {}).items():
            msg += f"Full > {_short_model(m)}  p={t['p']:.0e}\n"
        note_box(ax, msg.strip(), fontsize=6.5)
    else:
        ax.axis("off")

    case = cases[2]
    for c, mname in enumerate(others):
        ax = fig.add_subplot(gs[1, c])
        diff = attrs[2][full_name]["heat"] - attrs[2][mname]["heat"]
        im = mid_slices(case["masked"])["ax"][1]
        d = mid_slices(diff)["ax"][1]
        d = d / (np.abs(d).max() + 1e-8)
        imn = normalize_img(im)
        rgb = np.stack([imn, imn, imn], -1)
        pos = np.clip(d, 0, 1); neg = np.clip(-d, 0, 1)
        rgb = rgb * (1 - 0.45 * (pos + neg)[..., None])
        rgb[..., 0] += 0.55 * pos
        rgb[..., 2] += 0.50 * neg
        ax.imshow(np.rot90(np.clip(rgb, 0, 1)))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(f"Full vs {_short_model(mname)}", fontsize=8, pad=2)
        if c == 0:
            ax.set_ylabel("AD difference", fontsize=8, fontweight="bold")
    ax = fig.add_subplot(gs[1, 3])
    note_box(ax, "Red: Full stronger\nBlue: baseline stronger\n\n"
             "Full = atlas attention\nBaselines = Grad-CAM", fontsize=7)

    ax = fig.add_subplot(gs[2, :])
    x = np.arange(21)
    width = 0.78 / len(model_order)
    for i, mname in enumerate(model_order):
        vals = attrs[2][mname]["region_att"]
        vals = vals / (vals.sum() + 1e-8)
        ax.bar(x + i * width, vals, width=width, label=_short_model(mname),
               color=colors[i], edgecolor=_INK, linewidth=0.2, alpha=0.92)
    ax.set_xticks(x + width * (len(model_order) - 1) / 2)
    ax.set_xticklabels(REGION_NAMES, rotation=90, fontsize=6)
    ax.set_ylabel("Normalized importance")
    ax.set_title("AD case — region profile (atlas attn vs Grad-CAM pooled)", pad=4)
    ax.legend(fontsize=6.5, ncol=4, loc="upper right", frameon=False)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for rid in [MARKERS["L-Hipp"], MARKERS["R-Hipp"], MARKERS["L-Vent"], MARKERS["R-Vent"]]:
        ax.axvline(rid, color=NATURE[2], ls=":", lw=0.7, alpha=0.55)

    fig.suptitle("Figure 5. Cross-model attribution quantification & differences",
                 fontsize=11, y=0.97)
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("wrote", out)


def draw_fig06(out: Path, cases, attrs, model_order):
    set_style()
    fig = plt.figure(figsize=(11.6, 7.8))
    gs = gridspec.GridSpec(
        3, 4, figure=fig, width_ratios=[1, 1, 1, 1.35],
        wspace=0.16, hspace=0.16, left=0.06, right=0.88, top=0.90, bottom=0.05,
    )
    full_name = [m for m in model_order if "Full" in m][0]
    colors = model_color_list(model_order)
    for r, lab in enumerate([0, 1, 2]):
        case = cases[lab]
        for c, view in enumerate(["ax", "cor", "sag"]):
            ax = fig.add_subplot(gs[r, c])
            show_panel(ax, case, attrs[lab][full_name]["heat"], view=view,
                       markers=False,
                       title=["Axial", "Coronal", "Sagittal"][c] if r == 0 else "",
                       ylabel=CLASS_NAMES[lab] if c == 0 else "")
            if view == "cor":
                draw_markers(ax, mid_slices(case["seg"])["cor"][1],
                             which=("L-Hipp", "L-Vent"), view="cor")
            elif view == "ax":
                draw_markers(ax, mid_slices(case["seg"])["ax"][1],
                             which=("L-Vent", "R-Vent"), view="ax")
        ax = fig.add_subplot(gs[r, 3])
        x = np.arange(3)
        width = 0.78 / len(model_order)
        for i, mname in enumerate(model_order):
            ax.bar(x + i * width, attrs[lab][mname]["probs"], width=width,
                   color=colors[i], edgecolor=_INK, linewidth=0.25,
                   label=_short_model(mname) if r == 0 else None)
        ax.set_xticks(x + width * (len(model_order) - 1) / 2)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.axhline(0.33, color=_MUTED, ls=":", lw=0.7)
        ax.set_axisbelow(True)
        recv = attrs[lab][full_name]["region_att"]
        top = np.argsort(recv)[::-1][:5]
        txt = "Full top-5:\n" + "\n".join(f"{REGION_NAMES[j]} {recv[j]:.3f}" for j in top)
        ax.text(1.04, 0.5, txt, transform=ax.transAxes, fontsize=6, va="center",
                color=_INK, clip_on=False)
        if r == 0:
            ax.legend(fontsize=5.5, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.02))
        ax.set_title(case["stem"][:24], fontsize=6.5, pad=2)
    fig.suptitle("Figure 6. Individual cases — NeuroGate maps vs multi-model predictions",
                 fontsize=11, y=0.97)
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("wrote", out)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=Path, default=ROOT / "local_assets/sample_data/cache_real")
    p.add_argument("--full_ckpt", type=Path,
                   default=ROOT / "experiment_results/6gpu_fast/phaseA_ours_full/seed_42/best_model_seed42_fold0.pth")
    p.add_argument("--viz_dir", type=Path, default=ROOT / "experiment_results/viz_baselines")
    p.add_argument("--out_dir", type=Path, default=ROOT / "assets")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip_train_check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case_paths = pick_cases(args.cache_dir)
    cases = {lab: load_npz_case(p) for lab, p in case_paths.items()}
    print("cases:", {CLASS_NAMES[k]: v["stem"] for k, v in cases.items()})

    models = {}
    # NeuroGate Full — atlas attention
    m_full, miss, _ = load_ckpt_model("ours", True, args.full_ckpt, device)
    print(f"Full loaded (missing={len(miss)})")
    models["NeuroGate (Full)"] = {
        "model": m_full, "model_name": "ours", "use_atlas": True, "kind": "attn",
    }

    viz_specs = [
        ("Plain CNN", "plaincnn", False, "plaincnn_seed42_fold0.pth"),
        ("3D ResNet-18", "resnet3d", False, "resnet3d_seed42_fold0.pth"),
        ("NeuroGate (−Atlas)", "ours", False, "no_atlas_seed42_fold0.pth"),
    ]
    for display, mname, use_atlas, fname in viz_specs:
        ckpt = args.viz_dir / fname
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Missing viz checkpoint {ckpt}. Run scripts/train_viz_baselines.py first."
            )
        model, miss, _ = load_ckpt_model(mname, use_atlas, ckpt, device)
        layer = target_layer_for(model, mname)
        if mname == "plaincnn":
            layer = find_last_conv(model)
        models[display] = {
            "model": model, "model_name": mname, "use_atlas": use_atlas,
            "kind": "gradcam", "layer": layer,
        }
        print(f"loaded {display} from {ckpt.name}")

    model_order = ["Plain CNN", "3D ResNet-18", "NeuroGate (−Atlas)", "NeuroGate (Full)"]
    attrs = compute_attributions(cases, models, device)
    quant = _load_quant(args.viz_dir)
    if quant:
        print(f"loaded quant n={quant.get('n_cases')}")
    else:
        print("WARNING: attribution_quant_summary.json not found — figures without stats bars")

    # persist case meta for reproducibility
    meta = {
        "cases": {CLASS_NAMES[k]: {"stem": v["stem"], "path": str(v["path"])} for k, v in cases.items()},
        "models": model_order,
        "full_ckpt": str(args.full_ckpt),
        "viz_dir": str(args.viz_dir),
        "quant_n": None if quant is None else quant.get("n_cases"),
    }
    (args.out_dir / "fig_case_model_meta.json").write_text(json.dumps(meta, indent=2))

    draw_fig03(args.out_dir / "fig3_spatial_attention.png", cases, attrs, model_order, quant)
    draw_fig04(args.out_dir / "fig4_disease_gradient.png", cases, attrs, model_order, quant)
    draw_fig05(args.out_dir / "fig5_cross_dataset.png", cases, attrs, model_order, quant)
    draw_fig06(args.out_dir / "fig6_case_study.png", cases, attrs, model_order)

    # sync names used in submission package
    mapping = {
        "fig3_spatial_attention.png": "Fig03_spatial_attention.png",
        "fig4_disease_gradient.png": "Fig04_disease_gradient.png",
        "fig5_cross_dataset.png": "Fig05_cross_dataset.png",
        "fig6_case_study.png": "Fig06_case_study.png",
    }
    sub = ROOT / "submission/NeuroImage_package/03_figures"
    sub.mkdir(parents=True, exist_ok=True)
    for src, dst in mapping.items():
        data = (args.out_dir / src).read_bytes()
        (args.out_dir / src).write_bytes(data)
        (sub / dst).write_bytes(data)
        print("synced", sub / dst)


if __name__ == "__main__":
    main()
