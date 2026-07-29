#!/usr/bin/env python3
"""Generate annotated real-case MRI slice images for the Streamlit demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO = ROOT / "artifacts" / "streamlit_demo_cases.csv"
DEFAULT_LABEL_MAP = ROOT.parent / "chapter2_disentangle" / "data" / "dkt_label_map.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "streamlit_case_images"

ROI_GROUPS = {
    "Hippocampus": {"fs_labels": [17, 53], "color": (56, 189, 248), "note": "memory-related atrophy marker"},
    "Amygdala": {"fs_labels": [18, 54], "color": (244, 114, 182), "note": "medial temporal structure"},
    "Entorhinal": {"fs_labels": [1006, 2006], "color": (250, 204, 21), "note": "early AD-vulnerable cortex"},
    "Parahippocampal": {"fs_labels": [1016, 2016], "color": (167, 139, 250), "note": "temporal memory network"},
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(64, bold=True)
SUBTITLE_FONT = load_font(40)
LABEL_FONT = load_font(44, bold=True)
BODY_FONT = load_font(48)
SMALL_FONT = load_font(32)
TRI_LABEL_FONT = load_font(38, bold=True)
TRI_NOTE_FONT = load_font(26)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-cases", default=str(DEFAULT_DEMO))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-cases", type=int, default=20)
    return parser.parse_args()


def normalize_slice(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    lo, hi = np.percentile(values, [1, 99])
    out = np.clip((image - lo) / max(hi - lo, 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def best_slice(seg: np.ndarray, labels: list[int]) -> int:
    mask = np.isin(seg, labels)
    counts = mask.sum(axis=(0, 1))
    if counts.max() == 0:
        return int(seg.shape[2] // 2)
    return int(np.argmax(counts))


def best_slice_for_axis(seg: np.ndarray, labels: list[int], axis: int) -> int:
    mask = np.isin(seg, labels)
    axes = tuple(i for i in range(mask.ndim) if i != axis)
    counts = mask.sum(axis=axes)
    if counts.max() == 0:
        return int(seg.shape[axis] // 2)
    return int(np.argmax(counts))


def find_centroid(mask2d: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask2d)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def draw_callout(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    side: str,
    slot: int,
) -> None:
    x, y = xy
    box_w = 460
    box_h = 104
    left_slots = [520, 665, 810, 955]
    right_slots = [620, 765, 910, 1055]
    if side == "left":
        tx, ty = 80, left_slots[min(slot, len(left_slots) - 1)]
        anchor_x = tx + box_w
    else:
        tx, ty = 1060, right_slots[min(slot, len(right_slots) - 1)]
        anchor_x = tx
    draw.line((x, y, anchor_x, ty + box_h // 2), fill=color, width=6)
    draw.rounded_rectangle((tx, ty, tx + box_w, ty + box_h), radius=12, fill=(17, 24, 39), outline=color, width=4)
    draw.text((tx + 22, ty + 24), label, fill=(245, 247, 250), font=BODY_FONT)


def crop_to_brain(base: np.ndarray, seg_slice: np.ndarray, pad: int = 8) -> tuple[np.ndarray, np.ndarray]:
    mask = seg_slice > 0
    if not mask.any():
        mask = base > np.percentile(base, 15)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return base, seg_slice
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(base.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(base.shape[1], int(xs.max()) + pad + 1)
    h = y1 - y0
    w = x1 - x0
    side = max(h, w)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = max(0, cy - side // 2)
    y1 = min(base.shape[0], y0 + side)
    x0 = max(0, cx - side // 2)
    x1 = min(base.shape[1], x0 + side)
    y0 = max(0, y1 - side)
    x0 = max(0, x1 - side)
    return base[y0:y1, x0:x1], seg_slice[y0:y1, x0:x1]


def prepare_display_slice(base: np.ndarray, seg_slice: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=np.uint8).copy()
    if (seg_slice > 0).any():
        out[seg_slice <= 0] = 12
    return out


def oriented_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(volume[index, :, :])
    if axis == 1:
        return np.rot90(volume[:, index, :])
    return np.rot90(volume[:, :, index])


def roi_overlay(
    base: np.ndarray,
    seg_slice: np.ndarray,
    roi_cache_labels: dict[str, list[int]],
    alpha: float = 0.64,
) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    for name, spec in ROI_GROUPS.items():
        labels = roi_cache_labels.get(name, [])
        if not labels:
            continue
        mask = np.isin(seg_slice, labels)
        color = np.asarray(spec["color"], dtype=np.float32)
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
        inner = mask
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            inner = inner & np.roll(mask, shift=(dy, dx), axis=(0, 1))
        edge = mask & ~inner
        rgb[edge] = color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_panel_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + 150, y + 44), radius=8, fill=(15, 23, 42), outline=(71, 85, 105), width=2)
    draw.text((x + 14, y + 9), label, fill=(248, 250, 252), font=LABEL_FONT)


def resized_overlay_image(
    base: np.ndarray,
    seg_slice: np.ndarray,
    roi_cache_labels: dict[str, list[int]],
    size: int,
) -> tuple[Image.Image, np.ndarray]:
    base_img = (
        Image.fromarray(base)
        .resize((size, size), Image.Resampling.LANCZOS)
        .filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
    )
    seg_img = Image.fromarray(seg_slice.astype(np.int32), mode="I").resize((size, size), Image.Resampling.NEAREST)
    base_resized = np.asarray(base_img)
    seg_resized = np.asarray(seg_img, dtype=np.int32)
    return Image.fromarray(roi_overlay(base_resized, seg_resized, roi_cache_labels)), seg_resized


def make_view_image(
    image: np.ndarray,
    seg: np.ndarray,
    axis: int,
    all_labels: list[int],
    roi_cache_labels: dict[str, list[int]],
    size: int,
    pad: int = 14,
) -> tuple[Image.Image, np.ndarray]:
    idx = best_slice_for_axis(seg, all_labels, axis)
    seg_slice = oriented_slice(seg, axis, idx)
    base = normalize_slice(oriented_slice(image, axis, idx))
    base, seg_slice = crop_to_brain(base, seg_slice, pad=pad)
    base = prepare_display_slice(base, seg_slice)
    return resized_overlay_image(base, seg_slice, roi_cache_labels, size)


def render_case(row: pd.Series, label_map: dict[int, int], output_dir: Path) -> str | None:
    cache_path = row.get("cache_seg_path")
    if not isinstance(cache_path, str) or not cache_path or not Path(cache_path).exists():
        return None
    data = np.load(cache_path)
    image = np.asarray(data["image"], dtype=float)
    seg = np.asarray(data["seg"], dtype=np.int16)
    roi_cache_labels: dict[str, list[int]] = {}
    all_labels: list[int] = []
    for name, spec in ROI_GROUPS.items():
        labels = [label_map.get(int(fs)) for fs in spec["fs_labels"]]
        labels = [int(label) for label in labels if label is not None]
        roi_cache_labels[name] = labels
        all_labels.extend(labels)
    z = best_slice(seg, all_labels)
    seg_slice = np.rot90(seg[:, :, z])
    base = normalize_slice(np.rot90(image[:, :, z]))
    base, seg_slice = crop_to_brain(base, seg_slice, pad=12)
    base = prepare_display_slice(base, seg_slice)
    panel = Image.new("RGB", (1600, 1840), (8, 13, 22))
    image_size = 1420
    mri, seg_display = resized_overlay_image(base, seg_slice, roi_cache_labels, image_size)
    panel.paste(mri, (90, 190))
    draw = ImageDraw.Draw(panel)
    title = f"{row.get('PTID', 'case')} | {row.get('diagnosis_pair', 'NA')} | {row.get('conflict_type', 'none')}"
    draw.text((60, 24), title, fill=(248, 250, 252), font=TITLE_FONT)
    draw.text((60, 92), "Zoomed real MRI model input with segmented ROI overlays", fill=(148, 163, 184), font=SUBTITLE_FONT)
    draw.rounded_rectangle((60, 1736, 1540, 1800), radius=16, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
    x_legend = 92
    for name, spec in ROI_GROUPS.items():
        color = spec["color"]
        draw.rounded_rectangle((x_legend, 1752, x_legend + 32, 1784), radius=7, fill=color)
        draw.text((x_legend + 44, 1750), name, fill=(226, 232, 240), font=SMALL_FONT)
        x_legend += 360
    centroids = []
    for name, spec in ROI_GROUPS.items():
        labels = roi_cache_labels.get(name, [])
        mask = np.isin(seg_slice, labels)
        centroid = find_centroid(mask)
        if centroid is not None:
            centroids.append((name, centroid, spec["color"]))
    left_slot = 0
    right_slot = 0
    for idx, (name, centroid, color) in enumerate(centroids[:4]):
        x, y0 = centroid
        px = 90 + int(x * image_size / seg_slice.shape[1])
        py = 190 + int(y0 * image_size / seg_slice.shape[0])
        side = "left" if name in {"Entorhinal", "Hippocampus"} else "right"
        if side == "left":
            draw_callout(draw, (px, py), name, color, side, left_slot)
            left_slot += 1
        else:
            draw_callout(draw, (px, py), name, color, side, right_slot)
            right_slot += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(row.get("demo_case_id", f"case_{int(row.name) + 1:02d}"))
    filename = f"{case_id}_{row.get('PTID', 'case')}_annotated.png".replace("/", "_")
    path = output_dir / filename
    panel.save(path)
    return str(path)


def render_case_triplanar(row: pd.Series, label_map: dict[int, int], output_dir: Path) -> str | None:
    cache_path = row.get("cache_seg_path")
    if not isinstance(cache_path, str) or not cache_path or not Path(cache_path).exists():
        return None
    data = np.load(cache_path)
    image = np.asarray(data["image"], dtype=float)
    seg = np.asarray(data["seg"], dtype=np.int16)
    roi_cache_labels: dict[str, list[int]] = {}
    all_labels: list[int] = []
    for name, spec in ROI_GROUPS.items():
        labels = [label_map.get(int(fs)) for fs in spec["fs_labels"]]
        labels = [int(label) for label in labels if label is not None]
        roi_cache_labels[name] = labels
        all_labels.extend(labels)

    view_size = 560
    views = [
        ("Axial", *make_view_image(image, seg, 2, all_labels, roi_cache_labels, view_size)),
        ("Coronal", *make_view_image(image, seg, 1, all_labels, roi_cache_labels, view_size)),
        ("Sagittal", *make_view_image(image, seg, 0, all_labels, roi_cache_labels, view_size)),
    ]
    panel = Image.new("RGB", (1800, 1220), (8, 13, 22))
    draw = ImageDraw.Draw(panel)
    title = f"{row.get('PTID', 'case')} | {row.get('diagnosis_pair', 'NA')} | {row.get('conflict_type', 'none')}"
    draw.text((44, 24), title, fill=(248, 250, 252), font=TITLE_FONT)
    draw.text((44, 92), "Three-view real MRI context with segmented AD-relevant ROIs", fill=(148, 163, 184), font=SUBTITLE_FONT)
    x_positions = [44, 622, 1200]
    for (label, view, _seg_slice), x in zip(views, x_positions):
        panel.paste(view, (x, 128))
        draw_panel_label(draw, (x + 18, 146), label)
    legend_y = 742
    draw.rounded_rectangle((44, legend_y, 1756, 1176), radius=18, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
    draw.text((78, legend_y + 30), "ROI legend", fill=(248, 250, 252), font=LABEL_FONT)
    legend_positions = [(78, legend_y + 112), (910, legend_y + 112), (78, legend_y + 242), (910, legend_y + 242)]
    for (name, spec), (x, y) in zip(ROI_GROUPS.items(), legend_positions):
        color = spec["color"]
        draw.rounded_rectangle((x, y + 6, x + 42, y + 48), radius=8, fill=color)
        draw.text((x + 62, y - 2), name, fill=(241, 245, 249), font=TRI_LABEL_FONT)
        draw.text((x + 62, y + 44), str(spec["note"]), fill=(148, 163, 184), font=TRI_NOTE_FONT)
    draw.text(
        (78, legend_y + 370),
        "These images explain model inputs only and do not replace formal radiology interpretation.",
        fill=(203, 213, 225),
        font=TRI_NOTE_FONT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(row.get("demo_case_id", f"case_{int(row.name) + 1:02d}"))
    filename = f"{case_id}_{row.get('PTID', 'case')}_triplanar.png".replace("/", "_")
    path = output_dir / filename
    panel.save(path)
    return str(path)


def main() -> None:
    args = parse_args()
    demo = pd.read_csv(args.demo_cases)
    raw_map = json.loads(Path(args.label_map).read_text(encoding="utf-8"))
    label_map = {int(k): int(v) for k, v in raw_map.items()}
    output = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for idx, row in demo.head(args.max_cases).iterrows():
        path = render_case(row, label_map, output)
        if path:
            tri_path = render_case_triplanar(row, label_map, output)
            rows.append(
                {
                    "demo_case_id": row.get("demo_case_id"),
                    "PTID": row.get("PTID"),
                    "image_path": path,
                    "triplanar_path": tri_path or "",
                }
            )
    pd.DataFrame(rows).to_csv(output / "case_image_index.csv", index=False)
    print(json.dumps({"output_dir": str(output), "images": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
