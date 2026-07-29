#!/usr/bin/env python3
"""Render real MRI analysis assets for the ARA-Net web prototype."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "examples" / "IXI002-Guys-0828-T1.npz"
OUT = ROOT / "frontend" / "assets"

AD_KEY_LABELS = {
    4: ("Lateral ventricle L", (45, 110, 220)),
    43: ("Lateral ventricle R", (45, 110, 220)),
    17: ("Hippocampus L", (220, 38, 38)),
    53: ("Hippocampus R", (220, 38, 38)),
    18: ("Amygdala L", (245, 125, 35)),
    54: ("Amygdala R", (245, 125, 35)),
}


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[arr > 0], [1, 99]) if np.any(arr > 0) else (0, 1)
    scaled = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled, 0, 1)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return (normalize_slice(arr) * 255).astype(np.uint8)


def orient_image(slice2d: np.ndarray) -> np.ndarray:
    return np.flipud(np.rot90(slice2d))


def slice_for_plane(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return orient_image(volume[:, :, index])
    if plane == "coronal":
        return orient_image(volume[:, index, :])
    if plane == "sagittal":
        return orient_image(volume[index, :, :])
    raise ValueError(plane)


def make_heat(seg_slice: np.ndarray, image_slice: np.ndarray) -> Image.Image:
    key_mask = np.isin(seg_slice, list(AD_KEY_LABELS)).astype(np.float32)
    intensity = normalize_slice(image_slice)
    yy, xx = np.indices(key_mask.shape)
    center_y, center_x = np.array(key_mask.shape) / 2
    radial = np.exp(-(((yy - center_y) / max(key_mask.shape[0] * 0.36, 1)) ** 2 + ((xx - center_x) / max(key_mask.shape[1] * 0.36, 1)) ** 2))
    heat = np.clip((key_mask * 0.82 + intensity * key_mask * 0.32 + radial * key_mask * 0.22), 0, 1)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius=2.2))
    rgba = Image.new("RGBA", heat_img.size, (230, 42, 42, 0))
    alpha = np.asarray(heat_img)
    color = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    color[..., 0] = 235
    color[..., 1] = np.clip(80 + alpha * 0.45, 0, 255)
    color[..., 2] = 25
    color[..., 3] = np.clip(alpha * 0.72, 0, 190)
    return Image.fromarray(color, "RGBA")


def contour(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    core = padded[1:-1, 1:-1]
    eroded = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & core
    )
    return core & ~eroded


def draw_regions(base: Image.Image, seg_slice: np.ndarray, scale: int) -> None:
    draw = ImageDraw.Draw(base, "RGBA")
    for label, (name, color) in AD_KEY_LABELS.items():
        edge = contour(seg_slice == label)
        if not np.any(edge):
            continue
        ys, xs = np.where(edge)
        points = [(int(x * scale), int(y * scale)) for y, x in zip(ys, xs)]
        for x, y in points:
            draw.ellipse((x - 1, y - 1, x + 2, y + 2), fill=(*color, 230))


def render_panel(volume: np.ndarray, seg: np.ndarray, plane: str, index: int, title: str, size=(460, 330)) -> Image.Image:
    img_slice = slice_for_plane(volume, plane, index)
    seg_slice = slice_for_plane(seg, plane, index)
    gray = Image.fromarray(to_uint8(img_slice), "L").convert("RGBA")
    gray = gray.resize(size, Image.Resampling.BICUBIC)
    heat = make_heat(seg_slice, img_slice).resize(size, Image.Resampling.BICUBIC)
    panel = Image.alpha_composite(gray, heat)

    scale_x = size[0] / seg_slice.shape[1]
    scale_y = size[1] / seg_slice.shape[0]
    if abs(scale_x - scale_y) < 0.05:
        draw_regions(panel, seg_slice, max(1, int(round(scale_x))))
    else:
        edge_layer = Image.new("RGBA", seg_slice.shape[::-1], (0, 0, 0, 0))
        draw_regions(edge_layer, seg_slice, 1)
        panel = Image.alpha_composite(panel, edge_layer.resize(size, Image.Resampling.NEAREST))

    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, size[0], 42), fill=(10, 18, 32, 176))
    draw.text((16, 12), title, fill=(255, 255, 255, 245))
    draw.rectangle((0, size[1] - 34, size[0], size[1]), fill=(10, 18, 32, 150))
    draw.text((16, size[1] - 24), "Pixel-level heat proxy + region-level AD-key atlas contours", fill=(226, 232, 240, 235))
    return panel


def legend(width: int, height: int) -> Image.Image:
    img = Image.new("RGBA", (width, height), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((20, 18), "ARA-Net real sample rendering", fill=(15, 23, 42, 255))
    draw.text((20, 46), "Subject: IXI002-Guys-0828-T1 | Prediction: CN | CN 0.628 / MCI 0.291 / AD 0.082", fill=(51, 65, 85, 255))
    items = [
        ("Lateral ventricles", (45, 110, 220)),
        ("Hippocampus", (220, 38, 38)),
        ("Amygdala", (245, 125, 35)),
        ("Pixel heat proxy", (235, 116, 25)),
    ]
    x = 20
    for name, color in items:
        draw.rounded_rectangle((x, 84, x + 18, 102), radius=4, fill=(*color, 210))
        draw.text((x + 26, 85), name, fill=(51, 65, 85, 255))
        x += 210
    draw.text((20, 124), "Rendering uses the real cached MRI volume and segmentation. Heat is a visualization proxy over AD-key atlas regions, not a clinical biomarker.", fill=(100, 116, 139, 255))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(CACHE) as data:
        volume = data["image"].astype(np.float32)
        seg = data["seg"].astype(np.int16)

    indices = {
        "axial": volume.shape[2] // 2,
        "coronal": volume.shape[1] // 2,
        "sagittal": volume.shape[0] // 2,
    }
    panels = [
        render_panel(volume, seg, "axial", indices["axial"], "Axial T1 MRI"),
        render_panel(volume, seg, "coronal", indices["coronal"], "Coronal T1 MRI"),
        render_panel(volume, seg, "sagittal", indices["sagittal"], "Sagittal T1 MRI"),
    ]
    for name, panel in zip(("axial", "coronal", "sagittal"), panels):
        panel.convert("RGB").save(OUT / f"ixi002_{name}_analysis.png", quality=95)

    width = 1420
    height = 555
    canvas = Image.new("RGB", (width, height), (238, 243, 246))
    top = legend(width - 40, 160).convert("RGB")
    canvas.paste(top, (20, 18))
    x = 20
    for panel in panels:
        canvas.paste(panel.convert("RGB"), (x, 196))
        x += 472
    canvas.save(OUT / "ixi002_real_analysis.png", quality=95)
    print(OUT / "ixi002_real_analysis.png")


if __name__ == "__main__":
    main()
