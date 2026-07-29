#!/usr/bin/env python3
"""Create a quick PNG preview for saved PathwayPro Grad-CAM volumes."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_adapter(path: str):
    spec = importlib.util.spec_from_file_location("pathwaypro_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_numpy_volume(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().squeeze().float().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape {arr.shape}")
    return arr


def load_cam(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        import nibabel as nib  # type: ignore

        arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D CAM, got shape {arr.shape}")
    return normalize(arr)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def best_indices(cam: np.ndarray) -> Dict[str, int]:
    return {
        "sagittal": int(np.argmax(cam.sum(axis=(1, 2)))),
        "coronal": int(np.argmax(cam.sum(axis=(0, 2)))),
        "axial": int(np.argmax(cam.sum(axis=(0, 1)))),
    }


def crop_to_mask(img: np.ndarray, cam: np.ndarray, mask: np.ndarray, pad: int = 8):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img, cam, mask
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, img.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return img[slices], cam[slices], mask[slices]


def slice_pair(img: np.ndarray, cam: np.ndarray, plane: str, index: int):
    if plane == "sagittal":
        return img[index, :, :].T, cam[index, :, :].T
    if plane == "coronal":
        return img[:, index, :].T, cam[:, index, :].T
    if plane == "axial":
        return img[:, :, index].T, cam[:, :, index].T
    raise ValueError(plane)


def find_sample(adapter, split_csv: str, data_root: str, subject_id: str):
    for sample in adapter.iter_samples(split_csv, data_root, max_subjects=None):
        if str(sample["subject_id"]) == subject_id:
            return sample
    raise ValueError(f"Subject not found in split: {subject_id}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--gradcam-dir", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--branches", nargs="+", default=["t1", "flair"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    adapter = load_adapter(args.adapter)
    sample = find_sample(adapter, args.split_csv, args.data_root, args.subject_id)
    mask = to_numpy_volume(sample["roi_mask"]) > 0
    gradcam_dir = Path(args.gradcam_dir)
    branches = list(args.branches)

    fig, axes = plt.subplots(len(branches), 3, figsize=(10, 3.6 * len(branches)), squeeze=False)
    planes = ["sagittal", "coronal", "axial"]
    for row, branch in enumerate(branches):
        img_key = "t1" if branch == "t1" else "flair"
        img = normalize(to_numpy_volume(sample[img_key]))
        cam_path = gradcam_dir / f"{args.subject_id}_{branch}_ad_logit.nii.gz"
        if not cam_path.exists():
            cam_path = gradcam_dir / f"{args.subject_id}_{branch}_ad_logit.npy"
        cam = load_cam(cam_path)
        if cam.shape != img.shape:
            raise ValueError(f"Shape mismatch for {branch}: image={img.shape}, cam={cam.shape}")
        if mask.shape != img.shape:
            raise ValueError(f"Shape mismatch for mask: image={img.shape}, mask={mask.shape}")
        img_c, cam_c, mask_c = crop_to_mask(img, cam * mask, mask)
        cam_c = normalize(cam_c)
        indices = best_indices(cam_c)
        for col, plane in enumerate(planes):
            img_sl, cam_sl = slice_pair(img_c, cam_c, plane, indices[plane])
            cam_sl = np.ma.masked_less(cam_sl, 0.06)
            ax = axes[row, col]
            ax.imshow(img_sl, cmap="gray", origin="lower")
            ax.imshow(cam_sl, cmap="inferno", alpha=0.48, origin="lower", vmin=0.0, vmax=1.0)
            ax.set_title(f"{branch.upper()} {plane} #{indices[plane]}")
            ax.axis("off")

    fig.suptitle(f"Grad-CAM preview: {args.subject_id}", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
