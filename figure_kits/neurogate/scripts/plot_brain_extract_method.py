#!/usr/bin/env python3
"""Draw a method schematic + before/after panels for brain extraction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chapter1_foundation.preprocess_brain_extract import (  # noqa: E402
    brain_extract,
    load_volume,
    plot_orthogonal_qc,
)


STEPS = [
    "1. Reorient\nto RAS",
    "2. Threshold /\nmodel mask",
    "3. Largest\nconnected component",
    "4. Dilate /\nerode",
    "5. Apply mask\n(brain ROI)",
    "6. Volume\n(mL)",
]


def draw_flowchart(out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Brain extraction preprocessing pipeline", fontsize=13, pad=8)

    xs = np.linspace(0.6, 10.6, len(STEPS))
    for i, (x, text) in enumerate(zip(xs, STEPS)):
        box = mpatches.FancyBboxPatch(
            (x - 0.75, 0.9),
            1.5,
            1.3,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor="#E8F1F8",
            edgecolor="#2F5D8C",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x, 1.55, text, ha="center", va="center", fontsize=8.5)
        if i < len(STEPS) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.85, 1.55),
                xytext=(x + 0.85, 1.55),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--percentile", type=float, default=40.0)
    p.add_argument("--dilate", type=int, default=0)
    p.add_argument("--peel", type=int, default=4)
    p.add_argument("--grow", type=int, default=5)
    p.add_argument("--no_model_mask", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_flowchart(args.output_dir / "Fig_brain_extract_flowchart.png")

    image, affine, model_mask, meta = load_volume(args.input)
    result = brain_extract(
        image,
        affine,
        model_mask=None if args.no_model_mask else model_mask,
        percentile=args.percentile,
        dilate=args.dilate,
        peel=args.peel,
        grow=args.grow,
        use_model_mask=not args.no_model_mask and model_mask is not None,
        do_reorient=meta.get("format") != "npz",
        source_path=str(args.input),
    )
    qc = plot_orthogonal_qc(
        result.image_ras,
        result.mask_ras,
        result.masked_ras,
        args.output_dir / "Fig_brain_extract_qc.png",
        title=f"Brain extraction QC — {args.input.name}",
        volume_ml=result.brain_mask_volume_ml,
        affine=result.affine_ras,
    )
    print(f"flowchart: {args.output_dir / 'Fig_brain_extract_flowchart.png'}")
    print(f"qc: {qc}")
    print(f"brain_mask_volume_ml: {result.brain_mask_volume_ml:.2f}")


if __name__ == "__main__":
    main()
