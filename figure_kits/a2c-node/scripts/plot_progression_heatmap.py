#!/usr/bin/env python3
"""Color-coded longitudinal progression heatmap.

Reads a per-subject NPZ produced by ``predict_to_npz.py`` (or
``ensemble_npz.py``) and renders a heatmap with one row per subject and one
column per category, colour-coded by predicted P(AD) (or P(MCI), P(CN)).
Subjects are sorted by descending P(AD) so the figure reads top-down as
"how confident the model is that this person is on the AD spectrum".

This is the "clinical dashboard" view: a single image lets a clinician scan
50–100 subjects and see who needs urgent follow-up.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="prediction NPZ from predict_to_npz.py / ensemble_npz.py")
    p.add_argument("--out_prefix", required=True, type=Path)
    p.add_argument("--max_subjects", type=int, default=120)
    p.add_argument("--sort_by", choices=["AD", "MCI", "CN", "label"], default="AD")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input, allow_pickle=True)
    prob = d["prob"].astype(np.float64)                                # (N, 3)
    lab = d["label"].astype(np.int64)
    ptid = d["ptid"]
    age = d["age"].astype(np.float64) if "age" in d.files else np.full(len(lab), np.nan)
    mmse = d["mmse"].astype(np.float64) if "mmse" in d.files else np.full(len(lab), np.nan)

    # Sort
    if args.sort_by == "label":
        order = np.argsort(lab)
    else:
        col = {"CN": 0, "MCI": 1, "AD": 2}[args.sort_by]
        order = np.argsort(-prob[:, col])
    order = order[: args.max_subjects]

    prob = prob[order]; lab = lab[order]; ptid = ptid[order]
    age = age[order]; mmse = mmse[order]

    from _plot_style import apply as _style
    _style(extra=("nature",))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(4.6, max(4.0, 0.075 * len(prob) + 1.2)))
    im = ax.imshow(prob, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["P(CN)", "P(MCI)", "P(AD)"])
    ax.set_yticks([]); ax.set_ylabel(f"Subjects (sorted by P({args.sort_by}))")
    ax.set_title("Predicted progression probabilities")

    # Side band for true label
    ax2 = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
    LABEL_C = ["#2980b9", "#f39c12", "#c0392b"]                        # CN MCI AD
    for i, l in enumerate(lab):
        if 0 <= l <= 2:
            ax2.add_patch(Rectangle((0, len(lab) - 1 - i), 1, 1,
                                    color=LABEL_C[int(l)], linewidth=0))
    ax2.set_xlim(0, 1); ax2.set_ylim(0, len(lab))
    ax2.axis("off"); ax2.set_title("true", fontsize=8)

    # Bottom colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.05)
    cbar.set_label("Probability", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] wrote {args.out_prefix}.png / .pdf  N={len(lab)}",
          flush=True)


if __name__ == "__main__":
    main()
