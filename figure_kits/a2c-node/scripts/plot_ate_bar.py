#!/usr/bin/env python3
"""Plot a publication-quality bar chart of per-region ATE on AD probability.

Reads the CSV produced by ``compute_ate.py`` and emits

    ate_AD_bar.{png,pdf}

with the SciencePlots "science+nature" style (graceful fallback if
SciencePlots is missing). Hippocampus, Entorhinal and other temporal-lobe
regions are highlighted in red because they are the strongest a-priori
biological candidates.
"""
from __future__ import annotations

import argparse
import csv as _csv
from pathlib import Path
from typing import List

import numpy as np


HIGHLIGHT = {"Hippocampus", "Entorhinal", "Amygdala", "Thalamus"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="ate CSV from compute_ate.py")
    p.add_argument("--out_prefix", required=True, type=Path,
                   help="output path prefix; emits .png and .pdf")
    p.add_argument("--top_k", type=int, default=21,
                   help="show the top-K regions by |ATE_AD|")
    return p.parse_args()


def load(csv_path: Path):
    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            r["ate_AD"] = float(r["ate_AD"])
            r["ate_CN"] = float(r["ate_CN"])
            r["ate_MCI"] = float(r["ate_MCI"])
            rows.append(r)
    rows.sort(key=lambda r: -abs(r["ate_AD"]))
    return rows


def main() -> None:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    from _plot_style import apply as _style
    _style(extra=("nature",))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(args.input)[: args.top_k]
    names = [r["region_name"] for r in rows]
    vals = np.array([r["ate_AD"] for r in rows], dtype=np.float64)
    colors = ["#c0392b" if any(h in n for h in HIGHLIGHT) else "#34495e"
              for n in names]

    h = max(3.0, 0.32 * len(rows))
    fig, ax = plt.subplots(figsize=(5.6, h))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"ATE on $P(\mathrm{AD})$  [do$(h_k=0)$ - factual]")
    ax.set_title("Per-region average treatment effect (AD)", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=300)
    fig.savefig(f"{args.out_prefix}.pdf")
    plt.close(fig)
    print(f"[plot] wrote {args.out_prefix}.png / .pdf  K={len(rows)}")


if __name__ == "__main__":
    main()
