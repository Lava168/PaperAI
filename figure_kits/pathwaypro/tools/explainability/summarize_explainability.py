#!/usr/bin/env python3
"""Summarize PathwayPro explainability outputs into manuscript-ready tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_roi(rows: Sequence[Mapping[str, str]], top_k: int) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        key = (row["branch"], row["roi_id"], row.get("roi_name", row["roi_id"]))
        grouped.setdefault(key, []).append(float(row["delta_p_ad"]))
    summary: List[Dict[str, object]] = []
    for (branch, roi_id, roi_name), vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) > 1 else (arr[0], arr[0])
        summary.append(
            {
                "branch": branch,
                "roi_id": roi_id,
                "roi_name": roi_name,
                "n": len(vals),
                "delta_p_ad_mean": round(float(arr.mean()), 6),
                "delta_p_ad_sd": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
                "ci95_low": round(float(lo), 6),
                "ci95_high": round(float(hi), 6),
            }
        )
    out: List[Dict[str, object]] = []
    for branch in sorted({str(r["branch"]) for r in summary}):
        branch_rows = [r for r in summary if r["branch"] == branch]
        branch_rows.sort(key=lambda r: -float(r["delta_p_ad_mean"]))
        for rank, row in enumerate(branch_rows[:top_k], 1):
            row = dict(row)
            row["rank"] = rank
            out.append(row)
    return out


def summarize_deletion(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        key = (row["branch"], row["mode"], row["fraction_deleted"])
        grouped.setdefault(key, []).append(float(row["delta_p_ad"]))
    out: List[Dict[str, object]] = []
    for (branch, mode, frac), vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "branch": branch,
                "mode": mode,
                "fraction_deleted": frac,
                "n": len(vals),
                "delta_p_ad_mean": round(float(arr.mean()), 6),
                "delta_p_ad_sd": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
            }
        )
    out.sort(key=lambda r: (str(r["branch"]), float(r["fraction_deleted"]), str(r["mode"])))
    return out


def maybe_plot_deletion(rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    if not rows:
        return
    branches = sorted({str(r["branch"]) for r in rows})
    fig, axes = plt.subplots(1, len(branches), figsize=(5 * len(branches), 4), squeeze=False)
    for ax, branch in zip(axes[0], branches):
        for mode, marker in [("top_attr", "o"), ("random", "s")]:
            sub = [r for r in rows if r["branch"] == branch and r["mode"] == mode]
            sub.sort(key=lambda r: float(r["fraction_deleted"]))
            if not sub:
                continue
            x = [float(r["fraction_deleted"]) * 100 for r in sub]
            y = [float(r["delta_p_ad_mean"]) for r in sub]
            yerr = [float(r["delta_p_ad_sd"]) for r in sub]
            ax.errorbar(x, y, yerr=yerr, marker=marker, label=mode)
        ax.set_title(branch)
        ax.set_xlabel("Deleted attribution volume (%)")
        ax.set_ylabel("ΔP(AD)")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    roi = summarize_roi(read_csv(in_dir / "roi_ablation.csv"), top_k=args.top_k)
    deletion = summarize_deletion(read_csv(in_dir / "deletion_curves.csv"))
    write_csv(out_dir / "table_top_roi_ablation.csv", roi)
    write_csv(out_dir / "table_deletion_curve_summary.csv", deletion)
    maybe_plot_deletion(deletion, out_dir / "figure_deletion_curve.png")
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()

