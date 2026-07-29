#!/usr/bin/env python3
"""Parse a training log produced by ``a2c_node.train.fit`` and plot the
loss/accuracy curves.

Usage::

    python -m scripts.plot_training --log runs/main/train.log
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


_EPOCH_RE = re.compile(
    r"epoch\s+(?P<ep>\d+)\s+"
    r"train=(?P<train>[\d\.eE+\-nan]+)\s+"
    r"val=(?P<val>[\d\.eE+\-nan]+)\s+"
    r"val\(drop\)=(?P<vald>[\d\.eE+\-nan]+)\s+"
    r"acc=(?P<acc>[\d\.eE+\-nan]+)\s+"
    r"lr=(?P<lr>[\d\.eE+\-nan]+)\s+"
    r"t=(?P<t>[\d\.eE+\-nan]+)s"
)


def parse_log(log_path: Path) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, float]] = []
    with open(log_path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            try:
                rows.append({k: float(v) for k, v in d.items() if k != "ep"} | {"ep": int(d["ep"])})
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No epoch lines parsed from {log_path}")
    return {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}


def plot(parsed: Dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(parsed["ep"], parsed["train"], label="train", lw=2)
    ax.plot(parsed["ep"], parsed["val"],   label="val (full)", lw=2)
    ax.plot(parsed["ep"], parsed["vald"],  label="val (drop visits)",
            lw=2, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("A2C-NODE loss curve")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(parsed["ep"], parsed["acc"] * 100.0, lw=2, color="#1f77b4")
    ax.axhline(33.3, color="grey", ls=":", lw=1, label="random (33.3%)")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Validation classification accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(parsed["ep"], parsed["lr"], lw=2, color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate (log)")
    ax.set_title("Learning-rate schedule")
    ax.grid(alpha=0.3, which="both")

    fig.savefig(out_path, dpi=160)
    print(f"saved: {out_path}")
    print(f"summary: best val={parsed['val'].min():.4f}  "
          f"best acc={parsed['acc'].max()*100:.2f}%  "
          f"final t={parsed['t'][-1]:.1f}s/epoch")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="path to train.log")
    p.add_argument("--out", default=None, help="output PNG (default: same dir/curves.png)")
    args = p.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_name("curves.png")

    parsed = parse_log(log_path)
    plot(parsed, out_path)


if __name__ == "__main__":
    main()
