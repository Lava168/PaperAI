#!/usr/bin/env python3
"""Predictor-association network for A2C-NODE.

Builds a small graph that captures *Spearman* correlations among:
* the 3-class predicted probabilities (CN / MCI / AD)
* the 12 / 15 ADNIMERGE clinical features used by the tabular branch
* the 21 per-region ATE values on AD probability

Nodes are coloured by group and laid out with the spring algorithm so that
strongly-correlated predictors cluster together. Edge thickness ∝ |ρ|, and
positive vs negative correlations are coloured red vs blue.
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_npz", required=True, type=Path,
                   help="prediction NPZ from predict_to_npz.py / ensemble_npz.py")
    p.add_argument("--ate_csv", type=Path, default=None,
                   help="optional ATE CSV from compute_ate.py")
    p.add_argument("--clinical_csv", default="",
                   help="optional ADNIMERGE-style CSV providing per-subject "
                        "values of MMSE/CDR/etc for correlation")
    p.add_argument("--csf_csv", default="")
    p.add_argument("--out_prefix", required=True, type=Path)
    p.add_argument("--top_edges", type=int, default=80,
                   help="keep the strongest |ρ| edges (avoids hairball)")
    p.add_argument("--min_abs_rho", type=float, default=0.10)
    return p.parse_args()


def spearman_pair(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return float("nan")
    xs = x[mask].argsort().argsort().astype(np.float64)
    ys = y[mask].argsort().argsort().astype(np.float64)
    xs -= xs.mean(); ys -= ys.mean()
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float((xs * ys).mean() / (xs.std() * ys.std()))


def main() -> None:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.pred_npz, allow_pickle=True)
    prob = d["prob"].astype(np.float64)
    lab = d["label"].astype(np.int64)
    ptid = d["ptid"]
    N = prob.shape[0]

    # Build feature matrix: P(CN), P(MCI), P(AD), then any clinical fields
    # in the prediction NPZ that were carried through.
    feats: Dict[str, np.ndarray] = {}
    feats["P(CN)"] = prob[:, 0]
    feats["P(MCI)"] = prob[:, 1]
    feats["P(AD)"] = prob[:, 2]
    for k in ("age", "sex", "apoe4", "mmse"):
        if k in d.files:
            feats[k.upper()] = d[k].astype(np.float64)

    # Optionally enrich with ADNIMERGE per-subject fields.
    if args.clinical_csv:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from a2c_node.data.clinical_table import (                       # type: ignore
            ClinicalTable, FEATURE_NAMES_NO_CSF, FEATURE_NAMES_CSF,
        )
        clin = ClinicalTable.from_adnimerge(
            args.clinical_csv,
            csf_csv=args.csf_csv or None,
            include_csf=bool(args.csf_csv))
        wanted = [n for n in FEATURE_NAMES_NO_CSF + (
            FEATURE_NAMES_CSF if args.csf_csv else ()
        ) if n.upper() not in feats]
        # baseline value per subject
        for name in wanted:
            arr = np.full(N, np.nan, dtype=np.float64)
            for i, pt in enumerate(ptid):
                v, m = clin.get(str(pt), "bl")
                idx = clin.feature_names.index(name)
                if m[idx]:
                    arr[i] = float(v[idx])
            feats[name] = arr

    # ATE-derived features (one summary per region: ate_AD).
    if args.ate_csv and args.ate_csv.is_file():
        with open(args.ate_csv, newline="") as f:
            rdr = _csv.DictReader(f)
            for r in rdr:
                # one row per region; each region becomes a node, value = its
                # cohort-mean ate_AD broadcast to every subject (no per-
                # subject ATE in this CSV). We just include them as anchor
                # nodes that form their own cluster.
                feats[f"ATE:{r['region_name']}"] = np.full(N, float(r["ate_AD"]))

    feature_names = list(feats.keys())
    X = np.stack([feats[n] for n in feature_names], axis=1)
    K = X.shape[1]

    # Spearman matrix
    rho = np.full((K, K), np.nan, dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            rho[i, j] = spearman_pair(X[:, i], X[:, j])
            rho[j, i] = rho[i, j]

    # Build edge list
    edges = []
    for i in range(K):
        for j in range(i + 1, K):
            v = rho[i, j]
            if np.isnan(v) or abs(v) < args.min_abs_rho:
                continue
            edges.append((i, j, float(v)))
    edges.sort(key=lambda e: -abs(e[2]))
    edges = edges[: args.top_edges]
    print(f"[net] N={N}  features={K}  kept_edges={len(edges)}",
          flush=True)

    # ------------------ render with networkx ------------------------------
    from _plot_style import apply as _style
    _style(extra=("nature",))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    for i, n in enumerate(feature_names):
        G.add_node(i, label=n)
    for i, j, v in edges:
        G.add_edge(i, j, weight=abs(v), sign=np.sign(v))

    # Group / colour nodes
    def group(name: str) -> str:
        if name.startswith("P("): return "pred"
        if name.startswith("ATE:"): return "ate"
        if name.upper() in {"MMSE", "CDRSB", "ADAS11", "ADAS13", "FAQ"}: return "cog"
        if name in {"AGE", "PTGENDER", "PTEDUCAT", "SEX"}: return "demo"
        if name in {"APOE4"}: return "gen"
        if name in {"ABETA", "TAU", "PTAU"}: return "csf"
        return "img"
    palette = {"pred": "#c0392b", "ate": "#8e44ad", "cog": "#27ae60",
               "demo": "#7f8c8d", "gen": "#d35400", "csf": "#2980b9",
               "img": "#16a085"}
    node_colors = [palette[group(feature_names[i])] for i in G.nodes()]

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    pos = nx.spring_layout(G, seed=42, k=0.85, iterations=150)
    edge_w = [G[u][v]["weight"] * 6.0 for u, v in G.edges()]
    edge_c = ["#c0392b" if G[u][v]["sign"] > 0 else "#2980b9"
              for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_w, edge_color=edge_c,
                           alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=320, node_color=node_colors,
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(
        G, pos,
        labels={i: feature_names[i].replace("ATE:", "") for i in G.nodes()},
        font_size=6.5, ax=ax,
    )
    # legend
    handles = []
    for k, c in palette.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=c, markersize=8, label=k))
    ax.legend(handles=handles, fontsize=7, loc="upper left",
              frameon=True, framealpha=0.9, ncols=3)
    ax.set_axis_off()
    ax.set_title("Predictor-association network "
                 r"(top-$k$ Spearman edges; red = +, blue = $-$)")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[net] wrote {args.out_prefix}.png / .pdf",
          flush=True)


if __name__ == "__main__":
    main()
