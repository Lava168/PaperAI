#!/usr/bin/env python3
"""Papez-inspired circuit atrophy score vs cognitive / pathology markers.

Step-1 analysis for ARA-Net Ch1 biological validation:
  - DKT/FastSurfer ROI volumes (AIBL: full Papez circuit nodes)
  - Coarse 21-region proxy (all cohorts from atlas feature cache)
  - Spearman correlations vs MMSE, CDR-SB, amyloid proxies
  - Head-to-head vs hippocampus-only and legacy AD-key consistency score
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kruskal, spearmanr

from analyze_atlas_feature_biomarkers import ad_key_score, region_gradient_tests
from train_atlas_feature_baseline import AD_KEY_REGIONS, REGION_NAMES
from train_hybrid_atlas_clinical_baseline import (
    CLINICAL_FEATURES,
    build_adni_index,
    build_aibl_index,
    enrich_rows,
    read_csv_rows,
)


AIBL_SCAN_RE = re.compile(r"^AIBL_([0-9]+)_([^_]+)_I([0-9]+)$")

# Five circuit nodes (bilateral structures grouped). DKT StructName from aseg+DKT.VINN.stats.
PAPEZ_CIRCUIT_NODES_DKT: Dict[str, List[str]] = {
    "hippocampus": ["Left-Hippocampus", "Right-Hippocampus"],
    "thalamus": ["Left-Thalamus", "Right-Thalamus"],
    "entorhinal": ["ctx-lh-entorhinal", "ctx-rh-entorhinal"],
    "parahippocampal": ["ctx-lh-parahippocampal", "ctx-rh-parahippocampal"],
    "cingulate": [
        "ctx-lh-rostralanteriorcingulate",
        "ctx-rh-rostralanteriorcingulate",
        "ctx-lh-caudalanteriorcingulate",
        "ctx-rh-caudalanteriorcingulate",
        "ctx-lh-posteriorcingulate",
        "ctx-rh-posteriorcingulate",
    ],
}

# Coarse 21-region proxy when DKT stats are unavailable.
PAPEZ_CIRCUIT_NODES_COARSE21: Dict[str, List[str]] = {
    "hippocampus": ["L-Hippocampus", "R-Hippocampus"],
    "thalamus": ["L-Thalamus", "R-Thalamus"],
    "limbic_proxy_amygdala": ["L-Amygdala", "R-Amygdala"],
    "cingulate_proxy_cortex": ["L-Cortex", "R-Cortex"],
    "ventricle_expansion": ["L-Lat-Ventricle", "R-Lat-Ventricle"],
}

CORRELATION_TARGETS = [
    ("clin_mmse", "MMSE", True),
    ("clin_cdrsb", "CDR-SB", False),
    ("clin_amy_centiloids", "Amyloid centiloids", False),
    ("clin_amy_suvr", "Amyloid SUVR", False),
    ("clin_csf_abeta42", "CSF Aβ42", True),
    ("clin_csf_tau", "CSF tau", False),
    ("clin_entorhinal_icv_ratio", "Entorhinal/ICV (clinical FS)", False),
    ("label", "CN/MCI/AD label", False),
]


def clean_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_dkt_stats(path: Path) -> Tuple[float, Dict[str, float]]:
    brain_seg_vol = math.nan
    volumes: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# Measure BrainSeg, BrainSegVol"):
            parts = line.split(",")
            if len(parts) >= 4:
                brain_seg_vol = clean_float(parts[3])
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            struct = parts[4].strip()
            vol = clean_float(parts[3])
        except (ValueError, IndexError):
            continue
        if struct and math.isfinite(vol):
            volumes[struct] = vol
    return brain_seg_vol, volumes


def aibl_fs_subject_id(scan_id: str) -> Optional[str]:
    match = AIBL_SCAN_RE.match(scan_id)
    if not match:
        return None
    return f"AIBL_{match.group(1)}_I{match.group(3)}"


def node_normalized_volumes(
    volumes: Dict[str, float],
    brain_seg_vol: float,
    nodes: Dict[str, List[str]],
    prefix: str = "vol",
) -> Dict[str, float]:
    if not math.isfinite(brain_seg_vol) or brain_seg_vol <= 0:
        return {name: math.nan for name in nodes}
    out: Dict[str, float] = {}
    for node, structs in nodes.items():
        vals = [volumes[s] for s in structs if s in volumes]
        if not vals:
            out[node] = math.nan
        else:
            out[node] = float(sum(vals) / brain_seg_vol)
    return out


def coarse21_volumes(row: dict, nodes: Dict[str, List[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for node, regions in nodes.items():
        vals = [clean_float(row.get(f"vol_{region}")) for region in regions]
        vals = [v for v in vals if math.isfinite(v)]
        out[node] = float(np.mean(vals)) if vals else math.nan
    return out


def cn_reference_stats(rows: Sequence[dict], node_names: Sequence[str], field_prefix: str) -> Dict[str, Tuple[float, float]]:
    cn_rows = [r for r in rows if int(r["label"]) == 0]
    ref: Dict[str, Tuple[float, float]] = {}
    for node in node_names:
        key = f"{field_prefix}_{node}"
        vals = np.array([clean_float(r.get(key)) for r in cn_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            ref[node] = (math.nan, math.nan)
        else:
            ref[node] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0)
    return ref


INVERTED_ATROPHY_NODES = {"ventricle_expansion"}


def atrophy_scores(row: dict, node_names: Sequence[str], field_prefix: str, ref: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    z_vals = []
    per_node = {}
    for node in node_names:
        key = f"{field_prefix}_{node}"
        val = clean_float(row.get(key))
        mu, sd = ref.get(node, (math.nan, math.nan))
        if not math.isfinite(val) or not math.isfinite(mu) or not math.isfinite(sd) or sd < 1e-12:
            per_node[node] = math.nan
            continue
        z = (val - mu) / sd if node in INVERTED_ATROPHY_NODES else (mu - val) / sd
        per_node[node] = float(z)
        z_vals.append(float(z))
    circuit = float(np.mean(z_vals)) if z_vals else math.nan
    per_node["papez_circuit_atrophy"] = circuit
    return per_node


def hippocampus_only_score(row: dict, field_prefix: str, ref: Dict[str, Tuple[float, float]]) -> float:
    node = "hippocampus"
    key = f"{field_prefix}_{node}"
    val = clean_float(row.get(key))
    mu, sd = ref.get(node, (math.nan, math.nan))
    if not math.isfinite(val) or not math.isfinite(mu) or not math.isfinite(sd) or sd < 1e-12:
        return math.nan
    return float((mu - val) / sd)


def spearman_pair(x: Sequence[float], y: Sequence[float]) -> dict:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if len(xv) < 8:
        return {"n": int(len(xv)), "rho": None, "p": None}
    rho, p = spearmanr(xv, yv)
    return {"n": int(len(xv)), "rho": float(rho), "p": float(p)}


def correlation_table(rows: Sequence[dict], score_key: str, cohort_name: str) -> List[dict]:
    out = []
    scores = [clean_float(r.get(score_key)) for r in rows]
    for col, label, positive_expected in CORRELATION_TARGETS:
        ys = []
        for r in rows:
            if col == "label":
                ys.append(float(int(r["label"])))
            else:
                ys.append(clean_float(r.get(col)))
        res = spearman_pair(scores, ys)
        out.append({
            "cohort": cohort_name,
            "score": score_key,
            "target": label,
            "target_col": col,
            "positive_expected": positive_expected,
            "n": res["n"],
            "rho": res["rho"],
            "p": res["p"],
            "abs_rho": abs(res["rho"]) if res["rho"] is not None else None,
        })
    return out


def enrich_dkt_rows(
    rows: Sequence[dict],
    dkt_root: Path,
    nodes: Dict[str, List[str]],
    field_prefix: str,
) -> Tuple[List[dict], dict]:
    enriched = []
    stats = Counter()
    for row in rows:
        copied = dict(row)
        fs_id = aibl_fs_subject_id(row["scan_id"]) if row["dataset"] == "AIBL" else None
        stats_path = None
        if fs_id:
            stats_path = dkt_root / fs_id / "stats" / "aseg+DKT.VINN.stats"
        if stats_path and stats_path.exists():
            brain_seg, vols = parse_dkt_stats(stats_path)
            node_vols = node_normalized_volumes(vols, brain_seg, nodes, prefix=field_prefix)
            for node, val in node_vols.items():
                copied[f"{field_prefix}_{node}"] = val
            copied["brain_seg_vol_mm3"] = brain_seg
            copied["dkt_source"] = str(stats_path)
            stats["dkt_hit"] += 1
        else:
            stats["dkt_miss"] += 1
            for node in nodes:
                copied[f"{field_prefix}_{node}"] = math.nan
        enriched.append(copied)
    return enriched, dict(stats)


def enrich_coarse21_rows(rows: Sequence[dict], nodes: Dict[str, List[str]], field_prefix: str) -> List[dict]:
    enriched = []
    for row in rows:
        copied = dict(row)
        node_vols = coarse21_volumes(row, nodes)
        for node, val in node_vols.items():
            copied[f"{field_prefix}_{node}"] = val
        enriched.append(copied)
    return enriched


def apply_scores(rows: Sequence[dict], node_names: Sequence[str], field_prefix: str, ref_rows: Sequence[dict]) -> List[dict]:
    ref = cn_reference_stats(ref_rows, node_names, field_prefix)
    out = []
    for row in rows:
        copied = dict(row)
        per_node = atrophy_scores(copied, node_names, field_prefix, ref)
        for k, v in per_node.items():
            copied[f"{field_prefix}_{k}" if k != "papez_circuit_atrophy" else f"{field_prefix}_papez_circuit_atrophy"] = v
        copied[f"{field_prefix}_hippocampus_only_atrophy"] = hippocampus_only_score(copied, field_prefix, ref)
        out.append(copied)
    return out


def split_groups(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    groups = {
        "all_labeled": [r for r in rows if r["split"] != "ixi_external"],
        "aibl_heldout": [r for r in rows if r["split"] == "aibl_heldout"],
        "aibl_all": [r for r in rows if r["dataset"] == "AIBL"],
        "adni_train_val_test": [r for r in rows if r["dataset"] == "ADNI" and r["split"] in {"train", "val", "internal_test"}],
    }
    return {k: v for k, v in groups.items() if v}


def summarize_group(rows: Sequence[dict], score_keys: Sequence[str]) -> dict:
    labels = np.array([int(r["label"]) for r in rows], dtype=int)
    summary = {"n": len(rows), "label_counts": dict(Counter(r["label_name"] for r in rows))}
    for key in score_keys:
        vals = np.array([clean_float(r.get(key)) for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        by_label = {}
        for lab, name in enumerate(["CN", "MCI", "AD"]):
            g = vals[labels == lab]
            g = g[np.isfinite(g)]
            by_label[name] = float(np.mean(g)) if len(g) else None
        summary[key] = {
            "mean": float(np.mean(finite)) if len(finite) else None,
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "by_label_mean": by_label,
        }
        if key.endswith("papez_circuit_atrophy") or key.endswith("hippocampus_only_atrophy"):
            groups = [vals[labels == i] for i in range(3)]
            groups = [g[np.isfinite(g)] for g in groups]
            if all(len(g) for g in groups):
                try:
                    _, kw_p = kruskal(*groups)
                except Exception:
                    kw_p = math.nan
            else:
                kw_p = math.nan
            rho, sp_p = spearmanr(labels, vals, nan_policy="omit")
            summary[key]["kruskal_p_label"] = float(kw_p) if math.isfinite(kw_p) else None
            summary[key]["spearman_rho_label"] = float(rho) if np.isfinite(rho) else None
            summary[key]["spearman_p_label"] = float(sp_p) if np.isfinite(sp_p) else None
    return summary


def write_markdown_report(payload: dict, path: Path) -> None:
    lines = [
        "# Papez Circuit Atrophy Analysis",
        "",
        "Circuit nodes (DKT): hippocampus, thalamus, entorhinal, parahippocampal, cingulate (rostral/caudal/posterior).",
        "Score: CN-referenced z-atrophy averaged across nodes (higher = more atrophy).",
        "",
        "## DKT coverage",
        "",
        f"- AIBL DKT hits: {payload['dkt_coverage'].get('dkt_hit', 0)}",
        f"- AIBL DKT misses: {payload['dkt_coverage'].get('dkt_miss', 0)}",
        "",
        "## Group summaries (DKT / AIBL)",
        "",
    ]
    for name, item in payload.get("group_summaries_dkt_aibl", {}).items():
        s = item.get("dkt_papez_circuit_atrophy", {})
        lines.append(f"### {name} (n={item['n']})")
        lines.append(f"- Papez circuit atrophy mean: {s.get('mean')}")
        lines.append(f"- Spearman vs label: rho={s.get('spearman_rho_label')}, p={s.get('spearman_p_label')}")
        lines.append("")
    lines += ["## Correlations (DKT Papez vs targets)", "", "| cohort | score | target | n | rho | p |", "|---|---|---|---:|---:|---:|"]
    for row in payload.get("correlations", []):
        if row.get("score") != "dkt_papez_circuit_atrophy":
            continue
        lines.append(
            f"| {row['cohort']} | {row['score']} | {row['target']} | {row['n']} | "
            f"{row['rho'] if row['rho'] is not None else 'NA'} | {row['p'] if row['p'] is not None else 'NA'} |"
        )
    lines += ["", "## Head-to-head |rho| on AIBL heldout (MMSE / CDR-SB / label)", ""]
    hh = payload.get("head_to_head_aibl_heldout", {})
    for metric, vals in hh.items():
        parts = ", ".join(f"{k}={v}" for k, v in sorted(vals.items(), key=lambda x: -(x[1] or -1)))
        lines.append(f"- **{metric}**: {parts}")
    lines += [
        "",
        "## Legacy AD-key consistency (21-region vol)",
        "",
    ]
    for name, item in payload.get("legacy_ad_key", {}).items():
        s = item.get("ad_key_volume_score", {})
        lines.append(
            f"- {name}: score={s.get('ad_key_score')}, null={s.get('uniform_null')}, p={s.get('permutation_p_greater')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("chapter1_foundation/outputs/v4/atlas_feature_cache_v4.csv"),
    )
    parser.add_argument(
        "--aibl-clinical",
        type=Path,
        default=Path("data/external/aibl/aibl_adnimergelike.csv"),
    )
    parser.add_argument(
        "--adni-clinical",
        type=Path,
        default=Path("chapter2_disentangle/data/master_subjects_v2.csv"),
    )
    parser.add_argument(
        "--dkt-root",
        type=Path,
        default=Path("data/external/aibl/fastsurfer_seg/fs_subjects"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("chapter1_foundation/ARA-Net/reports/v6_final_model/papez_circuit"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    feature_csv = args.feature_csv if args.feature_csv.is_absolute() else project_root / args.feature_csv
    aibl_clinical = args.aibl_clinical if args.aibl_clinical.is_absolute() else project_root / args.aibl_clinical
    adni_clinical = args.adni_clinical if args.adni_clinical.is_absolute() else project_root / args.adni_clinical
    dkt_root = args.dkt_root if args.dkt_root.is_absolute() else project_root / args.dkt_root
    out_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(feature_csv)
    for row in rows:
        row["label"] = int(row["label"])

    adni_index = build_adni_index(adni_clinical)
    aibl_index = build_aibl_index(aibl_clinical)
    rows, clinical_meta = enrich_rows(rows, adni_index, aibl_index)

    node_names_dkt = list(PAPEZ_CIRCUIT_NODES_DKT.keys())
    node_names_coarse = list(PAPEZ_CIRCUIT_NODES_COARSE21.keys())

    aibl_rows = [r for r in rows if r["dataset"] == "AIBL"]
    dkt_rows, dkt_cov = enrich_dkt_rows(aibl_rows, dkt_root, PAPEZ_CIRCUIT_NODES_DKT, "dkt")
    cn_ref_aibl = [r for r in dkt_rows if r["split"] in {"aibl_adapt_train", "aibl_adapt_val", "train", "val"} and int(r["label"]) == 0]
    if not cn_ref_aibl:
        cn_ref_aibl = [r for r in dkt_rows if int(r["label"]) == 0]
    dkt_scored = apply_scores(dkt_rows, node_names_dkt, "dkt", cn_ref_aibl)

    coarse_all = enrich_coarse21_rows(rows, PAPEZ_CIRCUIT_NODES_COARSE21, "coarse21")
    cn_ref_all = [r for r in coarse_all if r["split"] in {"train", "val", "aibl_adapt_train", "aibl_adapt_val"} and int(r["label"]) == 0]
    if not cn_ref_all:
        cn_ref_all = [r for r in coarse_all if int(r["label"]) == 0]
    coarse_scored = apply_scores(coarse_all, node_names_coarse, "coarse21", cn_ref_all)

    dkt_by_scan = {r["scan_id"]: r for r in dkt_scored}
    merged = []
    for r in coarse_scored:
        m = dict(r)
        if r["scan_id"] in dkt_by_scan:
            for k, v in dkt_by_scan[r["scan_id"]].items():
                if k.startswith("dkt_") or k in {"brain_seg_vol_mm3", "dkt_source"}:
                    m[k] = v
        merged.append(m)

    score_keys_dkt = ["dkt_papez_circuit_atrophy", "dkt_hippocampus_only_atrophy"]
    score_keys_coarse = ["coarse21_papez_circuit_atrophy", "coarse21_hippocampus_only_atrophy"]

    group_summaries_dkt = {
        name: summarize_group(subset, score_keys_dkt)
        for name, subset in split_groups(dkt_scored).items()
    }
    group_summaries_coarse = {
        name: summarize_group(subset, score_keys_coarse)
        for name, subset in split_groups(coarse_scored).items()
    }

    correlations: List[dict] = []
    for name, subset in split_groups(dkt_scored).items():
        correlations.extend(correlation_table(subset, "dkt_papez_circuit_atrophy", f"dkt_aibl/{name}"))
        correlations.extend(correlation_table(subset, "dkt_hippocampus_only_atrophy", f"dkt_aibl/{name}"))
    for name, subset in split_groups(coarse_scored).items():
        correlations.extend(correlation_table(subset, "coarse21_papez_circuit_atrophy", f"coarse21/{name}"))
        correlations.extend(correlation_table(subset, "coarse21_hippocampus_only_atrophy", f"coarse21/{name}"))

    heldout = [r for r in dkt_scored if r["split"] == "aibl_heldout"]
    head_to_head = {}
    for target_col, target_label, _ in CORRELATION_TARGETS:
        if target_col == "label":
            ys = [float(int(r["label"])) for r in heldout]
        else:
            ys = [clean_float(r.get(target_col)) for r in heldout]
        head_to_head[target_label] = {}
        for score_key in ["dkt_papez_circuit_atrophy", "dkt_hippocampus_only_atrophy"]:
            xs = [clean_float(r.get(score_key)) for r in heldout]
            res = spearman_pair(xs, ys)
            head_to_head[target_label][score_key] = abs(res["rho"]) if res["rho"] is not None else None

    legacy = {}
    for name, subset in split_groups(coarse_scored).items():
        legacy[name] = {
            "ad_key_volume_score": ad_key_score(subset, "vol", args.seed),
            "volume_gradients_hippocampus": region_gradient_tests(subset, "vol")["regions"].get("L-Hippocampus"),
        }

    payload = {
        "config": {
            "feature_csv": str(feature_csv),
            "aibl_clinical": str(aibl_clinical),
            "adni_clinical": str(adni_clinical),
            "dkt_root": str(dkt_root),
            "papez_nodes_dkt": PAPEZ_CIRCUIT_NODES_DKT,
            "papez_nodes_coarse21": PAPEZ_CIRCUIT_NODES_COARSE21,
            "note": "DKT Papez is primary for AIBL; coarse21 proxy covers all cohorts when DKT unavailable.",
        },
        "clinical_meta": clinical_meta,
        "dkt_coverage": dkt_cov,
        "group_summaries_dkt_aibl": group_summaries_dkt,
        "group_summaries_coarse21": group_summaries_coarse,
        "correlations": correlations,
        "head_to_head_aibl_heldout": head_to_head,
        "legacy_ad_key": legacy,
    }

    json_path = out_dir / "papez_circuit_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_dir / "papez_circuit_report.md")

    corr_csv = out_dir / "papez_correlations.csv"
    with corr_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlations[0].keys()) if correlations else [])
        if correlations:
            writer.writeheader()
            writer.writerows(correlations)

    print(f"[saved] {json_path}")
    print(f"[saved] {out_dir / 'papez_circuit_report.md'}")
    try:
        import matplotlib.pyplot as plt

        heldout_rows = [r for r in dkt_scored if r["split"] == "aibl_heldout"]
        scores = {
            "Papez circuit (DKT)": [clean_float(r.get("dkt_papez_circuit_atrophy")) for r in heldout_rows],
            "Hippocampus only (DKT)": [clean_float(r.get("dkt_hippocampus_only_atrophy")) for r in heldout_rows],
        }
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (target_col, target_label) in zip(axes, [
            ("clin_mmse", "MMSE"),
            ("clin_cdrsb", "CDR-SB"),
        ]):
            ys = [clean_float(r.get(target_col)) for r in heldout_rows]
            for name, xs in scores.items():
                xv = np.array(xs, dtype=float)
                yv = np.array(ys, dtype=float)
                mask = np.isfinite(xv) & np.isfinite(yv)
                if mask.sum() < 8:
                    continue
                rho, p = spearmanr(xv[mask], yv[mask])
                ax.scatter(xv[mask], yv[mask], s=10, alpha=0.35, label=f"{name} |rho|={abs(rho):.3f}")
            ax.set_xlabel("Atrophy z-score")
            ax.set_ylabel(target_label)
            ax.set_title(f"AIBL heldout: {target_label}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = out_dir / "papez_aibl_heldout_scatter.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        print(f"[saved] {fig_path}")
    except Exception as exc:
        print(f"[warn] figure skipped: {exc}")

    print("DKT coverage:", dkt_cov)
    if "aibl_heldout" in group_summaries_dkt:
        s = group_summaries_dkt["aibl_heldout"]["dkt_papez_circuit_atrophy"]
        print("AIBL heldout Papez circuit:", s)
    print("Head-to-head AIBL heldout:", head_to_head)


if __name__ == "__main__":
    main()
