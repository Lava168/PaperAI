#!/usr/bin/env python3
"""Build the per-subject brain-structural profile table for Figure 5(g).

For every AIBL subject in the 5-seed external-validation ensemble
(``runs/aibl_eval/aibl_ensemble.npz``) this script reconstructs the
21-region FreeSurfer/FastSurfer volumes from the cached segmentation
(``/home/lry/aibl/cache_real/AIBL_<RID>_<vc>_I<id>.npz``) and joins them to
the model prediction, decision confidence/margin and clinical severity
(MMSE / global-CDR / APOE4 / age) so that all of Figure-5(f)+(g) can be drawn
from a single table.

Key derived quantities (one row per subject):

    hippo_vol_norm   (L+R hippocampus) / ICV-proxy
    vent_vol_norm    (L+R lateral ventricle) / ICV-proxy
    cortex_vol_norm  (L+R cerebral cortex) / ICV-proxy
    <region>_z       z-score of each normalised region volume vs the
                     *true-CN* group (mean/SD of correctly-defined CN cohort)
    ad_like_z        atlas-guided AD-likeness: sign-corrected, |ATE_AD|-weighted
                     mean of region z-scores (higher = more AD-like:
                     ventricular expansion + global atrophy)

The visit used per subject matches ``predict_to_npz`` (latest valid visit).

Usage
-----
    .venv/bin/python scripts/build_fig5_brain_profile.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from a2c_node.data.legacy_cache_dataset import (  # noqa: E402
    _Record,
    parse_aranet_filename,
    remap_freesurfer_labels,
)

# Remapped (contiguous, 1-based) region order — matches
# trajectory_atrophy.npz["region_names"] and ate_21regions.csv["region_idx"]+1.
REGION_NAMES = [
    "Left-Cerebral-WM", "Left-Cerebral-Cortex", "Left-Lateral-Ventricle",
    "Left-Thalamus", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Brain-Stem", "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens",
    "Right-Cerebral-WM", "Right-Cerebral-Cortex", "Right-Lateral-Ventricle",
    "Right-Thalamus", "Right-Caudate", "Right-Putamen", "Right-Pallidum",
    "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens",
]
NREG = len(REGION_NAMES)  # 21
# remapped labels are 1..21 (label k -> REGION_NAMES[k-1])
LBL_HIPPO = (9, 19)        # Left/Right-Hippocampus
LBL_VENT = (3, 14)         # Left/Right-Lateral-Ventricle
LBL_CORTEX = (2, 13)       # Left/Right-Cerebral-Cortex
CLASSES = ["CN", "MCI", "AD"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("/home/lry/aibl/cache_real"))
    p.add_argument("--ensemble", type=Path,
                   default=PROJ_ROOT / "runs/aibl_eval/aibl_ensemble.npz")
    p.add_argument("--preds_seed", type=Path,
                   default=PROJ_ROOT / "runs/aibl_eval/preds_seed42.npz",
                   help="any per-seed npz carrying age/sex/apoe4/mmse")
    p.add_argument("--clinical_csv", type=Path,
                   default=Path("/home/lry/aibl/aibl_adnimergelike.csv"))
    p.add_argument("--ate_csv", type=Path,
                   default=PROJ_ROOT / "runs/v4_mm_analysis_npj/ate_21regions.csv")
    p.add_argument("--min_visits", type=int, default=1)
    p.add_argument("--out_csv", type=Path,
                   default=PROJ_ROOT / "runs/aibl_eval/fig5_brain_profile.csv")
    p.add_argument("--out_npz", type=Path,
                   default=PROJ_ROOT / "runs/aibl_eval/fig5_brain_profile.npz")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def build_subject_visits(cache: Path, min_visits: int) -> Dict[str, _Record]:
    """Replicate LegacyCacheRealDataset grouping (parse + dedup + sort)."""
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {cache}")
    records: Dict[str, _Record] = {}
    for fp in files:
        parsed = parse_aranet_filename(fp.name)
        if parsed is None:
            continue
        ptid, months, img_id = parsed
        if months == -1.0:               # skip screening (keep_screening=False)
            continue
        rec = records.get(ptid) or _Record(ptid)
        rec.add(months, img_id, fp)
        records[ptid] = rec
    for r in records.values():
        r.finalise(dedup=True)
    return {pt: r for pt, r in records.items() if len(r.visits) >= min_visits}


def region_volumes_from_seg(seg_path: Path) -> np.ndarray:
    """Voxel count per remapped region 1..21 for the given cache npz."""
    with np.load(seg_path, allow_pickle=True) as z:
        seg = remap_freesurfer_labels(np.asarray(z["seg"]))
    vols = np.zeros(NREG, dtype=np.float64)
    flat = seg.reshape(-1)
    counts = np.bincount(flat, minlength=NREG + 1)
    vols[:] = counts[1:NREG + 1]          # drop background (label 0)
    return vols


def load_clinical_bl(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """RID -> {cdrsb, mmse, apoe4, age} using the baseline ('bl') visit when
    available, else the earliest visit row."""
    by_rid: Dict[str, Dict[str, dict]] = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = str(r["RID"]).strip()
            vc = (r.get("VISCODE") or "bl").strip() or "bl"
            by_rid[rid][vc] = r

    def _f(s: str) -> float:
        s = (s or "").strip()
        try:
            return float(s)
        except Exception:
            return float("nan")

    out: Dict[str, Dict[str, float]] = {}
    for rid, visits in by_rid.items():
        row = visits.get("bl") or next(iter(visits.values()))
        out[rid] = {
            "cdrsb": _f(row.get("CDRSB", "")),
            "mmse": _f(row.get("MMSE", "")),
            "apoe4": _f(row.get("APOE4", "")),
            "age": _f(row.get("AGE", "")),
        }
    return out


def load_ate_weights(ate_csv: Path) -> np.ndarray:
    """|ATE_AD| per region in REGION_NAMES order (normalised to sum 1)."""
    w = np.zeros(NREG, dtype=np.float64)
    name2idx = {n: i for i, n in enumerate(REGION_NAMES)}
    with open(ate_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nm = r["region_name"].strip()
            if nm in name2idx:
                w[name2idx[nm]] = abs(float(r["ate_AD"]))
    if w.sum() <= 0:
        w[:] = 1.0
    return w / w.sum()


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    ens = np.load(args.ensemble, allow_pickle=True)
    prob = ens["prob"].astype(np.float64)            # (N,3)
    label = ens["label"].astype(int)                 # (N,)
    ptid = np.asarray([str(p) for p in ens["ptid"]])
    N = len(ptid)
    pred = prob.argmax(-1)

    # clinical vars carried by the per-seed prediction npz (already aligned by
    # ptid order); we re-map defensively by ptid.
    seed = np.load(args.preds_seed, allow_pickle=True)
    seed_ptid = np.asarray([str(p) for p in seed["ptid"]])
    seed_lut = {p: i for i, p in enumerate(seed_ptid)}
    age = np.full(N, np.nan); sex = np.full(N, np.nan)
    apoe4 = np.full(N, np.nan); mmse = np.full(N, np.nan)
    for i, p in enumerate(ptid):
        j = seed_lut.get(p)
        if j is not None:
            age[i] = float(seed["age"][j]); sex[i] = float(seed["sex"][j])
            apoe4[i] = float(seed["apoe4"][j]); mmse[i] = float(seed["mmse"][j])

    clin = load_clinical_bl(args.clinical_csv)
    cdrsb = np.array([clin.get(p, {}).get("cdrsb", np.nan) for p in ptid])
    # backfill mmse/apoe4 from clinical csv if missing in npz
    for i, p in enumerate(ptid):
        c = clin.get(p, {})
        if np.isnan(mmse[i]):
            mmse[i] = c.get("mmse", np.nan)
        if np.isnan(apoe4[i]):
            apoe4[i] = c.get("apoe4", np.nan)
        if np.isnan(age[i]):
            age[i] = c.get("age", np.nan)

    # --- regional volumes from seg (latest visit per subject) --------------- #
    recs = build_subject_visits(args.cache, args.min_visits)
    print(f"[fig5] cache subjects (min_visits={args.min_visits}): {len(recs)}; "
          f"ensemble subjects: {N}", flush=True)

    vols = np.full((N, NREG), np.nan, dtype=np.float64)
    viscode_used: List[str] = ["?"] * N
    n_visits = np.zeros(N, dtype=int)
    missing = []
    for i, p in enumerate(ptid):
        rec = recs.get(p)
        if rec is None:
            missing.append(p); continue
        t_last, _img_id, fp_last = rec.visits[-1]       # latest valid visit
        vols[i] = region_volumes_from_seg(fp_last)
        n_visits[i] = len(rec.visits)
        m = int(round(t_last))
        viscode_used[i] = "bl" if m == 0 else f"m{m:02d}"
    if missing:
        print(f"[fig5] WARNING: {len(missing)} ptids not found in cache "
              f"(e.g. {missing[:5]})", flush=True)

    icv = np.nansum(vols, axis=1)                         # ICV proxy
    icv_safe = np.where(icv > 0, icv, np.nan)
    vol_norm = vols / icv_safe[:, None]                  # fraction of ICV

    hippo = vols[:, [LBL_HIPPO[0] - 1, LBL_HIPPO[1] - 1]].sum(1)
    vent = vols[:, [LBL_VENT[0] - 1, LBL_VENT[1] - 1]].sum(1)
    cortex = vols[:, [LBL_CORTEX[0] - 1, LBL_CORTEX[1] - 1]].sum(1)
    hippo_norm = hippo / icv_safe
    vent_norm = vent / icv_safe
    cortex_norm = cortex / icv_safe

    # --- z-score vs true-CN group ------------------------------------------ #
    cn = (label == 0) & np.isfinite(icv_safe)
    mu = np.nanmean(vol_norm[cn], axis=0)
    sd = np.nanstd(vol_norm[cn], axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = (vol_norm - mu[None, :]) / sd[None, :]           # (N,21)

    def _z1(arr_norm: np.ndarray) -> np.ndarray:
        m = np.nanmean(arr_norm[cn]); s = np.nanstd(arr_norm[cn])
        s = s if s > 1e-12 else 1.0
        return (arr_norm - m) / s

    hippo_z = _z1(hippo_norm)
    vent_z = _z1(vent_norm)
    cortex_z = _z1(cortex_norm)

    # AD-like atlas z-score: sign-corrected (ventricles +, atrophy -),
    # weighted by |ATE_AD| over the 21 atlas regions. Higher = more AD-like.
    w = load_ate_weights(args.ate_csv)
    sign = np.full(NREG, -1.0)
    sign[LBL_VENT[0] - 1] = 1.0
    sign[LBL_VENT[1] - 1] = 1.0
    ad_like_z = np.nansum(sign[None, :] * z * w[None, :], axis=1)
    # rescale to unit SD over true-CN for interpretability
    s_ad = np.nanstd(ad_like_z[cn]); s_ad = s_ad if s_ad > 1e-12 else 1.0
    ad_like_z = (ad_like_z - np.nanmean(ad_like_z[cn])) / s_ad

    # --- prediction confidence / margin ------------------------------------ #
    conf = prob.max(1)
    sp = np.sort(prob, axis=1)
    margin = sp[:, -1] - sp[:, -2]
    correct = (pred == label)

    # --- error-structure group label --------------------------------------- #
    def grp(t: int, pr: int) -> str:
        if t == pr:
            return f"correct_{CLASSES[t]}"
        return f"{CLASSES[t]}_to_{CLASSES[pr]}"
    group = np.array([grp(int(t), int(p)) for t, p in zip(label, pred)])

    # --- write CSV ---------------------------------------------------------- #
    base_cols = [
        "ptid", "true", "pred", "correct", "group",
        "p_CN", "p_MCI", "p_AD", "confidence", "margin",
        "age", "sex", "apoe4", "mmse", "cdrsb",
        "n_visits", "viscode_used", "icv_vox",
        "hippo_vol_vox", "hippo_vol_norm", "hippo_z",
        "vent_vol_vox", "vent_vol_norm", "vent_z",
        "cortex_vol_vox", "cortex_vol_norm", "cortex_z",
        "ad_like_z",
    ]
    reg_cols = ([f"{n}_vox" for n in REGION_NAMES]
                + [f"{n}_z" for n in REGION_NAMES])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(base_cols + reg_cols)
        for i in range(N):
            row = [
                ptid[i], CLASSES[label[i]], CLASSES[pred[i]], int(correct[i]),
                group[i],
                f"{prob[i,0]:.4f}", f"{prob[i,1]:.4f}", f"{prob[i,2]:.4f}",
                f"{conf[i]:.4f}", f"{margin[i]:.4f}",
                f"{age[i]:.1f}", f"{sex[i]:.0f}", f"{apoe4[i]:.0f}",
                f"{mmse[i]:.0f}", f"{cdrsb[i]:.2f}",
                int(n_visits[i]), viscode_used[i], f"{icv[i]:.0f}",
                f"{hippo[i]:.0f}", f"{hippo_norm[i]:.6f}", f"{hippo_z[i]:.3f}",
                f"{vent[i]:.0f}", f"{vent_norm[i]:.6f}", f"{vent_z[i]:.3f}",
                f"{cortex[i]:.0f}", f"{cortex_norm[i]:.6f}", f"{cortex_z[i]:.3f}",
                f"{ad_like_z[i]:.3f}",
            ]
            row += [f"{vols[i,k]:.0f}" for k in range(NREG)]
            row += [f"{z[i,k]:.3f}" for k in range(NREG)]
            wr.writerow(row)

    np.savez(
        args.out_npz,
        ptid=ptid, true=label, pred=pred, prob=prob.astype(np.float32),
        confidence=conf.astype(np.float32), margin=margin.astype(np.float32),
        group=group, age=age.astype(np.float32), sex=sex.astype(np.float32),
        apoe4=apoe4.astype(np.float32), mmse=mmse.astype(np.float32),
        cdrsb=cdrsb.astype(np.float32),
        vols_vox=vols.astype(np.float32), vol_norm=vol_norm.astype(np.float32),
        region_z=z.astype(np.float32), region_names=np.asarray(REGION_NAMES),
        icv_vox=icv.astype(np.float32),
        hippo_norm=hippo_norm.astype(np.float32), hippo_z=hippo_z.astype(np.float32),
        vent_norm=vent_norm.astype(np.float32), vent_z=vent_z.astype(np.float32),
        cortex_norm=cortex_norm.astype(np.float32), cortex_z=cortex_z.astype(np.float32),
        ad_like_z=ad_like_z.astype(np.float32),
        n_visits=n_visits, viscode_used=np.asarray(viscode_used),
    )

    # --- console summary: the four Figure-5(g) groups ---------------------- #
    print(f"[fig5] wrote {args.out_csv}")
    print(f"[fig5] wrote {args.out_npz}")
    focus = ["correct_MCI", "MCI_to_AD", "correct_AD", "AD_to_MCI",
             "MCI_to_CN", "AD_to_CN", "correct_CN"]
    hdr = (f"{'group':<14}{'n':>4}{'hippo_z':>9}{'vent_z':>9}"
           f"{'cortex_z':>10}{'ad_like_z':>11}{'mmse':>7}{'cdrsb':>7}"
           f"{'conf':>7}{'margin':>8}")
    print("\n[fig5] structural / clinical profile by group")
    print(hdr)
    print("-" * len(hdr))
    for g in focus:
        m = (group == g)
        if not m.any():
            continue

        def _mn(a):
            return float(np.nanmean(a[m])) if m.any() else float("nan")
        print(f"{g:<14}{int(m.sum()):>4}{_mn(hippo_z):>9.2f}"
              f"{_mn(vent_z):>9.2f}{_mn(cortex_z):>10.2f}"
              f"{_mn(ad_like_z):>11.2f}{_mn(mmse):>7.1f}{_mn(cdrsb):>7.2f}"
              f"{_mn(conf):>7.2f}{_mn(margin):>8.2f}")
    # AD->CN sanity (the headline of Figure 5)
    print(f"\n[fig5] AD->CN count = {int(((label==2)&(pred==0)).sum())}  "
          f"(should be 0 for the disease-boundary narrative)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
