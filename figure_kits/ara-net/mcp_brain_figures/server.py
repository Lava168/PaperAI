#!/usr/bin/env python3
"""ARA-Net Brain Figures MCP server (fcHMRF-style neuroimaging panels).

Exposes tools to list map assets and render glass-brain / montage / suite figures
using the locked BRAIN_FIGURE_SPEC and nilearn overlays.

Run (stdio):
  python chapter1_foundation/ARA-Net/mcp_brain_figures/server.py

Cursor MCP config example is in README.md / project .cursor/mcp.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve repo roots
SERVER_DIR = Path(__file__).resolve().parent
ARA_ROOT = SERVER_DIR.parent
SCRIPTS = ARA_ROOT / "scripts"
ASSETS = ARA_ROOT / "reports/brain_figures_fcstyle/assets"
MAPS = ASSETS / "maps"
FIG_DIR = ARA_ROOT / "reports/brain_figures_fcstyle/figures"
SPEC = ARA_ROOT / "docs/BRAIN_FIGURE_SPEC.md"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ara-brain-figures",
    instructions=(
        "Render fcHMRF-comparable atlas brain figures for ARA-Net "
        "(glass brain, axial/coronal/sagittal montages, locked map suite). "
        "Outputs PNG/PDF under reports/brain_figures_fcstyle/figures/. "
        "Do NOT claim voxel FDR, Braak staging, or clinical attention biomarkers."
    ),
)


CUTS = {
    "axial": (-28, -12, 0, 12, 28, 40),
    "coronal": (-52, -28, -8, 12, 32),
    "sagittal": (-36, -18, 0, 18, 36),
}

DEFAULT_CLIMS = {
    "atrophy": (-2.5, 2.5, "cold_hot"),
    "mrf_q": (0.0, 1.0, "plasma"),
    "assoc": (-0.6, 0.6, "cold_hot"),
}


def _guess_clim(name: str) -> tuple:
    n = name.lower()
    if "mrf_q" in n and "minus" not in n and "assoc" not in n:
        return DEFAULT_CLIMS["mrf_q"]
    if "assoc" in n or "minus" in n:
        return DEFAULT_CLIMS["assoc"] if "mrf_q" in n or "assoc" in n else DEFAULT_CLIMS["atrophy"]
    if "mrf_q" in n:
        return (-0.6, 0.6, "cold_hot")
    return DEFAULT_CLIMS["atrophy"]


def _ensure_libs():
    try:
        import nibabel  # noqa: F401
        import nilearn  # noqa: F401
        import matplotlib  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Missing neuroimaging deps (nibabel/nilearn/matplotlib): {exc}"
        ) from exc


@mcp.tool()
def brain_list_maps() -> str:
    """List available NIfTI map assets for fcHMRF-style brain figures."""
    if not MAPS.exists():
        return json.dumps(
            {
                "ok": False,
                "error": f"Maps directory missing: {MAPS}. Run build_brain_figure_assets.py first.",
            },
            indent=2,
        )
    maps = sorted(p.name for p in MAPS.glob("*.nii.gz"))
    return json.dumps(
        {
            "ok": True,
            "maps_dir": str(MAPS),
            "n_maps": len(maps),
            "maps": maps,
            "spec": str(SPEC) if SPEC.exists() else None,
            "claim_boundary": (
                "Atlas-filled / association maps only. Not voxel FDR, not Braak, "
                "not subject-specific clinical attention biomarkers."
            ),
        },
        indent=2,
    )


@mcp.tool()
def brain_get_spec() -> str:
    """Return the locked BRAIN_FIGURE_SPEC markdown (cuts, colormaps, allowed claims)."""
    if not SPEC.exists():
        return json.dumps({"ok": False, "error": f"Missing {SPEC}"})
    return SPEC.read_text(encoding="utf-8")


@mcp.tool()
def brain_render_glassbrain(
    map_name: str = "assoc_v6_rcspe",
    out_name: str = "mcp_glassbrain",
    display_mode: str = "lyrz",
    threshold: float = 0.05,
) -> str:
    """Render a nilearn glass-brain for one map asset (fcHMRF-comparable summary).

    Args:
        map_name: Stem of maps/*.nii.gz (e.g. atrophy_AD_minus_CN, assoc_v6_rcspe).
        out_name: Output basename under figures/ (without extension).
        display_mode: nilearn glass-brain display_mode (e.g. lyrz, ortho, xz).
        threshold: Absolute threshold for display.
    """
    _ensure_libs()
    import matplotlib

    matplotlib.use("Agg")
    import nibabel as nib
    from nilearn import plotting

    from brain_figure_align import plot_native_ortho_glass

    nii = MAPS / f"{map_name}.nii.gz"
    if not nii.exists():
        # allow bare name with extension
        nii = MAPS / map_name
    if not nii.exists():
        return json.dumps({"ok": False, "error": f"Map not found: {map_name}", "hint": "Call brain_list_maps"})

    vmin, vmax, cmap = _guess_clim(map_name)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / f"{out_name}.png"
    out_pdf = FIG_DIR / f"{out_name}.pdf"

    # Native ortho — do NOT use plot_glass_brain (fake MNI affine vs true silhouette).
    display = plot_native_ortho_glass(
        nib.load(str(nii)),
        title=map_name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        colorbar=True,
    )
    display.savefig(str(out_png), dpi=300)
    display.savefig(str(out_pdf))
    display.close()

    return json.dumps(
        {
            "ok": True,
            "map": str(nii),
            "vmin": vmin,
            "vmax": vmax,
            "cmap": cmap,
            "png": str(out_png),
            "pdf": str(out_pdf),
            "note": "Native-space ortho (not MNI glass): crop affine is approximate MNI-like only.",
            "claim_boundary": "Supporting spatial map only; not voxel-wise FDR discovery.",
        },
        indent=2,
    )


@mcp.tool()
def brain_render_ortho_montage(
    map_name: str = "atrophy_AD_minus_CN",
    out_name: str = "mcp_ortho_montage",
    plane: str = "axial",
    underlay: str = "atlas_template",
) -> str:
    """Render multi-slice ortho montage (axial/coronal/sagittal) like fcHMRF paper panels.

    Args:
        map_name: maps/*.nii.gz stem.
        out_name: output basename.
        plane: axial | coronal | sagittal
        underlay: atlas_template (from assets/) or empty for overlay-only.
    """
    _ensure_libs()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nilearn import plotting

    if plane not in CUTS:
        return json.dumps({"ok": False, "error": f"plane must be one of {list(CUTS)}"})

    nii = MAPS / f"{map_name}.nii.gz"
    if not nii.exists():
        return json.dumps({"ok": False, "error": f"Map not found: {map_name}"})

    bg = ASSETS / "atlas_template.nii.gz"
    vmin, vmax, cmap = _guess_clim(map_name)
    cuts = CUTS[plane]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / f"{out_name}.png"

    n = len(cuts)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4))
    if n == 1:
        axes = [axes]
    img = nib.load(str(nii))
    # Never use MNI/anat underlay with approximate crop affine (white-shadow look).
    bg_img = None
    cut_coords_key = {"axial": "z", "coronal": "y", "sagittal": "x"}[plane]

    for ax, cut in zip(axes, cuts):
        kwargs: Dict[str, Any] = dict(
            colorbar=False,
            display_mode=cut_coords_key,
            cut_coords=[cut],
            axes=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annotate=False,
            draw_cross=False,
            black_bg=True,
            bg_img=None,
        )
        plotting.plot_stat_map(img, threshold=0.0, **kwargs)
        ax.set_title(f"{cut_coords_key}={cut}", fontsize=8)

    fig.suptitle(f"{map_name} ({plane})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return json.dumps(
        {
            "ok": True,
            "png": str(out_png),
            "plane": plane,
            "cuts": list(cuts),
            "vmin": vmin,
            "vmax": vmax,
            "cmap": cmap,
        },
        indent=2,
    )


@mcp.tool()
def brain_render_locked_suite(phase: str = "fcstyle") -> str:
    """Run the locked paper brain-figure suite (B1–B4 or phase3 extras).

    Args:
        phase: 'fcstyle' runs render_brain_figures_fcstyle.py;
               'phase3' runs render_brain_figures_phase3.py (DKT/Papez/cases).
    """
    import subprocess

    if phase == "fcstyle":
        script = SCRIPTS / "render_brain_figures_fcstyle.py"
    elif phase == "phase3":
        script = SCRIPTS / "render_brain_figures_phase3.py"
    else:
        return json.dumps({"ok": False, "error": "phase must be fcstyle|phase3"})

    if not script.exists():
        return json.dumps({"ok": False, "error": f"Missing script {script}"})

    # Prefer running from workspace root that contains chapter1_foundation
    cwd = ARA_ROOT
    # scripts often assume cwd is ARA-Net or repo root; try ARA_ROOT
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
    )
    figs = sorted(str(p) for p in FIG_DIR.glob("figure_B*.png")) if FIG_DIR.exists() else []
    return json.dumps(
        {
            "ok": proc.returncode == 0,
            "phase": phase,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "figures": figs,
        },
        indent=2,
    )


@mcp.tool()
def brain_build_assets() -> str:
    """Rebuild NIfTI map assets from atlas features (build_brain_figure_assets.py)."""
    import subprocess

    script = SCRIPTS / "build_brain_figure_assets.py"
    if not script.exists():
        return json.dumps({"ok": False, "error": f"Missing {script}"})
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ARA_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    maps = sorted(p.name for p in MAPS.glob("*.nii.gz")) if MAPS.exists() else []
    return json.dumps(
        {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "n_maps": len(maps),
            "maps": maps,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
