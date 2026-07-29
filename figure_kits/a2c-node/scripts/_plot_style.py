"""Shared matplotlib styling for all paper figures.

Uses SciencePlots (https://github.com/garrettj403/SciencePlots) when
available, falling back to a minimal "science-ish" rcParams config when
not. Call :func:`apply` at the top of any plotting script.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Optional


def apply(extra: Optional[Iterable[str]] = None,
          fallback_no_latex: bool = True) -> None:
    """Apply the standard "science" + (optional) extras style.

    Parameters
    ----------
    extra : iterable of str, optional
        Additional style names to stack on top, e.g. ("nature",), ("ieee",),
        ("notebook",) or ("grid",).
    fallback_no_latex : bool
        If True (default), append "no-latex" to the style chain so that
        figures render even on hosts without a working LaTeX install
        (typical on remote training servers).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = ["science"]
    if extra:
        chosen.extend(extra)
    if fallback_no_latex:
        chosen.append("no-latex")

    try:
        import scienceplots                                            # noqa: F401
        plt.style.use(chosen)
    except Exception as e:                                             # pragma: no cover
        warnings.warn(f"[plot-style] SciencePlots unavailable, using "
                      f"minimal fallback: {e}")
        plt.rcParams.update({
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        })
