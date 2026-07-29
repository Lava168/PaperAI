# ARA-Net MedIA figure kit

Publication-style matplotlib figure scripts for **ARA-Net** (Chapter 1 / MedIA).

See also the full workspace archive: [`../MANIFEST.md`](../MANIFEST.md) (NeuroGate, PathwayPro, A2C-NODE, CUED-AD, shared).

## What is included

| Script | Figure |
|--------|--------|
| `scripts/generate_figure2_rcspe_atlas.py` | Fig 2 — RC-SPE atlas ensemble |
| `scripts/generate_figure3_atlas_directionality.py` | Fig 3 — atlas structural directionality |
| `scripts/generate_figure3_extended_pathology_laser.py` | Fig 3X — extended pathology + laser guide |
| `scripts/generate_figure3_rcspe_nbe_style.py` | Fig 3 alt — RC-SPE architecture (NBE style) |
| `scripts/generate_figure4_error_group_structure.py` | Fig 4 — error-group atlas structure |
| `scripts/generate_figure6_ensemble_comparison.py` | Fig 6 — ensemble comparison |
| `scripts/rebuild_pipeline_from_contrast.py` | Fig 1 — pipeline vector rebuild |
| `scripts/render_pipeline_flowchart_pdf.py` | Fig 1 — pipeline flowchart |
| `scripts/vectorize_pipeline_source.py` | Fig 1 — vectorization helper |
| `scripts/render_atlas_staging_triptych.py` | FastSurfer / 21-region atlas utils |
| `scripts/brain_figure_style.py` / `brain_figure_align.py` | Shared style / alignment |
| `scripts/render_brain_figures_*.py` | fcstyle / journal / phase3 brain suites |
| `scripts/generate_manuscript_figures.py` / `generate_v6_final_figures.py` | Batch manuscript figures |
| `mcp_brain_figures/server.py` | MCP brain-figure server |

## Data dependency

Expects the **ARA-Net project tree**. Run from live `ARA-Net/`, then sync scripts here.

## Style rules (MedIA)

- Times / Liberation Serif; white background; `pdf.fonttype=42`
- Laser = thin pathology-core parcel boundaries (Hipp / Amyg / Vent)
- Do **not** mix NeuroGate metrics into ARA-Net claims

## Cursor skill

[`.cursor/skills/aranet-media-figures/SKILL.md`](../../.cursor/skills/aranet-media-figures/SKILL.md)
