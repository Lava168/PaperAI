# ARA-Net MedIA figure kit

Publication-style matplotlib figure scripts distilled from the ARA-Net / Alzheimer’s Disease Dynamics figure work (MedIA-oriented).

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
| `scripts/render_pipeline_flowchart_pdf.py` | Fig 1 — pipeline flowchart (matplotlib) |
| `scripts/vectorize_pipeline_source.py` | Fig 1 — source vectorization helper |
| `scripts/render_atlas_staging_triptych.py` | Shared FastSurfer / 21-region atlas utils |
| `scripts/brain_figure_style.py` | Shared style helpers |
| `scripts/brain_figure_align.py` | Shared alignment helpers |

## Data dependency

These scripts expect the **ARA-Net project tree** (FastSurfer cases, enriched CSV, ROI matrices). They resolve `ROOT` as the parent of `scripts/` when run inside ARA-Net.

To use from this kit without rewriting every path:

```bash
export ARA_NET_ROOT="/path/to/ARA-Net"
# either symlink scripts into ARA-Net/scripts, or run from ARA-Net after copying
cd "$ARA_NET_ROOT"
python3 scripts/generate_figure3_atlas_directionality.py
```

Recommended: keep this kit as the **canonical copy** in PaperAI, and sync/copy into the ARA-Net `scripts/` directory when regenerating paper figures.

## Style rules (MedIA)

- Times / Liberation Serif; white background; `pdf.fonttype=42`
- **Laser outlines** = thin neon **parcel boundaries** (not landmark points)
- Laser only on pathology core: Hippocampus / Amygdala / Lat. ventricle
- Secondary pathology (Accumbens, Cortex): fill allowed; usually no laser
- Do **not** mix NeuroGate metrics into ARA-Net figure claims

## Cursor skill

See [`.cursor/skills/aranet-media-figures/SKILL.md`](../../.cursor/skills/aranet-media-figures/SKILL.md).
