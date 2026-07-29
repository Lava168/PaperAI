---
name: aranet-media-figures
description: >-
  Regenerates and edits ARA-Net / MedIA-style publication figures (pipeline,
  RC-SPE ensemble, atlas directionality, laser pathology outlines, error-group
  structure). Use when the user asks for Fig1–Fig6, PaperAI figure kits,
  AD-key/laser outlines, FastSurfer atlas overlays, or ARA-Net MedIA figures.
---

# ARA-Net MedIA figures

## When to use

- Regenerating / editing ARA-Net paper figures (Fig 1–6 / 3X)
- Laser pathology outlines, AD-key vs extended pathology
- FastSurfer coronal montages + AD-like z fills
- Pushing figure scripts into the PaperAI repo kit

## Canonical locations

| Item | Path |
|------|------|
| PaperAI kit | `figure_kits/ara-net/` (this repo) |
| Live ARA-Net scripts | `chapter1_foundation/ARA-Net/scripts/` |
| Outputs | `ARA-Net/reports/v6_final_model/figures/` + sync `reports/brain_figures_fcstyle/figures/FigXX_*` |

Prefer editing the live ARA-Net scripts, then syncing copies into `figure_kits/ara-net/scripts/` before committing to PaperAI.

## Script map

| Goal | Run |
|------|-----|
| Fig 1 pipeline (vector HQ) | `rebuild_pipeline_from_contrast.py` |
| Fig 2 RC-SPE atlas | `generate_figure2_rcspe_atlas.py` |
| Fig 3 directionality | `generate_figure3_atlas_directionality.py` |
| Fig 3X laser guide | `generate_figure3_extended_pathology_laser.py` |
| Fig 4 error groups | `generate_figure4_error_group_structure.py` |
| Fig 6 ensemble | `generate_figure6_ensemble_comparison.py` |
| Shared atlas I/O | `render_atlas_staging_triptych.py` |

Always `cd` into `ARA-Net` and run `python3 scripts/<name>.py`.

## Style rules (do not violate)

1. **Fonts**: Times New Roman / Liberation Serif; `pdf.fonttype=42`
2. **Background**: pure white for paper panels; black only inside MRI tiles
3. **Laser ≠ landmarks**: neon contours from `find_contours` on aseg masks
4. **Laser only pathology core**: Hipp (17/53), Amyg (18/54), Vent (4/43); thin (`lw≈0.55`); no fat glow
5. **Secondary pathology**: Accumbens (26/58), Cortex (3/42) — fill OK, usually **no** laser
6. **Fig 4 AD-like fill**: shared YlOrRd from subject `atlas_ad_like_z`; region ID via colored outlines
7. **Claims**: ARA-Net metrics only — never mix NeuroGate numbers into captions

## Edit workflow

1. Identify figure ID (1 / 2 / 3 / 3X / 4 / 6)
2. Open matching script under `ARA-Net/scripts/`
3. Change ROI sets / layout / captions as requested
4. Regenerate PNG+PDF; sync to `brain_figures_fcstyle/figures/FigXX_*`
5. Copy updated scripts into `PaperAI/figure_kits/ara-net/scripts/`
6. Commit on a feature branch; push to `Lava168/PaperAI`

## Pathology ROI quick reference

```text
Core (laser):     4,43 vent | 17,53 hipp | 18,54 amyg
Secondary (fill): 26,58 accumbens | 3,42 cortex
```

## More detail

- Kit README: [figure_kits/ara-net/README.md](../../../figure_kits/ara-net/README.md)
- Conventions: [reference.md](reference.md)
