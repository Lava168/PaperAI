# ARA-Net MedIA figure conventions

## Narrative per figure

| Figure | Story |
|--------|--------|
| Fig 1 | End-to-end ARA-Net pipeline (a–g); prefer vector rebuild over upscaled screenshots |
| Fig 2 | RC-SPE probability streams + atlas-guided brain tiles |
| Fig 3 | Structural directionality: 21-atlas vs extended pathology; multi-region trends; Cohen’s d |
| Fig 3X | Explains laser outlines; core vs extended pathology; ranked AD−CN Δ |
| Fig 4 | Real error-group cases: CN✓ / AD✓ / AD→MCI / MCI→AD; AD-like z fill; e1/e2 heatmaps |
| Fig 6 | Ensemble / comparator performance |

## Laser outlines

Implementation pattern:

```python
from skimage import measure
for contour in measure.find_contours(mask.astype(float), 0.5):
    ax.plot(contour[:, 1], contour[:, 0], color="white", lw=lw + 0.55, alpha=0.35)
    ax.plot(contour[:, 1], contour[:, 0], color=color, lw=lw, alpha=0.95)
```

Meaning for captions: *thin laser = pathology-core parcel boundary, not landmark points.*

## AD-like z color (Fig 4)

- Scale typically `ZMIN, ZMAX = -0.8, 2.3`
- Fill pathology parcels with `z_to_rgb(z)` matching side chip + e2 col1
- Outlines identify parcels when fill color is shared

## Data tables commonly used

- `reports/v6_final_model/tables/final_subject_predictions_enriched.csv`
- `reports/v6_final_model/tables/aibl_heldout_error_group_features.csv`
- `reports/brain_figures_fcstyle/assets/matrices/methods_roi_matrix.csv`
- `reports/brain_figures_fcstyle/assets/matrices/roi_vectors.json`
- FastSurfer under workspace `data/external/aibl/fastsurfer_seg/fs_subjects/`

## Sync checklist after regenerate

```bash
# from ARA-Net
python3 scripts/generate_figureN_....py
# scripts usually copy to reports/brain_figures_fcstyle/figures/FigNN_*
cp scripts/*.py /path/to/PaperAI/figure_kits/ara-net/scripts/
```
