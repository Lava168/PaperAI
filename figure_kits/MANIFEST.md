# Paper figure kits (full workspace inventory)

Canonical backup of publication / demo figure scripts from the Alzheimer’s Disease Dynamics thesis workspace, packaged for [PaperAI](https://github.com/Lava168/PaperAI).

Scripts are **copies** of live project code. Most still resolve data paths relative to their original chapter trees — regenerate figures from the source project, then sync updated scripts here.

## Layout

```text
figure_kits/
  MANIFEST.md                 ← this file
  ara-net/                    ← Chapter 1 ARA-Net / MedIA
  neurogate/                  ← Chapter 1 NeuroGate (do not mix metrics into ARA-Net)
  pathwaypro/                 ← Chapter 2 PathwayPro / disentangle
  a2c-node/                   ← Chapter 3 A2C-NODE
  cued-ad/                    ← Chapter 4 CUED-AD generalization
  shared/                     ← workspace-level visualization + PPT builders
```

## Counts (≈95 `.py` files)

| Kit | Contents |
|-----|----------|
| **ara-net** | Fig 1–6 generators, brain fcstyle suite, pipeline vector rebuild, atlas triptych, manuscript embed |
| **neurogate** | NeuroImage redraw, case comparison, brain-extract method, foundation viz |
| **pathwaypro** | Main-text fig1–6, Grad-CAM / 3D ROI renders, Nature viz utils |
| **a2c-node** | Paper plot suite (`plot_*`, `compose_paper_figures`), Grad-CAM / NBE render scripts |
| **cued-ad** | Manuscript packaging assets, OOD viz, real-data demos |
| **shared** | `utils/visualization.py`, slide/PPT builders |

## Cursor skill

`.cursor/skills/aranet-media-figures/` — MedIA / ARA-Net conventions (laser outlines, pathology ROIs).  
For other chapters, treat this tree as the **script archive**; chapter-specific skills can be added later.

## Sync rule

1. Edit & run scripts in the live chapter project.
2. Copy changed `.py` files into the matching `figure_kits/<kit>/` folder.
3. Commit on a PaperAI feature branch and push.
