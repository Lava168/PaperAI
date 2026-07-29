#!/usr/bin/env python3
"""High-fidelity raster→SVG→PDF via VTracer (white background).

Refs:
  - https://github.com/visioncortex/vtracer  (best practical color vectorizer)
  - https://github.com/sjtuplayer/SuperSVG   (CVPR 2024, needs GPU/weights)
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image
import vtracer
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'v6_final_model' / 'figures'
ALT = ROOT / 'reports' / 'brain_figures_fcstyle' / 'figures'
SRC = ALT / 'image.png'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ALT.mkdir(parents=True, exist_ok=True)
    im = Image.open(SRC).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    hi = Image.alpha_composite(bg, im).convert('RGB')
    hi = hi.resize((hi.width * 4, hi.height * 4), Image.Resampling.LANCZOS)
    png = OUT / 'figure1_pipeline_source_white.png'
    svg = OUT / 'figure1_pipeline_source.svg'
    pdf_v = OUT / 'figure1_pipeline_source_vector.pdf'
    pdf_r = OUT / 'figure1_pipeline_source_white.pdf'
    hi.save(png, dpi=(300, 300))
    vtracer.convert_image_to_svg_py(
        str(png), str(svg), colormode='color', hierarchical='stacked', mode='spline',
        filter_speckle=4, color_precision=8, layer_difference=8, corner_threshold=60,
        length_threshold=4.0, max_iterations=10, splice_threshold=45, path_precision=3,
    )
    cairosvg.svg2pdf(url=str(svg), write_to=str(pdf_v))
    fig = plt.figure(figsize=(11, 11 * hi.height / hi.width), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(np.asarray(hi)); ax.axis('off')
    fig.savefig(pdf_r, dpi=300, facecolor='white'); plt.close(fig)
    for p in (png, svg, pdf_v, pdf_r):
        (ALT / p.name).write_bytes(p.read_bytes())
    print('done', pdf_r, pdf_v)

if __name__ == '__main__':
    main()
