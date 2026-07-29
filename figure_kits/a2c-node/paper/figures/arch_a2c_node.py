#!/usr/bin/env python3
"""Generate a 3D-style architecture figure for A2C-NODE using PlotNeuralNet.

Pipeline:

    bash A2C-NODE/paper/figures/arch_a2c_node.sh
    # -> arch_a2c_node.tex  -> arch_a2c_node.pdf  (paper/figures/)

This script writes the .tex via PlotNeuralNet's pycore.tikzeng helpers;
the matching shell script then calls pdflatex.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
PNN_ROOT = PROJ_ROOT / "tools" / "PlotNeuralNet"
sys.path.insert(0, str(PNN_ROOT))

from pycore.tikzeng import (                                            # type: ignore
    to_head, to_cor, to_begin, to_input,
    to_Conv, to_ConvConvRelu, to_Pool,
    to_SoftMax, to_connection, to_skip, to_end, to_generate,
)


# Custom helper: 3D box for tabular branch / GNN module / NODE module.
# IMPORTANT: caption is wrapped in `{...}` so it survives commas inside
# math (e.g. ``$f_\theta(h, A)$`` would otherwise break pgfkeys parsing).
def to_box(name, caption, height=18, width=2, depth=18, offset="(0,0,0)",
           to="(0,0,0)", color=r"\FcReluColor", xlabel="", zlabel=""):
    return rf"""
\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={{{caption}}},
        xlabel={{{{ {xlabel} }}}},
        zlabel={{{zlabel}}},
        fill={color},
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""


def build_arch():
    arch = []
    arch.append(to_head(str(PNN_ROOT)))
    # Add a tabular-branch color and a NODE color in addition to the defaults.
    arch.append(to_cor() + r"""
\def\TabColor{rgb:green,3;black,2;white,5}
\def\NodeColor{rgb:red,1;magenta,3;white,5}
""")
    arch.append(to_begin())

    # ---------------- 1. Input T1 + 21-region mask ---------------------------
    arch.append(to_Conv(
        name="t1", s_filer=96, n_filer=1,
        offset="(0,0,0)", to="(0,0,0)",
        height=40, depth=40, width=2, caption="T1 / mask"))

    # ---------------- 2. ARA-Net SSL encoder (4 stages) ----------------------
    stages = [
        ("enc1", 64, 32, 32, 4),
        ("enc2", 128, 24, 24, 5),
        ("enc3", 192, 16, 16, 6),
        ("enc4", 256, 12, 12, 8),
    ]
    prev = "t1"
    for i, (name, ch, h, d, w) in enumerate(stages):
        offset = "(2,0,0)" if i == 0 else "(1.6,0,0)"
        arch.append(to_ConvConvRelu(
            name=name, s_filer=h, n_filer=(ch, ch),
            offset=offset, to=f"({prev}-east)",
            height=h, depth=d, width=(w, w), caption=f"stage {i+1}"))
        prev = name
    arch.append(to_skip(of="t1", to="enc4", pos=1.25))

    # ---------------- 3. Atlas-guided 21 region tokens ----------------------
    arch.append(to_box(
        name="region", caption=r"21 region tokens",
        offset="(2.4,0,0)", to=f"({prev}-east)",
        height=22, depth=22, width=4,
        color=r"\FcColor", xlabel="21", zlabel="C=256"))

    # ---------------- 4. Graph Neural ODE ------------------------------------
    arch.append(to_box(
        name="ode", caption=r"Graph Neural ODE  $\dot h = f_\theta(h, A)$",
        offset="(2.0,0,0)", to="(region-east)",
        height=22, depth=22, width=4,
        color=r"\NodeColor", xlabel="GCN $\\times$ 2", zlabel="t"))

    arch.append(to_connection("region", "ode"))

    # ---------------- 5. Tabular MLP branch (clinical) ----------------------
    arch.append(to_box(
        name="tab", caption=r"clinical MLP",
        offset="(2.0,-6,0)", to="(region-east)",
        height=8, depth=14, width=4,
        color=r"\TabColor", xlabel="MMSE/CDR/", zlabel="K=15"))
    arch.append(to_connection("region", "tab"))

    # ---------------- 6. Multi-task heads -----------------------------------
    arch.append(to_SoftMax("cls", 3, "(2.0,2.5,0)", "(ode-east)",
                            caption="CN / MCI / AD"))
    arch.append(to_box(
        name="atrophy", caption=r"atrophy rate",
        offset="(2.0,-2.5,0)", to="(ode-east)",
        height=10, depth=14, width=2,
        color=r"\PoolColor", xlabel="K=21", zlabel=""))
    arch.append(to_connection("ode", "cls"))
    arch.append(to_connection("ode", "atrophy"))
    arch.append(to_connection("tab", "cls"))

    # ---------------- 7. Counterfactual branch ------------------------------
    arch.append(to_box(
        name="cf", caption=r"do$(h_k=c)$ counterfactual",
        offset="(2.0,-6.0,0)", to="(ode-east)",
        height=10, depth=14, width=2,
        color=r"\NodeColor", xlabel="ATE", zlabel=""))
    arch.append(to_connection("ode", "cf"))

    arch.append(to_end())
    return arch


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    name = out_dir / "arch_a2c_node"
    arch = build_arch()
    to_generate(arch, str(name) + ".tex")
    print(f"[arch] wrote {name}.tex; "
          f"compile with `bash {out_dir}/arch_a2c_node.sh`")


if __name__ == "__main__":
    main()
