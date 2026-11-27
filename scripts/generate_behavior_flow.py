"""Render the AMI behavioral workflow as a readable 3–3–2 diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "Curva diária AMI\n96/288 pontos",
    "Qualidade mínima\ne regularização",
    "Representação dupla\nnormalizada + magnitude",
    "PCA / redução\nde redundância",
    "Arquétipo diário\ncluster comportamental",
    "Sequência de dias\npor UC e dia da semana",
    "Dias atípicos + mudança\nde distribuição",
    "S_behavior\nmudança de rotina",
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.1, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width, height = 0.27, 0.14
    positions = [
        (0.05, 0.76), (0.365, 0.76), (0.68, 0.76),
        (0.68, 0.43), (0.365, 0.43), (0.05, 0.43),
        (0.05, 0.10), (0.365, 0.10),
    ]
    for index, ((x, y), label) in enumerate(zip(positions, STEPS, strict=True)):
        box = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.016,rounding_size=0.025",
            edgecolor="#245678", facecolor="#ecf4f8" if index < 7 else "#d5e9dc",
            linewidth=1.3,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                fontsize=9.1, linespacing=1.2, color="#162b3d")

    for start, end in zip(positions[:-1], positions[1:], strict=True):
        sx, sy = start
        ex, ey = end
        if sy == ey:
            moving_right = ex > sx
            origin = (sx + width + 0.005, sy + height / 2) if moving_right else (sx - 0.005, sy + height / 2)
            target = (ex - 0.012, ey + height / 2) if moving_right else (ex + width + 0.012, ey + height / 2)
        else:
            origin = (sx + width / 2, sy - 0.008)
            target = (ex + width / 2, ey + height + 0.008)
        ax.add_patch(FancyArrowPatch(origin, target, arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.5, color="#245678"))

    output = ROOT / "docs/TCC_AMI_FINAL_BANCA_eekBR_FONTES/figures/behavior.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    main()
