# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})

# --- PAINEL (A): Sensibilidade à cobertura ---
thresholds = ["50%", "60%", "70%", "80%", "90%"]
retained = np.array([5945, 5793, 5788, 5728, 5531])
intercepted = np.array([784, 907, 979, 1176, 1867])
x = np.arange(len(thresholds))
width = 0.52

c_retained = "#1f4e79"     # Azul escuro institucional
c_intercepted = "#c00000"  # Vermelho escuro / vinho

bars1 = ax1.bar(x, retained, width, label="Retidos (elegíveis)", color=c_retained, edgecolor="none", alpha=0.92)
bars2 = ax1.bar(x, intercepted, width, bottom=retained, label="Interceptados (quarentena)", color=c_intercepted, edgecolor="none", alpha=0.88)

for i in range(len(x)):
    ax1.text(x[i], retained[i] / 2, f"{retained[i]:,}".replace(",", "."), ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax1.text(x[i], retained[i] + intercepted[i] / 2, f"+{intercepted[i]:,}".replace(",", "."), ha="center", va="center", color="white", fontsize=8.0, fontweight="bold")
    total = retained[i] + intercepted[i]
    ax1.text(x[i], total + 140, f"{total:,}".replace(",", "."), ha="center", va="bottom", color="#222222", fontsize=8.5)

ax1.set_title("(A) Sensibilidade à cobertura (corte 19/11)", pad=10, fontweight="bold")
ax1.set_xlabel("Limiar mínimo de cobertura")
ax1.set_ylabel("UCs sinalizadas no ensaio")
ax1.set_xticks(x)
ax1.set_xticklabels(thresholds)
ax1.set_ylim(0, 9600)  # Expandido para 9.600: margem limpa para legenda e totais sem colisão
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{int(val):,}".replace(",", ".")))
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.set_axisbelow(True)
ax1.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#cccccc")

# --- PAINEL (B): Funil sequencial M0 -> M5 ---
stages = [
    "M0\nJanelas",
    "M1\nUCs Únicas",
    "M2\nCob. ≥ 70%",
    "M3\nAlimentador",
    "M4\nAnom. Elét.",
    "M5\nCand. Vist.",
]
counts = np.array([65686, 65686, 62857, 58857, 4311, 4311])
x2 = np.arange(len(stages))
width2 = 0.58

colors2 = ["#4a6572", "#4a6572", "#23689b", "#1f4e79", "#d35400", "#c0392b"]

bars_b = ax2.bar(x2, counts, width2, color=colors2, edgecolor="none", alpha=0.92)

for i, bar in enumerate(bars_b):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + 1200, f"{h:,}".replace(",", "."), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")

ax2.set_title("(B) Funil sequencial M0 → M5 (execução 25/11)", pad=10, fontweight="bold")
ax2.set_xlabel("Estágio do funil de triagem")
ax2.set_ylabel("Número de UCs")
ax2.set_xticks(x2)
ax2.set_xticklabels(stages)
ax2.set_ylim(0, 75000)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{int(val):,}".replace(",", ".")))
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.set_axisbelow(True)

plt.tight_layout()

out_paths = [
    Path("docs/TCC_AMI_FINAL_BANCA_eekBR_FONTES/figuras/funil.pdf"),
    Path("docs/manuscrito_final/figuras/funil.pdf"),
    Path("docs/figures/funil.pdf")
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Salvo: {p} ({p.stat().st_size} bytes)")

plt.close(fig)
print("Sucesso!")
