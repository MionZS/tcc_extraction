"""Render the article from the frozen November campaign and masked EDA extracts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OOT = ROOT / "experiments/runs/20261125_134559_972_oot_multiview"
OUTPUT = ROOT / "docs/manuscrito_final/figuras"
DETECTORS = ["iso", "lof", "ecod", "copod", "hbos", "pca"]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / name, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def detector_figures() -> None:
    columns = [f"{name}_score" for name in DETECTORS] + ["s_elec"]
    scores = pl.read_parquet(OOT / "calibrated_scores.parquet", columns=columns)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    bins = ax.hexbin(scores["iso_score"], scores["lof_score"], gridsize=40,
                     mincnt=1, bins="log", cmap="viridis", rasterized=True)
    ax.set(xlabel="Escore do Isolation Forest", ylabel="Escore do LOF")
    fig.colorbar(bins, ax=ax, label="UCs por célula (escala logarítmica)")
    save(fig, "if_lof_oot.pdf")

    ranks = scores.select(pl.all().rank(method="average")).to_numpy()
    corr = np.corrcoef(ranks.T)
    labels = ["IF", "LOF", "ECOD", "COPOD", "HBOS", "PCA", "Elétrico"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4), layout="constrained")
    ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for (i, j), value in np.ndenumerate(corr):
        ax.text(j, i, f"{value:.2f}".replace(".", ","), ha="center", va="center",
                color="white" if abs(value) > 0.65 else "black")
    save(fig, "correlacao.pdf")


def measured_profiles() -> None:
    frame = pl.read_csv(ROOT / "docs/tcc_evidence/eda/06_curvas_carga_por_perfil.csv", separator=";")
    values = frame.select([f"h{h:02}" for h in range(24)]).to_numpy()
    fractions = values / values.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    ax.plot(range(24), fractions.T * 100, linewidth=1.2, alpha=0.85)
    ax.set(xlabel="Hora do dia", ylabel="Parcela do perfil diário (%)", xticks=range(0, 24, 2))
    save(fig, "perfis_reais.pdf")

    path = ROOT / "docs/figures/eda/exemplos/examples_metadata.json"
    pair = json.loads(path.read_text(encoding="utf-8"))["contrasting_pair"]
    week = pair["weekly_timeline"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    groups = [pair["uc_stable"], pair["uc_volatile"]]
    ids = [group["uc_censored"] for group in groups]
    for ax, values, title, ylabel in zip(
        axes, [[g["mad_kwh_day"] for g in groups], [week["max_kwh_a"], week["max_kwh_b"]]],
        ["Dispersão diária (MAD)", "Máximo em 168 horas"],
        ["Energia diária (kWh/dia)", "Energia em uma hora (kWh)"], strict=True,
    ):
        bars = ax.bar(ids, values, color=["#216686", "#ae5729"], width=0.52)
        ax.bar_label(bars, labels=[f"{v:.3f}".replace(".", ",") for v in values], padding=3)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.22))
    save(fig, "semana.pdf")


def case_series() -> None:
    source = ROOT / "docs/manuscript_figures/real_case_studies/fig_caso_908xxx00_fa_ra_completo_data.csv"
    daily = pl.read_csv(source, separator=";", try_parse_dates=True).sort("report_day")
    dates = daily["report_day"].to_list()
    fig, (energy, coverage) = plt.subplots(2, 1, sharex=True, figsize=(7.4, 4.5),
                                          height_ratios=[2, 1], layout="constrained")
    for col, label, color in [("fa_kwh_day", "Direta (FA)", "#225a9a"),
                              ("ra_kwh_day", "Reversa (RA)", "#15815d")]:
        energy.plot(dates, daily[col].to_numpy(), label=label, color=color, lw=1.4)
    energy.legend(ncols=2)
    energy.set_ylabel("Energia diária (kWh)")
    coverage.bar(dates, daily["interval_coverage"].to_numpy() * 100, color="#537487")
    coverage.set(ylabel="Cobertura (%)", xlabel="Data em 2026", ylim=(0, 105))
    coverage.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    save(fig, "fig_caso_908xxx00_fa_ra_completo.pdf")


def generate_article_figures() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    detector_figures()
    measured_profiles()
    case_series()


if __name__ == "__main__":
    generate_article_figures()
