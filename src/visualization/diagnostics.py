"""Diagnostic Visualizations for Anomaly Detection & MoE Model Evaluation.

Generates diagnostic charts to inspect model performance, score distributions,
feature anomalies, per-feeder breakdown, and spurious filter categorizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt


def generate_diagnostic_plots(
    scores_df: pl.DataFrame,
    output_dir: Path | str = "output/reports/figures",
    title_prefix: str = "Modelo Composto",
) -> dict[str, Path]:
    """Generate diagnostic visualization charts for anomaly model evaluation.

    Args:
        scores_df: DataFrame containing prediction scores and anomaly flags.
                   Expected columns: 'anomaly_score' / 'consensus_score',
                   'is_anomaly', and optionally 'meta__feeder' / 'ALIMENTADOR',
                   'ANOMALY_CATEGORY', 'iso_score', 'lof_score', 'ecod_score', etc.
        output_dir: Directory where PNG plots will be saved.
        title_prefix: Prefix string for chart titles.

    Returns:
        Dict mapping plot name to saved file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_plots: dict[str, Path] = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    score_col = "consensus_score" if "consensus_score" in scores_df.columns else (
        "anomaly_score" if "anomaly_score" in scores_df.columns else (
            "FILTERED_ANOMALY_SCORE" if "FILTERED_ANOMALY_SCORE" in scores_df.columns else None
        )
    )

    # 1. Anomaly Score Distribution Histogram
    if score_col and score_col in scores_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        scores = scores_df.get_column(score_col).drop_nulls().to_numpy()

        anom_col = "FINAL_IS_ANOMALY" if "FINAL_IS_ANOMALY" in scores_df.columns else (
            "is_anomaly" if "is_anomaly" in scores_df.columns else None
        )

        if anom_col and anom_col in scores_df.columns:
            anom_mask = scores_df.get_column(anom_col).to_numpy().astype(bool)
            ax.hist(scores[~anom_mask], bins=50, alpha=0.6, label="Normal", color="#2ecc71", edgecolor="none")
            ax.hist(scores[anom_mask], bins=50, alpha=0.8, label="Anomalia", color="#e74c3c", edgecolor="none")
        else:
            ax.hist(scores, bins=50, alpha=0.7, color="#3498db", edgecolor="none")

        ax.set_title(f"{title_prefix} - Distribuição dos Scores de Anomalia", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score de Anomalia [0.0 - 1.0]", fontsize=10)
        ax.set_ylabel("Frequência (Nº de UCs)", fontsize=10)
        ax.legend(loc="upper right")
        plt.tight_layout()

        p1 = out_path / "score_distribution.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        saved_plots["score_distribution"] = p1

    # 2. Sub-Model Score Comparison (IsoForest, LOF, ECOD, PCA)
    sub_cols = [c for c in ["iso_score", "lof_score", "ecod_score", "pca_score"] if c in scores_df.columns]
    if len(sub_cols) >= 2:
        fig, ax = plt.subplots(figsize=(9, 5))
        means_normal = []
        means_anom = []
        labels = [c.replace("_score", "").upper() for c in sub_cols]

        anom_col = "is_anomaly" if "is_anomaly" in scores_df.columns else "FINAL_IS_ANOMALY"
        has_anom = anom_col in scores_df.columns
        anom_mask = scores_df.get_column(anom_col).to_numpy().astype(bool) if has_anom else np.zeros(scores_df.height, dtype=bool)

        for c in sub_cols:
            arr = scores_df.get_column(c).to_numpy()
            means_normal.append(float(np.mean(arr[~anom_mask])) if len(arr[~anom_mask]) > 0 else 0.0)
            means_anom.append(float(np.mean(arr[anom_mask])) if len(arr[anom_mask]) > 0 else 0.0)

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, means_normal, width, label="Normal", color="#3498db")
        ax.bar(x + width / 2, means_anom, width, label="Anomalia", color="#e74c3c")

        ax.set_title(f"{title_prefix} - Score Médio por Sub-Modelo Especialista", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score Médio Normalizado", fontsize=10)
        ax.legend()
        plt.tight_layout()

        p2 = out_path / "model_consensus_breakdown.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        saved_plots["model_consensus_breakdown"] = p2

    # 3. Anomaly Rate per Feeder
    feeder_col = "meta__feeder" if "meta__feeder" in scores_df.columns else (
        "ALIMENTADOR" if "ALIMENTADOR" in scores_df.columns else None
    )
    anom_col = "FINAL_IS_ANOMALY" if "FINAL_IS_ANOMALY" in scores_df.columns else (
        "is_anomaly" if "is_anomaly" in scores_df.columns else None
    )

    if feeder_col and anom_col and feeder_col in scores_df.columns and anom_col in scores_df.columns:
        summary = (
            scores_df.group_by(feeder_col)
            .agg([
                pl.len().alias("total"),
                pl.col(anom_col).sum().alias("anomalias"),
            ])
            .with_columns((pl.col("anomalias") / pl.col("total")).alias("taxa_anomalia"))
            .sort("taxa_anomalia", descending=True)
        )

        if summary.height > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            feeders = [str(f) for f in summary.get_column(feeder_col).to_list()]
            rates = (summary.get_column("taxa_anomalia") * 100.0).to_numpy()

            bars = ax.barh(feeders[::-1], rates[::-1], color="#9b59b6")
            ax.set_title(f"{title_prefix} - Taxa de Anomalias por Alimentador (%)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Proporção de UCs Anômalas (%)", fontsize=10)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9)

            plt.tight_layout()
            p3 = out_path / "feeder_anomaly_rates.png"
            fig.savefig(p3, dpi=150)
            plt.close(fig)
            saved_plots["feeder_anomaly_rates"] = p3

    # 4. Spurious vs Genuine Anomaly Classification (if available)
    if "ANOMALY_CATEGORY" in scores_df.columns:
        cat_counts = scores_df.group_by("ANOMALY_CATEGORY").len()
        fig, ax = plt.subplots(figsize=(7, 5))
        cats = [str(c) for c in cat_counts.get_column("ANOMALY_CATEGORY").to_list()]
        vals = cat_counts.get_column("len").to_list()
        colors = ["#2ecc71" if c == "NORMAL" else ("#e74c3c" if c == "GENUINE_UC_ANOMALY" else "#f39c12") for c in cats]

        ax.pie(vals, labels=cats, autopct="%1.1f%%", startangle=140, colors=colors)
        ax.set_title("Classificação Pós-Filtro de Medidas Expúrias & Grade", fontsize=12, fontweight="bold")
        plt.tight_layout()

        p4 = out_path / "spurious_filter_breakdown.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        saved_plots["spurious_filter_breakdown"] = p4

    return saved_plots
