"""Unit tests for sequential training pipeline and Web Cockpit API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import polars as pl
import pytest

from scripts.run_sequential_training import run_sequential_training
from scripts.run_web_cockpit import CockpitHandler


def test_sequential_training_pipeline_end_to_end(tmp_path: Path) -> None:
    base_dir = tmp_path / "output"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    input_dir = base_dir / "model_input" / "v1"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic training dataset with 60 UCs across 2 months
    rows = []
    feeders = ["Fonte_Nova", "Angelica", "CSN"]
    for i in range(60):
        f = feeders[i % len(feeders)]
        is_anom = i in (5, 12, 28)
        rows.append({
            "id__uc_id": f"UC_{i:04d}",
            "meta__cutoff_date": "2026-08-31" if i < 30 else "2026-10-31",
            "meta__feeder": f,
            "meta__is_clean_for_train": True,
            "x__hist_fa_sum_median": 0.02 if is_anom else 12.5,
            "x__hist_fa_sum_mad": 0.01 if is_anom else 1.8,
            "x__trend_fa_slope": -2.5 if is_anom else 0.05,
            "x__hist_current_imbalance_max": 0.85 if is_anom else 0.08,
            "x__hist_voltage_imbalance_max": 1.2,
            "x__hist_ra_reversal_ratio_max": 0.80 if i == 5 else 0.0,
            "x__hist_ra_reversal_days": 10 if i == 5 else 0,
            "x__avg_interval_coverage": 0.95,
            "x__coverage_reliable_days": 28,
            "x__alarm_count_window": 3 if i == 12 else 0,
            "x__meter_age_days": 450,
            "x__tensao_base": 127.0,
            "x__fase_circuito": "BI",
            "x__hist_load_factor_mean": 0.42,
        })

    ds = pl.DataFrame(rows)
    ds.write_parquet(input_dir / "training_dataset.parquet")

    # Run sequential training
    summary = run_sequential_training(base_dir=base_dir, contamination=0.05, random_state=42)

    assert summary["total_ucs"] == 60
    assert summary["clean_train_samples"] > 0
    assert summary["anomalies_detected"] > 0
    assert (reports_dir / "classified_ucs_latest.parquet").exists()
    assert (reports_dir / "model_training_summary_latest.json").exists()

    # Verify classified dataframe contains diagnostic categories
    classified_df = pl.read_parquet(reports_dir / "classified_ucs_latest.parquet")
    assert "CATEGORIA_DIAGNOSTICA" in classified_df.columns
    assert "ANOMALY_SCORE" in classified_df.columns
    cats = classified_df.get_column("CATEGORIA_DIAGNOSTICA").unique().to_list()
    assert len(cats) >= 2

