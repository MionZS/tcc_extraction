"""Tests for scripts/demo_ml_methods.py real-data pipeline (no synthetic fallback)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts import demo_ml_methods as dmm


def test_load_real_benchmark_raises_without_real_data(tmp_path: Path) -> None:
    """Ensure no synthetic/mock fallback exists when Data Lake files are missing."""
    with pytest.raises(FileNotFoundError):
        dmm.load_real_benchmark(base_dir=tmp_path)


def test_demo_ml_methods_end_to_end_on_real_schema_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run demo_ml_methods against a temporary Data Lake using the canonical real schema."""
    base_dir = tmp_path / "output"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    n_ucs = 60
    target_day = date(2026, 11, 4)
    feeders = ["Fonte Nova", "CSN", "Polonia"]
    cats = [
        "REGULAR",
        "REGULAR",
        "REGULAR",
        "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT",
        "POSSIVEL_GD_NAO_CADASTRADA",
        "DESEQUILIBRIO_CARGA_OU_BYPASS",
    ]

    uc_rows = []
    mh_rows = []
    interval_rows = []
    day_feat_rows_by_day: dict[date, list[dict]] = {
        target_day - timedelta(days=d): [] for d in range(14)
    }

    for i in range(n_ucs):
        uc_id = f"{10000000 + i}"
        nio_id = f"4100{i:04d}"
        cat = cats[i % len(cats)] if i < 18 else "REGULAR"
        is_anom = cat != "REGULAR"
        is_cand = i in (3, 4, 5, 9, 10, 11)
        feeder = feeders[i % len(feeders)]

        med_kwh = float(rng.uniform(10.0, 40.0)) if i != 3 else 180.0
        mad_kwh = float(rng.uniform(0.5, 4.0)) if not is_anom else float(rng.uniform(6.0, 15.0))
        cv_val = mad_kwh / max(med_kwh, 1.0)
        lf_val = float(rng.uniform(0.35, 0.85)) if i != 4 else 0.12
        ci_val = float(rng.uniform(0.01, 0.15)) if cat != "DESEQUILIBRIO_CARGA_OU_BYPASS" else 0.82
        vi_val = float(rng.uniform(0.005, 0.03))
        slope_val = float(rng.uniform(-0.1, 0.1)) if i != 5 else -1.8
        ra_ratio = 0.0 if cat != "POSSIVEL_GD_NAO_CADASTRADA" else 0.45

        uc_rows.append({
            "id__uc_id": uc_id,
            "id__nio": nio_id,
            "meta__cutoff_date": target_day.isoformat(),
            "meta__feeder": feeder,
            "meta__is_clean_for_train": True,
            "CATEGORIA_DIAGNOSTICA": cat,
            "IS_ANOMALY": is_anom,
            "IS_CANDIDATO_VISTORIA": is_cand,
            "ANOMALY_SCORE": 0.78 if is_anom else 0.25,
            "DIAS_ANOMALIA_SEGUIDOS": 12 if is_cand else 0,
            "x__hist_fa_sum_median": med_kwh,
            "x__hist_fa_sum_mad": mad_kwh,
            "x__fa_cv_window": cv_val,
            "x__hist_load_factor_mean": lf_val,
            "x__hist_current_imbalance_max": ci_val,
            "x__hist_voltage_imbalance_max": vi_val,
            "x__trend_fa_slope": slope_val,
            "x__hist_ra_reversal_ratio_max": ra_ratio,
        })
        mh_rows.append({"UC": uc_id, "NIO": nio_id})

        # 96 x 15-min intervals for target_day
        for slot in range(96):
            ts = datetime.combine(target_day, datetime.min.time()) + timedelta(minutes=15 * slot)
            hour = slot / 4.0
            base_kw = med_kwh / 24.0
            if cat == "POSSIVEL_GD_NAO_CADASTRADA" and 10 <= hour <= 15:
                fa_val = 0.02
                ra_val = base_kw * 0.6
            elif i == 4 and 2 <= hour <= 4:
                fa_val = base_kw * 2.5
                ra_val = 0.0
            else:
                fa_val = max(0.01, base_kw * (0.25 + 0.1 * np.sin(2 * np.pi * hour / 24.0)) + float(rng.normal(0, 0.02)))
                ra_val = 0.0
            interval_rows.append({
                "NIO": nio_id,
                "TIMESTAMP_LOCAL": ts,
                "FA_INTERVAL": fa_val,
                "RA_INTERVAL": ra_val,
            })

        # 14 daily feature partitions for longitudinal Case F
        for d_off in range(14):
            dt = target_day - timedelta(days=13 - d_off)
            day_kwh = med_kwh if (i != 5 or d_off < 7) else med_kwh * 0.25
            day_feat_rows_by_day[dt].append({
                "UC": uc_id,
                "NIO": nio_id,
                "REPORT_DAY": dt,
                "INTERVAL_COVERAGE": 0.98,
                "FA_INTERVAL_SUM": day_kwh,
                "RA_INTERVAL_SUM": 0.0,
            })

    pl.DataFrame(uc_rows).write_parquet(reports_dir / "classified_ucs_latest.parquet")

    ctx_dir = base_dir / "context"
    ctx_dir.mkdir(parents=True)
    pl.DataFrame(mh_rows).write_parquet(ctx_dir / "meter_installation_history.parquet")

    int_dir = (
        base_dir / "measurements" / "ami_interval"
        / f"report_year={target_day.year:04d}"
        / f"report_month={target_day.month:02d}"
        / f"report_day={target_day.day:02d}"
    )
    int_dir.mkdir(parents=True)
    pl.DataFrame(interval_rows).write_parquet(int_dir / "data.parquet")

    for dt, d_rows in day_feat_rows_by_day.items():
        df_dir = (
            base_dir / "features" / "uc_day_features"
            / f"report_year={dt.year:04d}"
            / f"report_month={dt.month:02d}"
            / f"report_day={dt.day:02d}"
        )
        df_dir.mkdir(parents=True)
        pl.DataFrame(d_rows).write_parquet(df_dir / "data.parquet")

    out_dir = tmp_path / "ml_out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "demo_ml_methods.py",
            "--base-dir", str(base_dir),
            "--out-dir", str(out_dir),
            "--n-samples", "60",
            "--no-sync-docs",
        ],
    )
    dmm.main()

    expected_figs = [
        "01_raw_profiles.png",
        "02_kmeans_profiles.png",
        "03_pca_projection.png",
        "04_pca_residual.png",
        "05_isolation_forest.png",
        "06_lof.png",
        "07_ecod.png",
        "08_detector_comparison.png",
        "09_ablation_funnel.png",
        "10_case_studies.png",
    ]
    for fig_name in expected_figs:
        assert (out_dir / fig_name).exists()
        assert (out_dir / fig_name).stat().st_size > 0

    metrics_df = pl.read_csv(out_dir / "metrics.csv", separator=";")
    assert metrics_df.height == n_ucs
    assert "uc_censored" in metrics_df.columns
    assert all("*" in u for u in metrics_df["uc_censored"].to_list())
    assert (out_dir / "method_summary.csv").exists()
