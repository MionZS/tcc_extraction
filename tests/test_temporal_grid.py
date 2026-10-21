"""Tests for Temporal Grid Regularization, Communication Gap Metrics, and Micro-Interpolation."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import numpy as np
import polars as pl
import pytest

from src.features.temporal_grid import (
    build_daily_time_grid,
    regularize_interval_telemetry,
)


def test_build_daily_time_grid():
    d = date(2026, 6, 1)
    grid_5m = build_daily_time_grid(d, cadence_minutes=5)
    assert grid_5m.height == 288
    assert grid_5m["TIMESTAMP_UTC"][0] == datetime(2026, 6, 1, 0, 0, 0)
    assert grid_5m["TIMESTAMP_UTC"][-1] == datetime(2026, 6, 1, 23, 55, 0)

    grid_15m = build_daily_time_grid(d, cadence_minutes=15)
    assert grid_15m.height == 96


def test_regularize_telemetry_gap_metrics_and_interpolation():
    d = date(2026, 6, 1)
    # Generate 288 grid points with a micro-gap (2 missing points) and a macro-gap (10 missing points)
    grid = build_daily_time_grid(d, cadence_minutes=5)
    ts = grid["TIMESTAMP_UTC"].to_list()

    rows = []
    for i, t in enumerate(ts):
        # Micro gap at i in [10, 11] (2 slots missing)
        if 10 <= i <= 11:
            continue
        # Macro gap at i in [50, 59] (10 slots missing)
        if 50 <= i <= 59:
            continue
        rows.append({
            "TIMESTAMP_UTC": t,
            "FA_INTERVAL": 10.0 + (i % 5),
            "U_L1": 220.0,
            "CADENCE_MINUTES": 5,
        })

    raw_df = pl.DataFrame(rows)
    reg_df, comm_metrics = regularize_interval_telemetry(raw_df, d, cadence_minutes=5, max_interp_slots=3)

    assert reg_df.height == 288
    assert "m_t" in reg_df.columns

    # Gap metrics verification
    # Total missing = 2 + 10 = 12 intervals
    # Total observed = 288 - 12 = 276
    assert comm_metrics["gap_count"] == 2
    assert comm_metrics["maior_gap_min"] == 50  # 10 intervals * 5 min
    assert comm_metrics["intervalos_consecutivos_ausentes"] == 10
    assert 0.95 < comm_metrics["completude_1d"] < 0.96

    # Micro-gap interpolation verification: slots 10 and 11 should be interpolated!
    interp_fa = reg_df["FA_INTERVAL"].to_numpy()
    assert not np.isnan(interp_fa[10])
    assert not np.isnan(interp_fa[11])

    # Macro-gap policy: slots 50 to 59 MUST remain NaN (no hallucination of hours of consumption)
    for k in range(50, 60):
        assert np.isnan(interp_fa[k])
