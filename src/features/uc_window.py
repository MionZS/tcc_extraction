"""Build windowed feature tables for UCs ending at a cutoff date.

Grain: UC × CUTOFF_DATE × WINDOW_DAYS.
Calculates historical rolling statistics, trend slopes, meter age,
and event counts without looking ahead into the future.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from src.datasets.schemas import UC_WINDOW_FEATURES_SCHEMA


def build_uc_window_features(
    daily_features_df: pl.DataFrame,
    meter_history_df: pl.DataFrame | None = None,
    alarm_events_df: pl.DataFrame | None = None,
    *,
    cutoff_date: date,
    window_days: int = 30,
    feature_set_version: str = "v1",
) -> pl.DataFrame:
    """Compute rolling window features for each UC up to cutoff_date.

    Args:
        daily_features_df: Historical daily features table (must have REPORT_DAY and UC).
        meter_history_df: Optional meter_installation_history table.
        alarm_events_df: Optional alarm_events table.
        cutoff_date: Upper temporal boundary for feature computation.
        window_days: Size of the historical window in days (e.g., 7 or 30).
        feature_set_version: Version tag.

    Returns:
        DataFrame following UC_WINDOW_FEATURES_SCHEMA.
    """
    if daily_features_df.is_empty():
        return pl.DataFrame(schema=UC_WINDOW_FEATURES_SCHEMA)

    window_start = cutoff_date - timedelta(days=window_days)

    # Filter strictly to the window: window_start < REPORT_DAY <= cutoff_date
    window_daily = daily_features_df.filter(
        (pl.col("REPORT_DAY") > window_start) & (pl.col("REPORT_DAY") <= cutoff_date)
    )

    if window_daily.is_empty():
        return pl.DataFrame(schema=UC_WINDOW_FEATURES_SCHEMA)

    # Extract unique UCs present
    ucs = window_daily["UC"].unique().sort().to_list()
    rows: list[dict[str, Any]] = []

    for uc in ucs:
        uc_daily = window_daily.filter(pl.col("UC") == uc).sort("REPORT_DAY")

        # ── Coverage & Quality ──────────────────────────────────────────
        cov_col = uc_daily.get_column("INTERVAL_COVERAGE") if "INTERVAL_COVERAGE" in uc_daily.columns else pl.Series([0.0])
        reliable_days = int((cov_col >= 0.8).sum())
        avg_cov = float(cov_col.mean()) if cov_col.len() > 0 else 0.0

        # ── Electrical aggregations ─────────────────────────────────────
        fa_sums = (
            uc_daily.get_column("FA_INTERVAL_SUM").to_numpy()
            if "FA_INTERVAL_SUM" in uc_daily.columns
            else np.array([0.0])
        )
        fa_median = float(np.median(fa_sums)) if len(fa_sums) > 0 else 0.0
        fa_mad = float(np.median(np.abs(fa_sums - fa_median))) if len(fa_sums) > 0 else 0.0

        ra_rev = (
            uc_daily.get_column("RA_REVERSAL_RATIO").to_numpy()
            if "RA_REVERSAL_RATIO" in uc_daily.columns
            else np.array([0.0])
        )
        rev_days = int((ra_rev > 0.0).sum()) if len(ra_rev) > 0 else 0
        rev_max = float(np.max(ra_rev)) if len(ra_rev) > 0 else 0.0

        load_factors = (
            uc_daily.get_column("LOAD_FACTOR").to_numpy()
            if "LOAD_FACTOR" in uc_daily.columns
            else np.array([0.0])
        )
        lf_mean = float(np.nanmean(load_factors)) if len(load_factors) > 0 else 0.0

        u_imb = (
            uc_daily.get_column("VOLTAGE_IMBALANCE_MAX").to_numpy()
            if "VOLTAGE_IMBALANCE_MAX" in uc_daily.columns
            else np.array([0.0])
        )
        u_imb_max = float(np.nanmax(u_imb)) if len(u_imb) > 0 else 0.0

        i_imb = (
            uc_daily.get_column("CURRENT_IMBALANCE_MAX").to_numpy()
            if "CURRENT_IMBALANCE_MAX" in uc_daily.columns
            else np.array([0.0])
        )
        i_imb_max = float(np.nanmax(i_imb)) if len(i_imb) > 0 else 0.0

        # Linear trend slope
        if len(fa_sums) >= 3:
            x_vals = np.arange(len(fa_sums))
            # simple linear regression slope
            cov_xy = np.cov(x_vals, fa_sums)[0, 1]
            var_x = np.var(x_vals)
            trend_slope = float(cov_xy / var_x) if var_x > 0 else 0.0
        else:
            trend_slope = 0.0

        # ── Meter installation & age context ─────────────────────────────
        meter_age_days = 0
        meter_changed_30d = False
        meter_changed_90d = False

        if meter_history_df is not None and not meter_history_df.is_empty():
            uc_meters = meter_history_df.filter(pl.col("UC") == uc)
            if not uc_meters.is_empty() and "DATA_INSTALACAO" in uc_meters.columns:
                install_dates = uc_meters["DATA_INSTALACAO"].drop_nulls().to_list()
                if install_dates:
                    latest_install = max(install_dates)
                    meter_age_days = max(0, (cutoff_date - latest_install).days)
                    meter_changed_30d = meter_age_days <= 30
                    meter_changed_90d = meter_age_days <= 90

        # ── Alarm events over window ─────────────────────────────────────
        alarm_count = 0
        avg_alarm_latency = 0.0

        if alarm_events_df is not None and not alarm_events_df.is_empty():
            # Join alarm events via NIO if applicable
            nio_val = uc_daily["NIO"][0] if "NIO" in uc_daily.columns else None
            if nio_val:
                uc_alarms = alarm_events_df.filter(
                    (pl.col("NIO") == nio_val)
                    & (pl.col("ORIGIN_TIMESTAMP") >= window_start)
                    & (pl.col("ORIGIN_TIMESTAMP") <= cutoff_date)
                )
                alarm_count = uc_alarms.height
                if alarm_count > 0 and "LATENCY_SECONDS" in uc_alarms.columns:
                    lat_vals = uc_alarms["LATENCY_SECONDS"].drop_nulls()
                    if lat_vals.len() > 0:
                        avg_alarm_latency = float(lat_vals.mean())

        row = {
            "UC": uc,
            "CUTOFF_DATE": cutoff_date,
            "WINDOW_DAYS": window_days,
            "FEATURE_SET_VERSION": feature_set_version,
            "COVERAGE_RELIABLE_DAYS": reliable_days,
            "AVG_INTERVAL_COVERAGE": round(avg_cov, 4),
            "HIST_FA_SUM_MEDIAN": round(fa_median, 4),
            "HIST_FA_SUM_MAD": round(fa_mad, 4),
            "HIST_RA_REVERSAL_DAYS": rev_days,
            "HIST_RA_REVERSAL_RATIO_MAX": round(rev_max, 4),
            "HIST_LOAD_FACTOR_MEAN": round(lf_mean, 4),
            "HIST_VOLTAGE_IMBALANCE_MAX": round(u_imb_max, 4),
            "HIST_CURRENT_IMBALANCE_MAX": round(i_imb_max, 4),
            "TREND_FA_SLOPE": round(trend_slope, 4),
            "METER_AGE_DAYS": meter_age_days,
            "METER_CHANGED_30D": meter_changed_30d,
            "METER_CHANGED_90D": meter_changed_90d,
            "ALARM_COUNT_WINDOW": alarm_count,
            "AVG_ALARM_LATENCY_SEC": round(avg_alarm_latency, 2),
        }
        rows.append(row)

    return pl.DataFrame(rows, schema=UC_WINDOW_FEATURES_SCHEMA)

