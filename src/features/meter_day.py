"""Build the meter_day_features table from normalized datasets.

One row per NIO × report_day, with features available up to the
prediction cutoff.  This is the scikit-learn interface table
(§5.5 of the architecture decision).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from src.features.electrical import (
    compute_voltage_imbalance,
    compute_current_imbalance,
    compute_load_factor,
    compute_ra_reversal_ratio,
    compute_energy_ramp,
)
from src.features.quality import compute_coverage, compute_null_ratio


def build_meter_day_features(
    interval_df: pl.DataFrame,
    instantaneous_df: pl.DataFrame,
    register_df: pl.DataFrame,
    metadata_df: pl.DataFrame | None,
    *,
    report_day: date,
    feature_set_version: str = "v1",
    meter_to_uc_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Build feature rows for all NIOs/UCs on a given report_day.

    This function computes interpretable physical and statistical features
    from the normalized tables. It does NOT impute missing values —
    absence flags and coverage ratios are preserved as features.

    Args:
        interval_df: Normalized interval data for the report day.
        instantaneous_df: Normalized instantaneous data for the report day.
        register_df: Normalized register snapshot data for the report day.
        metadata_df: Optional metadata/context table for context features.
        report_day: The reference date.
        feature_set_version: Version tag for reproducibility.
        meter_to_uc_map: Optional mapping from NIO -> UC.

    Returns:
        DataFrame with one row per UC/NIO and feature columns.
    """
    if interval_df.is_empty():
        return _empty_features_frame()

    # Group by NIO
    nios = interval_df["NIO"].unique().sort().to_list()
    feature_rows: list[dict[str, Any]] = []

    for nio in nios:
        nio_interval = interval_df.filter(pl.col("NIO") == nio)
        nio_instant = (
            instantaneous_df.filter(pl.col("NIO") == nio)
            if not instantaneous_df.is_empty()
            else pl.DataFrame()
        )
        nio_register = (
            register_df.filter(pl.col("NIO") == nio)
            if not register_df.is_empty()
            else pl.DataFrame()
        )

        uc_val = meter_to_uc_map.get(nio, nio) if meter_to_uc_map else nio

        row = _compute_nio_features(
            nio=nio,
            uc=uc_val,
            interval_df=nio_interval,
            instantaneous_df=nio_instant,
            register_df=nio_register,
            report_day=report_day,
            feature_set_version=feature_set_version,
        )

        # Merge metadata context if available
        if metadata_df is not None and not metadata_df.is_empty():
            meta_row = metadata_df.filter(pl.col("NIO") == nio)
            if meta_row.height > 0:
                for col in ("PHASE_TYPE", "CONSUMER_CLASS", "METER_TYPE", "INSTALLED_KVA"):
                    if col in meta_row.columns:
                        row[col] = meta_row[col][0]

        feature_rows.append(row)

    return pl.DataFrame(feature_rows, strict=False)


def _compute_nio_features(
    *,
    nio: str,
    uc: str | None = None,
    interval_df: pl.DataFrame,
    instantaneous_df: pl.DataFrame,
    register_df: pl.DataFrame,
    report_day: date,
    feature_set_version: str,
) -> dict[str, Any]:
    """Compute features for a single NIO/UC on a single day."""
    from src.datasets.normalize import compute_expected_points, compute_coverage as compute_cov

    row: dict[str, Any] = {
        "UC": uc or nio,
        "NIO": nio,
        "REPORT_DAY": report_day,
        "FEATURE_SET_VERSION": feature_set_version,
    }

    # ── Coverage & quality ───────────────────────────────────────────────
    interval_cov = compute_coverage_from_df(interval_df, "FA_INTERVAL")
    instant_cov = compute_coverage_from_df(instantaneous_df, "U_L1") if instantaneous_df.height > 0 else 0.0
    register_cov = compute_coverage_from_df(register_df, "FA_TOTAL") if register_df.height > 0 else 0.0


    row["INTERVAL_COVERAGE"] = round(interval_cov, 4)
    row["INSTANTANEOUS_COVERAGE"] = round(instant_cov, 4)
    row["REGISTER_COVERAGE"] = round(register_cov, 4)
    row["NULL_RATIO_INTERVAL"] = round(
        compute_null_ratio(interval_df, "FA_INTERVAL"), 4
    )

    # ── Energy & demand ──────────────────────────────────────────────────
    if "FA_INTERVAL" in interval_df.columns:
        fa_values = interval_df["FA_INTERVAL"].drop_nulls()
        row["FA_INTERVAL_SUM"] = round(fa_values.sum(), 2) if fa_values.len() > 0 else 0.0
    else:
        row["FA_INTERVAL_SUM"] = 0.0

    if "RA_INTERVAL" in interval_df.columns:
        ra_values = interval_df["RA_INTERVAL"].drop_nulls()
        row["RA_INTERVAL_SUM"] = round(ra_values.sum(), 2) if ra_values.len() > 0 else 0.0
    else:
        row["RA_INTERVAL_SUM"] = 0.0

    if "FA_MD" in register_df.columns:
        md_values = register_df["FA_MD"].drop_nulls()
        row["FA_MD_MAX"] = round(md_values.max(), 2) if md_values.len() > 0 else 0.0
    else:
        row["FA_MD_MAX"] = 0.0

    # ── Load factor ──────────────────────────────────────────────────────
    row["LOAD_FACTOR"] = round(compute_load_factor(interval_df), 4)

    # ── Energy reversal ──────────────────────────────────────────────────
    row["RA_REVERSAL_RATIO"] = round(compute_ra_reversal_ratio(interval_df), 4)

    # ── Voltage imbalance ────────────────────────────────────────────────
    row["VOLTAGE_IMBALANCE_MAX"] = round(
        compute_voltage_imbalance(interval_df, instantaneous_df), 4
    )

    # ── Current imbalance ────────────────────────────────────────────────
    row["CURRENT_IMBALANCE_MAX"] = round(
        compute_current_imbalance(interval_df, instantaneous_df), 4
    )

    # ── Robust statistics (median, IQR, p05, p95) ───────────────────────
    for prefix, source in [("FA_INTERVAL", interval_df), ("U_L1", interval_df)]:
        if prefix in source.columns:
            vals = source[prefix].drop_nulls()
            if vals.len() > 0:
                row[f"{prefix}_MEDIAN"] = round(float(vals.median()), 4)
                q25 = float(vals.quantile(0.25))
                q75 = float(vals.quantile(0.75))
                row[f"{prefix}_IQR"] = round(q75 - q25, 4)
                row[f"{prefix}_P05"] = round(float(vals.quantile(0.05)), 4)
                row[f"{prefix}_P95"] = round(float(vals.quantile(0.95)), 4)
            else:
                for suffix in ("_MEDIAN", "_IQR", "_P05", "_P95"):
                    row[f"{prefix}{suffix}"] = 0.0
        else:
            for suffix in ("_MEDIAN", "_IQR", "_P05", "_P95"):
                row[f"{prefix}{suffix}"] = 0.0

    # ── Variations & ramps ───────────────────────────────────────────────
    row["FA_RAMP_MAX"] = round(compute_energy_ramp(interval_df, "FA_INTERVAL"), 4)
    row["U_L1_RAMP_MAX"] = round(compute_energy_ramp(interval_df, "U_L1_AVG"), 4)

    # ── Peak hour encoded cyclically ─────────────────────────────────────
    peak_hour = _detect_peak_hour(interval_df)
    import math
    row["PEAK_HOUR_SIN"] = round(math.sin(2 * math.pi * peak_hour / 24), 4)
    row["PEAK_HOUR_COS"] = round(math.cos(2 * math.pi * peak_hour / 24), 4)

    # ── Temporal variability ─────────────────────────────────────────────
    row["FA_INTERVAL_AUTOCORR_LAG1"] = round(
        _autocorr_lag1(interval_df, "FA_INTERVAL"), 4
    )
    if "FA_INTERVAL" in interval_df.columns:
        fa_vals = interval_df["FA_INTERVAL"].drop_nulls()
        row["FA_INTERVAL_STD"] = round(float(fa_vals.std()), 4) if fa_vals.len() > 1 else 0.0
    else:
        row["FA_INTERVAL_STD"] = 0.0

    # ── Context (defaults if no metadata) ────────────────────────────────
    for col in ("PHASE_TYPE", "CONSUMER_CLASS", "METER_TYPE"):
        if col not in row:
            row[col] = None
    if "INSTALLED_KVA" not in row:
        row["INSTALLED_KVA"] = None

    return row


def compute_coverage_from_df(df: pl.DataFrame, column: str) -> float:
    """Compute coverage ratio for a column in a DataFrame.

    Coverage = non-null points / total expected points (288 for 5-min).
    """
    if df.is_empty() or column not in df.columns:
        return 0.0

    total = df.height
    non_null = df[column].null_count()
    observed_valid = total - non_null

    # Use 288 as default expected points (5-min cadence)
    expected = 288
    if "CADENCE_MINUTES" in df.columns:
        cadence = df["CADENCE_MINUTES"].drop_nulls()
        if cadence.len() > 0:
            predominant = int(cadence.mode()[0]) if cadence.mode().len() > 0 else 5
            expected = 1440 // predominant

    return min(observed_valid / expected, 1.0) if expected > 0 else 0.0


def _detect_peak_hour(df: pl.DataFrame) -> int:
    """Detect the hour with maximum FA_INTERVAL value."""
    if "FA_INTERVAL" not in df.columns or "TIMESTAMP_UTC" not in df.columns:
        return 12  # default to noon

    try:
        temp = df.select([
            pl.col("FA_INTERVAL"),
            pl.col("TIMESTAMP_UTC").dt.hour().alias("hour"),
        ]).drop_nulls()

        if temp.is_empty():
            return 12

        hourly = temp.group_by("hour").agg(pl.col("FA_INTERVAL").mean())
        peak_row = hourly.sort("FA_INTERVAL", descending=True).head(1)
        return int(peak_row["hour"][0])
    except Exception:
        return 12


def _autocorr_lag1(df: pl.DataFrame, column: str) -> float:
    """Compute lag-1 autocorrelation for a column."""
    if column not in df.columns:
        return 0.0

    vals = df[column].drop_nulls().to_list()
    if len(vals) < 3:
        return 0.0

    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    if var == 0:
        return 0.0

    cov = sum((vals[i] - mean) * (vals[i + 1] - mean) for i in range(n - 1)) / (n - 1)
    return cov / var


def _empty_features_frame() -> pl.DataFrame:
    """Return an empty features DataFrame with the correct schema."""
    from src.datasets.schemas import METER_DAY_FEATURES_SCHEMA
    return pl.DataFrame(schema=METER_DAY_FEATURES_SCHEMA)
