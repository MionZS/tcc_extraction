"""Temporal Regularization and Communication Gap Analytics for AMI Smart Meters.

Constructs explicit daily temporal grids (288 intervals for 5-min cadence or 96 for 15-min).
Extracts fine-grained missingness features for the Communication Anomaly Model ($S_{comm}$):
- Completeness ratio ($C_{1d}$)
- Missing runs / gap counts
- Maximum, mean, and P95 gap duration in minutes
- Consecutive missing interval peak
- Nocturnal vs. diurnal gap distributions (periodic modem resets vs. random telemetry loss)

Enforces the strict physical interpolation policy for the Electrical Anomaly Model ($S_{elec}$):
- Micro-gaps (<= 3 intervals / <= 15 minutes): Linearly interpolated for continuity.
- Macro-gaps (> 3 intervals / > 15 minutes): Maintained strictly as NaN. Never invent data.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Sequence
import numpy as np
import polars as pl


def build_daily_time_grid(
    report_day: date,
    cadence_minutes: int = 5,
) -> pl.DataFrame:
    """Generate a complete, regularized temporal grid for a single 24-hour day.

    Args:
        report_day: Reference calendar date.
        cadence_minutes: Expected interval duration (default 5 min = 288 intervals).

    Returns:
        DataFrame with TIMESTAMP_UTC spanning 00:00 to 23:55 (or 23:45).
    """
    start_dt = datetime.combine(report_day, time.min)
    end_dt = datetime.combine(report_day, time(23, 59, 59))
    expected_intervals = 1440 // cadence_minutes

    timestamps = [start_dt + timedelta(minutes=i * cadence_minutes) for i in range(expected_intervals)]
    return pl.DataFrame({
        "TIMESTAMP_UTC": timestamps,
        "INTERVAL_INDEX": list(range(expected_intervals)),
        "INTERVAL_HOUR": [t.hour for t in timestamps],
    })


def regularize_interval_telemetry(
    interval_df: pl.DataFrame,
    report_day: date,
    *,
    cadence_minutes: int = 5,
    max_interp_slots: int = 3,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Regularize interval series against full daily grid, extracting gap features and micro-interpolating.

    Args:
        interval_df: Raw interval observations for a single meter on report_day.
        report_day: The target day.
        cadence_minutes: Expected cadence in minutes (5 or 15).
        max_interp_slots: Maximum consecutive missing slots permitted for linear interpolation.

    Returns:
        tuple of:
        - regularized_df: Grid-aligned DataFrame with mask $m_t$ and physical features micro-interpolated.
        - comm_metrics: Dict with rich missingness metrics for Communication Anomaly Detection.
    """
    grid_df = build_daily_time_grid(report_day, cadence_minutes=cadence_minutes)
    expected_total = grid_df.height

    if interval_df.is_empty() or "TIMESTAMP_UTC" not in interval_df.columns:
        # 100% missing day
        empty_res = grid_df.with_columns([
            pl.lit(0).alias("m_t"),  # 0 = missing, 1 = observed
            pl.lit(None, dtype=pl.Float64).alias("FA_INTERVAL"),
        ])
        metrics = {
            "completude_1d": 0.0,
            "gap_count": 1,
            "maior_gap_min": expected_total * cadence_minutes,
            "media_gap_min": float(expected_total * cadence_minutes),
            "p95_gap_min": float(expected_total * cadence_minutes),
            "intervalos_consecutivos_ausentes": expected_total,
            "nocturnal_missing_ratio": 1.0,
            "gap_hour_mode": 12,
        }
        return empty_res, metrics

    # Sort and join against full time grid
    joined = grid_df.join(
        interval_df.select([c for c in interval_df.columns if c in ("TIMESTAMP_UTC", "FA_INTERVAL", "RA_INTERVAL", "U_L1_AVG", "U_L1", "CADENCE_MINUTES")]),
        on="TIMESTAMP_UTC",
        how="left",
    )

    # Missingness indicator: 1 if FA_INTERVAL is present and non-null, else 0
    val_col = "FA_INTERVAL" if "FA_INTERVAL" in joined.columns else joined.columns[3]
    m_t = (~joined[val_col].is_null()).cast(pl.Int8).to_numpy()
    joined = joined.with_columns(pl.Series("m_t", m_t))

    observed_count = int(np.sum(m_t))
    completude_1d = observed_count / expected_total if expected_total > 0 else 0.0

    # ── Gap analysis (Runs of zeros) ──────────────────────────────────────────
    gap_lengths: list[int] = []
    gap_hours: list[int] = []
    current_run = 0
    current_start_hour = 0
    hours = joined["INTERVAL_HOUR"].to_numpy()

    for idx, is_present in enumerate(m_t):
        if is_present == 0:
            if current_run == 0:
                current_start_hour = hours[idx]
            current_run += 1
        else:
            if current_run > 0:
                gap_lengths.append(current_run)
                gap_hours.append(current_start_hour)
                current_run = 0
    if current_run > 0:
        gap_lengths.append(current_run)
        gap_hours.append(current_start_hour)

    if gap_lengths:
        gap_durations_min = [g * cadence_minutes for g in gap_lengths]
        gap_count = len(gap_lengths)
        maior_gap_min = int(max(gap_durations_min))
        media_gap_min = float(np.mean(gap_durations_min))
        p95_gap_min = float(np.percentile(gap_durations_min, 95))
        consecutive_missing = int(max(gap_lengths))

        # Check nocturnal missing (hours 0 to 5 inclusive)
        nocturnal_gaps = sum(1 for h in gap_hours if 0 <= h < 6)
        nocturnal_missing_ratio = round(nocturnal_gaps / gap_count, 3)

        # Gap hour mode
        vals, counts = np.unique(gap_hours, return_counts=True)
        gap_hour_mode = int(vals[np.argmax(counts)])
    else:
        gap_count = 0
        maior_gap_min = 0
        media_gap_min = 0.0
        p95_gap_min = 0.0
        consecutive_missing = 0
        nocturnal_missing_ratio = 0.0
        gap_hour_mode = 0

    comm_metrics = {
        "completude_1d": round(completude_1d, 4),
        "gap_count": gap_count,
        "maior_gap_min": maior_gap_min,
        "media_gap_min": round(media_gap_min, 2),
        "p95_gap_min": round(p95_gap_min, 2),
        "intervalos_consecutivos_ausentes": consecutive_missing,
        "nocturnal_missing_ratio": nocturnal_missing_ratio,
        "gap_hour_mode": gap_hour_mode,
    }

    # ── Micro-Interpolation Policy for Physical Channels ──────────────────────
    # Interpolate only short runs <= max_interp_slots.
    # Runs > max_interp_slots remain NaN.
    physical_cols = [c for c in ("FA_INTERVAL", "RA_INTERVAL", "U_L1_AVG", "U_L1") if c in joined.columns]
    interp_exprs = []

    for col in physical_cols:
        arr = joined[col].to_numpy().astype(float)
        nan_mask = np.isnan(arr)
        if not np.any(nan_mask) or np.all(nan_mask):
            continue

        # Identify run lengths of NaNs
        interpolated_arr = arr.copy()
        n = len(arr)
        run_start = None

        for i in range(n):
            if nan_mask[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_len = i - run_start
                    if run_len <= max_interp_slots and run_start > 0:
                        # Linear interpolation between run_start - 1 and i
                        val_before = arr[run_start - 1]
                        val_after = arr[i]
                        for step_k in range(run_len):
                            weight = (step_k + 1) / (run_len + 1)
                            interpolated_arr[run_start + step_k] = val_before + weight * (val_after - val_before)
                    run_start = None

        # Boundary at end of day
        if run_start is not None:
            # Leave trailing NaNs intact
            pass

        interp_exprs.append(pl.Series(col, interpolated_arr))

    if interp_exprs:
        joined = joined.with_columns(interp_exprs)

    return joined, comm_metrics
