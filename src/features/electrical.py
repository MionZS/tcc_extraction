"""Electrical domain feature computations.

Physical feature calculations for smart-meter data, following
standard electrical engineering definitions.
"""

from __future__ import annotations

import polars as pl


def compute_voltage_imbalance(
    interval_df: pl.DataFrame,
    instantaneous_df: pl.DataFrame,
) -> float | None:
    """Compute maximum dynamic voltage imbalance across active phases.

    Follows standard electrical engineering standards (PRODIST Módulo 8 / IEEE 1159):
    Computes instantaneous imbalance at each timestamp:
        V_avg(t) = (1/K) * sum(V_k(t))
        imbalance(t) = max_k |V_k(t) - V_avg(t)| / V_avg(t)
    Returns max(imbalance(t)) across the day.

    For single-phase installations (only 1 active phase), inter-phase imbalance
    is physically undefined, so None is returned.
    For two-phase (bifásico), computes imbalance between the 2 active phases without
    dropping rows due to missing third phase.

    Returns:
        Maximum imbalance ratio (0.0 to 1.0+), or None if single-phase or empty.
    """
    source = instantaneous_df if instantaneous_df.height > 0 else interval_df
    if source.is_empty():
        return None

    candidate_cols = [c for c in ("U_L1", "U_L2", "U_L3") if c in source.columns]
    if not candidate_cols:
        candidate_cols = [c for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG") if c in source.columns]

    # Filter to active phases that actually contain measurements for this meter
    active_cols = [c for c in candidate_cols if source[c].drop_nulls().len() > 0]
    if len(active_cols) < 2:
        return None

    try:
        valid_rows = source.select(active_cols).drop_nulls()
        if valid_rows.is_empty():
            return None

        # Vectorized dynamic calculation at each timestamp
        avg_expr = sum(pl.col(c) for c in active_cols) / len(active_cols)
        max_dev_expr = pl.max_horizontal([(pl.col(c) - avg_expr).abs() for c in active_cols])
        imb_expr = pl.when(avg_expr > 0).then(max_dev_expr / avg_expr).otherwise(0.0)

        max_imb = valid_rows.select(imb_expr.max()).item()
        return float(max_imb) if max_imb is not None else None
    except Exception:
        return None


def compute_current_imbalance(
    interval_df: pl.DataFrame,
    instantaneous_df: pl.DataFrame,
) -> float | None:
    """Compute maximum dynamic current imbalance across active phases.

    Follows PRODIST Módulo 8 / IEEE Std 1159 dynamic instantaneous calculation.
    Returns None for single-phase installations.

    Returns:
        Maximum imbalance ratio (0.0 to 1.0+), or None if single-phase or empty.
    """
    source = instantaneous_df if instantaneous_df.height > 0 else interval_df
    if source.is_empty():
        return None

    candidate_cols = [c for c in ("I_INSTANT_L1", "I_INSTANT_L2", "I_INSTANT_L3") if c in source.columns]
    if not candidate_cols:
        candidate_cols = [c for c in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG") if c in source.columns]

    active_cols = [c for c in candidate_cols if source[c].drop_nulls().len() > 0]
    if len(active_cols) < 2:
        return None

    try:
        valid_rows = source.select(active_cols).drop_nulls()
        if valid_rows.is_empty():
            return None

        avg_expr = sum(pl.col(c) for c in active_cols) / len(active_cols)
        max_dev_expr = pl.max_horizontal([(pl.col(c) - avg_expr).abs() for c in active_cols])
        imb_expr = pl.when(avg_expr > 0).then(max_dev_expr / avg_expr).otherwise(0.0)

        max_imb = valid_rows.select(imb_expr.max()).item()
        return float(max_imb) if max_imb is not None else None
    except Exception:
        return None


def compute_load_factor(interval_df: pl.DataFrame) -> float:
    """Compute load factor from interval energy measurements.

    Load factor = average demand / peak demand.
    Here we use FA_INTERVAL as a proxy for demand.

    Returns:
        Load factor between 0.0 and 1.0.
    """
    if interval_df.is_empty() or "FA_INTERVAL" not in interval_df.columns:
        return 0.0

    try:
        fa_vals = interval_df["FA_INTERVAL"].drop_nulls()
        if fa_vals.len() == 0:
            return 0.0

        mean_val = fa_vals.mean()
        max_val = fa_vals.max()

        if max_val is None or max_val == 0:
            return 0.0

        return float(mean_val / max_val) if mean_val is not None else 0.0
    except Exception:
        return 0.0


def compute_ra_reversal_ratio(interval_df: pl.DataFrame) -> float:
    """Compute the ratio of negative RA (reverse energy) to total RA.

    RA reversal indicates energy flowing back to the grid.
    A high ratio may indicate generation or measurement issues.

    Returns:
        Ratio of negative RA values to total non-null RA values (0.0 to 1.0).
    """
    if interval_df.is_empty() or "RA_INTERVAL" not in interval_df.columns:
        return 0.0

    try:
        ra_vals = interval_df["RA_INTERVAL"].drop_nulls()
        if ra_vals.len() == 0:
            return 0.0

        negative_count = (ra_vals < 0).sum()
        return float(negative_count / ra_vals.len())
    except Exception:
        return 0.0


def compute_energy_ramp(df: pl.DataFrame, column: str) -> float:
    """Compute maximum absolute ramp (difference between consecutive values).

    Useful for detecting sudden changes in energy consumption or voltage.

    Returns:
        Maximum absolute difference between consecutive non-null values.
    """
    if df.is_empty() or column not in df.columns:
        return 0.0

    try:
        vals = df[column].drop_nulls()
        if vals.len() < 2:
            return 0.0

        # Compute differences between consecutive values
        diff = vals.diff().abs()
        max_diff = diff.max()
        return float(max_diff) if max_diff is not None else 0.0
    except Exception:
        return 0.0
