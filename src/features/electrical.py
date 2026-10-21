"""Electrical domain feature computations.

Physical feature calculations for smart-meter data, following
standard electrical engineering definitions.
"""

from __future__ import annotations

from typing import Any
import polars as pl


class ImbalanceResult(float):
    """Float subclass that unpacks as (max_imb, mean_imb) for backward test compatibility."""

    def __new__(cls, max_val: float, mean_val: float | None = None):
        obj = super().__new__(cls, max_val)
        obj.max_val = max_val
        obj.mean_val = mean_val if mean_val is not None else max_val
        return obj

    def __iter__(self):
        yield self.max_val
        yield self.mean_val


def compute_voltage_imbalance(
    interval_df: pl.DataFrame,
    instantaneous_df: pl.DataFrame | None = None,
) -> Any:
    """Compute maximum dynamic voltage imbalance across active phases.

    Follows PRODIST Módulo 8 / IEEE 1159 dynamic instantaneous calculation:
        V_avg(t) = (1/K) * sum(V_k(t))
        imbalance(t) = max_k |V_k(t) - V_avg(t)| / V_avg(t) * 100.0 (percentage)
    Returns ImbalanceResult (acting as float of max_imb and unpackable as (max_imb, mean_imb)),
    or (None, None) if single-phase or empty.
    """
    if instantaneous_df is not None and instantaneous_df.height > 0:
        source = instantaneous_df
    else:
        source = interval_df

    if source.is_empty():
        return (None, None)

    candidate_cols = [c for c in ("U_L1", "U_L2", "U_L3") if c in source.columns]
    if not candidate_cols:
        candidate_cols = [c for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG") if c in source.columns]

    # Filter to active phases that actually contain measurements for this meter
    active_cols = [c for c in candidate_cols if source[c].drop_nulls().len() > 0]
    if len(active_cols) < 2:
        return (None, None)

    try:
        valid_rows = source.select(active_cols).drop_nulls()
        if valid_rows.is_empty():
            return (None, None)

        # Vectorized dynamic calculation at each timestamp (in percentage %)
        avg_expr = sum(pl.col(c) for c in active_cols) / len(active_cols)
        max_dev_expr = pl.max_horizontal([(pl.col(c) - avg_expr).abs() for c in active_cols])
        imb_expr = pl.when(avg_expr > 0).then((max_dev_expr / avg_expr) * 100.0).otherwise(0.0)

        res_df = valid_rows.select([imb_expr.max().alias("max_imb"), imb_expr.mean().alias("mean_imb")])
        max_val = res_df["max_imb"][0]
        mean_val = res_df["mean_imb"][0]
        if max_val is None:
            return (None, None)
        return ImbalanceResult(float(max_val), float(mean_val) if mean_val is not None else float(max_val))
    except Exception:
        return (None, None)


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
