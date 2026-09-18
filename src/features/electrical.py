"""Electrical domain feature computations.

Physical feature calculations for smart-meter data, following
standard electrical engineering definitions.
"""

from __future__ import annotations

import polars as pl


def compute_voltage_imbalance(
    interval_df: pl.DataFrame,
    instantaneous_df: pl.DataFrame,
) -> float:
    """Compute maximum voltage imbalance across phases.

    Voltage imbalance is defined as the maximum deviation from the
    average across L1, L2, L3, divided by the average.

    Uses instantaneous voltage if available, falls back to interval averages.

    Returns:
        Maximum imbalance ratio (0.0 to 1.0+).
    """
    source = instantaneous_df if instantaneous_df.height > 0 else interval_df

    if source.is_empty():
        return 0.0

    u_cols = [c for c in ("U_L1", "U_L2", "U_L3") if c in source.columns]
    if not u_cols:
        u_cols = [c for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG") if c in source.columns]

    if len(u_cols) < 2:
        return 0.0

    try:
        temp = source.select(u_cols).drop_nulls()
        if temp.is_empty():
            return 0.0

        max_imbalance = 0.0
        for i in range(len(u_cols)):
            for j in range(i + 1, len(u_cols)):
                v1 = temp[u_cols[i]].mean()
                v2 = temp[u_cols[j]].mean()
                avg = (v1 + v2) / 2
                if avg > 0:
                    imbalance = abs(v1 - v2) / avg
                    max_imbalance = max(max_imbalance, imbalance)

        return max_imbalance
    except Exception:
        return 0.0


def compute_current_imbalance(
    interval_df: pl.DataFrame,
    instantaneous_df: pl.DataFrame,
) -> float:
    """Compute maximum current imbalance across phases.

    Similar to voltage imbalance but for current measurements.

    Returns:
        Maximum imbalance ratio (0.0 to 1.0+).
    """
    source = instantaneous_df if instantaneous_df.height > 0 else interval_df

    if source.is_empty():
        return 0.0

    i_cols = [c for c in ("I_INSTANT_L1", "I_INSTANT_L2", "I_INSTANT_L3") if c in source.columns]
    if not i_cols:
        i_cols = [c for c in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG") if c in source.columns]

    if len(i_cols) < 2:
        return 0.0

    try:
        temp = source.select(i_cols).drop_nulls()
        if temp.is_empty():
            return 0.0

        max_imbalance = 0.0
        for i in range(len(i_cols)):
            for j in range(i + 1, len(i_cols)):
                v1 = temp[i_cols[i]].mean()
                v2 = temp[i_cols[j]].mean()
                avg = (v1 + v2) / 2
                if avg > 0:
                    imbalance = abs(v1 - v2) / avg
                    max_imbalance = max(max_imbalance, imbalance)

        return max_imbalance
    except Exception:
        return 0.0


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
