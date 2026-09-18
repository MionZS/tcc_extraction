"""Data quality feature computations.

Functions for computing coverage, null ratios, and other
quality-related features from normalized datasets.
"""

from __future__ import annotations

import polars as pl


def compute_coverage(
    df: pl.DataFrame,
    column: str,
    *,
    expected_points: int | None = None,
    cadence_column: str = "CADENCE_MINUTES",
) -> float:
    """Compute coverage ratio for a column.

    Coverage = observed_valid_points / expected_points.

    If ``expected_points`` is not provided, it is inferred from
    the predominant cadence in the DataFrame.

    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if df.is_empty() or column not in df.columns:
        return 0.0

    total = df.height
    non_null = df[column].null_count()
    observed_valid = total - non_null

    if expected_points is None:
        if cadence_column in df.columns:
            cadences = df[cadence_column].drop_nulls()
            if cadences.len() > 0:
                predominant = int(cadences.mode()[0]) if cadences.mode().len() > 0 else 5
                expected_points = 1440 // predominant
            else:
                expected_points = 288  # default for 5-min cadence
        else:
            expected_points = 288

    if expected_points <= 0:
        return 0.0

    return min(observed_valid / expected_points, 1.0)


def compute_null_ratio(df: pl.DataFrame, column: str) -> float:
    """Compute the ratio of null values to total rows.

    Returns:
        Null ratio between 0.0 (no nulls) and 1.0 (all nulls).
    """
    if df.is_empty() or column not in df.columns:
        return 1.0

    total = df.height
    if total == 0:
        return 1.0

    null_count = df[column].null_count()
    return null_count / total


def compute_quality_score(interval_df: pl.DataFrame) -> float:
    """Compute a composite quality score for a meter's interval data.

    The score combines:
    - Coverage ratio (weight: 0.4)
    - Non-null ratio of FA_INTERVAL (weight: 0.3)
    - Physical range compliance (weight: 0.3)

    Returns:
        Quality score between 0.0 and 1.0.
    """
    if interval_df.is_empty():
        return 0.0

    # Coverage
    coverage = compute_coverage(interval_df, "FA_INTERVAL")

    # Null ratio (inverted: 1 - null_ratio)
    null_ratio = compute_null_ratio(interval_df, "FA_INTERVAL")
    completeness = 1.0 - null_ratio

    # Physical range compliance
    compliance = _compute_physical_compliance(interval_df)

    return round(0.4 * coverage + 0.3 * completeness + 0.3 * compliance, 4)


def _compute_physical_compliance(df: pl.DataFrame) -> float:
    """Check if values fall within physically plausible ranges.

    Checks:
    - FA_INTERVAL: 0 to 100 kWh (typical 5-min slot)
    - U_L1_AVG: 0 to 500 V
    - I_L1_AVG: 0 to 1000 A
    """
    checks: list[bool] = []

    if "FA_INTERVAL" in df.columns:
        vals = df["FA_INTERVAL"].drop_nulls()
        if vals.len() > 0:
            in_range = ((vals >= 0) & (vals <= 100)).all()
            checks.append(bool(in_range))

    for col, min_val, max_val in [
        ("U_L1_AVG", 0, 500),
        ("U_L2_AVG", 0, 500),
        ("U_L3_AVG", 0, 500),
    ]:
        if col in df.columns:
            vals = df[col].drop_nulls()
            if vals.len() > 0:
                in_range = ((vals >= min_val) & (vals <= max_val)).all()
                checks.append(bool(in_range))

    for col, min_val, max_val in [
        ("I_L1_AVG", 0, 1000),
        ("I_L2_AVG", 0, 1000),
        ("I_L3_AVG", 0, 1000),
    ]:
        if col in df.columns:
            vals = df[col].drop_nulls()
            if vals.len() > 0:
                in_range = ((vals >= min_val) & (vals <= max_val)).all()
                checks.append(bool(in_range))

    if not checks:
        return 1.0  # no data to check → assume compliant

    return sum(checks) / len(checks)
