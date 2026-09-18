"""Register reset detection for accumulated energy meters.

Register resets occur when accumulated energy values decrease
(negative difference between consecutive readings), which can
indicate meter replacement, rollover, or data quality issues.

This module provides production-ready detection functions used by
both the pipeline checks and the test suite.
"""

from __future__ import annotations

import polars as pl


def detect_register_resets(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Detect register resets in a DataFrame.

    A reset is defined as a negative difference between consecutive
    non-null values of the same column, grouped by NIO.

    Args:
        df: DataFrame with NIO, TIMESTAMP_UTC, and the target column.
        column: Name of the register column to check for resets.

    Returns:
        DataFrame with NIO, TIMESTAMP_UTC, and {column}_RESET_DELTA columns.
        Empty DataFrame with the same schema if no resets found.
    """
    if df.is_empty() or column not in df.columns:
        return pl.DataFrame(schema={
            "NIO": pl.String,
            "TIMESTAMP_UTC": pl.Datetime,
            f"{column}_RESET_DELTA": pl.Float64,
        })

    # Sort by NIO, then timestamp
    sorted_df = df.sort(["NIO", "TIMESTAMP_UTC"])

    # Compute differences between consecutive values per NIO
    result = sorted_df.with_columns([
        pl.col(column)
        .diff()
        .over("NIO")
        .alias(f"{column}_diff"),
    ])

    # Filter negative differences (resets)
    resets = result.filter(
        pl.col(f"{column}_diff").is_not_null() & (pl.col(f"{column}_diff") < 0)
    ).select([
        "NIO",
        "TIMESTAMP_UTC",
        pl.col(f"{column}_diff").abs().alias(f"{column}_RESET_DELTA"),
    ])

    return resets


def count_resets_per_nio(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Count register resets per NIO.

    Args:
        df: DataFrame with NIO, TIMESTAMP_UTC, and the target column.
        column: Name of the register column to check for resets.

    Returns:
        DataFrame with NIO and reset_count columns.
    """
    resets = detect_register_resets(df, column)
    if resets.is_empty():
        return pl.DataFrame(schema={"NIO": pl.String, "reset_count": pl.Int64})

    return (
        resets.group_by("NIO")
        .agg(pl.len().alias("reset_count"))
        .sort("reset_count", descending=True)
    )


def max_reset_delta_per_nio(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Get maximum reset delta per NIO.

    Args:
        df: DataFrame with NIO, TIMESTAMP_UTC, and the target column.
        column: Name of the register column to check for resets.

    Returns:
        DataFrame with NIO and max_reset_delta columns.
    """
    delta_col = f"{column}_RESET_DELTA"
    resets = detect_register_resets(df, column)
    if resets.is_empty():
        return pl.DataFrame(schema={"NIO": pl.String, "max_reset_delta": pl.Float64})

    return (
        resets.group_by("NIO")
        .agg(pl.col(delta_col).max().alias("max_reset_delta"))
        .sort("max_reset_delta", descending=True)
    )
