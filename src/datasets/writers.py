"""Parquet/CSV writers with manifest integration.

Writes context, measurements, events, features, and model input tables
following the multi-layer architecture.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from src.datasets.partitioning import (
    context_path,
    ensure_partition_dirs,
    events_partition_path,
    feature_partition_path,
    label_partition_path,
    measurements_partition_path,
    model_input_path,
    partition_path,
    window_feature_partition_path,
)


def write_context_table(
    df: pl.DataFrame,
    table_name: str,
    *,
    base_dir: Path | str = "data",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write context table (e.g. uc_context, meter_installation_history, electrical_hierarchy)."""
    if df.is_empty():
        return Path()

    out_file = context_path(table_name, base_dir=base_dir)
    ensure_partition_dirs(out_file.parent)

    df.write_parquet(out_file, compression=compression)

    if write_csv:
        csv_path = out_file.with_suffix(".csv")
        df.write_csv(csv_path, separator=";")

    return out_file


def write_measurements_table(
    df: pl.DataFrame,
    table_name: str,
    report_day: date,
    *,
    base_dir: Path | str = "data",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write measurements table (ami_interval, ami_instantaneous, ami_registers)."""
    if df.is_empty():
        return Path()

    out_dir = measurements_partition_path(table_name, report_day, base_dir=base_dir)
    ensure_partition_dirs(out_dir)

    parquet_path = out_dir / "data.parquet"
    df.write_parquet(parquet_path, compression=compression)

    if write_csv:
        csv_path = out_dir / "data.csv"
        df.write_csv(csv_path, separator=";")

    return parquet_path


def write_events_table(
    df: pl.DataFrame,
    table_name: str,
    report_day: date,
    *,
    base_dir: Path | str = "data",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write events table (alarm_events)."""
    if df.is_empty():
        return Path()

    out_dir = events_partition_path(table_name, report_day, base_dir=base_dir)
    ensure_partition_dirs(out_dir)

    parquet_path = out_dir / "data.parquet"
    df.write_parquet(parquet_path, compression=compression)

    if write_csv:
        csv_path = out_dir / "data.csv"
        df.write_csv(csv_path, separator=";")

    return parquet_path


def write_normalized_table(
    df: pl.DataFrame,
    table_name: str,
    report_day: date,
    *,
    base_dir: Path | str = "data/normalized",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write a normalized DataFrame to a Hive-style partitioned Parquet file."""
    if df.is_empty():
        return Path()

    out_dir = partition_path(table_name, report_day, base_dir=base_dir)
    ensure_partition_dirs(out_dir)

    parquet_path = out_dir / "data.parquet"
    df.write_parquet(parquet_path, compression=compression)

    if write_csv:
        csv_path = out_dir / "data.csv"
        df.write_csv(csv_path, separator=";")

    return parquet_path


def write_normalized_batch(
    tables: dict[str, pl.DataFrame],
    report_day: date,
    *,
    base_dir: Path | str = "data/normalized",
    write_csv: bool = False,
    compression: str = "zstd",
) -> dict[str, Path]:
    """Write multiple normalized tables for a single report_day."""
    paths: dict[str, Path] = {}
    for table_name, df in tables.items():
        paths[table_name] = write_normalized_table(
            df,
            table_name,
            report_day,
            base_dir=base_dir,
            write_csv=write_csv,
            compression=compression,
        )
    return paths


def write_features(
    df: pl.DataFrame,
    feature_set_version: str,
    report_day: date,
    *,
    base_dir: Path | str = "data",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write daily feature DataFrame to partitioned Parquet."""
    if df.is_empty():
        return Path()

    out_dir = feature_partition_path(feature_set_version, report_day, base_dir=base_dir)
    ensure_partition_dirs(out_dir)

    parquet_path = out_dir / "data.parquet"
    df.write_parquet(parquet_path, compression=compression)

    if write_csv:
        csv_path = out_dir / "data.csv"
        df.write_csv(csv_path, separator=";")

    return parquet_path


def write_window_features(
    df: pl.DataFrame,
    feature_set_version: str,
    cutoff_date: date,
    *,
    base_dir: Path | str = "data",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write window feature DataFrame to partitioned Parquet."""
    if df.is_empty():
        return Path()

    out_dir = window_feature_partition_path(feature_set_version, cutoff_date, base_dir=base_dir)
    ensure_partition_dirs(out_dir)

    parquet_path = out_dir / "data.parquet"
    df.write_parquet(parquet_path, compression=compression)

    if write_csv:
        csv_path = out_dir / "data.csv"
        df.write_csv(csv_path, separator=";")

    return parquet_path


def write_training_dataset(
    df: pl.DataFrame,
    dataset_version: str = "v1",
    *,
    base_dir: Path | str = "data",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write final model input dataset (UC × cutoff_date)."""
    if df.is_empty():
        return Path()

    out_path = model_input_path(dataset_version, base_dir=base_dir)
    ensure_partition_dirs(out_path.parent)

    df.write_parquet(out_path, compression=compression)

    if write_csv:
        csv_path = out_path.with_suffix(".csv")
        df.write_csv(csv_path, separator=";")

    return out_path


def write_labels(
    df: pl.DataFrame,
    label_version: str,
    *,
    base_dir: Path | str = "data",
    write_csv: bool = False,
    compression: str = "zstd",
) -> Path:
    """Write labels DataFrame to partitioned Parquet."""
    if df.is_empty():
        return Path()

    out_dir = label_partition_path(label_version, base_dir=base_dir)
    ensure_partition_dirs(out_dir)

    parquet_path = out_dir / "data.parquet"
    df.write_parquet(parquet_path, compression=compression)

    if write_csv:
        csv_path = out_dir / "data.csv"
        df.write_csv(csv_path, separator=";")

    return parquet_path


def compute_file_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_dataframe_stats(df: pl.DataFrame) -> dict[str, Any]:
    """Compute quality statistics for a DataFrame."""
    stats: dict[str, Any] = {
        "row_count": df.height,
        "column_count": df.width,
    }

    null_counts = {}
    for col in df.columns:
        nc = df[col].null_count()
        if nc > 0:
            null_counts[col] = nc
    if null_counts:
        stats["null_counts"] = null_counts

    for ts_col in ("TIMESTAMP_UTC", "TIMESTAMP_LOCAL", "REPORT_DAY", "CUTOFF_DATE"):
        if ts_col in df.columns:
            try:
                min_val = df[ts_col].min()
                max_val = df[ts_col].max()
                if min_val is not None:
                    stats[f"min_{ts_col.lower()}"] = str(min_val)
                    stats[f"max_{ts_col.lower()}"] = str(max_val)
            except Exception:
                pass

    if "NIO" in df.columns:
        stats["unique_nio"] = df["NIO"].n_unique()
    if "UC" in df.columns:
        stats["unique_uc"] = df["UC"].n_unique()

    return stats
