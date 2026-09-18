"""Hive-style partitioning helpers for normalized Parquet datasets.

Follows the semantic multi-layer architecture:

    data/
    ├── context/
    │   ├── uc_context.parquet
    │   ├── meter_installation_history.parquet
    │   └── electrical_hierarchy.parquet
    ├── measurements/
    │   ├── ami_interval/report_year=2026/report_month=06/report_day=23/
    │   ├── ami_instantaneous/report_year=2026/report_month=06/report_day=23/
    │   └── ami_registers/report_year=2026/report_month=06/report_day=23/
    ├── events/
    │   └── alarm_events/report_year=2026/report_month=06/report_day=23/
    ├── features/
    │   ├── uc_day_features/feature_set_version=v1/report_year=2026/report_month=06/
    │   └── uc_window_features/feature_set_version=v1/cutoff_year=2026/cutoff_month=06/
    ├── labels/
    │   └── uc_labels/label_version=v1/
    └── model_input/
        └── v1/
            └── training_dataset.parquet
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence


def context_path(
    table_name: str,
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build path for context tables (1 table file or partitioned directory)."""
    return Path(base_dir) / "context" / f"{table_name}.parquet"


def measurements_partition_path(
    table_name: str,
    report_day: date,
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build Hive-style partition directory for measurements."""
    return (
        Path(base_dir)
        / "measurements"
        / table_name
        / f"report_year={report_day.year:04d}"
        / f"report_month={report_day.month:02d}"
        / f"report_day={report_day.day:02d}"
    )


def events_partition_path(
    table_name: str,
    report_day: date,
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build Hive-style partition directory for events."""
    return (
        Path(base_dir)
        / "events"
        / table_name
        / f"report_year={report_day.year:04d}"
        / f"report_month={report_day.month:02d}"
        / f"report_day={report_day.day:02d}"
    )


def partition_path(
    table_name: str,
    report_day: date,
    *,
    base_dir: Path | str = "data/normalized",
    extra_partitions: dict[str, str] | None = None,
) -> Path:
    """Build a Hive-style partition directory for a table and report_day."""
    base = Path(base_dir)
    parts: list[str] = [table_name]

    if extra_partitions:
        for key, value in sorted(extra_partitions.items()):
            parts.append(f"{key}={value}")

    parts.append(f"report_year={report_day.year:04d}")
    parts.append(f"report_month={report_day.month:02d}")
    parts.append(f"report_day={report_day.day:02d}")

    return base.joinpath(*parts)


def feature_partition_path(
    feature_set_version: str,
    report_day: date,
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build a partition path for the daily features table."""
    return (
        Path(base_dir)
        / "features"
        / "uc_day_features"
        / f"feature_set_version={feature_set_version}"
        / f"report_year={report_day.year:04d}"
        / f"report_month={report_day.month:02d}"
        / f"report_day={report_day.day:02d}"
    )


def window_feature_partition_path(
    feature_set_version: str,
    cutoff_date: date,
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build a partition path for the windowed features table."""
    return (
        Path(base_dir)
        / "features"
        / "uc_window_features"
        / f"feature_set_version={feature_set_version}"
        / f"cutoff_year={cutoff_date.year:04d}"
        / f"cutoff_month={cutoff_date.month:02d}"
        / f"cutoff_day={cutoff_date.day:02d}"
    )


def label_partition_path(
    label_version: str,
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build a partition path for the labels table."""
    return Path(base_dir) / "labels" / f"label_version={label_version}"


def model_input_path(
    dataset_version: str = "v1",
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build the path for the training dataset."""
    return Path(base_dir) / "model_input" / dataset_version / "training_dataset.parquet"


def manifest_path(
    run_id: str,
    *,
    base_dir: Path | str = "data",
) -> Path:
    """Build the path for a run manifest file."""
    return Path(base_dir) / "manifests" / f"run_id={run_id}.json"


def ensure_partition_dirs(path: Path) -> None:
    """Create partition directories (including parents) if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)


def discover_partitions(
    table_name: str,
    *,
    base_dir: Path | str = "data/normalized",
) -> list[Path]:
    """Discover all existing partition directories for a table."""
    table_dir = Path(base_dir) / table_name
    if not table_dir.exists():
        return []

    partitions: list[Path] = []
    for year_dir in sorted(table_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith("report_year="):
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.startswith("report_month="):
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir() or not day_dir.name.startswith("report_day="):
                    continue
                partitions.append(day_dir)

    return partitions


def discover_partition_dates(
    table_name: str,
    *,
    base_dir: Path | str = "data/normalized",
) -> list[date]:
    """Discover all dates that have data for a table."""
    dates: list[date] = []
    for partition_dir in discover_partitions(table_name, base_dir=base_dir):
        d = _date_from_partition(partition_dir)
        if d is not None:
            dates.append(d)
    return sorted(dates)


def _date_from_partition(path: Path) -> date | None:
    """Extract a date from a Hive-style partition path."""
    parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in path.parts if "=" in part}

    year_str = parts.get("report_year") or parts.get("cutoff_year")
    month_str = parts.get("report_month") or parts.get("cutoff_month")
    day_str = parts.get("report_day") or parts.get("cutoff_day")

    if not all([year_str, month_str, day_str]):
        return None

    try:
        return date(int(year_str), int(month_str), int(day_str))
    except (ValueError, TypeError):
        return None
