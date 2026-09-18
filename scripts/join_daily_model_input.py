"""Versioned builder of meter_day_features from normalized datasets.

Replaces the old ad-hoc CSV join script.  Now uses the new architecture:
1. Reads normalized Parquet datasets (or CSV fallback)
2. Computes features via src.features.meter_day.build_meter_day_features
3. Writes versioned Parquet + CSV output
4. Generates run manifest with per-table stats
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

from src.datasets.partitioning import (
    partition_path,
    feature_partition_path,
    ensure_partition_dirs,
)
from src.datasets.writers import (
    write_features,
    compute_dataframe_stats,
)
from src.features.meter_day import build_meter_day_features
from src.run_manifest import create_manifest


DEFAULT_DATA_DIR = Path("data")


def _guess_report_day(path: Path) -> date | None:
    """Try to extract a date from a path or filename."""
    match = re.search(r"(20\d{6})", path.stem)
    if match:
        try:
            return date(int(match.group(1)[:4]), int(match.group(1)[4:6]), int(match.group(1)[6:8]))
        except ValueError:
            pass
    return None


def build_features_from_datasets(
    *,
    report_day: date,
    interval_path: Path | None = None,
    instantaneous_path: Path | None = None,
    register_path: Path | None = None,
    metadata_path: Path | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    feature_set_version: str = "v1",
    output_dir: Path | None = None,
    write_csv: bool = True,
) -> Path:
    """Build versioned meter_day_features from normalized Parquet datasets.

    Args:
        report_day: Reference date for features.
        interval_path: Optional override for meter_interval parquet.
        instantaneous_path: Optional override for meter_instantaneous parquet.
        register_path: Optional override for meter_register_snapshot parquet.
        metadata_path: Optional override for meter_day_metadata parquet.
        data_dir: Root data directory (default: ``data/``).
        feature_set_version: Version tag for the feature set.
        output_dir: Override output directory.
        write_csv: Also write CSV companion file.

    Returns:
        Path to the written feature file.
    """
    import polars as pl

    manifest = create_manifest(report_day, sample_size=0, schema_version=feature_set_version)
    manifest.record_step("build_features", method="dataset", version=feature_set_version)

    # Locate data using partition convention
    def _resolve(path_override: Path | None, table: str) -> Path | None:
        if path_override is not None:
            return path_override if path_override.exists() else None
        p = partition_path(table, report_day, base_dir=data_dir / "normalized") / "data.parquet"
        return p if p.exists() else None

    interval_p = _resolve(interval_path, "meter_interval")
    instant_p = _resolve(instantaneous_path, "meter_instantaneous")
    register_p = _resolve(register_path, "meter_register_snapshot")
    meta_p = _resolve(metadata_path, "meter_day_metadata")

    if interval_p is None:
        raise FileNotFoundError(
            f"meter_interval not found for {report_day}. "
            "Run the pipeline first or provide --interval-path."
        )

    # Read normalized datasets
    interval_df = pl.read_parquet(interval_p)
    instant_df = pl.read_parquet(instant_p) if instant_p else pl.DataFrame()
    register_df = pl.read_parquet(register_p) if register_p else pl.DataFrame()
    meta_df = pl.read_parquet(meta_p) if meta_p else pl.DataFrame()

    manifest.record_table(
        "meter_interval", interval_df,
        file_path=str(interval_p),
    )
    if not instant_df.is_empty():
        manifest.record_table("meter_instantaneous", instant_df)
    if not register_df.is_empty():
        manifest.record_table("meter_register_snapshot", register_df)
    if not meta_df.is_empty():
        manifest.record_table("meter_day_metadata", meta_df)

    # Build features
    features_df = build_meter_day_features(
        interval_df=interval_df,
        instantaneous_df=instant_df,
        register_df=register_df,
        metadata_df=meta_df if not meta_df.is_empty() else None,
        report_day=report_day,
        feature_set_version=feature_set_version,
    )

    manifest.record_step(
        "features_computed",
        nio_count=features_df.height,
        feature_count=len(features_df.columns),
    )

    # Write output
    if output_dir is not None:
        # Custom output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / f"features_{feature_set_version}_{report_day:%Y%m%d}.parquet"
        features_df.write_parquet(parquet_path)
        if write_csv:
            csv_path = output_dir / f"features_{feature_set_version}_{report_day:%Y%m%d}.csv"
            features_df.write_csv(csv_path, separator=";")
    else:
        # Use standard partition layout
        parquet_path = write_features(
            features_df, feature_set_version, report_day,
            base_dir=data_dir,
            write_csv=write_csv,
        )

    manifest.record_step("features_written", path=str(parquet_path))
    manifest.record_table("meter_day_features", features_df, file_path=str(parquet_path))

    # Finalize manifest
    manifest.finalize()
    manifest_path = manifest.save()
    print(f"Manifest saved: {manifest_path}")
    print(f"Features written: {parquet_path}")
    print(f"NIOs     : {features_df.height:,}")
    print(f"Features : {len(features_df.columns)}")

    return parquet_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build versioned meter_day_features from normalized Parquet datasets.",
    )
    parser.add_argument(
        "--report-day",
        type=date.fromisoformat,
        required=True,
        help="Reference date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--interval-path",
        type=Path,
        default=None,
        help="Override path for meter_interval Parquet.",
    )
    parser.add_argument(
        "--instantaneous-path",
        type=Path,
        default=None,
        help="Override path for meter_instantaneous Parquet.",
    )
    parser.add_argument(
        "--register-path",
        type=Path,
        default=None,
        help="Override path for meter_register_snapshot Parquet.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Override path for meter_day_metadata Parquet.",
    )
    parser.add_argument(
        "--feature-version",
        default="v1",
        help="Feature set version tag (default: v1).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (default: data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (overrides standard partition layout).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV companion file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    build_features_from_datasets(
        report_day=args.report_day,
        interval_path=args.interval_path,
        instantaneous_path=args.instantaneous_path,
        register_path=args.register_path,
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        feature_set_version=args.feature_version,
        output_dir=args.output_dir,
        write_csv=not args.no_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
