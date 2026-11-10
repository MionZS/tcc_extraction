"""Retrospective Data Cleaning, Entity Reconciliation & Non-Overlapping Window Consolidation.

Implements the offline sanitation stage between the Raw/Normalized Data Lake (`output/measurements`,
`output/events`, `output/context`) and the Model Input Matrix (`output/model_input/v1/training_dataset`).

Solves five structural distortions identified in real smart-meter (AMI) field data:
1. Canonical Entity Reconciliation (`NIO <-> UC` stripping leading-zero mismatches between CIS 10-digit
   serials and ORCA/MDM 8-digit serials, respecting meter replacement history).
2. Physical Noise Sanitation (filtering unenergized <50V channels in phase voltage unbalance, filtering
   <1.0A nocturnal standby currents in phase current unbalance, and fixing U_L1_AVG column resolution).
3. RF Mesh Packet-Loss Isolation (excluding degraded communication days `< min_day_coverage` from
   corrupting electrical medians, MAD, and linear consumption slopes, while normalizing window coverage
   over calendar extracted days to eliminate survival bias).
4. Decoupled Canonical Cutoff Windows (replacing both degenerate 1-day snapshots where MAD=0/Slope=0
   and 96.7%-overlapping daily sliding windows with non-overlapping monthly/biweekly windows).
5. Training Eligibility Quarantine Gate (`meta__is_clean_for_train`), ensuring estimators learn normal
   electrical density strictly over complete telemetry while routing intermittent RF Mesh meters to an
   explicit Telecom Quarantine audit report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math
from pathlib import Path
from typing import Callable

import polars as pl

from src.datasets.build_training_dataset import build_training_dataset
from src.datasets.partitioning import (
    ensure_partition_dirs,
    events_partition_path,
    feature_partition_path,
    measurements_partition_path,
    window_feature_partition_path,
)
from src.datasets.schemas import UC_DAY_FEATURES_SCHEMA, UC_WINDOW_FEATURES_SCHEMA
from src.datasets.writers import (
    write_features,
    write_training_dataset,
    write_window_features,
)


@dataclass(frozen=True)
class RetroactiveCleanReport:
    """Audit summary emitted by the retrospective sanitation & consolidation pipeline."""

    days_discovered: int
    days_recomputed: int
    days_reused_cache: int
    cutoff_dates: tuple[date, ...]
    window_days: int
    stride_mode: str
    total_uc_windows: int
    clean_train_eligible_rows: int
    telecom_quarantine_rows: int
    reconciled_feeder_ratio: float
    reconciled_context_ratio: float
    nonzero_mad_ratio: float
    nonzero_slope_ratio: float
    dataset_path: Path
    quarantine_path: Path | None


def _read_table_flexible(parquet_or_csv_path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Read Parquet (preferred) or semicolon CSV peer with optional column projection."""
    p_path = parquet_or_csv_path.with_suffix(".parquet")
    c_path = parquet_or_csv_path.with_suffix(".csv")

    if p_path.exists() and p_path.stat().st_size > 0:
        if columns is None:
            return pl.read_parquet(p_path)
        schema_cols = pl.read_parquet_schema(p_path)
        valid_cols = [c for c in columns if c in schema_cols]
        return pl.read_parquet(p_path, columns=valid_cols) if valid_cols else pl.DataFrame()

    if c_path.exists() and c_path.stat().st_size > 0:
        df = pl.read_csv(c_path, separator=";", infer_schema_length=5000, ignore_errors=True)
        if columns is not None:
            valid_cols = [c for c in columns if c in df.columns]
            return df.select(valid_cols) if valid_cols else pl.DataFrame()
        return df

    return pl.DataFrame()


def _extract_date_from_partition(path: Path) -> date | None:
    """Parse Hive partition path .../report_year=YYYY/report_month=MM/report_day=DD/..."""
    try:
        parts = path.parts
        y = next(int(p.split("=", 1)[1]) for p in parts if p.startswith("report_year="))
        m = next(int(p.split("=", 1)[1]) for p in parts if p.startswith("report_month="))
        d = next(int(p.split("=", 1)[1]) for p in parts if p.startswith("report_day="))
        return date(y, m, d)
    except Exception:
        return None


def discover_measurement_days(
    base_dir: Path | str = "output",
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[date]:
    """Discover all extracted calendar dates in `measurements/ami_interval` (or `ami_instantaneous`)."""
    root = Path(base_dir)
    found: set[date] = set()

    for layer in ("ami_interval", "ami_instantaneous"):
        layer_dir = root / "measurements" / layer
        if not layer_dir.exists():
            continue
        for f in layer_dir.rglob("data.parquet"):
            if f.stat().st_size > 0:
                dt = _extract_date_from_partition(f)
                if dt is not None:
                    found.add(dt)
        if not found:
            for f in layer_dir.rglob("data.csv"):
                if f.stat().st_size > 0:
                    dt = _extract_date_from_partition(f)
                    if dt is not None:
                        found.add(dt)

    days = sorted(found)
    if start_date is not None:
        days = [d for d in days if d >= start_date]
    if end_date is not None:
        days = [d for d in days if d <= end_date]
    return days


def select_non_overlapping_cutoffs(
    available_days: list[date],
    *,
    stride_mode: str = "monthly",
    min_days_per_window: int = 1,
) -> tuple[list[date], int]:
    """Select decoupled non-overlapping cutoff dates and matching window horizon.

    Args:
        available_days: Sorted list of extracted telemetry dates in the Data Lake.
        stride_mode:
            - ``"monthly"`` (default): 1 consolidated 30-day window at the last available
              date of each calendar month (0% overlap across billing cycles).
            - ``"biweekly"``: 2 non-overlapping 15-day windows per calendar month
              (day <= 15 and day > 15).
        min_days_per_window: Minimum extracted days required in a bucket to emit a cutoff
            (if only 1 bucket exists overall, it is always emitted).

    Returns:
        Tuple of ``(cutoff_dates, window_days)``.
    """
    if not available_days:
        return [], 30

    sorted_days = sorted(set(available_days))
    mode = stride_mode.lower().strip()

    buckets: dict[tuple[int, int, int], list[date]] = {}
    if mode == "biweekly":
        window_days = 15
        for d in sorted_days:
            half = 1 if d.day <= 15 else 2
            buckets.setdefault((d.year, d.month, half), []).append(d)
    else:
        window_days = 30
        for d in sorted_days:
            buckets.setdefault((d.year, d.month, 1), []).append(d)

    cutoffs: list[date] = []
    for key in sorted(buckets.keys()):
        b_days = buckets[key]
        if len(b_days) >= min_days_per_window or len(buckets) == 1:
            cutoffs.append(b_days[-1])

    return cutoffs, window_days


def _build_nio_to_uc_bridge(
    meter_history_df: pl.DataFrame,
    hierarchy_df: pl.DataFrame,
    report_day: date | None = None,
) -> pl.DataFrame:
    """Build deterministic `_NORM_NIO -> UC` mapping respecting installation dates."""
    candidates: list[pl.DataFrame] = []

    if not meter_history_df.is_empty() and "NIO" in meter_history_df.columns and "UC" in meter_history_df.columns:
        mh = meter_history_df.with_columns([
            pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_NIO"),
            pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("UC"),
            (
                pl.col("DATA_INSTALACAO").cast(pl.Date, strict=False)
                if "DATA_INSTALACAO" in meter_history_df.columns
                else pl.lit(None, dtype=pl.Date)
            ).alias("_DT_INST"),
            (
                pl.col("DATA_REMOCAO").cast(pl.Date, strict=False)
                if "DATA_REMOCAO" in meter_history_df.columns
                else pl.lit(None, dtype=pl.Date)
            ).alias("_DT_REM"),
        ]).filter(
            pl.col("_NORM_NIO").is_not_null()
            & (pl.col("_NORM_NIO") != "")
            & pl.col("UC").is_not_null()
            & (pl.col("UC") != "")
        )

        if report_day is not None:
            active_mh = (
                mh.filter(
                    (pl.col("_DT_INST").is_null() | (pl.col("_DT_INST") <= report_day))
                    & (pl.col("_DT_REM").is_null() | (pl.col("_DT_REM") >= report_day))
                )
                .sort("_DT_INST", nulls_last=False)
                .unique(subset=["_NORM_NIO"], keep="last")
                .select(["_NORM_NIO", "UC"])
            )
            candidates.append(active_mh)

        static_mh = (
            mh.sort("_DT_INST", nulls_last=False)
            .unique(subset=["_NORM_NIO"], keep="last")
            .select(["_NORM_NIO", "UC"])
        )
        candidates.append(static_mh)

    if not hierarchy_df.is_empty() and "NIO" in hierarchy_df.columns and "UC" in hierarchy_df.columns:
        hier_bridge = (
            hierarchy_df.select([
                pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_NIO"),
                pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("UC"),
            ])
            .filter(
                pl.col("_NORM_NIO").is_not_null()
                & (pl.col("_NORM_NIO") != "")
                & pl.col("UC").is_not_null()
                & (pl.col("UC") != "")
            )
            .unique(subset=["_NORM_NIO"], keep="first")
        )
        candidates.append(hier_bridge)

    if not candidates:
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "UC": pl.String})

    return pl.concat(candidates, how="vertical").unique(subset=["_NORM_NIO"], keep="first")


def _compute_sanitized_voltage_imbalance(source_df: pl.DataFrame) -> pl.DataFrame:
    """Vectorized PRODIST M8 dynamic voltage unbalance (P95 over energized >=50V phases)."""
    if source_df.is_empty() or "_NORM_NIO" not in source_df.columns:
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "VOLTAGE_IMBALANCE_MAX": pl.Float64})

    u_cols = [c for c in ("U_L1", "U_L2", "U_L3") if c in source_df.columns]
    if not u_cols:
        u_cols = [c for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG") if c in source_df.columns]
    if len(u_cols) < 2:
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "VOLTAGE_IMBALANCE_MAX": pl.Float64})

    energ_exprs = [
        pl.when(pl.col(c).cast(pl.Float64, strict=False) >= 50.0)
        .then(pl.col(c).cast(pl.Float64, strict=False))
        .otherwise(None)
        .alias(f"_E_{idx}")
        for idx, c in enumerate(u_cols)
    ]
    e_names = [f"_E_{idx}" for idx in range(len(u_cols))]

    work = source_df.select(["_NORM_NIO"] + energ_exprs)
    k_expr = pl.sum_horizontal([pl.col(en).is_not_null().cast(pl.Int8) for en in e_names])
    sum_expr = pl.sum_horizontal([pl.col(en).fill_null(0.0) for en in e_names])

    work = work.with_columns([
        k_expr.alias("_K_ENERG"),
        pl.when(k_expr >= 2).then(sum_expr / k_expr.cast(pl.Float64)).otherwise(None).alias("_U_AVG"),
    ]).filter(pl.col("_K_ENERG") >= 2)

    if work.is_empty():
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "VOLTAGE_IMBALANCE_MAX": pl.Float64})

    dev_exprs = [
        pl.when(pl.col(en).is_not_null())
        .then((pl.col(en) - pl.col("_U_AVG")).abs())
        .otherwise(0.0)
        for en in e_names
    ]
    imb_expr = (
        pl.when(pl.col("_U_AVG") > 10.0)
        .then((pl.max_horizontal(dev_exprs) / pl.col("_U_AVG")) * 100.0)
        .otherwise(None)
    )

    return (
        work.with_columns(imb_expr.alias("_V_IMB"))
        .filter(pl.col("_V_IMB").is_not_null())
        .group_by("_NORM_NIO")
        .agg(pl.col("_V_IMB").quantile(0.95).round(4).alias("VOLTAGE_IMBALANCE_MAX"))
    )


def _compute_sanitized_current_imbalance(source_df: pl.DataFrame) -> pl.DataFrame:
    """Vectorized loaded-regime current imbalance (P90 over intervals with mean I >= 1.0 A)."""
    if source_df.is_empty() or "_NORM_NIO" not in source_df.columns:
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "CURRENT_IMBALANCE_MAX": pl.Float64})

    i_cols = [c for c in ("I_INSTANT_L1", "I_INSTANT_L2", "I_INSTANT_L3") if c in source_df.columns]
    if not i_cols:
        i_cols = [c for c in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG") if c in source_df.columns]
    if len(i_cols) < 2:
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "CURRENT_IMBALANCE_MAX": pl.Float64})

    clean_exprs = [pl.col(c).cast(pl.Float64, strict=False).alias(f"_I_{idx}") for idx, c in enumerate(i_cols)]
    i_names = [f"_I_{idx}" for idx in range(len(i_cols))]
    work = source_df.select(["_NORM_NIO"] + clean_exprs)

    # Identify active phases per meter across the day (max > 0.1 A)
    phase_activity = work.group_by("_NORM_NIO").agg([
        (pl.col(inm).max().fill_null(0.0) > 0.1).alias(f"_ACT_{idx}")
        for idx, inm in enumerate(i_names)
    ])
    act_names = [f"_ACT_{idx}" for idx in range(len(i_names))]
    phase_activity = phase_activity.with_columns(
        pl.sum_horizontal([pl.col(an).cast(pl.Int8) for an in act_names]).alias("_K_PHASES")
    ).filter(pl.col("_K_PHASES") >= 2)

    if phase_activity.is_empty():
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "CURRENT_IMBALANCE_MAX": pl.Float64})

    work = work.join(phase_activity, on="_NORM_NIO", how="inner")

    masked_i = [
        pl.when(pl.col(f"_ACT_{idx}") & pl.col(inm).is_not_null())
        .then(pl.col(inm).clip(lower_bound=0.0))
        .otherwise(None)
        for idx, inm in enumerate(i_names)
    ]
    m_names = [f"_MI_{idx}" for idx in range(len(i_names))]
    work = work.with_columns([expr.alias(m_names[idx]) for idx, expr in enumerate(masked_i)])

    k_valid = pl.sum_horizontal([pl.col(mn).is_not_null().cast(pl.Int8) for mn in m_names])
    sum_i = pl.sum_horizontal([pl.col(mn).fill_null(0.0) for mn in m_names])
    work = work.with_columns([
        k_valid.alias("_K_VALID"),
        pl.when(k_valid >= 2).then(sum_i / k_valid.cast(pl.Float64)).otherwise(0.0).alias("_I_AVG"),
    ]).filter((pl.col("_K_VALID") >= 2) & (pl.col("_I_AVG") >= 1.0))

    if work.is_empty():
        return pl.DataFrame(schema={"_NORM_NIO": pl.String, "CURRENT_IMBALANCE_MAX": pl.Float64})

    dev_i = [
        pl.when(pl.col(mn).is_not_null())
        .then((pl.col(mn) - pl.col("_I_AVG")).abs())
        .otherwise(0.0)
        for mn in m_names
    ]
    imb_i = (pl.max_horizontal(dev_i) / pl.col("_I_AVG")).clip(0.0, 2.0)

    return (
        work.with_columns(imb_i.alias("_C_IMB"))
        .group_by("_NORM_NIO")
        .agg(pl.col("_C_IMB").quantile(0.90).round(4).alias("CURRENT_IMBALANCE_MAX"))
    )


def compute_clean_day_features_vectorized(
    report_day: date,
    *,
    base_dir: Path | str = "output",
    uc_context_df: pl.DataFrame,
    meter_history_df: pl.DataFrame,
    hierarchy_df: pl.DataFrame,
    feature_set_version: str = "v1",
) -> pl.DataFrame:
    """Compute vectorized, physically sanitized `uc_day_features` for all meters on `report_day`."""
    root = Path(base_dir)
    int_dir = measurements_partition_path("ami_interval", report_day, base_dir=root)
    inst_dir = measurements_partition_path("ami_instantaneous", report_day, base_dir=root)
    reg_dir = measurements_partition_path("ami_registers", report_day, base_dir=root)

    interval_df = _read_table_flexible(
        int_dir / "data.parquet",
        columns=[
            "NIO", "TIMESTAMP_LOCAL", "TIMESTAMP_UTC", "CADENCE_MINUTES",
            "FA_INTERVAL", "RA_INTERVAL",
            "I_L1_AVG", "I_L2_AVG", "I_L3_AVG",
            "U_L1_AVG", "U_L2_AVG", "U_L3_AVG", "U_L1",
        ],
    )
    instant_df = _read_table_flexible(
        inst_dir / "data.parquet",
        columns=["NIO", "U_L1", "U_L2", "U_L3", "I_INSTANT_L1", "I_INSTANT_L2", "I_INSTANT_L3"],
    )
    register_df = _read_table_flexible(
        reg_dir / "data.parquet",
        columns=["NIO", "FA_TOTAL", "FA_MD"],
    )

    if interval_df.is_empty() and instant_df.is_empty():
        return pl.DataFrame(schema=UC_DAY_FEATURES_SCHEMA)

    if not interval_df.is_empty() and "NIO" in interval_df.columns:
        interval_df = interval_df.with_columns(
            pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_NIO")
        ).filter(pl.col("_NORM_NIO") != "")

    if not instant_df.is_empty() and "NIO" in instant_df.columns:
        instant_df = instant_df.with_columns(
            pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_NIO")
        ).filter(pl.col("_NORM_NIO") != "")

    if not register_df.is_empty() and "NIO" in register_df.columns:
        register_df = register_df.with_columns(
            pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_NIO")
        ).filter(pl.col("_NORM_NIO") != "")

    # 1. Interval aggregations
    if not interval_df.is_empty() and "_NORM_NIO" in interval_df.columns:
        ts_col = next((c for c in ("TIMESTAMP_LOCAL", "TIMESTAMP_UTC") if c in interval_df.columns), None)
        if ts_col:
            interval_df = interval_df.sort(["_NORM_NIO", ts_col])
            if interval_df[ts_col].dtype == pl.String:
                interval_df = interval_df.with_columns(
                    pl.col(ts_col).str.to_datetime(strict=False).dt.hour().fill_null(0).alias("_SLOT_HOUR")
                )
            else:
                interval_df = interval_df.with_columns(
                    pl.col(ts_col).dt.hour().fill_null(0).alias("_SLOT_HOUR")
                )
        else:
            interval_df = interval_df.with_columns(pl.lit(0).alias("_SLOT_HOUR"))

        for num_c in ("FA_INTERVAL", "RA_INTERVAL", "U_L1_AVG", "U_L1"):
            if num_c in interval_df.columns and not interval_df[num_c].dtype.is_numeric():
                interval_df = interval_df.with_columns(pl.col(num_c).cast(pl.Float64, strict=False))

        if "FA_INTERVAL" not in interval_df.columns:
            interval_df = interval_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("FA_INTERVAL"))
        if "RA_INTERVAL" not in interval_df.columns:
            interval_df = interval_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("RA_INTERVAL"))
        if "CADENCE_MINUTES" not in interval_df.columns:
            interval_df = interval_df.with_columns(pl.lit(5, dtype=pl.Int16).alias("CADENCE_MINUTES"))

        u1_int_col = "U_L1_AVG" if "U_L1_AVG" in interval_df.columns else (
            "U_L1" if "U_L1" in interval_df.columns else None
        )
        if u1_int_col is None:
            interval_df = interval_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("_U1_VAL"))
        else:
            interval_df = interval_df.with_columns(
                pl.when(pl.col(u1_int_col) >= 10.0).then(pl.col(u1_int_col)).otherwise(None).alias("_U1_VAL")
            )

        interval_df = interval_df.with_columns(
            pl.col("FA_INTERVAL").shift(1).over("_NORM_NIO").alias("_FA_PREV")
        )

        int_agg = interval_df.group_by("_NORM_NIO").agg([
            pl.len().alias("_TOTAL_ROWS"),
            pl.col("CADENCE_MINUTES").cast(pl.Float64, strict=False).median().fill_null(5.0).clip(1.0, 60.0).alias("_CADENCE"),
            pl.col("FA_INTERVAL").drop_nulls().len().alias("_VALID_FA"),
            pl.col("FA_INTERVAL").sum().fill_null(0.0).round(4).alias("FA_INTERVAL_SUM"),
            pl.col("RA_INTERVAL").sum().fill_null(0.0).round(4).alias("RA_INTERVAL_SUM"),
            pl.col("FA_INTERVAL").mean().fill_null(0.0).alias("_FA_MEAN"),
            pl.col("FA_INTERVAL").max().fill_null(0.0).alias("_FA_MAX"),
            (
                (pl.col("RA_INTERVAL").drop_nulls().abs() > 1e-4).sum().cast(pl.Float64)
                / pl.col("RA_INTERVAL").drop_nulls().len().clip(lower_bound=1).cast(pl.Float64)
            ).fill_null(0.0).round(4).alias("RA_REVERSAL_RATIO"),
            pl.col("FA_INTERVAL").drop_nulls().median().fill_null(0.0).round(4).alias("FA_INTERVAL_MEDIAN"),
            (
                pl.col("FA_INTERVAL").drop_nulls().quantile(0.75).fill_null(0.0)
                - pl.col("FA_INTERVAL").drop_nulls().quantile(0.25).fill_null(0.0)
            ).round(4).alias("FA_INTERVAL_IQR"),
            pl.col("FA_INTERVAL").drop_nulls().quantile(0.05).fill_null(0.0).round(4).alias("FA_INTERVAL_P05"),
            pl.col("FA_INTERVAL").drop_nulls().quantile(0.95).fill_null(0.0).round(4).alias("FA_INTERVAL_P95"),
            pl.col("FA_INTERVAL").drop_nulls().std(ddof=0).fill_null(0.0).round(4).alias("FA_INTERVAL_STD"),
            pl.col("_U1_VAL").drop_nulls().median().fill_null(0.0).round(4).alias("U_L1_MEDIAN"),
            (
                pl.col("_U1_VAL").drop_nulls().quantile(0.75).fill_null(0.0)
                - pl.col("_U1_VAL").drop_nulls().quantile(0.25).fill_null(0.0)
            ).round(4).alias("U_L1_IQR"),
            pl.col("_U1_VAL").drop_nulls().quantile(0.05).fill_null(0.0).round(4).alias("U_L1_P05"),
            pl.col("_U1_VAL").drop_nulls().quantile(0.95).fill_null(0.0).round(4).alias("U_L1_P95"),
            pl.col("FA_INTERVAL").drop_nulls().diff().abs().max().fill_null(0.0).round(4).alias("FA_RAMP_MAX"),
            pl.col("_U1_VAL").drop_nulls().diff().abs().max().fill_null(0.0).round(4).alias("U_L1_RAMP_MAX"),
            pl.col("_SLOT_HOUR").sort_by("FA_INTERVAL", descending=True).drop_nulls().first().fill_null(0).cast(pl.Float64).alias("_PEAK_H"),
            pl.corr("FA_INTERVAL", "_FA_PREV").fill_nan(0.0).fill_null(0.0).round(4).alias("FA_INTERVAL_AUTOCORR_LAG1"),
        ]).with_columns([
            ((pl.col("_VALID_FA").cast(pl.Float64) / (1440.0 / pl.col("_CADENCE"))).clip(0.0, 1.0)).round(4).alias("INTERVAL_COVERAGE"),
            (1.0 - (pl.col("_VALID_FA").cast(pl.Float64) / pl.col("_TOTAL_ROWS").clip(lower_bound=1).cast(pl.Float64))).clip(0.0, 1.0).round(4).alias("NULL_RATIO_INTERVAL"),
            pl.when(pl.col("_FA_MAX") > 0.0).then((pl.col("_FA_MEAN") / pl.col("_FA_MAX")).clip(0.0, 1.0)).otherwise(0.0).round(4).alias("LOAD_FACTOR"),
            ((pl.col("_PEAK_H") * (2.0 * math.pi / 24.0)).sin()).round(4).alias("PEAK_HOUR_SIN"),
            ((pl.col("_PEAK_H") * (2.0 * math.pi / 24.0)).cos()).round(4).alias("PEAK_HOUR_COS"),
        ]).drop(["_TOTAL_ROWS", "_CADENCE", "_VALID_FA", "_FA_MEAN", "_FA_MAX", "_PEAK_H"])
    else:
        int_agg = pl.DataFrame(schema={"_NORM_NIO": pl.String})

    # 2. Instantaneous coverage & fallback U_L1 stats
    if not instant_df.is_empty() and "_NORM_NIO" in instant_df.columns and "U_L1" in instant_df.columns:
        inst_agg = instant_df.group_by("_NORM_NIO").agg([
            (pl.col("U_L1").cast(pl.Float64, strict=False).drop_nulls().len().cast(pl.Float64) / 96.0)
            .clip(0.0, 1.0)
            .round(4)
            .alias("INSTANTANEOUS_COVERAGE"),
            pl.col("U_L1").cast(pl.Float64, strict=False).filter(pl.col("U_L1").cast(pl.Float64, strict=False) >= 10.0).median().fill_null(0.0).round(4).alias("_INST_U1_MED"),
        ])
    else:
        inst_agg = pl.DataFrame(
            schema={"_NORM_NIO": pl.String, "INSTANTANEOUS_COVERAGE": pl.Float64, "_INST_U1_MED": pl.Float64}
        )

    # 3. Register coverage & max demand
    if not register_df.is_empty() and "_NORM_NIO" in register_df.columns:
        fa_tot_expr = (
            (pl.col("FA_TOTAL").cast(pl.Float64, strict=False).drop_nulls().len() > 0).cast(pl.Float64)
            if "FA_TOTAL" in register_df.columns
            else pl.lit(0.0)
        )
        fa_md_expr = (
            pl.col("FA_MD").cast(pl.Float64, strict=False).max().fill_null(0.0).round(4)
            if "FA_MD" in register_df.columns
            else pl.lit(0.0)
        )
        reg_agg = register_df.group_by("_NORM_NIO").agg([
            fa_tot_expr.alias("REGISTER_COVERAGE"),
            fa_md_expr.alias("FA_MD_MAX"),
        ])
    else:
        reg_agg = pl.DataFrame(
            schema={"_NORM_NIO": pl.String, "REGISTER_COVERAGE": pl.Float64, "FA_MD_MAX": pl.Float64}
        )

    # 4. Sanitized voltage & current imbalances (preferring instantaneous if populated, with interval fallback)
    v_imb_inst = _compute_sanitized_voltage_imbalance(instant_df) if not instant_df.is_empty() else pl.DataFrame()
    v_imb_int = _compute_sanitized_voltage_imbalance(interval_df) if not interval_df.is_empty() else pl.DataFrame()
    if not v_imb_inst.is_empty() and not v_imb_int.is_empty():
        v_imb_df = (
            v_imb_int.rename({"VOLTAGE_IMBALANCE_MAX": "_V_INT"})
            .join(v_imb_inst.rename({"VOLTAGE_IMBALANCE_MAX": "_V_INST"}), on="_NORM_NIO", how="full", coalesce=True)
            .select([
                pl.col("_NORM_NIO"),
                pl.coalesce([pl.col("_V_INST"), pl.col("_V_INT")]).alias("VOLTAGE_IMBALANCE_MAX"),
            ])
        )
    elif not v_imb_inst.is_empty():
        v_imb_df = v_imb_inst
    else:
        v_imb_df = v_imb_int

    c_imb_inst = _compute_sanitized_current_imbalance(instant_df) if not instant_df.is_empty() else pl.DataFrame()
    c_imb_int = _compute_sanitized_current_imbalance(interval_df) if not interval_df.is_empty() else pl.DataFrame()
    if not c_imb_inst.is_empty() and not c_imb_int.is_empty():
        c_imb_df = (
            c_imb_int.rename({"CURRENT_IMBALANCE_MAX": "_C_INT"})
            .join(c_imb_inst.rename({"CURRENT_IMBALANCE_MAX": "_C_INST"}), on="_NORM_NIO", how="full", coalesce=True)
            .select([
                pl.col("_NORM_NIO"),
                pl.coalesce([pl.col("_C_INST"), pl.col("_C_INT")]).alias("CURRENT_IMBALANCE_MAX"),
            ])
        )
    elif not c_imb_inst.is_empty():
        c_imb_df = c_imb_inst
    else:
        c_imb_df = c_imb_int

    # 5. Merge all daily meter signals
    nio_frames = [
        f.select("_NORM_NIO")
        for f in (int_agg, inst_agg, reg_agg)
        if not f.is_empty() and "_NORM_NIO" in f.columns
    ]
    if not nio_frames:
        return pl.DataFrame(schema=UC_DAY_FEATURES_SCHEMA)

    base_nios = pl.concat(nio_frames, how="vertical").unique(subset=["_NORM_NIO"])
    day_df = base_nios
    for part_df in (int_agg, inst_agg, reg_agg, v_imb_df, c_imb_df):
        if not part_df.is_empty() and "_NORM_NIO" in part_df.columns and part_df.width > 1:
            day_df = day_df.join(part_df, on="_NORM_NIO", how="left")

    if "_INST_U1_MED" in day_df.columns:
        if "U_L1_MEDIAN" in day_df.columns:
            day_df = day_df.with_columns(
                pl.when((pl.col("U_L1_MEDIAN").fill_null(0.0) <= 0.0) & (pl.col("_INST_U1_MED").fill_null(0.0) > 0.0))
                .then(pl.col("_INST_U1_MED"))
                .otherwise(pl.col("U_L1_MEDIAN").fill_null(0.0))
                .alias("U_L1_MEDIAN")
            )
        else:
            day_df = day_df.with_columns(pl.col("_INST_U1_MED").fill_null(0.0).alias("U_L1_MEDIAN"))
        day_df = day_df.drop("_INST_U1_MED")

    # 6. Reconcile NIO -> UC and enrich context
    bridge_df = _build_nio_to_uc_bridge(meter_history_df, hierarchy_df, report_day=report_day)
    if not bridge_df.is_empty():
        day_df = day_df.join(bridge_df, on="_NORM_NIO", how="left")
        day_df = day_df.with_columns(pl.coalesce([pl.col("UC"), pl.col("_NORM_NIO")]).alias("UC"))
    else:
        day_df = day_df.with_columns(pl.col("_NORM_NIO").alias("UC"))

    day_df = day_df.with_columns([
        pl.col("_NORM_NIO").alias("NIO"),
        pl.lit(report_day).cast(pl.Date).alias("REPORT_DAY"),
        pl.lit(feature_set_version).cast(pl.String).alias("FEATURE_SET_VERSION"),
    ])

    # Context enrichment (PHASE_TYPE, CONSUMER_CLASS, INSTALLED_KVA)
    if not uc_context_df.is_empty() and "UC" in uc_context_df.columns:
        ctx_sel: list[pl.Expr] = [
            pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("UC")
        ]
        if "FASE_CIRCUITO" in uc_context_df.columns:
            ctx_sel.append(pl.col("FASE_CIRCUITO").cast(pl.String).alias("PHASE_TYPE"))
        if "CLASSE" in uc_context_df.columns:
            ctx_sel.append(pl.col("CLASSE").cast(pl.String).alias("CONSUMER_CLASS"))
        ctx_sub = uc_context_df.select(ctx_sel).unique(subset=["UC"], keep="first")
        day_df = day_df.join(ctx_sub, on="UC", how="left")

    if not hierarchy_df.is_empty() and "UC" in hierarchy_df.columns and "POT_INST_KVA" in hierarchy_df.columns:
        kva_sub = (
            hierarchy_df.select([
                pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("UC"),
                pl.col("POT_INST_KVA").cast(pl.Float64, strict=False).alias("INSTALLED_KVA"),
            ])
            .unique(subset=["UC"], keep="first")
        )
        day_df = day_df.join(kva_sub, on="UC", how="left")

    # Zero-out current imbalance on explicitly single-phase installations
    if "PHASE_TYPE" in day_df.columns and "CURRENT_IMBALANCE_MAX" in day_df.columns:
        day_df = day_df.with_columns(
            pl.when(pl.col("PHASE_TYPE").str.to_uppercase().str.starts_with("MO"))
            .then(None)
            .otherwise(pl.col("CURRENT_IMBALANCE_MAX"))
            .alias("CURRENT_IMBALANCE_MAX")
        )

    # Deduplicate to 1 row per UC x REPORT_DAY (keeping meter with highest interval coverage)
    if "INTERVAL_COVERAGE" in day_df.columns:
        day_df = day_df.sort("INTERVAL_COVERAGE", descending=True, nulls_last=True).unique(subset=["UC"], keep="first")
    else:
        day_df = day_df.unique(subset=["UC"], keep="first")

    # Guarantee full canonical schema order and defaults
    final_exprs: list[pl.Expr] = []
    numeric_zero_defaults = {
        "INTERVAL_COVERAGE", "INSTANTANEOUS_COVERAGE", "REGISTER_COVERAGE", "NULL_RATIO_INTERVAL",
        "FA_INTERVAL_SUM", "RA_INTERVAL_SUM", "FA_MD_MAX", "LOAD_FACTOR", "RA_REVERSAL_RATIO",
        "FA_INTERVAL_MEDIAN", "FA_INTERVAL_IQR", "FA_INTERVAL_P05", "FA_INTERVAL_P95",
        "U_L1_MEDIAN", "U_L1_IQR", "U_L1_P05", "U_L1_P95",
        "FA_RAMP_MAX", "U_L1_RAMP_MAX", "PEAK_HOUR_SIN", "PEAK_HOUR_COS",
        "FA_INTERVAL_AUTOCORR_LAG1", "FA_INTERVAL_STD",
    }
    for col_name, dtype in UC_DAY_FEATURES_SCHEMA.items():
        if col_name in day_df.columns:
            expr = pl.col(col_name).cast(dtype, strict=False)
            if col_name in numeric_zero_defaults:
                expr = expr.fill_null(0.0)
            final_exprs.append(expr.alias(col_name))
        else:
            default_val = 0.0 if col_name in numeric_zero_defaults else None
            final_exprs.append(pl.lit(default_val, dtype=dtype).alias(col_name))

    return day_df.select(final_exprs).sort("UC")


def _is_day_feature_cache_valid(
    part_dir: Path,
    known_ucs: set[str],
) -> bool:
    """Verify whether a cached `uc_day_features` partition already has reconciled UC IDs."""
    sample = _read_table_flexible(part_dir / "data.parquet", columns=["UC", "NIO"])
    if sample.is_empty() or "UC" not in sample.columns or "NIO" not in sample.columns:
        return False

    norm_uc = sample["UC"].cast(pl.String).str.strip_chars().str.replace(r"^0+", "")
    norm_nio = sample["NIO"].cast(pl.String).str.strip_chars().str.replace(r"^0+", "")
    same_ratio = float((norm_uc == norm_nio).mean() or 0.0)

    if known_ucs:
        uc_set = set(norm_uc.drop_nulls().to_list())
        overlap_ratio = len(uc_set & known_ucs) / max(1, len(uc_set))
        if overlap_ratio < 0.50 and same_ratio > 0.50:
            return False

    return True


def build_sanitized_window_features_vectorized(
    daily_features_df: pl.DataFrame,
    *,
    meter_history_df: pl.DataFrame,
    alarm_events_df: pl.DataFrame,
    cutoff_date: date,
    window_days: int = 30,
    min_day_coverage: float = 0.70,
    min_window_coverage: float = 0.60,
    min_reliable_days: int = 15,
    feature_set_version: str = "v1",
) -> pl.DataFrame:
    """Vectorized 30-day/15-day window aggregation with RF Mesh packet-drop isolation."""
    if daily_features_df.is_empty():
        return pl.DataFrame(schema=UC_WINDOW_FEATURES_SCHEMA)

    window_start = cutoff_date - timedelta(days=window_days)
    w_df = daily_features_df.with_columns(
        pl.col("REPORT_DAY").cast(pl.Date, strict=False)
    ).filter(
        (pl.col("REPORT_DAY") > window_start) & (pl.col("REPORT_DAY") <= cutoff_date)
    )

    if w_df.is_empty():
        return pl.DataFrame(schema=UC_WINDOW_FEATURES_SCHEMA)

    n_extracted_days = max(1, w_df.get_column("REPORT_DAY").n_unique())

    w_df = w_df.with_columns([
        pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("UC"),
        (pl.col("INTERVAL_COVERAGE").fill_null(0.0) >= min_day_coverage).alias("_IS_REL"),
        (pl.col("REPORT_DAY") - pl.lit(window_start)).dt.total_days().cast(pl.Float64).alias("_DAY_T"),
    ]).sort(["UC", "REPORT_DAY"])

    rel_fa = pl.col("FA_INTERVAL_SUM").filter(pl.col("_IS_REL"))
    rel_t = pl.col("_DAY_T").filter(pl.col("_IS_REL"))
    rel_lf = pl.col("LOAD_FACTOR").filter(pl.col("_IS_REL"))
    rel_u_imb = pl.col("VOLTAGE_IMBALANCE_MAX").filter(pl.col("_IS_REL")).drop_nulls()
    rel_c_imb = pl.col("CURRENT_IMBALANCE_MAX").filter(pl.col("_IS_REL")).drop_nulls()

    agg_df = w_df.group_by("UC").agg([
        pl.col("_IS_REL").sum().cast(pl.Int16).alias("COVERAGE_RELIABLE_DAYS"),
        (pl.col("INTERVAL_COVERAGE").fill_null(0.0).sum() / float(n_extracted_days))
        .clip(0.0, 1.0)
        .round(4)
        .alias("AVG_INTERVAL_COVERAGE"),
        # Reliable-only electrical medians & MAD (with fallback for 0-reliable-day quarantined UCs)
        pl.coalesce([rel_fa.median(), pl.col("FA_INTERVAL_SUM").median(), pl.lit(0.0)])
        .round(4)
        .alias("HIST_FA_SUM_MEDIAN"),
        pl.coalesce([
            (rel_fa - rel_fa.median()).abs().median(),
            (pl.col("FA_INTERVAL_SUM") - pl.col("FA_INTERVAL_SUM").median()).abs().median(),
            pl.lit(0.0),
        ]).round(4).alias("HIST_FA_SUM_MAD"),
        (pl.col("RA_REVERSAL_RATIO").fill_null(0.0) > 0.0).sum().cast(pl.Int16).alias("HIST_RA_REVERSAL_DAYS"),
        pl.col("RA_REVERSAL_RATIO").fill_null(0.0).max().fill_null(0.0).round(4).alias("HIST_RA_REVERSAL_RATIO_MAX"),
        pl.coalesce([rel_lf.mean(), pl.col("LOAD_FACTOR").mean(), pl.lit(0.0)])
        .round(4)
        .alias("HIST_LOAD_FACTOR_MEAN"),
        pl.coalesce([rel_u_imb.quantile(0.95), pl.col("VOLTAGE_IMBALANCE_MAX").drop_nulls().quantile(0.95), pl.lit(0.0)])
        .round(4)
        .alias("HIST_VOLTAGE_IMBALANCE_MAX"),
        pl.coalesce([rel_c_imb.quantile(0.90), pl.col("CURRENT_IMBALANCE_MAX").drop_nulls().quantile(0.90), pl.lit(0.0)])
        .round(4)
        .alias("HIST_CURRENT_IMBALANCE_MAX"),
        # Calendar-aware OLS slope across reliable days (requires >= 3 reliable days)
        pl.when((rel_t.len() >= 3) & (rel_t.var(ddof=0).fill_null(0.0) > 0.0))
        .then(pl.cov(rel_t, rel_fa, ddof=0) / rel_t.var(ddof=0))
        .otherwise(0.0)
        .fill_nan(0.0)
        .fill_null(0.0)
        .round(4)
        .alias("TREND_FA_SLOPE"),
    ])

    # Meter installation age, replacement flags & historical meter count (anti-leakage: DATA_INSTALACAO <= cutoff_date)
    if not meter_history_df.is_empty() and "UC" in meter_history_df.columns and "DATA_INSTALACAO" in meter_history_df.columns:
        nio_col_expr = (
            pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "")
            if "NIO" in meter_history_df.columns
            else pl.col("DATA_INSTALACAO").cast(pl.String)
        )
        mh_agg = (
            meter_history_df.select([
                pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("UC"),
                pl.col("DATA_INSTALACAO").cast(pl.Date, strict=False).alias("_DT_INST"),
                nio_col_expr.alias("_NORM_NIO"),
            ])
            .filter(pl.col("_DT_INST").is_not_null() & (pl.col("_DT_INST") <= cutoff_date))
            .group_by("UC")
            .agg([
                pl.col("_DT_INST").max().alias("_LATEST_INST"),
                pl.col("_NORM_NIO").drop_nulls().n_unique().clip(lower_bound=1).cast(pl.Int16).alias("N_METERS_HISTORICAL"),
            ])
            .with_columns(
                (pl.lit(cutoff_date) - pl.col("_LATEST_INST")).dt.total_days().clip(lower_bound=0).cast(pl.Int32).alias("METER_AGE_DAYS")
            )
            .with_columns([
                (pl.col("METER_AGE_DAYS") <= 30).alias("METER_CHANGED_30D"),
                (pl.col("METER_AGE_DAYS") <= 90).alias("METER_CHANGED_90D"),
            ])
            .select(["UC", "METER_AGE_DAYS", "METER_CHANGED_30D", "METER_CHANGED_90D", "N_METERS_HISTORICAL"])
        )
        agg_df = agg_df.join(mh_agg, on="UC", how="left")

    agg_df = agg_df.with_columns([
        pl.col("METER_AGE_DAYS").fill_null(0).cast(pl.Int32) if "METER_AGE_DAYS" in agg_df.columns else pl.lit(0, dtype=pl.Int32).alias("METER_AGE_DAYS"),
        pl.col("METER_CHANGED_30D").fill_null(False).cast(pl.Boolean) if "METER_CHANGED_30D" in agg_df.columns else pl.lit(False, dtype=pl.Boolean).alias("METER_CHANGED_30D"),
        pl.col("METER_CHANGED_90D").fill_null(False).cast(pl.Boolean) if "METER_CHANGED_90D" in agg_df.columns else pl.lit(False, dtype=pl.Boolean).alias("METER_CHANGED_90D"),
        pl.col("N_METERS_HISTORICAL").fill_null(1).cast(pl.Int16) if "N_METERS_HISTORICAL" in agg_df.columns else pl.lit(1, dtype=pl.Int16).alias("N_METERS_HISTORICAL"),
    ])

    # Window alarm events (mapped deterministically via _NORM_NIO -> UC)
    if not alarm_events_df.is_empty() and "NIO" in alarm_events_df.columns and "ORIGIN_TIMESTAMP" in alarm_events_df.columns:
        window_start_dt = datetime.combine(window_start + timedelta(days=1), time.min)
        cutoff_end_dt = datetime.combine(cutoff_date, time.max)
        al_df = alarm_events_df
        if al_df["ORIGIN_TIMESTAMP"].dtype == pl.String:
            al_df = al_df.with_columns(pl.col("ORIGIN_TIMESTAMP").str.to_datetime(strict=False))
        else:
            al_df = al_df.with_columns(pl.col("ORIGIN_TIMESTAMP").dt.replace_time_zone(None))

        al_df = al_df.filter(
            (pl.col("ORIGIN_TIMESTAMP") >= window_start_dt)
            & (pl.col("ORIGIN_TIMESTAMP") <= cutoff_end_dt)
        ).with_columns(
            pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_NIO")
        )

        if not al_df.is_empty():
            uc_nio_pairs = w_df.select([
                pl.col("UC"),
                pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_NIO"),
            ]).unique()
            al_joined = al_df.join(uc_nio_pairs, on="_NORM_NIO", how="inner")
            if not al_joined.is_empty():
                lat_expr = (
                    pl.col("LATENCY_SECONDS").cast(pl.Float64, strict=False).mean().fill_null(0.0).round(2)
                    if "LATENCY_SECONDS" in al_joined.columns
                    else pl.lit(0.0)
                )
                al_agg = al_joined.group_by("UC").agg([
                    pl.len().cast(pl.Int32).alias("ALARM_COUNT_WINDOW"),
                    lat_expr.alias("AVG_ALARM_LATENCY_SEC"),
                ])
                agg_df = agg_df.join(al_agg, on="UC", how="left")

    agg_df = agg_df.with_columns([
        pl.col("ALARM_COUNT_WINDOW").fill_null(0).cast(pl.Int32) if "ALARM_COUNT_WINDOW" in agg_df.columns else pl.lit(0, dtype=pl.Int32).alias("ALARM_COUNT_WINDOW"),
        pl.col("AVG_ALARM_LATENCY_SEC").fill_null(0.0).cast(pl.Float64) if "AVG_ALARM_LATENCY_SEC" in agg_df.columns else pl.lit(0.0, dtype=pl.Float64).alias("AVG_ALARM_LATENCY_SEC"),
    ])

    # Training Eligibility Gate (scales fairly with extracted days available in window)
    effective_min_days = min(min_reliable_days, max(1, int(math.ceil(0.50 * n_extracted_days))))
    agg_df = agg_df.with_columns([
        pl.lit(cutoff_date).cast(pl.Date).alias("CUTOFF_DATE"),
        pl.lit(window_days).cast(pl.Int16).alias("WINDOW_DAYS"),
        pl.lit(feature_set_version).cast(pl.String).alias("FEATURE_SET_VERSION"),
        (pl.col("COVERAGE_RELIABLE_DAYS").cast(pl.Float64) / float(n_extracted_days))
        .clip(0.0, 1.0)
        .round(4)
        .alias("RELIABLE_DAY_RATIO"),
        (
            (pl.col("COVERAGE_RELIABLE_DAYS") >= effective_min_days)
            & (pl.col("AVG_INTERVAL_COVERAGE") >= min_window_coverage)
        ).alias("IS_CLEAN_FOR_TRAIN"),
    ]).with_columns([
        pl.when(pl.col("METER_CHANGED_30D") & (pl.col("METER_AGE_DAYS") > 0) & (pl.col("N_METERS_HISTORICAL") <= 1))
        .then(pl.lit("PROFILE_INITIATION_NEW_UC"))
        .when(pl.col("METER_CHANGED_30D") & (pl.col("METER_AGE_DAYS") > 0) & (pl.col("N_METERS_HISTORICAL") >= 2))
        .then(pl.lit("METER_REPLACEMENT_TRANSITION"))
        .when(pl.col("IS_CLEAN_FOR_TRAIN"))
        .then(pl.lit("CLEAN"))
        .when(pl.col("AVG_INTERVAL_COVERAGE") < 0.20)
        .then(pl.lit("QUARANTINE_SILENT_MESH"))
        .otherwise(pl.lit("QUARANTINE_INTERMITTENT_MESH"))
        .alias("TELECOM_STATUS"),
        (pl.col("HIST_FA_SUM_MAD") / (pl.col("HIST_FA_SUM_MEDIAN").abs() + 0.1))
        .round(4)
        .alias("FA_CV_WINDOW"),
        pl.when(pl.col("IS_CLEAN_FOR_TRAIN") & (pl.col("HIST_FA_SUM_MEDIAN") < 0.05))
        .then(1.0)
        .otherwise(0.0)
        .alias("ZERO_CONSUMPTION_FLAG"),
    ])

    ordered_cols = list(UC_WINDOW_FEATURES_SCHEMA.keys()) + [
        "N_METERS_HISTORICAL",
        "IS_CLEAN_FOR_TRAIN",
        "TELECOM_STATUS",
        "RELIABLE_DAY_RATIO",
        "FA_CV_WINDOW",
        "ZERO_CONSUMPTION_FLAG",
    ]
    return agg_df.select([c for c in ordered_cols if c in agg_df.columns]).sort("UC")


class RetroactiveDataCleaner:
    """End-to-end offline curator over extracted Data Lake partitions."""

    def __init__(
        self,
        base_dir: Path | str = "output",
        *,
        min_day_coverage: float = 0.70,
        min_window_coverage: float = 0.60,
        min_reliable_days: int = 15,
        feature_set_version: str = "v1",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.min_day_coverage = min_day_coverage
        self.min_window_coverage = min_window_coverage
        self.min_reliable_days = min_reliable_days
        self.feature_set_version = feature_set_version

    def run(
        self,
        *,
        stride_mode: str = "monthly",
        start_date: date | None = None,
        end_date: date | None = None,
        force_recompute_days: bool = False,
        write_csv: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> RetroactiveCleanReport:
        """Execute retroactive entity reconciliation, daily feature backfill, and decoupled windowing."""
        uc_context = _read_table_flexible(self.base_dir / "context" / "uc_context.parquet")
        meter_history = _read_table_flexible(self.base_dir / "context" / "meter_installation_history.parquet")
        hierarchy = _read_table_flexible(self.base_dir / "context" / "electrical_hierarchy.parquet")

        known_ucs: set[str] = set()
        for ctx_df in (hierarchy, uc_context):
            if not ctx_df.is_empty() and "UC" in ctx_df.columns:
                known_ucs.update(
                    ctx_df["UC"].drop_nulls().cast(pl.String).str.strip_chars().str.replace(r"^0+", "").to_list()
                )

        available_days = discover_measurement_days(self.base_dir, start_date=start_date, end_date=end_date)
        if not available_days:
            raise FileNotFoundError(
                f"Nenhuma partição de medição encontrada em {self.base_dir / 'measurements'}. "
                "Execute a Opção 1 de extração primeiro."
            )

        cutoffs, window_days = select_non_overlapping_cutoffs(available_days, stride_mode=stride_mode)

        # Determine exact set of calendar days required to cover all decoupled windows
        needed_days_set: set[date] = set()
        for c_dt in cutoffs:
            w_start = c_dt - timedelta(days=window_days)
            for d in available_days:
                if w_start < d <= c_dt:
                    needed_days_set.add(d)
        needed_days = sorted(needed_days_set)

        days_recomputed = 0
        days_cached = 0
        day_frames_by_date: dict[date, pl.DataFrame] = {}

        for idx, d in enumerate(needed_days, start=1):
            if progress_callback is not None:
                progress_callback(f"Saneando features diárias ({d.isoformat()})", idx, len(needed_days))

            f_part_dir = feature_partition_path(self.feature_set_version, d, base_dir=self.base_dir)
            if not force_recompute_days and _is_day_feature_cache_valid(f_part_dir, known_ucs):
                day_df = _read_table_flexible(f_part_dir / "data.parquet")
                days_cached += 1
            else:
                day_df = compute_clean_day_features_vectorized(
                    d,
                    base_dir=self.base_dir,
                    uc_context_df=uc_context,
                    meter_history_df=meter_history,
                    hierarchy_df=hierarchy,
                    feature_set_version=self.feature_set_version,
                )
                if not day_df.is_empty():
                    write_features(day_df, self.feature_set_version, d, base_dir=self.base_dir, write_csv=False)
                days_recomputed += 1

            if not day_df.is_empty():
                day_frames_by_date[d] = day_df

        # Build decoupled canonical window datasets
        training_slices: list[pl.DataFrame] = []
        canonical_train_end = date(2026, 10, 31)

        for c_dt in cutoffs:
            w_start = c_dt - timedelta(days=window_days)
            w_frames = [df_d for d_key, df_d in sorted(day_frames_by_date.items()) if w_start < d_key <= c_dt]
            if not w_frames:
                continue

            combined_daily = pl.concat(w_frames, how="vertical_relaxed")

            # Collect alarm events strictly inside (w_start, c_dt]
            alarm_parts: list[pl.DataFrame] = []
            for d_key in sorted(day_frames_by_date.keys()):
                if w_start < d_key <= c_dt:
                    ev_dir = events_partition_path("alarm_events", d_key, base_dir=self.base_dir)
                    ev_df = _read_table_flexible(ev_dir / "data.parquet")
                    if not ev_df.is_empty():
                        alarm_parts.append(ev_df)
            window_alarms = pl.concat(alarm_parts, how="vertical_relaxed") if alarm_parts else pl.DataFrame()

            win_df = build_sanitized_window_features_vectorized(
                combined_daily,
                meter_history_df=meter_history,
                alarm_events_df=window_alarms,
                cutoff_date=c_dt,
                window_days=window_days,
                min_day_coverage=self.min_day_coverage,
                min_window_coverage=self.min_window_coverage,
                min_reliable_days=self.min_reliable_days,
                feature_set_version=self.feature_set_version,
            )
            if win_df.is_empty():
                continue

            # Persist standard schema columns to uc_window_features layer
            std_win_cols = [c for c in UC_WINDOW_FEATURES_SCHEMA.keys() if c in win_df.columns]
            write_window_features(
                win_df.select(std_win_cols),
                self.feature_set_version,
                c_dt,
                base_dir=self.base_dir,
                write_csv=write_csv,
            )

            split_role = "train" if c_dt <= canonical_train_end else "oot_eval"
            slice_model_df = build_training_dataset(
                win_df,
                uc_context_df=uc_context,
                hierarchy_df=hierarchy,
                split_policy=split_role,
            )
            if not slice_model_df.is_empty():
                training_slices.append(slice_model_df)

        if not training_slices:
            raise RuntimeError("Nenhuma janela consolidada pôde ser construída a partir das partições disponíveis.")

        full_dataset = pl.concat(training_slices, how="vertical_relaxed").sort(["meta__cutoff_date", "id__uc_id"])
        dataset_path = write_training_dataset(
            full_dataset,
            self.feature_set_version,
            base_dir=self.base_dir,
            write_csv=write_csv,
        )

        # Export Telecom Quarantine audit table
        quarantine_path: Path | None = None
        if "meta__is_clean_for_train" in full_dataset.columns:
            quarantine_df = full_dataset.filter(~pl.col("meta__is_clean_for_train").fill_null(True))
            reports_dir = self.base_dir / "reports"
            ensure_partition_dirs(reports_dir)
            quarantine_path = reports_dir / "telecom_quarantine_uc_windows.parquet"
            quarantine_df.write_parquet(quarantine_path, compression="zstd")
            if write_csv:
                quarantine_df.write_csv(quarantine_path.with_suffix(".csv"), separator=";")
            clean_count = int(full_dataset.get_column("meta__is_clean_for_train").fill_null(True).sum())
            quar_count = full_dataset.height - clean_count
        else:
            clean_count = full_dataset.height
            quar_count = 0

        feeder_ratio = (
            float(full_dataset.get_column("meta__feeder").is_not_null().mean() or 0.0)
            if "meta__feeder" in full_dataset.columns
            else 0.0
        )
        ctx_ratio = (
            float(full_dataset.get_column("x__tensao_base").is_not_null().mean() or 0.0)
            if "x__tensao_base" in full_dataset.columns
            else 0.0
        )
        mad_ratio = (
            float((full_dataset.get_column("x__hist_fa_sum_mad").fill_null(0.0) > 0.0).mean() or 0.0)
            if "x__hist_fa_sum_mad" in full_dataset.columns
            else 0.0
        )
        slope_ratio = (
            float((full_dataset.get_column("x__trend_fa_slope").fill_null(0.0).abs() > 0.0).mean() or 0.0)
            if "x__trend_fa_slope" in full_dataset.columns
            else 0.0
        )

        return RetroactiveCleanReport(
            days_discovered=len(available_days),
            days_recomputed=days_recomputed,
            days_reused_cache=days_cached,
            cutoff_dates=tuple(cutoffs),
            window_days=window_days,
            stride_mode=stride_mode,
            total_uc_windows=full_dataset.height,
            clean_train_eligible_rows=clean_count,
            telecom_quarantine_rows=quar_count,
            reconciled_feeder_ratio=round(feeder_ratio, 4),
            reconciled_context_ratio=round(ctx_ratio, 4),
            nonzero_mad_ratio=round(mad_ratio, 4),
            nonzero_slope_ratio=round(slope_ratio, 4),
            dataset_path=dataset_path,
            quarantine_path=quarantine_path,
        )

