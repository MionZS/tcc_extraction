"""Normalize raw MDM data into atomic per-timestamp rows.

The current pipeline stores MDM data as one row per NIO with JSON columns
containing 288 slots (5-min cadence).  This module splits those JSON blobs
into individual timestamp rows, separating interval, instantaneous, and
register data according to the architecture decision (§4, §5).

Usage::

    from src.datasets.normalize import normalize_mdm_day

    interval_df, instantaneous_df, register_df = normalize_mdm_day(
        mdm_row, report_day=..., run_id=...
    )
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path
from typing import Any, Sequence

import polars as pl


# ── Constants ────────────────────────────────────────────────────────────────

# Columns that belong to the interval table (physical quantities averaged over slot)
_INTERVAL_COLUMNS = [
    "FA_INTERVAL", "RA_INTERVAL",
    "I_L1_AVG", "I_L2_AVG", "I_L3_AVG",
    "U_L1_AVG", "U_L2_AVG", "U_L3_AVG",
    "R_Q1_INTERVAL", "R_Q2_INTERVAL", "R_Q3_INTERVAL", "R_Q4_INTERVAL",
]

# Columns that belong to the instantaneous table
_INSTANTANEOUS_COLUMNS = [
    "U_L1", "U_L2", "U_L3",
    "I_INSTANT_L1", "I_INSTANT_L2", "I_INSTANT_L3",
]

# Columns that belong to the register snapshot table (MDM CSV names)
_REGISTER_COLUMNS = [
    "FA", "FA_T1", "FA_T2", "FA_T3", "FA_T4",
    "RA_TOTAL", "RA_T1_TOTAL", "RA_T2_TOTAL", "RA_T3_TOTAL", "RA_T4_TOTAL",
    "FA_MD", "FA_MD_T1", "FA_MD_T2", "FA_MD_T3", "FA_MD_T4",
]

# Mapping from MDM CSV column names to schema column names
_REGISTER_COLUMN_MAP: dict[str, str] = {
    "FA": "FA_TOTAL",
    "FA_T1": "FA_T1_TOTAL",
    "FA_T2": "FA_T2_TOTAL",
    "FA_T3": "FA_T3_TOTAL",
    "FA_T4": "FA_T4_TOTAL",
    "RA_TOTAL": "RA_TOTAL",
    "RA_T1_TOTAL": "RA_T1_TOTAL",
    "RA_T2_TOTAL": "RA_T2_TOTAL",
    "RA_T3_TOTAL": "RA_T3_TOTAL",
    "RA_T4_TOTAL": "RA_T4_TOTAL",
    "FA_MD": "FA_MD",
    "FA_MD_T1": "FA_MD_T1",
    "FA_MD_T2": "FA_MD_T2",
    "FA_MD_T3": "FA_MD_T3",
    "FA_MD_T4": "FA_MD_T4",
}

# Expected number of 5-min slots in a full day
_SLOT_MINUTES = 5


# ── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_json_slot_map(value: Any, default_cadence_minutes: int = 5) -> dict[str, float]:
    """Parse a JSON column into a mapping from time string 'HH:MM' to float value.

    Preserves exact chronological timestamps, preventing time-shift and inter-column desynchronization.
    Handles:
    - JSON object: '{"08:00": 1.2, "08:10": 3.4}' -> {"08:00": 1.2, "08:10": 3.4}
    - JSON array: '[1.2, 3.4]' -> mapped to successive slots starting from 00:00
    - Plain numeric or None
    """
    if value is None:
        return {}
    if isinstance(value, (int, float)):
        fv = _coerce_float(value)
        return {"00:00": fv} if fv is not None else {}

    text = str(value).strip()
    if not text:
        return {}

    # 1. Try JSON object
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                result: dict[str, float] = {}
                for k, v in obj.items():
                    fv = _coerce_float(v)
                    if fv is not None:
                        k_str = str(k).strip()
                        if ":" in k_str:
                            parts = k_str.split(":", 1)
                            try:
                                h, m = int(parts[0]), int(parts[1])
                                norm_k = f"{h:02d}:{m:02d}"
                            except ValueError:
                                norm_k = k_str
                        else:
                            norm_k = k_str
                        result[norm_k] = fv
                return result
        except json.JSONDecodeError:
            pass

    # 2. Try JSON array
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                cadence = _infer_cadence_from_slot_count(len(arr)) if len(arr) > 0 else default_cadence_minutes
                result = {}
                for i, v in enumerate(arr):
                    fv = _coerce_float(v)
                    if fv is not None:
                        mins = i * cadence
                        h = (mins // 60) % 24
                        m = mins % 60
                        result[f"{h:02d}:{m:02d}"] = fv
                return result
        except json.JSONDecodeError:
            pass

    # 3. Plain numeric
    fv = _coerce_float(text)
    return {"00:00": fv} if fv is not None else {}


def _parse_json_slot(value: Any) -> list[float | None]:
    """Parse a JSON column containing hourly or 5-min slot values.

    Handles formats:
    - JSON object with "HH:MM" keys → list of 24 (hourly) or 288 (5-min)
    - JSON array
    - Plain numeric string (single value)
    - None / empty → list of Nones

    Returns a list of float | None values.
    """
    if value is None:
        return []

    if isinstance(value, (int, float)):
        return [float(value)]

    text = str(value).strip()
    if not text:
        return []

    # Try JSON object
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                # Sort by key to get chronological order
                sorted_items = sorted(obj.items(), key=lambda kv: kv[0])
                return [_coerce_float(v) for _, v in sorted_items]
        except json.JSONDecodeError:
            pass

    # Try JSON array
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [_coerce_float(v) for v in arr]
        except json.JSONDecodeError:
            pass

    # Plain numeric
    fv = _coerce_float(text)
    return [fv] if fv is not None else []


def _coerce_float(value: Any) -> float | None:
    """Convert a value to float, returning None for missing/invalid."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        fv = float(value)
        # Treat sentinel nulls as None
        if fv == -999 or fv == -1 or fv == 9999:
            return None
        return fv
    except (ValueError, TypeError):
        return None


# ── Cadence detection ────────────────────────────────────────────────────────

def detect_cadence(timestamps: list[datetime], *, default_minutes: int = 5) -> int:
    """Infer the predominant cadence from a list of timestamps.

    Returns the most common interval in minutes, or ``default_minutes``
    if detection fails.
    """
    if len(timestamps) < 2:
        return default_minutes

    deltas: dict[int, int] = {}
    for i in range(1, len(timestamps)):
        delta_seconds = int((timestamps[i] - timestamps[i - 1]).total_seconds())
        delta_minutes = round(delta_seconds / 60)
        if 1 <= delta_minutes <= 120:
            deltas[delta_minutes] = deltas.get(delta_minutes, 0) + 1

    if not deltas:
        return default_minutes

    return max(deltas, key=deltas.get)  # type: ignore[arg-type]


def compute_expected_points(cadence_minutes: int, duration_minutes: int = 1440) -> int:
    """Compute expected observation points for a given cadence.

    ``expected_points = duration_minutes / cadence_minutes``
    """
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be > 0")
    return duration_minutes // cadence_minutes


def compute_coverage(observed_valid: int, expected: int) -> float:
    """Compute coverage ratio.

    ``coverage = observed_valid_points / expected_points``
    """
    if expected <= 0:
        return 0.0
    return min(observed_valid / expected, 1.0)


# ── Normalization ────────────────────────────────────────────────────────────

def _slot_index_to_time(
    slot_index: int,
    report_day: date,
    cadence_minutes: int = _SLOT_MINUTES,
) -> datetime:
    """Convert a slot index to a datetime."""
    minutes = slot_index * cadence_minutes
    return datetime.combine(report_day, datetime.min.time()) + timedelta(minutes=minutes)


def _infer_cadence_from_slot_count(slot_count: int) -> int:
    """Infer cadence from the number of slots in a day.

    288 slots → 5 min, 96 slots → 15 min, 48 slots → 30 min, 24 slots → 60 min.
    """
    if slot_count <= 0:
        return _SLOT_MINUTES

    minutes_per_day = 1440
    raw_cadence = minutes_per_day / slot_count

    # Snap to standard cadences
    standard = [5, 10, 15, 30, 60]
    closest = min(standard, key=lambda c: abs(c - raw_cadence))
    return closest


def normalize_mdm_day(
    mdm_row: dict[str, Any],
    *,
    report_day: date,
    run_id: str = "",
    schema_version: str = "v1",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a single MDM row (with JSON columns) into three atomic DataFrames.

    Returns:
        (interval_df, instantaneous_df, register_df)
    """
    nio = mdm_row.get("NIO") or mdm_row.get("nio") or ""
    if not nio:
        empty_interval = pl.DataFrame(schema={
            "NIO": pl.String, "REPORT_DAY": pl.Date,
            "TIMESTAMP_UTC": pl.Datetime, "TIMESTAMP_LOCAL": pl.Datetime,
            "CADENCE_MINUTES": pl.Int16,
            "MEASUREMENT_SOURCE": pl.String,
            **{col: pl.Float64 for col in _INTERVAL_COLUMNS},
            "QUALITY_CODE": pl.Int16,
            "SOURCE_UPDATED_AT": pl.Datetime,
            "RUN_ID": pl.String,
        })
        empty_instant = empty_interval  # same empty shape
        empty_register = pl.DataFrame(schema={
            "NIO": pl.String, "REPORT_DAY": pl.Date,
            "TIMESTAMP_UTC": pl.Datetime,
            **{col: pl.Float64 for col in [
                "FA_TOTAL", "FA_T1_TOTAL", "FA_T2_TOTAL", "FA_T3_TOTAL", "FA_T4_TOTAL",
                "RA_TOTAL", "RA_T1_TOTAL", "RA_T2_TOTAL", "RA_T3_TOTAL", "RA_T4_TOTAL",
                "FA_MD", "FA_MD_T1", "FA_MD_T2", "FA_MD_T3", "FA_MD_T4",
            ]},
            "QUALITY_CODE": pl.Int16,
            "SOURCE_UPDATED_AT": pl.Datetime,
            "RUN_ID": pl.String,
        })
        return empty_interval, empty_instant, empty_register

    now = datetime.now(timezone.utc)

    # Parse interval columns as timestamp maps
    interval_maps: dict[str, dict[str, float]] = {}
    for col in _INTERVAL_COLUMNS:
        m = _parse_json_slot_map(mdm_row.get(col), default_cadence_minutes=_SLOT_MINUTES)
        if m:
            interval_maps[col] = m

    # Parse instantaneous columns as timestamp maps
    instant_maps: dict[str, dict[str, float]] = {}
    for col in _INSTANTANEOUS_COLUMNS:
        m = _parse_json_slot_map(mdm_row.get(col), default_cadence_minutes=60)
        if m:
            instant_maps[col] = m

    # Parse register columns as timestamp maps
    register_maps: dict[str, dict[str, float]] = {}
    for col in _REGISTER_COLUMNS:
        m = _parse_json_slot_map(mdm_row.get(col), default_cadence_minutes=360)
        if m:
            register_maps[col] = m

    # ── Build interval rows (exact chronological alignment) ──────────
    interval_keys = sorted(set().union(*(m.keys() for m in interval_maps.values()))) if interval_maps else []
    interval_rows = []
    if interval_keys:
        parsed_dt = []
        for k in interval_keys:
            if ":" in k:
                try:
                    h, m = map(int, k.split(":", 1))
                    parsed_dt.append(datetime.combine(report_day, time(h, m)))
                except ValueError:
                    pass
        interval_cadence = detect_cadence(parsed_dt, default_minutes=_SLOT_MINUTES) if len(parsed_dt) >= 2 else _SLOT_MINUTES

        for k in interval_keys:
            if ":" in k:
                try:
                    h, m = map(int, k.split(":", 1))
                    ts = datetime.combine(report_day, time(h, m))
                except ValueError:
                    continue
            else:
                continue

            row: dict[str, Any] = {
                "NIO": nio,
                "REPORT_DAY": report_day,
                "TIMESTAMP_UTC": ts,
                "TIMESTAMP_LOCAL": ts,
                "CADENCE_MINUTES": interval_cadence,
                "MEASUREMENT_SOURCE": "MDM",
            }
            has_data = False
            for col in _INTERVAL_COLUMNS:
                val = interval_maps.get(col, {}).get(k)
                row[col] = val
                if val is not None:
                    has_data = True
            row["QUALITY_CODE"] = 0
            row["SOURCE_UPDATED_AT"] = now
            row["RUN_ID"] = run_id

            if has_data:
                interval_rows.append(row)

    # ── Build instantaneous rows (exact chronological alignment) ─────
    instant_keys = sorted(set().union(*(m.keys() for m in instant_maps.values()))) if instant_maps else []
    instant_rows = []
    if instant_keys:
        parsed_dt = []
        for k in instant_keys:
            if ":" in k:
                try:
                    h, m = map(int, k.split(":", 1))
                    parsed_dt.append(datetime.combine(report_day, time(h, m)))
                except ValueError:
                    pass
        instant_cadence = detect_cadence(parsed_dt, default_minutes=60) if len(parsed_dt) >= 2 else 60

        for k in instant_keys:
            if ":" in k:
                try:
                    h, m = map(int, k.split(":", 1))
                    ts = datetime.combine(report_day, time(h, m))
                except ValueError:
                    continue
            else:
                continue

            row = {
                "NIO": nio,
                "REPORT_DAY": report_day,
                "TIMESTAMP_UTC": ts,
                "TIMESTAMP_LOCAL": ts,
                "CADENCE_MINUTES": instant_cadence,
                "MEASUREMENT_SOURCE": "MDM",
            }
            has_data = False
            for col in _INSTANTANEOUS_COLUMNS:
                val = instant_maps.get(col, {}).get(k)
                row[col] = val
                if val is not None:
                    has_data = True
            row["QUALITY_CODE"] = 0
            row["SOURCE_UPDATED_AT"] = now
            row["RUN_ID"] = run_id

            if has_data:
                instant_rows.append(row)

    # ── Build register rows (exact chronological alignment) ──────────
    register_keys = sorted(set().union(*(m.keys() for m in register_maps.values()))) if register_maps else []
    register_rows = []
    if register_keys:
        for k in register_keys:
            if ":" in k:
                try:
                    h, m = map(int, k.split(":", 1))
                    ts = datetime.combine(report_day, time(h, m))
                except ValueError:
                    continue
            else:
                continue

            row = {
                "NIO": nio,
                "REPORT_DAY": report_day,
                "TIMESTAMP_UTC": ts,
                "REGISTER_GROUP": "DAILY",
            }
            has_data = False
            for col in _REGISTER_COLUMNS:
                schema_col = _REGISTER_COLUMN_MAP[col]
                val = register_maps.get(col, {}).get(k)
                row[schema_col] = val
                if val is not None:
                    has_data = True
            row["QUALITY_CODE"] = 0
            row["SOURCE_UPDATED_AT"] = now
            row["RUN_ID"] = run_id

            if has_data:
                register_rows.append(row)

    # Build DataFrames
    interval_df = pl.DataFrame(
        interval_rows,
        schema={
            "NIO": pl.String, "REPORT_DAY": pl.Date,
            "TIMESTAMP_UTC": pl.Datetime, "TIMESTAMP_LOCAL": pl.Datetime,
            "CADENCE_MINUTES": pl.Int16,
            "MEASUREMENT_SOURCE": pl.String,
            **{col: pl.Float64 for col in _INTERVAL_COLUMNS},
            "QUALITY_CODE": pl.Int16,
            "SOURCE_UPDATED_AT": pl.Datetime,
            "RUN_ID": pl.String,
        },
        strict=False,
    ) if interval_rows else _empty_interval_frame()

    instant_df = pl.DataFrame(
        instant_rows,
        schema={
            "NIO": pl.String, "REPORT_DAY": pl.Date,
            "TIMESTAMP_UTC": pl.Datetime, "TIMESTAMP_LOCAL": pl.Datetime,
            "CADENCE_MINUTES": pl.Int16,
            "MEASUREMENT_SOURCE": pl.String,
            **{col: pl.Float64 for col in _INSTANTANEOUS_COLUMNS},
            "QUALITY_CODE": pl.Int16,
            "SOURCE_UPDATED_AT": pl.Datetime,
            "RUN_ID": pl.String,
        },
        strict=False,
    ) if instant_rows else _empty_instantaneous_frame()

    register_df = pl.DataFrame(
        register_rows,
        schema={
            "NIO": pl.String, "REPORT_DAY": pl.Date,
            "TIMESTAMP_UTC": pl.Datetime,
            "REGISTER_GROUP": pl.String,
            **{col: pl.Float64 for col in [
                "FA_TOTAL", "FA_T1_TOTAL", "FA_T2_TOTAL", "FA_T3_TOTAL", "FA_T4_TOTAL",
                "RA_TOTAL", "RA_T1_TOTAL", "RA_T2_TOTAL", "RA_T3_TOTAL", "RA_T4_TOTAL",
                "FA_MD", "FA_MD_T1", "FA_MD_T2", "FA_MD_T3", "FA_MD_T4",
            ]},
            "QUALITY_CODE": pl.Int16,
            "SOURCE_UPDATED_AT": pl.Datetime,
            "RUN_ID": pl.String,
        },
        strict=False,
    ) if register_rows else _empty_register_frame()

    return interval_df, instant_df, register_df


def _empty_interval_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "NIO": pl.String, "REPORT_DAY": pl.Date,
        "TIMESTAMP_UTC": pl.Datetime, "TIMESTAMP_LOCAL": pl.Datetime,
        "CADENCE_MINUTES": pl.Int16,
        "MEASUREMENT_SOURCE": pl.String,
        **{col: pl.Float64 for col in _INTERVAL_COLUMNS},
        "QUALITY_CODE": pl.Int16,
        "SOURCE_UPDATED_AT": pl.Datetime,
        "RUN_ID": pl.String,
    })


def _empty_instantaneous_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "NIO": pl.String, "REPORT_DAY": pl.Date,
        "TIMESTAMP_UTC": pl.Datetime, "TIMESTAMP_LOCAL": pl.Datetime,
        "CADENCE_MINUTES": pl.Int16,
        "MEASUREMENT_SOURCE": pl.String,
        **{col: pl.Float64 for col in _INSTANTANEOUS_COLUMNS},
        "QUALITY_CODE": pl.Int16,
        "SOURCE_UPDATED_AT": pl.Datetime,
        "RUN_ID": pl.String,
    })


def _empty_register_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "NIO": pl.String, "REPORT_DAY": pl.Date,
        "TIMESTAMP_UTC": pl.Datetime,
        "REGISTER_GROUP": pl.String,
        **{col: pl.Float64 for col in [
            "FA_TOTAL", "FA_T1_TOTAL", "FA_T2_TOTAL", "FA_T3_TOTAL", "FA_T4_TOTAL",
            "RA_TOTAL", "RA_T1_TOTAL", "RA_T2_TOTAL", "RA_T3_TOTAL", "RA_T4_TOTAL",
            "FA_MD", "FA_MD_T1", "FA_MD_T2", "FA_MD_T3", "FA_MD_T4",
        ]},
        "QUALITY_CODE": pl.Int16,
        "SOURCE_UPDATED_AT": pl.Datetime,
        "RUN_ID": pl.String,
    })


def normalize_mdm_dataframe(
    mdm_df: pl.DataFrame,
    *,
    report_day: date,
    run_id: str = "",
    schema_version: str = "v1",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Normalize an entire MDM DataFrame (multiple NIOs) into atomic tables in memory-safe chunks.

    Returns:
        (interval_df, instantaneous_df, register_df)
    """
    if mdm_df.is_empty():
        return _empty_interval_frame(), _empty_instantaneous_frame(), _empty_register_frame()

    interval_chunks: list[pl.DataFrame] = []
    instant_chunks: list[pl.DataFrame] = []
    register_chunks: list[pl.DataFrame] = []

    cur_interval: list[pl.DataFrame] = []
    cur_instant: list[pl.DataFrame] = []
    cur_register: list[pl.DataFrame] = []

    chunk_size = 500
    for idx, row_dict in enumerate(mdm_df.iter_rows(named=True), start=1):
        i_df, ins_df, reg_df = normalize_mdm_day(
            row_dict,
            report_day=report_day,
            run_id=run_id,
            schema_version=schema_version,
        )
        if i_df.height > 0:
            cur_interval.append(i_df)
        if ins_df.height > 0:
            cur_instant.append(ins_df)
        if reg_df.height > 0:
            cur_register.append(reg_df)

        if idx % chunk_size == 0:
            if cur_interval:
                interval_chunks.append(pl.concat(cur_interval, how="vertical_relaxed"))
                cur_interval.clear()
            if cur_instant:
                instant_chunks.append(pl.concat(cur_instant, how="vertical_relaxed"))
                cur_instant.clear()
            if cur_register:
                register_chunks.append(pl.concat(cur_register, how="vertical_relaxed"))
                cur_register.clear()

    if cur_interval:
        interval_chunks.append(pl.concat(cur_interval, how="vertical_relaxed"))
    if cur_instant:
        instant_chunks.append(pl.concat(cur_instant, how="vertical_relaxed"))
    if cur_register:
        register_chunks.append(pl.concat(cur_register, how="vertical_relaxed"))

    combined_interval = (
        pl.concat(interval_chunks, how="vertical_relaxed")
        if interval_chunks
        else _empty_interval_frame()
    )
    combined_instant = (
        pl.concat(instant_chunks, how="vertical_relaxed")
        if instant_chunks
        else _empty_instantaneous_frame()
    )
    combined_register = (
        pl.concat(register_chunks, how="vertical_relaxed")
        if register_chunks
        else _empty_register_frame()
    )

    return combined_interval, combined_instant, combined_register


def _normalize_single_batch_file(
    item: tuple[int, Path],
    *,
    report_day: date,
    run_id: str,
    temp_dir: Path,
    compression: str,
) -> tuple[int, Path | None, Path | None, Path | None, int, int, int]:
    idx, raw_path = item
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return idx, None, None, None, 0, 0, 0
    try:
        batch_df = pl.read_parquet(raw_path)
    except Exception:
        return idx, None, None, None, 0, 0, 0
    if batch_df.is_empty():
        return idx, None, None, None, 0, 0, 0

    i_df, ins_df, reg_df = normalize_mdm_dataframe(
        batch_df,
        report_day=report_day,
        run_id=run_id,
    )

    p_i: Path | None = None
    p_ins: Path | None = None
    p_reg: Path | None = None
    i_rows = i_df.height
    ins_rows = ins_df.height
    reg_rows = reg_df.height

    if i_rows > 0:
        p_i = temp_dir / f"interval_chunk_{idx:05d}.parquet"
        i_df.write_parquet(p_i, compression=compression)
    if ins_rows > 0:
        p_ins = temp_dir / f"instant_chunk_{idx:05d}.parquet"
        ins_df.write_parquet(p_ins, compression=compression)
    if reg_rows > 0:
        p_reg = temp_dir / f"register_chunk_{idx:05d}.parquet"
        reg_df.write_parquet(p_reg, compression=compression)

    return idx, p_i, p_ins, p_reg, i_rows, ins_rows, reg_rows


def normalize_and_sink_mdm_batches(
    raw_parquet_files: Sequence[Path],
    *,
    report_day: date,
    interval_final_parquet: Path,
    instant_final_parquet: Path,
    register_final_parquet: Path,
    temp_dir: Path,
    run_id: str = "",
    write_csv: bool = False,
    compression: str = "zstd",
    max_workers: int = 5,
) -> tuple[int, int, int]:
    """Lazy-sink normalizer: processes raw MDM batch files in parallel (up to max_workers),

    writes chunk parquets, and streams them to final partitioned parquets using Polars scan/sink.
    Memory footprint remains strictly bounded to one batch per worker.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    temp_dir.mkdir(parents=True, exist_ok=True)
    i_parts: list[Path] = []
    ins_parts: list[Path] = []
    reg_parts: list[Path] = []

    total_interval_rows = 0
    total_instant_rows = 0
    total_register_rows = 0

    items = list(enumerate(raw_parquet_files, start=1))
    results: list[tuple[int, Path | None, Path | None, Path | None, int, int, int]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _normalize_single_batch_file,
                item,
                report_day=report_day,
                run_id=run_id,
                temp_dir=temp_dir,
                compression=compression,
            )
            for item in items
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r[0])
    for _, p_i, p_ins, p_reg, i_rows, ins_rows, reg_rows in results:
        if p_i is not None:
            i_parts.append(p_i)
            total_interval_rows += i_rows
        if p_ins is not None:
            ins_parts.append(p_ins)
            total_instant_rows += ins_rows
        if p_reg is not None:
            reg_parts.append(p_reg)
            total_register_rows += reg_rows

    # Stream to final destination
    interval_final_parquet.parent.mkdir(parents=True, exist_ok=True)
    if i_parts:
        lf = pl.concat([pl.scan_parquet(str(p)) for p in i_parts], how="vertical_relaxed")
        lf.sink_parquet(interval_final_parquet, compression=compression)
        if write_csv:
            lf.sink_csv(interval_final_parquet.with_suffix(".csv"), separator=";")
    else:
        _empty_interval_frame().write_parquet(interval_final_parquet, compression=compression)

    instant_final_parquet.parent.mkdir(parents=True, exist_ok=True)
    if ins_parts:
        lf = pl.concat([pl.scan_parquet(str(p)) for p in ins_parts], how="vertical_relaxed")
        lf.sink_parquet(instant_final_parquet, compression=compression)
        if write_csv:
            lf.sink_csv(instant_final_parquet.with_suffix(".csv"), separator=";")
    else:
        _empty_instantaneous_frame().write_parquet(instant_final_parquet, compression=compression)

    register_final_parquet.parent.mkdir(parents=True, exist_ok=True)
    if reg_parts:
        lf = pl.concat([pl.scan_parquet(str(p)) for p in reg_parts], how="vertical_relaxed")
        lf.sink_parquet(register_final_parquet, compression=compression)
        if write_csv:
            lf.sink_csv(register_final_parquet.with_suffix(".csv"), separator=";")
    else:
        _empty_register_frame().write_parquet(register_final_parquet, compression=compression)

    return total_interval_rows, total_instant_rows, total_register_rows


# ── Context Layer Normalization ──────────────────────────────────────────────

def normalize_uc_context(
    cis_df: pl.DataFrame,
    geo_df: pl.DataFrame | None = None,
    *,
    run_id: str = "",
) -> pl.DataFrame:
    """Normalize CIS and GEO data into the canonical uc_context table.

    Grain: 1 row per UC.
    Excludes textual address fields as agreed in architectural rules.
    """
    if cis_df.is_empty():
        from src.datasets.schemas import build_empty_frame
        return build_empty_frame("uc_context")

    now = datetime.now(timezone.utc)
    # Ensure column lookup is case-insensitive
    cols = {c.upper(): c for c in cis_df.columns}

    uc_col = cols.get("UC")
    if not uc_col:
        from src.datasets.schemas import build_empty_frame
        return build_empty_frame("uc_context")

    # Select and rename available context columns
    exprs: list[pl.Expr] = [pl.col(uc_col).cast(pl.String).alias("UC")]

    mappings = {
        "COD_LOCALIDADE": ("COD_LOCALIDADE", "COD_LOC_UEE", "LOCALIDADE"),
        "COD_GRUPO_FAT": ("COD_GRUPO_FAT", "COD_GRU_TENS_FAT_UEE", "GRUPO"),
        "COD_SUB_GRUPO_FAT": ("COD_SUB_GRUPO_FAT", "COD_SUB_GRU_FAT_UEE", "SUB_GRUPO"),
        "TARIFA_FATURA": ("TARIFA_FATURA", "COD_TIPO_TAR_FAT_UEE", "TARIFA"),
        "CLASSE": ("CLASSE", "CLASSE_CONSUMO", "COD_CLAS_CONS_UEE"),
        "FASE_CIRCUITO": ("FASE_CIRCUITO", "TIPO_FASE", "COD_TIPO_FASE_UEE"),
        "TENSAO_BASE": ("TENSAO_BASE", "QTD_TENS_LIG_UEE", "TENSAO"),
        "TENSAO_SEC": ("TENSAO_SEC", "TENSAO_SECUNDARIA"),
        "DEMANDA": ("DEMANDA", "POTENCIA_CONTRATADA"),
        "STATUS_FAT": ("STATUS_FAT", "SITUACAO_FAT"),
        "DISJUNTOR": ("DISJUNTOR", "COD_TIPO_DISJ_UEE"),
        "FASE_LIGADA": ("FASE_LIGADA",),
        "CONSUMO_ESTM": ("CONSUMO_ESTM", "CONSUMO_ESTIMADO"),
        "INICIO_UC": ("INICIO_UC", "DTA_INIC_UEE"),
        "RECEBIMENTO_FATURA": ("RECEBIMENTO_FATURA", "COD_TIPO_ENTG_UEE", "TIPO_ENTREGA"),
        "STATUS_LIGACAO": ("STATUS_LIGACAO", "SITUACAO_UC", "COD_SITU_UEE"),
        "TIPO_UC": ("TIPO_UC",),
        "LOC_UB_RR": ("LOC_UB_RR", "URBANO_RURAL"),
        "MUNICIPIO": ("MUNICIPIO", "NOM_MUN_MUN"),
        "BAIRRO": ("BAIRRO", "NOM_BAI_BAI"),
        "LATITUDE": ("LATITUDE", "LAT", "NUM_COORY_XXX"),
        "LONGITUDE": ("LONGITUDE", "LON", "LONG", "NUM_COORX_XXX"),
    }

    for target_col, candidates in mappings.items():
        found = False
        for cand in candidates:
            if cand in cols:
                source_col = cols[cand]
                if target_col in ("TENSAO_BASE", "TENSAO_SEC", "DEMANDA", "CONSUMO_ESTM", "LATITUDE", "LONGITUDE"):
                    exprs.append(pl.col(source_col).cast(pl.Float64, strict=False).alias(target_col))
                elif target_col == "INICIO_UC":
                    exprs.append(pl.col(source_col).cast(pl.Date, strict=False).alias(target_col))
                else:
                    exprs.append(pl.col(source_col).cast(pl.String, strict=False).alias(target_col))
                found = True
                break
        if not found:
            if target_col in ("TENSAO_BASE", "TENSAO_SEC", "DEMANDA", "CONSUMO_ESTM", "LATITUDE", "LONGITUDE"):
                exprs.append(pl.lit(None, dtype=pl.Float64).alias(target_col))
            elif target_col == "INICIO_UC":
                exprs.append(pl.lit(None, dtype=pl.Date).alias(target_col))
            else:
                exprs.append(pl.lit(None, dtype=pl.String).alias(target_col))

    exprs.append(pl.lit(now).alias("SOURCE_UPDATED_AT"))
    exprs.append(pl.lit(run_id).alias("RUN_ID"))

    context_df = (
        cis_df.select(exprs)
        .unique(subset=["UC"], keep="first")
        .sort("UC")
    )

    return context_df


def normalize_meter_installation_history(
    cis_df: pl.DataFrame,
    *,
    run_id: str = "",
) -> pl.DataFrame:
    """Normalize meter installations linked to UCs.

    Grain: UC × installation (NIO).
    Preserves historical installation and removal dates for temporal association.
    """
    if cis_df.is_empty():
        from src.datasets.schemas import build_empty_frame
        return build_empty_frame("meter_installation_history")

    now = datetime.now(timezone.utc)
    cols = {c.upper(): c for c in cis_df.columns}

    uc_col = cols.get("UC")
    nio_col = cols.get("NIO")
    if not uc_col or not nio_col:
        from src.datasets.schemas import build_empty_frame
        return build_empty_frame("meter_installation_history")

    exprs: list[pl.Expr] = [
        pl.col(uc_col).cast(pl.String).alias("UC"),
        pl.col(nio_col).cast(pl.String).alias("NIO"),
    ]

    # DATA_INSTALACAO
    dta_ins = cols.get("DATA_INSTALACAO_MEDIDOR") or cols.get("DATA_INSTALACAO") or cols.get("DTA_INS_REU")
    if dta_ins:
        exprs.append(pl.col(dta_ins).cast(pl.Date, strict=False).alias("DATA_INSTALACAO"))
    else:
        exprs.append(pl.lit(None, dtype=pl.Date).alias("DATA_INSTALACAO"))

    # DATA_REMOCAO
    dta_ret = cols.get("DATA_RETIRADA_MEDIDOR") or cols.get("DATA_REMOCAO") or cols.get("DTA_RETI_REU")
    if dta_ret:
        exprs.append(pl.col(dta_ret).cast(pl.Date, strict=False).alias("DATA_REMOCAO"))
    else:
        exprs.append(pl.lit(None, dtype=pl.Date).alias("DATA_REMOCAO"))

    # COD_SUBTIPO
    cod_sub = cols.get("COD_SUBTIPO_MEDIDOR") or cols.get("COD_SUBTIPO") or cols.get("COD_SUB_TIPO_EQIP_EMD")
    exprs.append(pl.col(cod_sub).cast(pl.String, strict=False).alias("COD_SUBTIPO") if cod_sub else pl.lit(None, dtype=pl.String).alias("COD_SUBTIPO"))

    # DESCRICAO_TIPO
    des_tipo = cols.get("TIPO_MEDIDOR") or cols.get("DESCRICAO_TIPO") or cols.get("DES_SUB_TIPO_EQIP_STE")
    exprs.append(pl.col(des_tipo).cast(pl.String, strict=False).alias("DESCRICAO_TIPO") if des_tipo else pl.lit(None, dtype=pl.String).alias("DESCRICAO_TIPO"))

    # FAMILIA_SMART
    smart = cols.get("SMART") or cols.get("FAMILIA_SMART")
    exprs.append(pl.col(smart).cast(pl.String, strict=False).alias("FAMILIA_SMART") if smart else pl.lit(None, dtype=pl.String).alias("FAMILIA_SMART"))

    # FASE
    fase = cols.get("FASE") or cols.get("TIPO_FASE") or cols.get("COD_TIPO_FASE_UEE")
    exprs.append(pl.col(fase).cast(pl.String, strict=False).alias("FASE") if fase else pl.lit(None, dtype=pl.String).alias("FASE"))

    exprs.append(pl.lit(now).alias("SOURCE_UPDATED_AT"))
    exprs.append(pl.lit(run_id).alias("RUN_ID"))

    meter_df = (
        cis_df.select(exprs)
        .filter(pl.col("UC").is_not_null() & pl.col("NIO").is_not_null())
        .unique(subset=["UC", "NIO"], keep="first")
        .sort(["UC", "NIO"])
    )

    return meter_df


def normalize_electrical_hierarchy(
    cis_df: pl.DataFrame,
    geo_df: pl.DataFrame | None = None,
    *,
    run_id: str = "",
) -> pl.DataFrame:
    """Normalize topological network hierarchy:
    Subestação -> Alimentador -> Posto/Transformador -> UC -> NIO.
    """
    if cis_df.is_empty():
        from src.datasets.schemas import build_empty_frame
        return build_empty_frame("electrical_hierarchy")

    now = datetime.now(timezone.utc)
    source_df = cis_df

    if geo_df is not None and not geo_df.is_empty():
        join_col = "UC" if "UC" in geo_df.columns and "UC" in cis_df.columns else None
        if join_col:
            source_df = cis_df.join(
                geo_df.with_columns(pl.col(join_col).cast(pl.String)),
                on=join_col,
                how="left",
                suffix="_GEO",
            )

    cols = {c.upper(): c for c in source_df.columns}

    uc_col = cols.get("UC")
    nio_col = cols.get("NIO")

    exprs: list[pl.Expr] = [
        pl.col(uc_col).cast(pl.String).alias("UC") if uc_col else pl.lit("").alias("UC"),
        pl.col(nio_col).cast(pl.String).alias("NIO") if nio_col else pl.lit(None, dtype=pl.String).alias("NIO"),
    ]

    string_fields = [
        ("SUBESTACAO", ("SUBESTACAO", "GEO_SUBESTACAO")),
        ("SIGLA_SE", ("SIGLA_SE", "GEO_SIGLA_SE")),
        ("CAR_SE", ("CAR_SE", "GEO_CAR_SE")),
        ("MUNICIPIO_SE", ("MUNICIPIO_SE", "GEO_MUNICIPIO_SE")),
        ("COD_MUN_SE", ("COD_MUN_SE", "GEO_COD_MUN_SE")),
        ("GEDIS_ALIMENTADOR", ("GEDIS_ALIMENTADOR", "GEO_GEDIS_ALIMENTADOR")),
        ("ALIMENTADOR", ("ALIMENTADOR", "GEO_ALIMENTADOR", "FEEDER_ID")),
        ("POSTO_OPERACIONAL", ("POSTO_OPERACIONAL", "GEO_POSTO_OPERACIONAL")),
    ]

    float_fields = [
        ("TENSAO_SE", ("TENSAO_SE", "GEO_TENSAO_SE")),
        ("TENSAO_ALIMENTADOR", ("TENSAO_ALIMENTADOR", "GEO_TENSAO_ALIMENTADOR")),
        ("POT_INST_KVA", ("POT_INST_KVA", "GEO_POT_INST_KVA", "INSTALLED_KVA")),
        ("COORD_X_POSTE", ("COORD_X_POSTE", "GEO_COORD_X_POSTE")),
        ("COORD_Y_POSTE", ("COORD_Y_POSTE", "GEO_COORD_Y_POSTE")),
        ("LAT_POSTE_WGS84", ("LAT_POSTE_WGS84", "GEO_LAT_POSTE_WGS84")),
        ("LONG_POSTE_WGS84", ("LONG_POSTE_WGS84", "GEO_LONG_POSTE_WGS84")),
        ("LAT_UC", ("LAT_UC", "LAT", "LATITUDE")),
        ("LONG_UC", ("LONG_UC", "LON", "LONGITUDE")),
    ]

    for target_col, candidates in string_fields:
        found = next((cols[c] for c in candidates if c in cols), None)
        if found:
            exprs.append(pl.col(found).cast(pl.String, strict=False).alias(target_col))
        else:
            exprs.append(pl.lit(None, dtype=pl.String).alias(target_col))

    for target_col, candidates in float_fields:
        found = next((cols[c] for c in candidates if c in cols), None)
        if found:
            exprs.append(pl.col(found).cast(pl.Float64, strict=False).alias(target_col))
        else:
            exprs.append(pl.lit(None, dtype=pl.Float64).alias(target_col))

    exprs.append(pl.lit(now).alias("SOURCE_UPDATED_AT"))
    exprs.append(pl.lit(run_id).alias("RUN_ID"))

    hierarchy_df = (
        source_df.select(exprs)
        .unique(subset=["UC"], keep="first")
        .sort("UC")
    )

    return hierarchy_df


def normalize_alarm_events(
    alarm_df: pl.DataFrame,
    *,
    run_id: str = "",
) -> pl.DataFrame:
    """Normalize raw alarm events into the canonical alarm_events table.

    Grain: ALARM_ID.
    OBJ_ID represents 8-digit NIO without zero-padding.
    """
    if alarm_df.is_empty():
        from src.datasets.schemas import build_empty_frame
        return build_empty_frame("alarm_events")

    now = datetime.now(timezone.utc)
    cols = {c.upper(): c for c in alarm_df.columns}

    nio_col = cols.get("OBJ_ID") or cols.get("NIO")
    origin_col = cols.get("ORIGIN_TIMESTAMP") or cols.get("DATA_ORIGEM") or cols.get("TIMESTAMP_ORIGIN")
    received_col = cols.get("RECEIVED_TIMESTAMP") or cols.get("DATA_RECEPCAO") or cols.get("TIMESTAMP_RECEIVED")

    exprs: list[pl.Expr] = [
        (pl.col(cols["ALARM_ID"]).cast(pl.String) if "ALARM_ID" in cols else pl.lit("").alias("ALARM_ID")),
        (pl.col(nio_col).cast(pl.String).alias("NIO") if nio_col else pl.lit("").alias("NIO")),
        (pl.col(origin_col).cast(pl.Datetime).alias("ORIGIN_TIMESTAMP") if origin_col else pl.lit(None, dtype=pl.Datetime).alias("ORIGIN_TIMESTAMP")),
        (pl.col(received_col).cast(pl.Datetime).alias("RECEIVED_TIMESTAMP") if received_col else pl.lit(None, dtype=pl.Datetime).alias("RECEIVED_TIMESTAMP")),
    ]

    # Derived latency in seconds
    if origin_col and received_col:
        exprs.append(
            (pl.col(received_col).cast(pl.Datetime) - pl.col(origin_col).cast(pl.Datetime))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("LATENCY_SECONDS")
        )
    else:
        exprs.append(pl.lit(None, dtype=pl.Float64).alias("LATENCY_SECONDS"))

    for fld in ("SYSTEM_CODE", "CONTENT", "OBJECT_TYPE", "OBJECT_ORG"):
        found = cols.get(fld)
        exprs.append(pl.col(found).cast(pl.String).alias(fld) if found else pl.lit(None, dtype=pl.String).alias(fld))

    exprs.append(pl.lit(now).alias("SOURCE_UPDATED_AT"))
    exprs.append(pl.lit(run_id).alias("RUN_ID"))

    return alarm_df.select(exprs)

