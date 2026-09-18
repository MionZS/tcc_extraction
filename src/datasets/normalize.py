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
from datetime import date, datetime, timezone, timedelta
from typing import Any

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

    # Parse interval columns
    interval_slots: dict[str, list[float | None]] = {}
    for col in _INTERVAL_COLUMNS:
        raw = mdm_row.get(col)
        parsed = _parse_json_slot(raw)
        if parsed:
            interval_slots[col] = parsed

    # Parse instantaneous columns
    instant_slots: dict[str, list[float | None]] = {}
    for col in _INSTANTANEOUS_COLUMNS:
        raw = mdm_row.get(col)
        parsed = _parse_json_slot(raw)
        if parsed:
            instant_slots[col] = parsed

    # Parse register columns (typically 4 snapshots per day)
    register_slots: dict[str, list[float | None]] = {}
    for col in _REGISTER_COLUMNS:
        raw = mdm_row.get(col)
        parsed = _parse_json_slot(raw)
        if parsed:
            register_slots[col] = parsed

    # Determine slot count and cadence for interval data
    max_interval_slots = max((len(v) for v in interval_slots.values()), default=0)
    if max_interval_slots == 0:
        # Try instantaneous
        max_interval_slots = max((len(v) for v in instant_slots.values()), default=0)

    cadence = _infer_cadence_from_slot_count(max_interval_slots) if max_interval_slots > 0 else _SLOT_MINUTES
    slot_count = max_interval_slots if max_interval_slots > 0 else 288

    # Build interval rows
    interval_rows = []
    for i in range(slot_count):
        ts = _slot_index_to_time(i, report_day, cadence)
        row: dict[str, Any] = {
            "NIO": nio,
            "REPORT_DAY": report_day,
            "TIMESTAMP_UTC": ts,
            "TIMESTAMP_LOCAL": ts,  # TODO: timezone conversion
            "CADENCE_MINUTES": cadence,
            "MEASUREMENT_SOURCE": "MDM",
        }
        has_data = False
        for col in _INTERVAL_COLUMNS:
            values = interval_slots.get(col, [])
            val = values[i] if i < len(values) else None
            row[col] = val
            if val is not None:
                has_data = True
        row["QUALITY_CODE"] = 0
        row["SOURCE_UPDATED_AT"] = now
        row["RUN_ID"] = run_id

        if has_data:
            interval_rows.append(row)

    # Build instantaneous rows
    instant_cadence = _infer_cadence_from_slot_count(
        max((len(v) for v in instant_slots.values()), default=288)
    ) if instant_slots else 60
    instant_count = max((len(v) for v in instant_slots.values()), default=0)

    instant_rows = []
    for i in range(instant_count):
        ts = _slot_index_to_time(i, report_day, instant_cadence)
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
            values = instant_slots.get(col, [])
            val = values[i] if i < len(values) else None
            row[col] = val
            if val is not None:
                has_data = True
        row["QUALITY_CODE"] = 0
        row["SOURCE_UPDATED_AT"] = now
        row["RUN_ID"] = run_id

        if has_data:
            instant_rows.append(row)

    # Build register rows
    register_count = max((len(v) for v in register_slots.values()), default=0)
    # Registers are typically 4 snapshots at 00:00, 06:00, 12:00, 18:00
    register_cadence = 360  # 6 hours in minutes

    register_rows = []
    for i in range(register_count):
        ts = _slot_index_to_time(i, report_day, register_cadence)
        row = {
            "NIO": nio,
            "REPORT_DAY": report_day,
            "TIMESTAMP_UTC": ts,
            "REGISTER_GROUP": "DAILY",
        }
        has_data = False
        for col in _REGISTER_COLUMNS:
            schema_col = _REGISTER_COLUMN_MAP[col]
            values = register_slots.get(col, [])
            val = values[i] if i < len(values) else None
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
    """Normalize an entire MDM DataFrame (multiple NIOs) into atomic tables.

    This is the batch version of :func:`normalize_mdm_day`.

    Returns:
        (interval_df, instantaneous_df, register_df)
    """
    all_interval: list[pl.DataFrame] = []
    all_instant: list[pl.DataFrame] = []
    all_register: list[pl.DataFrame] = []

    for row_dict in mdm_df.iter_rows(named=True):
        interval_df, instant_df, register_df = normalize_mdm_day(
            row_dict,
            report_day=report_day,
            run_id=run_id,
            schema_version=schema_version,
        )
        if interval_df.height > 0:
            all_interval.append(interval_df)
        if instant_df.height > 0:
            all_instant.append(instant_df)
        if register_df.height > 0:
            all_register.append(register_df)

    combined_interval = (
        pl.concat(all_interval, how="vertical_relaxed")
        if all_interval
        else _empty_interval_frame()
    )
    combined_instant = (
        pl.concat(all_instant, how="vertical_relaxed")
        if all_instant
        else _empty_instantaneous_frame()
    )
    combined_register = (
        pl.concat(all_register, how="vertical_relaxed")
        if all_register
        else _empty_register_frame()
    )

    return combined_interval, combined_instant, combined_register


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

