"""Schema definitions for normalized smart-meter datasets.

Following the multi-layer semantic architecture:
- AMI is phenomenon (ami_interval, ami_instantaneous, ami_registers)
- Cadastro, Meter History, and GEO are context (uc_context, meter_installation_history, electrical_hierarchy)
- Alarms are events (alarm_events)
- Features & Training Data: projected onto unified analytical unit (UC × cutoff_date)

Column names are UPPER_CASE matching database & Polars standards.
"""

from __future__ import annotations

import polars as pl


# ── Context Layer ─────────────────────────────────────────────────────────────

# 1.1 uc_context
# Grain: UC (1 row per consumer unit)
# Purpose: Static / commercial and geographical context of the consumer unit.
UC_CONTEXT_SCHEMA = {
    "UC": pl.String,
    "COD_LOCALIDADE": pl.String,
    "COD_GRUPO_FAT": pl.String,
    "COD_SUB_GRUPO_FAT": pl.String,
    "TARIFA_FATURA": pl.String,
    "CLASSE": pl.String,
    "FASE_CIRCUITO": pl.String,
    "TENSAO_BASE": pl.Float64,
    "TENSAO_SEC": pl.Float64,
    "DEMANDA": pl.Float64,
    "STATUS_FAT": pl.String,
    "DISJUNTOR": pl.String,
    "FASE_LIGADA": pl.String,
    "CONSUMO_ESTM": pl.Float64,
    "INICIO_UC": pl.Date,
    "RECEBIMENTO_FATURA": pl.String,
    "STATUS_LIGACAO": pl.String,
    "TIPO_UC": pl.String,
    "LOC_UB_RR": pl.String,
    "MUNICIPIO": pl.String,
    "BAIRRO": pl.String,
    "LATITUDE": pl.Float64,
    "LONGITUDE": pl.Float64,
    "SOURCE_UPDATED_AT": pl.Datetime,
    "RUN_ID": pl.String,
}

UC_CONTEXT_KEYS = ("UC",)


# 1.2 meter_installation_history
# Grain: UC × installation of meter (NIO)
# Purpose: Historical record of meters linked to the UC, respecting temporal validity:
#          DATA_INSTALACAO <= timestamp_AMI < DATA_REMOCAO.
METER_INSTALLATION_HISTORY_SCHEMA = {
    "UC": pl.String,
    "NIO": pl.String,
    "DATA_INSTALACAO": pl.Date,
    "DATA_REMOCAO": pl.Date,
    "COD_SUBTIPO": pl.String,
    "DESCRICAO_TIPO": pl.String,
    "FAMILIA_SMART": pl.String,
    "FASE": pl.String,
    "SOURCE_UPDATED_AT": pl.Datetime,
    "RUN_ID": pl.String,
}

METER_INSTALLATION_HISTORY_KEYS = ("UC", "NIO", "DATA_INSTALACAO")


# 1.3 electrical_hierarchy
# Grain: UC
# Purpose: Topological hierarchy: Subestação -> Alimentador -> Posto/Transformador -> UC -> NIO.
ELECTRICAL_HIERARCHY_SCHEMA = {
    "UC": pl.String,
    "NIO": pl.String,
    "SUBESTACAO": pl.String,
    "SIGLA_SE": pl.String,
    "TENSAO_SE": pl.Float64,
    "CAR_SE": pl.String,
    "MUNICIPIO_SE": pl.String,
    "COD_MUN_SE": pl.String,
    "GEDIS_ALIMENTADOR": pl.String,
    "ALIMENTADOR": pl.String,
    "TENSAO_ALIMENTADOR": pl.Float64,
    "POSTO_OPERACIONAL": pl.String,
    "POT_INST_KVA": pl.Float64,
    "COORD_X_POSTE": pl.Float64,
    "COORD_Y_POSTE": pl.Float64,
    "LAT_POSTE_WGS84": pl.Float64,
    "LONG_POSTE_WGS84": pl.Float64,
    "LAT_UC": pl.Float64,
    "LONG_UC": pl.Float64,
    "SOURCE_UPDATED_AT": pl.Datetime,
    "RUN_ID": pl.String,
}

ELECTRICAL_HIERARCHY_KEYS = ("UC",)


# ── Measurements Layer (AMI Phenomenon) ───────────────────────────────────────

# 2.1 ami_interval (or meter_interval)
# Grain: NIO × timestamp_utc × measurement_source
# Format: long in time, wide by stable physical quantity.
METER_INTERVAL_SCHEMA = {
    "NIO": pl.String,
    "REPORT_DAY": pl.Date,
    "TIMESTAMP_UTC": pl.Datetime,
    "TIMESTAMP_LOCAL": pl.Datetime,
    "CADENCE_MINUTES": pl.Int16,
    "MEASUREMENT_SOURCE": pl.String,
    "FA_INTERVAL": pl.Float64,
    "RA_INTERVAL": pl.Float64,
    "I_L1_AVG": pl.Float64,
    "I_L2_AVG": pl.Float64,
    "I_L3_AVG": pl.Float64,
    "U_L1_AVG": pl.Float64,
    "U_L2_AVG": pl.Float64,
    "U_L3_AVG": pl.Float64,
    "R_Q1_INTERVAL": pl.Float64,
    "R_Q2_INTERVAL": pl.Float64,
    "R_Q3_INTERVAL": pl.Float64,
    "R_Q4_INTERVAL": pl.Float64,
    "QUALITY_CODE": pl.Int16,
    "IS_ESTIMATED": pl.Boolean,
    "SOURCE_UPDATED_AT": pl.Datetime,
    "RUN_ID": pl.String,
}

METER_INTERVAL_KEYS = ("NIO", "TIMESTAMP_UTC", "MEASUREMENT_SOURCE")
AMI_INTERVAL_SCHEMA = METER_INTERVAL_SCHEMA
AMI_INTERVAL_KEYS = METER_INTERVAL_KEYS


# 2.2 ami_instantaneous (or meter_instantaneous)
# Grain: NIO × timestamp_utc × measurement_source
METER_INSTANTANEOUS_SCHEMA = {
    "NIO": pl.String,
    "REPORT_DAY": pl.Date,
    "TIMESTAMP_UTC": pl.Datetime,
    "TIMESTAMP_LOCAL": pl.Datetime,
    "CADENCE_MINUTES": pl.Int16,
    "MEASUREMENT_SOURCE": pl.String,
    "U_L1": pl.Float64,
    "U_L2": pl.Float64,
    "U_L3": pl.Float64,
    "I_INSTANT_L1": pl.Float64,
    "I_INSTANT_L2": pl.Float64,
    "I_INSTANT_L3": pl.Float64,
    "QUALITY_CODE": pl.Int16,
    "SOURCE_UPDATED_AT": pl.Datetime,
    "RUN_ID": pl.String,
}

METER_INSTANTANEOUS_KEYS = ("NIO", "TIMESTAMP_UTC", "MEASUREMENT_SOURCE")
AMI_INSTANTANEOUS_SCHEMA = METER_INSTANTANEOUS_SCHEMA
AMI_INSTANTANEOUS_KEYS = METER_INSTANTANEOUS_KEYS


# 2.3 ami_registers (or meter_register_snapshot)
# Grain: NIO × timestamp_utc × register_group
METER_REGISTER_SNAPSHOT_SCHEMA = {
    "NIO": pl.String,
    "REPORT_DAY": pl.Date,
    "TIMESTAMP_UTC": pl.Datetime,
    "REGISTER_GROUP": pl.String,
    "FA_TOTAL": pl.Float64,
    "FA_T1_TOTAL": pl.Float64,
    "FA_T2_TOTAL": pl.Float64,
    "FA_T3_TOTAL": pl.Float64,
    "FA_T4_TOTAL": pl.Float64,
    "RA_TOTAL": pl.Float64,
    "RA_T1_TOTAL": pl.Float64,
    "RA_T2_TOTAL": pl.Float64,
    "RA_T3_TOTAL": pl.Float64,
    "RA_T4_TOTAL": pl.Float64,
    "FA_MD": pl.Float64,
    "FA_MD_T1": pl.Float64,
    "FA_MD_T2": pl.Float64,
    "FA_MD_T3": pl.Float64,
    "FA_MD_T4": pl.Float64,
    "QUALITY_CODE": pl.Int16,
    "SOURCE_UPDATED_AT": pl.Datetime,
    "RUN_ID": pl.String,
}

METER_REGISTER_SNAPSHOT_KEYS = ("NIO", "TIMESTAMP_UTC", "REGISTER_GROUP")
AMI_REGISTERS_SCHEMA = METER_REGISTER_SNAPSHOT_SCHEMA
AMI_REGISTERS_KEYS = METER_REGISTER_SNAPSHOT_KEYS


# ── Events Layer ──────────────────────────────────────────────────────────────

# 3.1 alarm_events
# Grain: ALARM_ID
# OBJ_ID represents 8-digit NIO without zero-padding.
ALARM_EVENTS_SCHEMA = {
    "ALARM_ID": pl.String,
    "NIO": pl.String,
    "ORIGIN_TIMESTAMP": pl.Datetime,
    "RECEIVED_TIMESTAMP": pl.Datetime,
    "LATENCY_SECONDS": pl.Float64,
    "SYSTEM_CODE": pl.String,
    "CONTENT": pl.String,
    "OBJECT_TYPE": pl.String,
    "OBJECT_ORG": pl.String,
    "SOURCE_UPDATED_AT": pl.Datetime,
    "RUN_ID": pl.String,
}

ALARM_EVENTS_KEYS = ("ALARM_ID",)


# ── Features Layer ────────────────────────────────────────────────────────────

# 4.1 uc_day_features (or meter_day_features)
# Grain: UC × REPORT_DAY × FEATURE_SET_VERSION
UC_DAY_FEATURES_SCHEMA = {
    "UC": pl.String,
    "NIO": pl.String,
    "REPORT_DAY": pl.Date,
    "FEATURE_SET_VERSION": pl.String,
    # Coverage & quality
    "INTERVAL_COVERAGE": pl.Float64,
    "INSTANTANEOUS_COVERAGE": pl.Float64,
    "REGISTER_COVERAGE": pl.Float64,
    "NULL_RATIO_INTERVAL": pl.Float64,
    # Energy & demand
    "FA_INTERVAL_SUM": pl.Float64,
    "RA_INTERVAL_SUM": pl.Float64,
    "FA_MD_MAX": pl.Float64,
    # Load factor
    "LOAD_FACTOR": pl.Float64,
    # Energy reversal
    "RA_REVERSAL_RATIO": pl.Float64,
    # Voltage imbalance
    "VOLTAGE_IMBALANCE_MAX": pl.Float64,
    # Current imbalance
    "CURRENT_IMBALANCE_MAX": pl.Float64,
    # Robust statistics (median, IQR, p05, p95)
    "FA_INTERVAL_MEDIAN": pl.Float64,
    "FA_INTERVAL_IQR": pl.Float64,
    "FA_INTERVAL_P05": pl.Float64,
    "FA_INTERVAL_P95": pl.Float64,
    "U_L1_MEDIAN": pl.Float64,
    "U_L1_IQR": pl.Float64,
    # Variations & ramps
    "FA_RAMP_MAX": pl.Float64,
    "U_L1_RAMP_MAX": pl.Float64,
    # Peak hour encoded cyclically
    "PEAK_HOUR_SIN": pl.Float64,
    "PEAK_HOUR_COS": pl.Float64,
    # Temporal variability
    "FA_INTERVAL_AUTOCORR_LAG1": pl.Float64,
    "FA_INTERVAL_STD": pl.Float64,
    # Context
    "PHASE_TYPE": pl.String,
    "CONSUMER_CLASS": pl.String,
    "METER_TYPE": pl.String,
    "INSTALLED_KVA": pl.Float64,
}

UC_DAY_FEATURES_KEYS = ("UC", "REPORT_DAY", "FEATURE_SET_VERSION")
METER_DAY_FEATURES_SCHEMA = UC_DAY_FEATURES_SCHEMA
METER_DAY_FEATURES_KEYS = UC_DAY_FEATURES_KEYS


# 4.2 uc_window_features
# Grain: UC × CUTOFF_DATE × WINDOW_DAYS
# Window aggregated features (7d / 30d) for anomaly detection / classification
UC_WINDOW_FEATURES_SCHEMA = {
    "UC": pl.String,
    "CUTOFF_DATE": pl.Date,
    "WINDOW_DAYS": pl.Int16,
    "FEATURE_SET_VERSION": pl.String,
    # Window coverage & quality
    "COVERAGE_RELIABLE_DAYS": pl.Int16,
    "AVG_INTERVAL_COVERAGE": pl.Float64,
    # Historical electrical aggregations
    "HIST_FA_SUM_MEDIAN": pl.Float64,
    "HIST_FA_SUM_MAD": pl.Float64,
    "HIST_RA_REVERSAL_DAYS": pl.Int16,
    "HIST_RA_REVERSAL_RATIO_MAX": pl.Float64,
    "HIST_LOAD_FACTOR_MEAN": pl.Float64,
    "HIST_VOLTAGE_IMBALANCE_MAX": pl.Float64,
    "HIST_CURRENT_IMBALANCE_MAX": pl.Float64,
    "TREND_FA_SLOPE": pl.Float64,
    # Meter installation & age context
    "METER_AGE_DAYS": pl.Int32,
    "METER_CHANGED_30D": pl.Boolean,
    "METER_CHANGED_90D": pl.Boolean,
    # Alarm events over window
    "ALARM_COUNT_WINDOW": pl.Int32,
    "AVG_ALARM_LATENCY_SEC": pl.Float64,
}

UC_WINDOW_FEATURES_KEYS = ("UC", "CUTOFF_DATE", "WINDOW_DAYS", "FEATURE_SET_VERSION")


# ── Labels Layer ──────────────────────────────────────────────────────────────

LABELS_SCHEMA = {
    "ENTITY_TYPE": pl.String,
    "ENTITY_ID": pl.String,
    "REFERENCE_START": pl.Date,
    "REFERENCE_END": pl.Date,
    "PREDICTION_HORIZON": pl.Int16,
    "LABEL_TYPE": pl.String,
    "LABEL_VALUE": pl.Boolean,
    "LABEL_SOURCE": pl.String,
    "CONFIDENCE": pl.Float64,
    "REVIEWER_ID": pl.String,
    "CREATED_AT": pl.Datetime,
    "LABEL_VERSION": pl.String,
    "NOTES": pl.String,
}

LABELS_KEYS = ("ENTITY_TYPE", "ENTITY_ID", "REFERENCE_START", "LABEL_TYPE", "LABEL_VERSION")


# ── Model Input Layer ─────────────────────────────────────────────────────────

# 5.1 training_dataset (for scikit-learn / Polars ML)
# Grain: UC × CUTOFF_DATE
TRAINING_DATASET_PREFIXES = {
    "ID": "id__",
    "META": "meta__",
    "FEATURE": "x__",
    "TARGET": "y__",
}


# ── Registry ─────────────────────────────────────────────────────────────────

TABLE_REGISTRY: dict[str, dict] = {
    "uc_context": {
        "schema": UC_CONTEXT_SCHEMA,
        "keys": UC_CONTEXT_KEYS,
    },
    "meter_installation_history": {
        "schema": METER_INSTALLATION_HISTORY_SCHEMA,
        "keys": METER_INSTALLATION_HISTORY_KEYS,
    },
    "electrical_hierarchy": {
        "schema": ELECTRICAL_HIERARCHY_SCHEMA,
        "keys": ELECTRICAL_HIERARCHY_KEYS,
    },
    "alarm_events": {
        "schema": ALARM_EVENTS_SCHEMA,
        "keys": ALARM_EVENTS_KEYS,
    },
    "ami_interval": {
        "schema": AMI_INTERVAL_SCHEMA,
        "keys": AMI_INTERVAL_KEYS,
    },
    "ami_instantaneous": {
        "schema": AMI_INSTANTANEOUS_SCHEMA,
        "keys": AMI_INSTANTANEOUS_KEYS,
    },
    "ami_registers": {
        "schema": AMI_REGISTERS_SCHEMA,
        "keys": AMI_REGISTERS_KEYS,
    },
    "meter_interval": {
        "schema": METER_INTERVAL_SCHEMA,
        "keys": METER_INTERVAL_KEYS,
    },
    "meter_instantaneous": {
        "schema": METER_INSTANTANEOUS_SCHEMA,
        "keys": METER_INSTANTANEOUS_KEYS,
    },
    "meter_register_snapshot": {
        "schema": METER_REGISTER_SNAPSHOT_SCHEMA,
        "keys": METER_REGISTER_SNAPSHOT_KEYS,
    },
    "uc_day_features": {
        "schema": UC_DAY_FEATURES_SCHEMA,
        "keys": UC_DAY_FEATURES_KEYS,
    },
    "meter_day_features": {
        "schema": METER_DAY_FEATURES_SCHEMA,
        "keys": METER_DAY_FEATURES_KEYS,
    },
    "uc_window_features": {
        "schema": UC_WINDOW_FEATURES_SCHEMA,
        "keys": UC_WINDOW_FEATURES_KEYS,
    },
    "labels": {
        "schema": LABELS_SCHEMA,
        "keys": LABELS_KEYS,
    },
}


def get_schema(table_name: str) -> dict[str, pl.DataType]:
    """Return the Polars schema dict for a named table."""
    entry = TABLE_REGISTRY.get(table_name)
    if entry is None:
        raise ValueError(
            f"Unknown table {table_name!r}. "
            f"Available: {sorted(TABLE_REGISTRY)}"
        )
    return entry["schema"]


def get_keys(table_name: str) -> tuple[str, ...]:
    """Return the candidate key columns for a named table."""
    entry = TABLE_REGISTRY.get(table_name)
    if entry is None:
        raise ValueError(
            f"Unknown table {table_name!r}. "
            f"Available: {sorted(TABLE_REGISTRY)}"
        )
    return entry["keys"]


def build_empty_frame(table_name: str) -> pl.DataFrame:
    """Return an empty Polars DataFrame with the correct schema."""
    schema = get_schema(table_name)
    return pl.DataFrame(schema=schema)
