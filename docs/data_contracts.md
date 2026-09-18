# Data Contracts — ARAUCARIA Smart Meter Datasets

**Projeto:** TCC — análise de dados de medidores inteligentes em Araucária  
**Status:** v1.0  
**Data:** 2026-07-27  
**Base:** `docs/dataset_architecture_decision.md`

## Overview

This document defines the formal contracts for each dataset table in the normalized Parquet architecture. Each contract specifies:

- **Table name** and **grain** (candidate key)
- **Column names**, **types**, and **units**
- **Allowed values** and **constraints**
- **Source** and **update cadence**

---

## 1. `meter_day_metadata`

**Grain:** `(REPORT_DAY, NIO)` — one row per NIO per day.  
**Purpose:** Temporal snapshot of the installation and electrical context at the reference instant.  
**Source:** CIS (`cad_uc_ee`, `rel_equip_uc`, `cad_equip_med`, `tab_sub_tipo_equip`) + GEO (`alimentador`, `subestação`).  
**Update:** Daily, append-only.

| Column | Type | Unit | Required | Description |
|--------|------|------|----------|-------------|
| `REPORT_DAY` | `DATE` | — | ✓ | Reference date |
| `NIO` | `STRING` | — | ✓ | Meter asset number (normalized, no leading zeros) |
| `UC_KEY` | `STRING` | — | | Consumer unit pseudonym |
| `METER_SERIAL_KEY` | `STRING` | — | | Meter serial number pseudonym |
| `METER_TYPE` | `STRING` | — | | Meter model/type description |
| `METER_SUBTYPE` | `STRING` | — | | Meter subtype code |
| `PHASE_TYPE` | `STRING` | — | | Phase configuration (M, B, T) |
| `SERVICE_STATUS` | `STRING` | — | | Service status code |
| `INSTALLATION_DATE` | `DATE` | — | | Meter installation date |
| `REMOVAL_DATE` | `DATE` | — | | Meter removal date |
| `CONSUMER_CLASS` | `STRING` | — | | Consumer class code |
| `SUBGROUP` | `STRING` | — | | Tariff subgroup |
| `INSTALLED_KVA` | `FLOAT` | kVA | | Installed transformer capacity |
| `FEEDER_ID` | `STRING` | — | | Feeder/network identifier |
| `SUBSTATION_ID` | `STRING` | — | | Substation name/ID |
| `NOMINAL_VOLTAGE` | `FLOAT` | V | | Nominal voltage |
| `MUNICIPALITY` | `STRING` | — | | Municipality name |
| `LATITUDE_BUCKET` | `STRING` | — | | Binned latitude (privacy) |
| `LONGITUDE_BUCKET` | `STRING` | — | | Binned longitude (privacy) |
| `SOURCE_UPDATED_AT` | `TIMESTAMP` | — | ✓ | Source system update timestamp |
| `SCHEMA_VERSION` | `STRING` | — | ✓ | Schema version tag |
| `RUN_ID` | `STRING` | — | ✓ | Pipeline run identifier |

**Constraints:**
- `NIO` must be non-null and unique per `REPORT_DAY`
- `INSTALLATION_DATE <= REPORT_DAY <= REMOVAL_DATE` (if removal date present)
- `INSTALLED_KVA` must be > 0 when populated

---

## 2. `meter_interval`

**Grain:** `(NIO, TIMESTAMP_UTC)` — one row per NIO per timestamp.  
**Purpose:** Interval energy, reactive energy, current and voltage averages (average over the interval).  
**Source:** MDM/ORCA (`AMI.biz_pub_data_f_energy_c`, `biz_pub_data_r_energy_c`, `biz_pub_data_current`, `biz_pub_data_voltage`, `biz_pub_data_q_energy_c`).  
**Cadence:** 5, 10, or 15 minutes (varies by meter).  
**Update:** Daily, append-only per partition.

| Column | Type | Unit | Required | Description |
|--------|------|------|----------|-------------|
| `NIO` | `STRING` | — | ✓ | Meter asset number |
| `REPORT_DAY` | `DATE` | — | ✓ | Reference date |
| `TIMESTAMP_UTC` | `TIMESTAMP` | — | ✓ | UTC timestamp of the interval end |
| `TIMESTAMP_LOCAL` | `TIMESTAMP` | — | | Local time (America/Sao_Paulo) |
| `CADENCE_MINUTES` | `INT16` | min | ✓ | Interval duration in minutes |
| `FA_INTERVAL` | `FLOAT` | kWh | | Active energy (interval) |
| `RA_INTERVAL` | `FLOAT` | kVArh | | Reactive energy (interval) |
| `I_L1_AVG` | `FLOAT` | A | | Average current, phase L1 |
| `I_L2_AVG` | `FLOAT` | A | | Average current, phase L2 |
| `I_L3_AVG` | `FLOAT` | A | | Average current, phase L3 |
| `U_L1_AVG` | `FLOAT` | V | | Average voltage, phase L1 |
| `U_L2_AVG` | `FLOAT` | V | | Average voltage, phase L2 |
| `U_L3_AVG` | `FLOAT` | V | | Average voltage, phase L3 |
| `R_Q1_INTERVAL` | `FLOAT` | kVArh | | Reactive energy Q1 (interval) |
| `R_Q2_INTERVAL` | `FLOAT` | kVArh | | Reactive energy Q2 (interval) |
| `R_Q3_INTERVAL` | `FLOAT` | kVArh | | Reactive energy Q3 (interval) |
| `R_Q4_INTERVAL` | `FLOAT` | kVArh | | Reactive energy Q4 (interval) |
| `QUALITY_CODE` | `INT16` | — | | Quality/validation code |
| `IS_ESTIMATED` | `BOOLEAN` | — | | Whether the value was estimated |
| `SOURCE_UPDATED_AT` | `TIMESTAMP` | — | ✓ | Source system update timestamp |
| `RUN_ID` | `STRING` | — | ✓ | Pipeline run identifier |

**Constraints:**
- `(NIO, TIMESTAMP_UTC)` must be unique
- `TIMESTAMP_UTC` must fall within `REPORT_DAY`
- `CADENCE_MINUTES` must be one of `{5, 10, 15, 30, 60}`
- `FA_INTERVAL >= 0` when present
- `U_L[123]_AVG` in `[0, 500]` V when present
- `I_L[123]_AVG` in `[0, 1000]` A when present

---

## 3. `meter_instantaneous`

**Grain:** `(NIO, TIMESTAMP_UTC)` — one row per NIO per timestamp.  
**Purpose:** Instantaneous voltage and current snapshots.  
**Source:** MDM/ORCA (`AMI.biz_pub_data_voltage_instant`, `biz_pub_data_current_instant`).  
**Cadence:** 15 or 60 minutes (varies by meter).  
**Update:** Daily, append-only per partition.

| Column | Type | Unit | Required | Description |
|--------|------|------|----------|-------------|
| `NIO` | `STRING` | — | ✓ | Meter asset number |
| `REPORT_DAY` | `DATE` | — | ✓ | Reference date |
| `TIMESTAMP_UTC` | `TIMESTAMP` | — | ✓ | UTC timestamp |
| `TIMESTAMP_LOCAL` | `TIMESTAMP` | — | | Local time |
| `CADENCE_MINUTES` | `INT16` | min | ✓ | Interval duration in minutes |
| `U_L1` | `FLOAT` | V | | Instantaneous voltage, phase L1 |
| `U_L2` | `FLOAT` | V | | Instantaneous voltage, phase L2 |
| `U_L3` | `FLOAT` | V | | Instantaneous voltage, phase L3 |
| `I_INSTANT_L1` | `FLOAT` | A | | Instantaneous current, phase L1 |
| `I_INSTANT_L2` | `FLOAT` | A | | Instantaneous current, phase L2 |
| `I_INSTANT_L3` | `FLOAT` | A | | Instantaneous current, phase L3 |
| `QUALITY_CODE` | `INT16` | — | | Quality/validation code |
| `SOURCE_UPDATED_AT` | `TIMESTAMP` | — | ✓ | Source system update timestamp |
| `RUN_ID` | `STRING` | — | ✓ | Pipeline run identifier |

**Constraints:**
- `(NIO, TIMESTAMP_UTC)` must be unique
- `U_L[123]` in `[0, 500]` V when present
- `I_INSTANT_L[123]` in `[0, 1000]` A when present

---

## 4. `meter_register_snapshot`

**Grain:** `(NIO, TIMESTAMP_UTC)` — one row per NIO per timestamp.  
**Purpose:** Accumulated active/reactive energy and demand register snapshots.  
**Source:** MDM/ORCA (`AMI.biz_pub_data_f_energy_d`, `biz_pub_data_r_energy_d`, `biz_pub_data_f_md_d`).  
**Cadence:** Typically 6 hours (00:00, 06:00, 12:00, 18:00).  
**Update:** Daily, append-only per partition.

| Column | Type | Unit | Required | Description |
|--------|------|------|----------|-------------|
| `NIO` | `STRING` | — | ✓ | Meter asset number |
| `REPORT_DAY` | `DATE` | — | ✓ | Reference date |
| `TIMESTAMP_UTC` | `TIMESTAMP` | — | ✓ | UTC timestamp of the snapshot |
| `FA_TOTAL` | `FLOAT` | kWh | | Accumulated active energy (total) |
| `FA_T1_TOTAL` | `FLOAT` | kWh | | Accumulated active energy, tariff T1 |
| `FA_T2_TOTAL` | `FLOAT` | kWh | | Accumulated active energy, tariff T2 |
| `FA_T3_TOTAL` | `FLOAT` | kWh | | Accumulated active energy, tariff T3 |
| `FA_T4_TOTAL` | `FLOAT` | kWh | | Accumulated active energy, tariff T4 |
| `RA_TOTAL` | `FLOAT` | kVArh | | Accumulated reactive energy (total) |
| `RA_T1_TOTAL` | `FLOAT` | kVArh | | Accumulated reactive energy, tariff T1 |
| `RA_T2_TOTAL` | `FLOAT` | kVArh | | Accumulated reactive energy, tariff T2 |
| `RA_T3_TOTAL` | `FLOAT` | kVArh | | Accumulated reactive energy, tariff T3 |
| `RA_T4_TOTAL` | `FLOAT` | kVArh | | Accumulated reactive energy, tariff T4 |
| `FA_MD` | `FLOAT` | kW | | Maximum demand (active) |
| `FA_MD_T1` | `FLOAT` | kW | | Maximum demand, tariff T1 |
| `FA_MD_T2` | `FLOAT` | kW | | Maximum demand, tariff T2 |
| `FA_MD_T3` | `FLOAT` | kW | | Maximum demand, tariff T3 |
| `FA_MD_T4` | `FLOAT` | kW | | Maximum demand, tariff T4 |
| `QUALITY_CODE` | `INT16` | — | | Quality/validation code |
| `SOURCE_UPDATED_AT` | `TIMESTAMP` | — | ✓ | Source system update timestamp |
| `RUN_ID` | `STRING` | — | ✓ | Pipeline run identifier |

**Constraints:**
- Register values are cumulative counters — differences between consecutive snapshots give interval consumption
- Rollover/reset detection should be applied before computing differences (see `test_register_resets.py`)
- `FA_TOTAL` should generally be monotonic increasing (resets indicate meter replacement or data issue)

---

## 5. `meter_day_features`

**Grain:** `(NIO, REPORT_DAY, FEATURE_SET_VERSION)` — one row per NIO per day per feature version.  
**Purpose:** Tabular feature matrix for scikit-learn modeling.  
**Source:** Derived from `meter_interval`, `meter_instantaneous`, `meter_register_snapshot`, and `meter_day_metadata`.  
**Nature:** **Derived, discardable, reproducible** — not source of truth.  
**Update:** On demand, versioned.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `NIO` | `STRING` | — | Meter asset number |
| `REPORT_DAY` | `DATE` | — | Reference date (feature cutoff) |
| `FEATURE_SET_VERSION` | `STRING` | — | Version tag for reproducibility |
| `INTERVAL_COVERAGE` | `FLOAT` | ratio | Coverage ratio (non-null / expected) |
| `INSTANTANEOUS_COVERAGE` | `FLOAT` | ratio | Coverage ratio for instantaneous data |
| `REGISTER_COVERAGE` | `FLOAT` | ratio | Coverage ratio for register snapshots |
| `NULL_RATIO_INTERVAL` | `FLOAT` | ratio | Null ratio in FA_INTERVAL |
| `FA_INTERVAL_SUM` | `FLOAT` | kWh | Sum of active energy |
| `RA_INTERVAL_SUM` | `FLOAT` | kVArh | Sum of reactive energy |
| `FA_MD_MAX` | `FLOAT` | kW | Maximum demand |
| `LOAD_FACTOR` | `FLOAT` | ratio | Average demand / peak demand |
| `RA_REVERSAL_RATIO` | `FLOAT` | ratio | Fraction of negative RA intervals |
| `VOLTAGE_IMBALANCE_MAX` | `FLOAT` | ratio | Max voltage imbalance across phases |
| `CURRENT_IMBALANCE_MAX` | `FLOAT` | ratio | Max current imbalance across phases |
| `FA_INTERVAL_MEDIAN` | `FLOAT` | kWh | Median active energy |
| `FA_INTERVAL_IQR` | `FLOAT` | kWh | Interquartile range of active energy |
| `FA_INTERVAL_P05` | `FLOAT` | kWh | 5th percentile active energy |
| `FA_INTERVAL_P95` | `FLOAT` | kWh | 95th percentile active energy |
| `U_L1_MEDIAN` | `FLOAT` | V | Median voltage L1 |
| `U_L1_IQR` | `FLOAT` | V | Voltage L1 IQR |
| `FA_RAMP_MAX` | `FLOAT` | kWh | Max absolute ramp in active energy |
| `U_L1_RAMP_MAX` | `FLOAT` | V | Max absolute voltage ramp |
| `PEAK_HOUR_SIN` | `FLOAT` | — | Peak hour, sine encoding |
| `PEAK_HOUR_COS` | `FLOAT` | — | Peak hour, cosine encoding |
| `FA_INTERVAL_AUTOCORR_LAG1` | `FLOAT` | — | Lag-1 autocorrelation |
| `FA_INTERVAL_STD` | `FLOAT` | kWh | Standard deviation |
| `PHASE_TYPE` | `STRING` | — | Context: phase configuration |
| `CONSUMER_CLASS` | `STRING` | — | Context: consumer class |
| `METER_TYPE` | `STRING` | — | Context: meter type |
| `INSTALLED_KVA` | `FLOAT` | kVA | Context: installed capacity |

**Constraints:**
- Features must use only data available up to `REPORT_DAY` (no future leakage)
- Do not impute missing values in this table — preserve nulls and coverage flags
- `FEATURE_SET_VERSION` allows side-by-side comparison of feature engineering approaches

---

## 6. `labels`

**Grain:** `(ENTITY_TYPE, ENTITY_ID, REFERENCE_START, LABEL_TYPE, LABEL_VERSION)`.  
**Purpose:** Research labels, independent of features.  
**Source:** MDM events, CIS status, field inspection, automated rules, expert review.  
**Update:** On demand, versioned.

| Column | Type | Unit | Required | Description |
|--------|------|------|----------|-------------|
| `ENTITY_TYPE` | `STRING` | — | ✓ | Entity type (`NIO`, `UC`, `ALIMENTADOR`, `SUBESTACAO`) |
| `ENTITY_ID` | `STRING` | — | ✓ | Entity identifier |
| `REFERENCE_START` | `DATE` | — | ✓ | Start of reference window |
| `REFERENCE_END` | `DATE` | — | ✓ | End of reference window |
| `PREDICTION_HORIZON` | `INT16` | days | | Days ahead from reference end |
| `LABEL_TYPE` | `STRING` | — | ✓ | Label type (see allowed values) |
| `LABEL_VALUE` | `BOOLEAN` | — | ✓ | Label value (True/False) |
| `LABEL_SOURCE` | `STRING` | — | ✓ | Source of the label |
| `CONFIDENCE` | `FLOAT` | — | | Confidence level (0.0 to 1.0) |
| `REVIEWER_ID` | `STRING` | — | | Identifier of human reviewer |
| `CREATED_AT` | `TIMESTAMP` | — | ✓ | Creation timestamp |
| `LABEL_VERSION` | `STRING` | — | ✓ | Version tag |
| `NOTES` | `STRING` | — | | Free-text notes |

**Allowed LABEL_TYPES:**

| Label Type | Description | Entity Types |
|------------|-------------|-------------|
| `falha_comunicacao` | Communication failure — meter stopped sending data | NIO, UC, ALIMENTADOR |
| `anomalia_eletrica` | Electrical anomaly — unusual voltage/current pattern | NIO, UC |
| `classe_perfil` | Consumer profile classification | NIO, UC |
| `troca_medidor` | Meter replacement event | NIO |
| `reset_registro` | Register reset or rollover detected | NIO |

**Allowed LABEL_SOURCES:**

| Source | Description |
|--------|-------------|
| `campo` | Field inspection or manual measurement |
| `alarme` | Automated alarm from SCADA/MDM |
| `os` | Work order (ordem de serviço) |
| `regra` | Automated rule-based derivation |
| `especialista` | Expert human review |
| `mdm` | MDM system event log |
| `cis` | CIS system event or status change |

**Constraints:**
- `REFERENCE_START <= REFERENCE_END`
- `CONFIDENCE` in `[0.0, 1.0]`
- Labels must be **independent** of features — computed from a different time window or data source
- Features and labels should share only entity identifiers (no overlap in time-dependent columns)

---

## Partitioning

### Normalized tables

```text
data/normalized/
├── meter_day_metadata/report_year=2026/report_month=06/report_day=23/
├── meter_interval/report_year=2026/report_month=06/report_day=23/
├── meter_instantaneous/report_year=2026/report_month=06/report_day=23/
└── meter_register_snapshot/report_year=2026/report_month=06/report_day=23/
```

### Features

```text
data/features/feature_set_version=v1/report_year=2026/report_month=06/report_day=23/
```

### Labels

```text
data/labels/label_version=v1/
```

### Manifests

```text
data/manifests/run_id=20260623_143000.json
```

**Compression:** Zstandard (`zstd`) for all Parquet files.

---

## File Naming Convention

| Entity | Pattern | Example |
|--------|---------|---------|
| CIS raw | `araucaria_cis_{YYYYMMDD}.{csv,parquet}` | `araucaria_cis_20260623.parquet` |
| GEO raw | `araucaria_geo_ucs_{YYYYMMDD}.{csv,parquet}` | `araucaria_geo_ucs_20260623.parquet` |
| MDM raw | `araucaria_mdm_{YYYYMMDD}.{csv,parquet}` | `araucaria_mdm_20260623.parquet` |
| Weather raw | `araucaria_weather_{YYYYMMDD}.{csv,parquet}` | `araucaria_weather_20260623.parquet` |
| Joined report | `araucaria_daily_report_{YYYYMMDD}.{csv,parquet}` | `araucaria_daily_report_20260623.parquet` |
| Normalized Parquet | `data/normalized/{table}/report_year=.../data.parquet` | See partitioning section |
| Run manifest | `run_{YYYYMMDD_HHMMSS}.json` | `run_20260623_143000.json` |

---

## Versioning

| Component | Version Scheme | Example |
|-----------|---------------|---------|
| Dataset schema | `v{major}` | `v1` |
| Feature set | `v{major}` | `v1` |
| Labels | `v{major}` | `v1` |
| Pipeline manifest | Run timestamp | `20260623_143000` |
