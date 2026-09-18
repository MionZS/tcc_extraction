---

title: "Practical Dataset Layout for scikit-learn"
subtitle: "UC-level anomaly detection for unregistered distributed generation and non-technical-loss indicators"
date: "29 July 2026"
lang: en
---

## Practical Dataset Layout for scikit-learn

## 1. Decision in one page

A scikit-learn dataset is not required to be a CSV. The model ultimately needs a rectangular matrix:

```text
rows    = samples
columns = features
X       = feature matrix
y       = optional target labels
```

For this TCC:

```text
one sample = one UC x cutoff date x historical window
```

Recommended initial windows:

```text
7 days  -> recent behavior
30 days -> persistent behavior and baseline comparison
```

`UC` is the stable entity. `NIO` is equipment and can change. Measurements must first be attached to the UC using installation validity at the measurement timestamp.

The current daily report is useful as an extraction prototype, but it is not a model-ready dataset. It has one row per meter/day and stores many complete time series as JSON strings. In the inspected Araucaria file:

```text
6,877 rows
77 columns
33 JSON-series columns
6,371 rows with MDM data
interval cadences observed: 5, 10 and 15 minutes
instantaneous cadences observed: mainly 15 and 60 minutes
```

Therefore the pipeline must separate source storage, normalized measurements, engineered features and the final model matrix.

## 2. What "standardized dataset" means here

It does **not** mean dividing current by an arbitrary ampere value. It means a fixed data contract:

- stable keys and data types;
- explicit units and time zones;
- null for missing values, never magic numbers;
- no nested JSON in analytical tables;
- one physical measurement per row in normalized measurement tables;
- one model sample per row in the final model table;
- deterministic feature names, versions and train/test assignment;
- preprocessing fitted only on training data.

Physical values remain physical values. Statistical scaling, when needed by PCA, K-means or SVM, belongs inside the scikit-learn `Pipeline`, after the train/test split.

## 3. Directory layout

```text
data/
├── raw/
│   └── <source>/event_date=YYYY-MM-DD/run_id=<uuid>/part-*.parquet
├── manifests/
│   └── extraction_run_id=<uuid>.json
├── normalized/
│   ├── uc_history/
│   ├── meter_installation_history/
│   ├── measurement_channel_config_history/
│   ├── meter_interval/
│   ├── meter_instantaneous/
│   ├── meter_register_snapshot/
│   ├── gd_registry_history/
│   ├── network_topology_history/
│   ├── weather_hourly/
│   └── inspection_events/
├── features/
│   ├── uc_day_features/
│   └── uc_window_features/
├── labels/
│   └── labels.parquet
├── model_input/
│   └── <dataset_version>/
│       ├── model_dataset.parquet
│       ├── feature_catalog.yml
│       └── dataset_manifest.json
└── outputs/
    └── anomaly_scores/model_version=<version>/*.parquet
```

Only `model_dataset.parquet` is opened by ordinary model-training code. Earlier layers exist to build it correctly, update it after late data, and explain every score.

## 4. Non-negotiable conventions

### 4.1 Identity

```text
uc_id           -> String; stable analytical entity
nio              -> String; replaceable meter identifier
installation_id  -> String/UUID; time-valid UC-NIO relationship
```

Never cast UC/NIO to numeric. They are identifiers, not quantities.

### 4.2 Time

Store:

```text
event_timestamp_utc
event_timestamp_local      # America/Sao_Paulo
source_arrived_at_utc
source_updated_at_utc
extracted_at_utc
```

Use event time for electrical behavior. Use arrival/update/extraction time for late-data reconciliation and reproducibility.

### 4.3 Missing data

```text
unknown / absent -> null
measured zero    -> 0.0
```

These are not equivalent. Coverage features must explain whether a zero is real or caused by missing telemetry.

### 4.4 Feature/target prefixes

```text
id__*    identifiers; never sent to estimator
meta__*  cutoff, split, version; never sent to estimator
x__*     permitted model features
y__*     labels/targets; never included in X
```

This naming rule prevents accidental leakage.

## 5. File-by-file specification

### 5.1 `raw/<source>/event_date=YYYY-MM-DD/run_id=<uuid>/part-*.parquet`

**Layer:** Raw

**Purpose:** Immutable copy of each source extraction. It preserves exactly what the pipeline observed, including duplicates, late arrivals and later corrections.

**Grain:** Same grain as the source query or source table.

**Key:** No destructive deduplication. Add extraction metadata to every row.

**Partitioning:** Source, event date when available, and run_id. Never partition by UC or NIO.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `source_system` | `String` | ORCA, MDM, CIS, GEO, weather, inspection system, etc. |
| `source_table_or_query` | `String` | Origin query/table identifier. |
| `source_primary_key` | `String or struct` | Natural source key, if available. |
| `source_created_at` | `Datetime[UTC], nullable` | When source created the record. |
| `source_updated_at` | `Datetime[UTC], nullable` | When source last corrected the record. |
| `extracted_at` | `Datetime[UTC]` | When this pipeline read the record. |
| `run_id` | `String/UUID` | Extraction execution identifier. |
| `raw source columns` | `Original types when reliable` | No feature engineering in this layer. |

**Built from:** Database/API source.

**Consumed by:** Normalization and reconciliation jobs only. Never directly by scikit-learn.

**Critical rule:** Prefer Parquet even for raw extracts. Keep the original CSV only as an audit export when necessary.

### 5.2 `manifests/extraction_run.json`

**Layer:** Raw

**Purpose:** Audit record for one extraction execution.

**Grain:** One JSON document per run.

**Key:** run_id.

**Partitioning:** One file named `run_id=<uuid>.json`.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `run_id` | `UUID` | Unique execution identifier. |
| `started_at / finished_at` | `Datetime[UTC]` | Execution interval. |
| `source_watermark` | `Datetime or value` | Incremental extraction boundary. |
| `queries_and_parameters` | `Object` | Queries, date ranges and feeder filters. |
| `row_counts` | `Object` | Rows read/written by table. |
| `file_hashes` | `Object` | Integrity hashes of outputs. |
| `code_version` | `String` | Git commit. |
| `status` | `String` | success, partial or failed. |

**Built from:** Pipeline runtime.

**Consumed by:** Integrity checks, reproducibility and dataset manifests.

**Critical rule:** A model result must be traceable to extraction runs.

**Layer:** Normalized

**Purpose:** Cumulative register readings and other irregular snapshots.

**Grain:** One register snapshot.

**Key:** (installation_id, channel_id, event_timestamp_utc).

**Partitioning:** event_month and measurement_type.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `uc_id / installation_id / nio` | `String` | Identity. |
| `measurement_type` | `Categorical/String` | Import/export register, demand register, etc. |
| `tariff_period` | `Categorical, nullable` | T1-T4. |
| `event_timestamp_utc / local` | `Datetime` | Snapshot time. |
| `value / unit` | `Float64 / String` | Cumulative or snapshot value. |
| `reset_or_rollover_flag` | `Boolean, nullable` | Detected or source-provided reset. |
| `source_updated_at / extracted_at / run_id` | `Audit fields` | Version lineage. |

**Built from:** Register tables or exploded `*_TOTAL` and `*_MD` JSON.

**Consumed by:** Energy deltas, meter-reset diagnostics and consistency features.

**Critical rule:** A difference between snapshots is valid only across the same installation/channel and without reset/rollover.

### 5.9 `normalized/gd_registry_history/`

**Layer:** Normalized

**Purpose:** Time-valid registry of declared/authorized distributed generation and beneficiary relationships.

**Grain:** One GD registration state during a validity interval.

**Key:** (uc_id, gd_registration_id, valid_from_utc).

**Partitioning:** valid_from_year or active flag.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `uc_id` | `String` | UC owning or receiving GD benefit. |
| `gd_registration_id` | `String, nullable` | Registration/process identifier. |
| `gd_type / modality` | `String, nullable` | Technology/modality. |
| `installed_generation_kw` | `Float64, nullable` | Registered capacity. |
| `beneficiary_flag` | `Boolean` | Receives credits from another generator. |
| `valid_from_utc / valid_to_utc` | `Datetime[UTC]` | Registration validity. |
| `source_updated_at / extracted_at` | `Datetime[UTC]` | Audit. |

**Built from:** GD cadastro and beneficiary records.

**Consumed by:** Exclusion rules, context features and future validated targets.

**Critical rule:** Registered GD is context, not an anomaly. Time validity prevents using future registration information.

### 5.10 `normalized/network_topology_history/`

**Layer:** Normalized

**Purpose:** Time-valid network context for peer comparison.

**Grain:** One UC-to-network assignment during a validity interval.

**Key:** (uc_id, valid_from_utc).

**Partitioning:** substation/feeder and validity year.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `uc_id` | `String` | Consumer unit. |
| `transformer_id / feeder_id / substation_id` | `String, nullable` | Network hierarchy. |
| `nominal_voltage_v` | `Float64, nullable` | Confirmed nominal voltage. |
| `transformer_kva` | `Float64, nullable` | Transformer context. |
| `latitude / longitude` | `Float64, nullable` | Operational coordinates; protect in publication. |
| `valid_from_utc / valid_to_utc` | `Datetime[UTC]` | Topology validity. |
| `source_updated_at / extracted_at` | `Datetime[UTC]` | Audit. |

**Built from:** GEO/network topology sources.

**Consumed by:** Peer-group, feeder/transformer residual and spatial features.

**Critical rule:** Raw IDs are metadata/context. Do not let a high-cardinality feeder or transformer ID become a shortcut feature without validation.

### 5.11 `normalized/weather_hourly/`

**Layer:** Normalized (optional but valuable for GD)

**Purpose:** Weather context, mainly irradiance/cloud/temperature proxies for solar-shaped behavior.

**Grain:** One station/grid cell per hour.

**Key:** (weather_location_id, event_hour_utc).

**Partitioning:** event_date and weather_location_id.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `weather_location_id` | `String` | Station/grid cell. |
| `event_hour_utc / local` | `Datetime` | Time. |
| `irradiance_w_m2` | `Float64, nullable` | Preferred solar signal. |
| `cloud_cover_pct` | `Float64, nullable` | Fallback/context. |
| `temperature_c` | `Float64, nullable` | Load/weather context. |
| `precipitation_mm` | `Float64, nullable` | Optional context. |
| `source_updated_at / extracted_at` | `Datetime[UTC]` | Audit. |

**Built from:** Weather API/station source.

**Consumed by:** Solar-correlation and weather-adjusted load features.

**Critical rule:** Join using event time and nearest valid location; preserve source and imputation flags.

### 5.12 `normalized/inspection_events/`

**Layer:** Normalized (required for model validation)

**Purpose:** Operational evidence: inspections, orders, confirmed irregularities and normal inspected cases.

**Grain:** One operational event or inspection conclusion.

**Key:** inspection_event_id.

**Partitioning:** event_year and event_type.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `inspection_event_id` | `String` | Unique event. |
| `uc_id` | `String` | Consumer unit. |
| `event_timestamp_utc` | `Datetime[UTC]` | Inspection/order time. |
| `event_type` | `Categorical/String` | Inspection, order, replacement, alarm, etc. |
| `result_code` | `String, nullable` | Operational result. |
| `confirmed_irregularity_type` | `String, nullable` | Only controlled vocabulary. |
| `evidence_quality` | `Categorical` | confirmed, probable, inconclusive, administrative. |
| `source_updated_at / extracted_at` | `Datetime[UTC]` | Audit. |

**Built from:** Work-order and inspection systems.

**Consumed by:** labels.parquet and retrospective validation.

**Critical rule:** A score is not proof of theft. Labels must originate from evidence, not from model output.

### 5.13 `features/uc_day_features/`

**Layer:** Curated features

**Purpose:** Atomic, auditable features for one UC and one local day.

**Grain:** One UC x feature_date.

**Key:** (uc_id, feature_date, feature_version, as_of_utc).

**Partitioning:** feature_date or feature_month.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `uc_id / feature_date` | `String / Date` | Sample identity at daily grain. |
| `feature_version` | `String` | Feature-code/schema version. |
| `as_of_utc` | `Datetime[UTC]` | Latest source knowledge included. |
| `coverage_*` | `Float/Int` | Expected count, observed count, coverage ratio, longest gap. |
| `energy_*` | `Float` | Import/export totals, reverse share, register consistency. |
| `current_*` | `Float` | Mean, p95, max, zero ratio, load factor, phase unbalance. |
| `voltage_*` | `Float` | Mean, min, max, dispersion, out-of-band ratios. |
| `shape_*` | `Float` | Day/night share, peak hour, ramps, variability. |
| `quality_*` | `Float/Int/Boolean` | Late records, corrections, estimates, missing channels. |
| `context_*` | `Typed` | Phase, class, registered-GD state and topology valid that day. |

**Built from:** Canonical normalized measurements plus time-valid context tables.

**Consumed by:** Window feature builder; also useful for plots and single-day diagnostics.

**Critical rule:** Keep absolute current features. Relative/scaled current is added later from history or inside the sklearn pipeline.

### 5.14 `features/uc_window_features/`

**Layer:** Curated features

**Purpose:** Final feature source for anomaly models. It represents behavior over a history window ending at a cutoff.

**Grain:** One UC x cutoff_date x window_days.

**Key:** (uc_id, cutoff_date, window_days, feature_version, as_of_utc).

**Partitioning:** cutoff_month and window_days.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `uc_id / cutoff_date / window_days` | `Identity` | Example: UC 123, cutoff 2026-06-30, 30-day window. |
| `feature_version / as_of_utc` | `Lineage` | Reproducible feature state. |
| `hist_*` | `Float` | 7/30-day medians, MAD, trends and recent-vs-baseline ratios. |
| `peer_*` | `Float` | Deviation from comparable UCs, transformer or feeder peers. |
| `solar_*` | `Float` | Daylight export, irradiance correlation, solar-shape residual. |
| `persistence_*` | `Float/Int` | Number/fraction of anomalous days and consecutive patterns. |
| `coverage_*` | `Float` | Window data completeness and reliable-day count. |
| `context_*` | `Typed` | Context valid at cutoff. |

**Built from:** uc_day_features plus weather, topology and history.

**Consumed by:** model_dataset builder.

**Critical rule:** Recommended initial windows: 7 and 30 days. A single day is too fragile for GD/theft screening.

### 5.15 `labels/labels.parquet`

**Layer:** Labels

**Purpose:** Human/operationally validated outcomes, stored separately from features.

**Grain:** One target assertion for one UC and validity/evidence interval.

**Key:** label_id.

**Partitioning:** target_name and label_year.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `label_id` | `String/UUID` | Unique label assertion. |
| `uc_id` | `String` | Consumer unit. |
| `target_name` | `Categorical` | unregistered_gd, ntl_irregularity, inspected_normal, etc. |
| `target_value` | `Categorical/Boolean` | Positive, negative, unknown or multiclass value. |
| `valid_from / valid_to` | `Date/Datetime, nullable` | Period to which label applies. |
| `evidence_timestamp` | `Datetime[UTC]` | When evidence was produced. |
| `evidence_source / evidence_id` | `String` | Inspection, order, cadastro, etc. |
| `label_status` | `Categorical` | confirmed, probable, inconclusive. |
| `label_confidence` | `Float, nullable` | Controlled score when justified. |
| `created_at / label_version` | `Audit fields` | Label lineage. |

**Built from:** Inspection events, GD registry reconciliation and expert review.

**Consumed by:** Evaluation and future supervised/semi-supervised training.

**Critical rule:** Never derive a training label from the same anomaly score being evaluated.

### 5.16 `model_input/<dataset_version>/model_dataset.parquet`

**Layer:** Model-ready

**Purpose:** The only table that normal scikit-learn training code needs to open.

**Grain:** One UC x cutoff_date x window_days.

**Key:** id__sample_id.

**Partitioning:** Usually one immutable file/dataset per version; optionally split by meta__split.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `id__sample_id` | `String` | Hash of UC, cutoff, window and version. |
| `id__uc_id` | `String` | Grouping/audit only; excluded from X. |
| `meta__cutoff_date` | `Date` | Prediction/scoring date. |
| `meta__window_days` | `Int16` | History window. |
| `meta__split` | `Categorical` | train, validation, test or score_only. |
| `meta__feature_version / dataset_version` | `String` | Lineage. |
| `x__*` | `Numeric or categorical` | Only columns permitted in model input X. |
| `y__unregistered_gd` | `Nullable target` | Present only when a valid label exists. |
| `y__ntl_irregularity` | `Nullable target` | Separate task/target. |

**Built from:** uc_window_features joined to labels and deterministic split assignment.

**Consumed by:** scikit-learn pipelines.

**Critical rule:** Null means missing. Never encode missing as 0 or -999. Feature prefixes make leakage-resistant selection simple.

### 5.17 `model_input/<dataset_version>/feature_catalog.yml`

**Layer:** Model-ready metadata

**Purpose:** Machine-readable contract for every x__ feature.

**Grain:** One YAML entry per feature.

**Key:** Feature name.

**Partitioning:** Stored beside model_dataset.parquet.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `name` | `String` | Exact x__ column name. |
| `dtype` | `String` | numeric, categorical, boolean. |
| `unit` | `String/null` | Physical unit when applicable. |
| `description` | `String` | Meaning. |
| `source_columns` | `List` | Lineage. |
| `window` | `String/int` | Aggregation window. |
| `missing_policy` | `String` | Null, imputation or exclusion rule. |
| `allowed_for_model` | `Boolean` | Blocks identifiers/leaky columns. |

**Built from:** Reviewed feature definitions.

**Consumed by:** Dataset validation and training code.

**Critical rule:** The training script should select features from this catalog, not an ad-hoc manual list.

### 5.18 `model_input/<dataset_version>/dataset_manifest.json`

**Layer:** Model-ready metadata

**Purpose:** Reproduces an exact model dataset.

**Grain:** One JSON document per immutable dataset version.

**Key:** dataset_version.

**Partitioning:** Stored beside model_dataset.parquet.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `dataset_version` | `String` | Immutable identifier. |
| `created_at` | `Datetime[UTC]` | Build time. |
| `git_commit` | `String` | Code version. |
| `source_run_ids` | `List` | Raw extraction lineage. |
| `feature_version / label_version` | `String` | Contracts used. |
| `as_of_utc` | `Datetime[UTC]` | Knowledge cutoff. |
| `date_range / feeder_scope` | `Object` | Population scope. |
| `row_count / feature_count` | `Integer` | Shape. |
| `split_policy` | `Object` | Chronological/group/purge rules. |
| `file_hashes` | `Object` | Integrity. |

**Built from:** Dataset build job.

**Consumed by:** Training reports, model registry and TCC reproducibility.

**Critical rule:** Never silently overwrite a published dataset version.

### 5.19 `outputs/anomaly_scores/model_version=<version>/*.parquet`

**Layer:** Outputs

**Purpose:** Stores scores separately from source features and labels.

**Grain:** One model score per sample.

**Key:** (id__sample_id, model_version).

**Partitioning:** score_date/model_version.

| Column | Logical type | Meaning |
| --- | --- | --- |
| `id__sample_id / id__uc_id` | `String` | Traceability. |
| `model_version / dataset_version` | `String` | Lineage. |
| `score_gd / score_ntl` | `Float, nullable` | Separate model scores. |
| `rank_percentile` | `Float` | Operational ranking. |
| `flag_for_review` | `Boolean` | Thresholded workflow flag. |
| `scored_at` | `Datetime[UTC]` | Scoring time. |

**Built from:** Trained model applied to model_dataset.

**Consumed by:** Case review, dashboards and label feedback loop.

**Critical rule:** Never write model output back into labels automatically.

## 6. How the current daily report is decomposed

The current report should be treated as a temporary wide export. Its fields are routed as follows:

| Current group | Destination | Treatment |
| --- | --- | --- |
| UC/cadastro columns | `uc_history` | Preserve time validity; remove personal identifier from model features. |
| NIO, installation/removal dates, meter subtype | `meter_installation_history` | Build temporal UC-NIO relationship. |
| Feeder/substation/transformer/geography | `network_topology_history` | Preserve validity and use for peer groups. |
| GD and beneficiary columns | `gd_registry_history` | Preserve registration validity. |
| `FA_INTERVAL`, `RA_INTERVAL`, `R_Q*_INTERVAL`, average current/voltage JSON | `meter_interval` | Explode JSON: one timestamp/value per row. Validate physical semantics and units. |
| `U_L*`, `I_INSTANT_L*` JSON | `meter_instantaneous` | Explode separately; do not mix with interval averages. |
| `*_TOTAL`, `*_MD*` JSON | `meter_register_snapshot` | Treat as snapshots/registers, not interval energy. |
| `HAS_MDM_DATA` | quality/context only | Never use same-day JSON-derived features to predict it. |
| `REPORT_DAY` | extraction context | Derive actual timestamps from JSON keys plus local day. |

Preferred future extraction: query each source family directly in long/tabular form. Do not aggregate a full day's time series into JSON unless the source forces it. If JSON is unavoidable, preserve the raw JSON, then explode it immediately into normalized Parquet.

## 7. Feature construction

### 7.1 Daily features

For each UC/day/channel, calculate data quality before electrical statistics:

```text
expected_count
observed_count
coverage_ratio
longest_missing_gap
estimated_ratio
late_record_count
corrected_record_count
```

Then calculate physical/behavioral features where coverage is sufficient:

```text
energy import/export totals
reverse-energy share
current mean, p95, max, zero ratio, load factor
voltage mean, min, max, dispersion, out-of-band ratio
phase unbalance
night/day consumption share
peak hour and ramp statistics
register-vs-interval consistency
```

No arbitrary current base is required. Keep absolute current. Add relative current only through meaningful comparisons:

```text
current versus the UC's own 30-day median
current versus same hour on previous comparable days
current versus similar peer UCs
current per confirmed installed/contracted capacity, when available
```

### 7.2 Window features

For each cutoff date, use only data available up to that cutoff and `as_of_utc`:

```text
7-day and 30-day medians/MAD
recent-vs-baseline ratios
trend slopes
number of persistent reverse-flow days
solar-hour export share
correlation with irradiance/daylight proxy
peer-group residuals
fraction of reliable days
```

The feature builder must not look at future days, later registrations or inspection results.

## 8. Late arrivals and corrections

Daily extraction must not permanently close a day after one run.

Recommended policy:

1. Append every extraction to `raw`.
2. Incrementally query by `source_updated_at` when available.
3. Also re-query a rolling event-time window, initially D-30 to D, until late-arrival statistics justify a smaller window.
4. Build a canonical measurement view by natural key, selecting the latest known source version as of a chosen `as_of_utc`.
5. Detect which UC-days changed.
6. Rebuild only affected `uc_day_features` and dependent 7/30-day windows.
7. Publish a new immutable dataset version when model input changes.

This allows two reproducible questions:

```text
What is the best value known now?
What value was known when model version X was trained?
```

## 9. Exact scikit-learn boundary

### 9.1 Unsupervised initial model

```python
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

path = "data/model_input/anomaly_v1/model_dataset.parquet"
lf = pl.scan_parquet(path)

schema = lf.collect_schema()
numeric_features = [
    name
    for name, dtype in schema.items()
    if name.startswith("x__") and dtype.is_numeric()
]

train = (
    lf.filter(pl.col("meta__split") == "train")
      .select("id__sample_id", "id__uc_id", *numeric_features)
      .collect()
)

X_train = train.select(numeric_features).to_numpy()

model = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    ),
)

model.fit(X_train)
```

`id__*`, `meta__*` and `y__*` never enter `X`.

### 9.2 PCA, K-means or SVM

These algorithms depend on numerical scale. Scaling stays inside the pipeline and is fitted only on training data:

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

model = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    RobustScaler(),
    PCA(n_components=0.95, random_state=42),
    KMeans(n_clusters=8, random_state=42, n_init="auto"),
)

model.fit(X_train)
```

This statistical scaling does not claim that current has a physical nominal value.

### 9.3 Later supervised model

After validated labels exist:

```python
labeled = (
    lf.filter(
        (pl.col("meta__split") == "train")
        & pl.col("y__unregistered_gd").is_not_null()
    )
    .select(*numeric_features, "y__unregistered_gd")
    .collect()
)

X_train = labeled.select(numeric_features).to_numpy()
y_train = labeled["y__unregistered_gd"].to_numpy()
```

GD and non-technical-loss targets must remain separate. They may share features but need separate models, thresholds and evaluation.

## 10. Train/validation/test policy

Do not randomly split overlapping UC windows.

Recommended initial policy:

```text
train      -> oldest cutoff dates
purge gap  -> at least maximum feature window (for example 30 days)
validation -> later dates
purge gap  -> 30 days
final test -> newest untouched dates
```

Why: a 30-day sample ending on 30 June and another ending on 1 July share 29 raw days. A random row split would leak almost the same history into train and test.

Use a UC-group holdout only when testing generalization to completely unseen UCs. Use chronological holdout when the operational goal is scoring future behavior of known UCs.

## 11. Minimum viable implementation

### Phase 1 - Build correct data

Required:

```text
raw extracts
extraction manifests
uc_history
meter_installation_history
measurement_channel_config_history
meter_interval
meter_instantaneous
meter_register_snapshot
gd_registry_history
uc_day_features
uc_window_features
model_dataset + manifest + feature catalog
```

Initial scope:

```text
one feeder
30-90 days
7-day and 30-day windows
numeric features
Isolation Forest baseline
manual review of highest and lowest scores
```

### Phase 2 - Obtain evaluation evidence

Add:

```text
inspection_events
validated normal inspected cases
confirmed GD registrations and registration dates
confirmed unregistered-GD cases, if available
confirmed irregularity cases
```

Then measure precision among reviewed top-k cases, recall on known cases, ranking stability and false-positive causes.

### Phase 3 - Expand context

Add weather, topology peers, more feeders and more seasons. Only then claim generalization beyond the pilot feeder/time period.

## 12. Recommended repository integration

```text
tcc_extraction/
├── configs/
│   ├── measurement_catalog.yml
│   ├── dataset_v1.yml
│   └── model_isolation_forest_v1.yml
├── data/                       # gitignored
├── docs/
│   └── dataset_layout_sklearn_tcc.md
├── src/
│   ├── extract/                # existing extraction adapters
│   ├── normalize/
│   │   ├── explode_mdm_series.py
│   │   ├── build_uc_history.py
│   │   ├── build_installation_history.py
│   │   └── reconcile_measurements.py
│   ├── features/
│   │   ├── build_uc_day_features.py
│   │   └── build_uc_window_features.py
│   ├── datasets/
│   │   ├── assign_temporal_splits.py
│   │   └── build_model_dataset.py
│   └── models/
│       ├── train_isolation_forest.py
│       └── score_anomalies.py
└── tests/
    ├── test_measurement_uniqueness.py
    ├── test_uc_meter_temporal_join.py
    ├── test_cadence_and_coverage.py
    ├── test_no_future_leakage.py
    ├── test_feature_schema.py
    └── test_dataset_reproducibility.py
```

Do not commit operational data, credentials, personal identifiers or model outputs containing UC identifiers.

## 13. What to show next

For each source/query, provide this mapping:

| Question | Required answer |
| --- | --- |
| Source system/table/query | Exact source name and current SQL/query file |
| Natural key | Columns that uniquely identify one source measurement/version |
| UC/NIO relationship | Table and start/end timestamps for installation |
| Measurement event time | Column representing when value was measured |
| Arrival/update time | Columns representing arrival and later correction |
| Measurement semantics | Import/export, average/instantaneous/register, phase, tariff period |
| Unit | A, V, kWh, kvarh, kW, etc. |
| Quality flags | Estimated, invalid, substituted, missing, meter alarm |
| Configured cadence | Table/column and validity timestamps |
| History availability | Earliest reliable date and backfill limits |
| GD registry | Registration/beneficiary validity and capacity |
| Inspection evidence | Event/result codes and dates usable as labels |

Start with these five artifacts:

```text
1. SQL/query that produces FA_INTERVAL/RA_INTERVAL
2. SQL/query that produces U_L*/I_INSTANT_L*
3. SQL/query that produces *_TOTAL/*_MD
4. UC-NIO installation history query
5. source columns for created/arrived/updated timestamps
```

With those, the exact normalized schemas and extraction SQL can be finalized without guessing measurement semantics.

## 14. Acceptance checklist

A dataset version is ready for scikit-learn only when:

- every row has one declared sample grain;
- all `x__*` columns have fixed types;
- units are documented;
- missing values are null;
- UC/NIO are strings;
- no JSON remains in `model_dataset.parquet`;
- no identifier/target column appears in X;
- features use only information available before cutoff;
- temporal windows do not overlap across split boundaries without a purge gap;
- late corrections are reproducible through `as_of_utc` and manifests;
- dataset and feature versions are immutable;
- model scores remain separate from labels.

## References

1. scikit-learn, *Getting Started* and `ColumnTransformer`: heterogeneous preprocessing inside a model pipeline.
2. scikit-learn, *Common pitfalls and recommended practices*: consistent preprocessing and prevention of data leakage.
3. scikit-learn, *Novelty and Outlier Detection*: distinction between outlier and novelty detection.
4. scikit-learn, `TimeSeriesSplit`: evaluation of time-ordered data without training on the future.
5. Polars, *Sources and sinks* and `scan_parquet`: lazy scans, projection/predicate pushdown and streaming execution.
6. Apache Parquet, *Overview* and *Concepts*: column-oriented storage, row groups and column chunks.
