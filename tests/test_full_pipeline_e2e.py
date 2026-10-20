"""End-to-End Integration Test for the Entire TCC Pipeline.

Tests the full data and modeling lifecycle:
1. Context Layer Normalization (UC, Meters, Electrical Hierarchy).
2. Measurements Layer Normalization (Interval & Instantaneous).
3. Events Layer Multi-Table Alarm Extraction (T_EVENT_YYYYMMDD across window).
4. Feature Engineering (Daily Features, Rolling 30d Window Features, Neighborhood Trafo Features).
5. Model Input Generation (training_dataset with strict id__*, meta__*, x__*, y__* prefixes).
6. Composite Anomaly Detection (IsolationForest + LOF + ECOD + PCA + Physics Rules).
7. Spurious Glitch and Grid-Wide Contingency Filtering.
8. Voltage Disturbance and Imbalance Classification (Baseline Heuristics + ML Classifier).
9. Data Lake Health Audit and Report Generation.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from src.automation.health_check import DataLakeHealthChecker
from src.datasets.build_training_dataset import build_training_dataset
from src.datasets.normalize import (
    normalize_alarm_events,
    normalize_electrical_hierarchy,
    normalize_meter_installation_history,
    normalize_uc_context,
)
from src.datasets.partitioning import (
    ensure_partition_dirs,
    events_partition_path,
    measurements_partition_path,
)
from src.datasets.writers import (
    write_context_table,
    write_events_table,
    write_features,
    write_training_dataset,
    write_window_features,
)
from src.events.alarm_extractor import AlarmExtractor
from src.features.meter_day import build_meter_day_features
from src.features.neighborhood import compute_neighborhood_features
from src.features.uc_window import build_uc_window_features
from src.models.composite import CompositeAnomalyDetector
from src.models.spurious_filter import filter_spurious_anomalies
from src.models.voltage_classifier import (
    MLVoltageDisturbanceClassifier,
    RuleBasedVoltageClassifier,
    VoltageDisturbanceType,
    extract_voltage_classifier_features,
)


@pytest.fixture
def simulated_database():
    """Create in-memory SQLite database simulating multi-day T_EVENT tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        # Day 2026-06-21
        conn.execute(text("""
            CREATE TABLE "AMI.T_EVENT_20260621" (
                EVENT_ID TEXT,
                OBJ_ID TEXT,
                ORIGIN_TIME TIMESTAMP,
                RECEIVE_TIME TIMESTAMP,
                CONTENT TEXT,
                SYSTEM_CODE TEXT,
                OBJECT_TYPE TEXT,
                OBJECT_ORG TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO "AMI.T_EVENT_20260621" VALUES
            ('EVT_101', '41000001', '2026-06-21 08:15:00', '2026-06-21 08:15:20', 'Falta de Energia (Last Gasp)', 'SYS01', 'METER', 'COPEL'),
            ('EVT_102', '41000002', '2026-06-21 08:15:00', '2026-06-21 08:15:25', 'Falta de Energia (Last Gasp)', 'SYS01', 'METER', 'COPEL'),
            ('EVT_103', '41000003', '2026-06-21 08:15:00', '2026-06-21 08:15:30', 'Falta de Energia (Last Gasp)', 'SYS01', 'METER', 'COPEL')
        """))

        # Day 2026-06-22
        conn.execute(text("""
            CREATE TABLE "AMI.T_EVENT_20260622" (
                EVENT_ID TEXT,
                OBJ_ID TEXT,
                ORIGIN_TIME TIMESTAMP,
                RECEIVE_TIME TIMESTAMP,
                CONTENT TEXT,
                SYSTEM_CODE TEXT,
                OBJECT_TYPE TEXT,
                OBJECT_ORG TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO "AMI.T_EVENT_20260622" VALUES
            ('EVT_201', '41000004', '2026-06-22 10:00:00', '2026-06-22 10:00:15', 'Subtensao de Fase', 'SYS01', 'METER', 'COPEL')
        """))

        # 2026-06-23 has no table (simulates unrecorded day / ORA-00942)
        conn.commit()

    return engine


def test_full_pipeline_e2e(tmp_path: Path, simulated_database):
    """Execute complete end-to-end integration run across all layers."""
    base_dir = tmp_path
    cutoff_date = date(2026, 6, 23)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Context Layer
    # ─────────────────────────────────────────────────────────────────────────
    cis_raw = pl.DataFrame({
        "UC": ["UC_01", "UC_02", "UC_03", "UC_04", "UC_05", "UC_06"],
        "NIO": ["41000001", "41000002", "41000003", "41000004", "41000005", "41000006"],
        "CLASSE_CONSUMO": ["RESIDENCIAL"] * 6,
        "TIPO_FASE": ["TRIFASICO", "TRIFASICO", "TRIFASICO", "TRIFASICO", "BIFASICO", "MONOFASICO"],
        "QTD_TENS_LIG_UEE": [127.0] * 6,
        "NOM_MUN_MUN": ["ARAUCARIA"] * 6,
        "DATA_INSTALACAO_MEDIDOR": [date(2025, 1, 1)] * 6,
        "DATA_RETIRADA_MEDIDOR": [None] * 6,
        "TIPO_MEDIDOR": ["SMART"] * 6,
        "SMART": ["1"] * 6,
        "FEEDER_ID": ["FONTE_NOVA"] * 6,
        "POT_INST_KVA": [15.0] * 6,
    })

    geo_raw = pl.DataFrame({
        "UC": ["UC_01", "UC_02", "UC_03", "UC_04", "UC_05", "UC_06"],
        "SUBESTACAO": ["SE ARAUCARIA"] * 6,
        "ALIMENTADOR": ["FONTE_NOVA"] * 6,
        "POSTO_OPERACIONAL": ["TR_100", "TR_100", "TR_100", "TR_200", "TR_200", "TR_200"],
        "TENSAO_ALIMENTADOR": [13800.0] * 6,
    })

    uc_ctx = normalize_uc_context(cis_raw, run_id="e2e_run")
    meter_hist = normalize_meter_installation_history(cis_raw, run_id="e2e_run")
    elec_hier = normalize_electrical_hierarchy(cis_raw, geo_raw, run_id="e2e_run")

    write_context_table(uc_ctx, "uc_context", base_dir=base_dir)
    write_context_table(meter_hist, "meter_installation_history", base_dir=base_dir)
    write_context_table(elec_hier, "electrical_hierarchy", base_dir=base_dir)

    assert (base_dir / "context" / "uc_context.parquet").exists()
    assert (base_dir / "context" / "electrical_hierarchy.parquet").exists()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Events Layer: Multi-Table Alarm Extraction across 7-day window
    # ─────────────────────────────────────────────────────────────────────────
    extractor = AlarmExtractor(engine=simulated_database)
    alarms_df = extractor.extract_window(
        cutoff_date=cutoff_date,
        window_days=7,
        nios=["41000001", "41000002", "41000003", "41000004", "41000005", "41000006"],
        save_partitions=True,
        base_dir=base_dir,
    )

    assert alarms_df.height == 4
    assert "EVT_101" in alarms_df["ALARM_ID"].to_list()
    assert "EVT_201" in alarms_df["ALARM_ID"].to_list()

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Measurements Layer (AMI Instantaneous and Interval)
    # ─────────────────────────────────────────────────────────────────────────
    # Simulates Day 2026-06-23 measurements
    inst_records = []
    for uc, nio in [
        ("UC_01", "41000001"),  # Normal
        ("UC_02", "41000002"),  # Normal
        ("UC_03", "41000003"),  # Outage / 0V
        ("UC_04", "41000004"),  # Sag
        ("UC_05", "41000005"),  # Bifásico normal
        ("UC_06", "41000006"),  # Monofásico normal
    ]:
        if uc == "UC_03":
            u1, u2, u3 = 0.0, 0.0, 0.0
        elif uc == "UC_04":
            u1, u2, u3 = 90.0, 92.0, 91.0
        elif uc == "UC_05":
            u1, u2, u3 = 127.0, 127.0, None
        elif uc == "UC_06":
            u1, u2, u3 = 127.0, None, None
        else:
            u1, u2, u3 = 127.0, 127.0, 127.0

        for h in range(4):
            inst_records.append({
                "NIO": nio,
                "REPORT_DAY": cutoff_date,
                "TIMESTAMP_UTC": datetime(2026, 6, 23, h, 0, 0),
                "CADENCE_MINUTES": 15,
                "MEASUREMENT_SOURCE": "MDM",
                "U_L1": u1,
                "U_L2": u2,
                "U_L3": u3,
                "I_INSTANT_L1": 5.0,
                "I_INSTANT_L2": 5.0,
                "I_INSTANT_L3": 5.0,
                "QUALITY_CODE": 0,
            })

    instantaneous_df = pl.DataFrame(inst_records)

    meas_part = measurements_partition_path("ami_instantaneous", cutoff_date, base_dir=base_dir)
    ensure_partition_dirs(meas_part)
    instantaneous_df.write_parquet(meas_part / "data.parquet")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────
    # Build daily features
    interval_empty = pl.DataFrame(schema={"NIO": pl.String, "FA_INTERVAL": pl.Float64, "RA_INTERVAL": pl.Float64})
    day_feats = build_meter_day_features(
        interval_df=interval_empty,
        instantaneous_df=instantaneous_df,
        registers_df=pl.DataFrame(),
        context_df=uc_ctx,
        report_day=cutoff_date,
    )
    write_features(day_feats, "uc_day_features", cutoff_date, base_dir=base_dir)

    # Build rolling window features with multi-day alarms
    window_feats = build_uc_window_features(
        daily_features_df=day_feats,
        meter_history_df=meter_hist,
        alarm_events_df=alarms_df,
        cutoff_date=cutoff_date,
        window_days=7,
    )
    write_window_features(window_feats, "uc_window_features", cutoff_date, base_dir=base_dir)

    # Compute spatial neighborhood features
    neighborhood_df = compute_neighborhood_features(
        day_features_df=day_feats,
        hierarchy_df=elec_hier,
    )

    assert window_feats.height == 6
    assert "ALARM_COUNT_WINDOW" in window_feats.columns

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Model Input Generation (id__*, meta__*, x__*, y__*)
    # ─────────────────────────────────────────────────────────────────────────
    train_ds = build_training_dataset(
        day_features_df=day_feats,
        window_features_df=window_feats,
        hierarchy_df=elec_hier,
        cutoff_date=cutoff_date,
    )
    write_training_dataset(train_ds, version="v1", base_dir=base_dir)

    feature_cols = [c for c in train_ds.columns if c.startswith("x__")]
    id_cols = [c for c in train_ds.columns if c.startswith("id__")]
    meta_cols = [c for c in train_ds.columns if c.startswith("meta__")]

    assert len(feature_cols) > 5
    assert len(id_cols) >= 1
    assert len(meta_cols) >= 1

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Composite Anomaly Model Training & Scoring
    # ─────────────────────────────────────────────────────────────────────────
    X = train_ds.select(feature_cols).to_numpy()
    detector = CompositeAnomalyDetector(contamination=0.15, random_state=42)
    detector.fit(X)

    scores = detector.score_samples(X)
    is_anomaly = detector.predict(X)

    scores_df = train_ds.select(id_cols).with_columns([
        pl.Series("anomaly_score", scores),
        pl.Series("is_anomaly", is_anomaly),
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Spurious Filter (Transformer Peer Consensus)
    # ─────────────────────────────────────────────────────────────────────────
    filtered_df = filter_spurious_anomalies(
        scores_df=scores_df,
        neighborhood_df=neighborhood_df,
    )
    assert "FILTERED_ANOMALY_SCORE" in filtered_df.columns
    assert "ANOMALY_CATEGORY" in filtered_df.columns

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Voltage Disturbance & Imbalance Classification
    # ─────────────────────────────────────────────────────────────────────────
    volt_features_df = extract_voltage_classifier_features(
        instantaneous_df=instantaneous_df,
        context_df=uc_ctx,
        neighborhood_df=neighborhood_df,
        alarms_df=alarms_df,
    )

    # Deterministic baseline labeling
    baseline_labeled = RuleBasedVoltageClassifier.classify(volt_features_df)
    assert "BASELINE_VOLTAGE_LABEL" in baseline_labeled.columns

    # ML Classifier training and probabilistic inference
    # Duplicate samples slightly to meet minimum training size
    train_pool = pl.concat([baseline_labeled] * 3, how="vertical")
    ml_clf = MLVoltageDisturbanceClassifier(random_state=42)
    ml_clf.fit(train_pool)

    clf_result = ml_clf.predict(volt_features_df)
    pred_df = clf_result.predictions_df

    assert "PREDICTED_DISTURBANCE_CLASS" in pred_df.columns
    assert "DISTURBANCE_CONFIDENCE" in pred_df.columns

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Data Lake Health Audit
    # ─────────────────────────────────────────────────────────────────────────
    checker = DataLakeHealthChecker(base_dir=base_dir)
    audit_report = checker.run_full_health_audit()

    assert audit_report["global_status"] in ("HEALTHY", "WARNING", "INCOMPLETE")
    assert (base_dir / "reports" / "daily_health_report_latest.md").exists()
