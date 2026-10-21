"""Unit and Integration Tests for Specialized Smart Meter Anomaly Models.

Verifies:
1. Complete separation of Communication (S_comm) and Electrical (S_elec) feature spaces.
2. Electrical Data Quality Gating (unreliable scores flagged rather than fabricated).
3. Tamper / Irregularity Scoring (S_tamper).
4. Behavioral Consumption Profile Clustering.
5. Multi-View Anomaly Pipeline end-to-end execution.
"""

from __future__ import annotations

from datetime import date
import numpy as np
import polars as pl
import pytest

from src.datasets.build_training_dataset import build_training_dataset
from src.models.specialized_detectors import (
    BehavioralClusterer,
    CommunicationAnomalyDetector,
    ElectricalAnomalyDetector,
    MultiViewAnomalyPipeline,
    TamperAnomalyDetector,
)


def test_build_training_dataset_feature_set_separation():
    # Synthetic window feature table
    window_df = pl.DataFrame({
        "UC": ["UC_001", "UC_002"],
        "CUTOFF_DATE": [date(2026, 6, 1), date(2026, 6, 1)],
        "WINDOW_DAYS": [30, 30],
        # Communication
        "COVERAGE_RELIABLE_DAYS": [30, 5],
        "AVG_INTERVAL_COVERAGE": [0.99, 0.20],
        "ALARM_COUNT_WINDOW": [2, 85],
        "AVG_ALARM_LATENCY_SEC": [10.5, 450.0],
        # Electrical
        "HIST_FA_SUM_MEDIAN": [120.0, 115.0],
        "HIST_FA_SUM_MAD": [15.0, 12.0],
        "HIST_RA_REVERSAL_DAYS": [0, 0],
        "HIST_RA_REVERSAL_RATIO_MAX": [0.0, 0.0],
        "HIST_LOAD_FACTOR_MEAN": [0.45, 0.42],
        "HIST_VOLTAGE_IMBALANCE_MAX": [1.2, 1.1],
        "HIST_CURRENT_IMBALANCE_MAX": [0.15, 0.14],
        "TREND_FA_SLOPE": [0.02, -0.01],
        # Context
        "METER_AGE_DAYS": [500, 30],
        "METER_CHANGED_30D": [False, True],
        "METER_CHANGED_90D": [False, True],
    })

    # 1. Electrical Feature Set: ZERO communication features permitted in x__*
    ds_elec = build_training_dataset(window_df, feature_set="electrical")
    elec_x_cols = [c for c in ds_elec.columns if c.startswith("x__")]
    assert "x__hist_fa_sum_median" in elec_x_cols
    assert "x__hist_voltage_imbalance_max" in elec_x_cols
    assert "x__avg_interval_coverage" not in elec_x_cols
    assert "x__coverage_reliable_days" not in elec_x_cols
    assert "x__alarm_count_window" not in elec_x_cols
    # meta__data_quality must be preserved for confidence weighting
    assert "meta__data_quality" in ds_elec.columns
    assert ds_elec["meta__data_quality"][1] == 0.20

    # 2. Communication Feature Set: only communication/availability features in x__*
    ds_comm = build_training_dataset(window_df, feature_set="communication")
    comm_x_cols = [c for c in ds_comm.columns if c.startswith("x__")]
    assert "x__avg_interval_coverage" in comm_x_cols
    assert "x__coverage_reliable_days" in comm_x_cols
    assert "x__hist_fa_sum_median" not in comm_x_cols
    assert "x__hist_voltage_imbalance_max" not in comm_x_cols


def test_electrical_data_quality_gating():
    # Train detector on normal electrical features
    np.random.seed(42)
    X_train = np.random.normal(loc=100.0, scale=10.0, size=(50, 4))
    detector = ElectricalAnomalyDetector(min_quality_threshold=0.70, contamination=0.10)
    detector.fit(X_train)

    # Test evaluation:
    # Meter 1: Electrical normal, High Quality (0.95)
    # Meter 2: Electrical extreme outlier, Low Quality (0.25) -> should be marked UNRELIABLE
    # Meter 3: Electrical extreme outlier, High Quality (0.95) -> should be confirmed anomaly
    X_test = np.array([
        [100.0, 100.0, 100.0, 100.0],
        [500.0, 500.0, 500.0, 500.0],
        [500.0, 500.0, 500.0, 500.0],
    ])
    quality = [0.95, 0.25, 0.95]

    res = detector.predict_electrical(X_test, data_quality=quality)
    assert bool(res["is_reliable"][0]) is True
    assert bool(res["is_reliable"][1]) is False  # Low quality: score is unreliable!
    assert bool(res["is_reliable"][2]) is True

    # Crucial assertion: Meter 2 cannot be declared an electrical anomaly because data was missing/insufficient!
    assert bool(res["is_electrical_anomaly"][1]) is False
    # Meter 3 had sufficient data and is an anomaly
    assert bool(res["is_electrical_anomaly"][2]) is True


def test_tamper_anomaly_scoring():
    scorer = TamperAnomalyDetector()
    # Meter 0: No alarms, normal slope -> S_tamper ~ 0.0
    # Meter 1: 3 tamper alarms (magnetic, cover open), sharp drop in consumption -> S_tamper > 0.8
    s_tamper = scorer.score(
        tamper_alarm_counts=[0, 3],
        fa_trend_slopes=[0.05, -4.5],
        meter_changed_30d=[False, True],
    )
    assert s_tamper[0] < 0.10
    assert s_tamper[1] > 0.80


def test_behavioral_clustering():
    np.random.seed(42)
    # Two distinct clusters of load profiles
    cluster_a = np.random.normal(loc=10.0, scale=1.0, size=(20, 5))
    cluster_b = np.random.normal(loc=80.0, scale=2.0, size=(20, 5))
    X = np.vstack([cluster_a, cluster_b])

    clusterer = BehavioralClusterer(n_clusters=2, random_state=42)
    labels = clusterer.fit_predict(X)
    assert len(labels) == 40
    # Samples in A should share label, samples in B should share label
    assert len(np.unique(labels[:20])) == 1
    assert len(np.unique(labels[20:])) == 1
    assert labels[0] != labels[20]


def test_multi_view_pipeline_end_to_end():
    np.random.seed(42)
    n = 20
    X_comm = np.random.uniform(0.8, 1.0, size=(n, 3))
    # Meter 0 has severe communication drop
    X_comm[0] = [0.1, 0.2, 0.1]

    X_elec = np.random.normal(loc=50.0, scale=5.0, size=(n, 4))
    # Meter 1 has genuine physical unbalance/overload
    X_elec[1] = [300.0, 300.0, 300.0, 300.0]

    data_quality = [0.15] + [0.95] * (n - 1)
    tamper_alarms = [0] * n
    # Meter 2 has tamper alarms
    tamper_alarms[2] = 4

    pipeline = MultiViewAnomalyPipeline(contamination=0.10, min_quality_threshold=0.70)
    scores_df = pipeline.fit_and_score(
        X_comm=X_comm,
        X_elec=X_elec,
        data_quality=data_quality,
        tamper_alarms=tamper_alarms,
        uc_ids=[f"UC_{i:03d}" for i in range(n)],
    )

    assert "s_comm" in scores_df.columns
    assert "s_elec" in scores_df.columns
    assert "s_tamper" in scores_df.columns
    assert "data_quality" in scores_df.columns
    assert "is_reliable_elec" in scores_df.columns
    assert "operational_diagnosis" in scores_df.columns

    # Verification of operational diagnoses:
    # Meter 0 has low quality and comm failure -> FALHA_CRITICA_TELEMETRIA or DADO_INSUFICIENTE
    diag_0 = scores_df["operational_diagnosis"][0]
    assert "INSUFICIENTE" in diag_0 or "TELEMETRIA" in diag_0

    # Meter 2 has tamper alarms -> INVESTIGACAO_FRAUDE_OU_TAMPER
    assert scores_df["operational_diagnosis"][2] == "INVESTIGACAO_FRAUDE_OU_TAMPER"
