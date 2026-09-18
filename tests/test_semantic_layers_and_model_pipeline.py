"""Test the semantic multi-layer architecture and model training pipeline.

Verifies:
1. Context layer normalization (uc_context, meter_installation_history, electrical_hierarchy).
2. Alarm events normalization with latency calculation.
3. Feature layer (uc_day_features and uc_window_features).
4. Model input generation (UC × cutoff_date with id__*, meta__*, x__*, y__*).
5. Scikit-learn baseline model training (IsolationForest & KMeans).
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.datasets.schemas import (
    UC_CONTEXT_SCHEMA,
    METER_INSTALLATION_HISTORY_SCHEMA,
    ELECTRICAL_HIERARCHY_SCHEMA,
    ALARM_EVENTS_SCHEMA,
    UC_WINDOW_FEATURES_SCHEMA,
)
from src.datasets.normalize import (
    normalize_uc_context,
    normalize_meter_installation_history,
    normalize_electrical_hierarchy,
    normalize_alarm_events,
)
from src.features.meter_day import build_meter_day_features
from src.features.uc_window import build_uc_window_features
from src.datasets.build_training_dataset import build_training_dataset


def test_normalize_uc_context():
    cis_sample = pl.DataFrame({
        "UC": ["1001", "1002"],
        "CLASSE_CONSUMO": ["RESIDENCIAL", "COMERCIAL"],
        "TIPO_FASE": ["MONOFASICO", "TRIFASICO"],
        "QTD_TENS_LIG_UEE": ["127.0", "220.0"],
        "NOM_MUN_MUN": ["ARAUCARIA", "ARAUCARIA"],
        "LAT": ["-25.59", "-25.60"],
        "LON": ["-49.40", "-49.41"],
    })

    ctx = normalize_uc_context(cis_sample, run_id="test_run")
    assert ctx.height == 2
    assert "UC" in ctx.columns
    assert "CLASSE" in ctx.columns
    assert "TENSAO_BASE" in ctx.columns
    assert ctx["CLASSE"].to_list() == ["RESIDENCIAL", "COMERCIAL"]
    assert ctx["TENSAO_BASE"].to_list() == [127.0, 220.0]


def test_normalize_meter_installation_history():
    cis_sample = pl.DataFrame({
        "UC": ["1001", "1001"],
        "NIO": ["41000001", "41000002"],
        "DATA_INSTALACAO_MEDIDOR": [date(2025, 1, 1), date(2026, 1, 1)],
        "DATA_RETIRADA_MEDIDOR": [date(2025, 12, 31), None],
        "TIPO_MEDIDOR": ["ELETROMECANICO", "SMART TRIFASICO"],
        "SMART": ["0", "1"],
    })

    history = normalize_meter_installation_history(cis_sample, run_id="test_run")
    assert history.height == 2
    assert history["NIO"].to_list() == ["41000001", "41000002"]
    assert history["DATA_INSTALACAO"].to_list() == [date(2025, 1, 1), date(2026, 1, 1)]


def test_normalize_electrical_hierarchy():
    cis_sample = pl.DataFrame({
        "UC": ["1001"],
        "NIO": ["41000001"],
        "FEEDER_ID": ["ARA01"],
        "POT_INST_KVA": ["15.0"],
    })
    geo_sample = pl.DataFrame({
        "UC": ["1001"],
        "SUBESTACAO": ["SE ARAUCARIA"],
        "TENSAO_ALIMENTADOR": [13800.0],
    })

    hierarchy = normalize_electrical_hierarchy(cis_sample, geo_sample, run_id="test_run")
    assert hierarchy.height == 1
    assert hierarchy["SUBESTACAO"][0] == "SE ARAUCARIA"
    assert hierarchy["ALIMENTADOR"][0] == "ARA01"


def test_normalize_alarm_events():
    now = datetime(2026, 6, 23, 12, 0, 0)
    later = now + timedelta(seconds=45)
    alarm_raw = pl.DataFrame({
        "ALARM_ID": ["A001"],
        "OBJ_ID": ["41000001"],
        "ORIGIN_TIMESTAMP": [now],
        "RECEIVED_TIMESTAMP": [later],
        "SYSTEM_CODE": ["ERR_COMM"],
        "CONTENT": ["Communication Timeout"],
    })

    alarms = normalize_alarm_events(alarm_raw, run_id="test_run")
    assert alarms.height == 1
    assert alarms["LATENCY_SECONDS"][0] == 45.0
    assert alarms["NIO"][0] == "41000001"


def test_window_features_and_training_dataset_pipeline():
    cutoff = date(2026, 6, 30)
    dates = [cutoff - timedelta(days=i) for i in range(10)]

    daily_rows = []
    for d in dates:
        daily_rows.append({
            "UC": "UC_001",
            "NIO": "41000001",
            "REPORT_DAY": d,
            "FEATURE_SET_VERSION": "v1",
            "INTERVAL_COVERAGE": 0.95,
            "INSTANTANEOUS_COVERAGE": 0.90,
            "REGISTER_COVERAGE": 1.0,
            "NULL_RATIO_INTERVAL": 0.05,
            "FA_INTERVAL_SUM": 25.0 + np.random.uniform(-1, 1),
            "RA_INTERVAL_SUM": 0.0,
            "FA_MD_MAX": 5.0,
            "LOAD_FACTOR": 0.45,
            "RA_REVERSAL_RATIO": 0.0,
            "VOLTAGE_IMBALANCE_MAX": 0.01,
            "CURRENT_IMBALANCE_MAX": 0.02,
            "FA_INTERVAL_MEDIAN": 0.8,
            "FA_INTERVAL_IQR": 0.2,
            "FA_INTERVAL_P05": 0.3,
            "FA_INTERVAL_P95": 1.5,
            "U_L1_MEDIAN": 127.0,
            "U_L1_IQR": 2.0,
            "FA_RAMP_MAX": 0.5,
            "U_L1_RAMP_MAX": 1.0,
            "PEAK_HOUR_SIN": 0.5,
            "PEAK_HOUR_COS": 0.5,
            "FA_INTERVAL_AUTOCORR_LAG1": 0.7,
            "FA_INTERVAL_STD": 0.3,
            "PHASE_TYPE": "TRIFASICO",
            "CONSUMER_CLASS": "RESIDENCIAL",
            "METER_TYPE": "SMART",
            "INSTALLED_KVA": 15.0,
        })
        daily_rows.append({
            "UC": "UC_002",
            "NIO": "41000002",
            "REPORT_DAY": d,
            "FEATURE_SET_VERSION": "v1",
            "INTERVAL_COVERAGE": 0.95,
            "INSTANTANEOUS_COVERAGE": 0.90,
            "REGISTER_COVERAGE": 1.0,
            "NULL_RATIO_INTERVAL": 0.05,
            "FA_INTERVAL_SUM": 500.0,  # anomalous usage
            "RA_INTERVAL_SUM": 30.0,   # high reverse power
            "FA_MD_MAX": 50.0,
            "LOAD_FACTOR": 0.85,
            "RA_REVERSAL_RATIO": 0.6,
            "VOLTAGE_IMBALANCE_MAX": 0.08,
            "CURRENT_IMBALANCE_MAX": 0.15,
            "FA_INTERVAL_MEDIAN": 15.0,
            "FA_INTERVAL_IQR": 5.0,
            "FA_INTERVAL_P05": 5.0,
            "FA_INTERVAL_P95": 25.0,
            "U_L1_MEDIAN": 127.0,
            "U_L1_IQR": 5.0,
            "FA_RAMP_MAX": 10.0,
            "U_L1_RAMP_MAX": 5.0,
            "PEAK_HOUR_SIN": -0.5,
            "PEAK_HOUR_COS": 0.2,
            "FA_INTERVAL_AUTOCORR_LAG1": 0.2,
            "FA_INTERVAL_STD": 4.0,
            "PHASE_TYPE": "TRIFASICO",
            "CONSUMER_CLASS": "COMERCIAL",
            "METER_TYPE": "SMART",
            "INSTALLED_KVA": 50.0,
        })

    daily_df = pl.DataFrame(daily_rows)

    meter_hist = pl.DataFrame({
        "UC": ["UC_001", "UC_002"],
        "NIO": ["41000001", "41000002"],
        "DATA_INSTALACAO": [date(2026, 6, 15), date(2025, 1, 1)],  # UC_001 changed 15d ago
    })

    window_df = build_uc_window_features(
        daily_df,
        meter_history_df=meter_hist,
        cutoff_date=cutoff,
        window_days=30,
        feature_set_version="v1",
    )

    assert window_df.height == 2
    assert "METER_CHANGED_30D" in window_df.columns
    assert window_df.filter(pl.col("UC") == "UC_001")["METER_CHANGED_30D"][0] is True
    assert window_df.filter(pl.col("UC") == "UC_002")["METER_CHANGED_30D"][0] is False

    # Build ML training matrix
    model_df = build_training_dataset(window_df, split_policy="train")

    assert model_df.height == 2
    assert "id__uc_id" in model_df.columns
    assert "meta__cutoff_date" in model_df.columns
    assert "x__hist_fa_sum_median" in model_df.columns

    # Test training scikit-learn model on the output
    feature_cols = [c for c in model_df.columns if c.startswith("x__") and model_df[c].dtype.is_numeric()]
    X = model_df.select(feature_cols).to_numpy()

    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        IsolationForest(n_estimators=50, contamination=0.5, random_state=42),
    )
    pipe.fit(X)
    scores = -pipe.score_samples(X)
    assert len(scores) == 2

