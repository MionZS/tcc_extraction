"""Unit tests for the Voltage Disturbance and Imbalance Classifier."""

from __future__ import annotations

import polars as pl
import pytest
from src.models.voltage_classifier import (
    ClassificationResult,
    MLVoltageDisturbanceClassifier,
    RuleBasedVoltageClassifier,
    VoltageDisturbanceType,
    extract_voltage_classifier_features,
)


def test_extract_voltage_classifier_features_basic():
    inst_df = pl.DataFrame({
        "UC": ["UC001", "UC002", "UC003"],
        "NIO": ["NIO001", "NIO002", "NIO003"],
        "U_L1": [127.0, 100.0, None],
        "U_L2": [127.0, 127.0, None],
        "U_L3": [127.0, 127.0, None],
    })

    context_df = pl.DataFrame({
        "UC": ["UC001", "UC002", "UC003"],
        "FASE_CIRCUITO": ["TRIFASICO", "TRIFASICO", "TRIFASICO"],
        "TENSAO_BASE": [127.0, 127.0, 127.0],
    })

    neighborhood_df = pl.DataFrame({
        "UC": ["UC001", "UC002", "UC003"],
        "NEIGHBOR_COVERAGE_RATIO": [1.0, 1.0, 0.1],
    })

    feat_df = extract_voltage_classifier_features(
        instantaneous_df=inst_df,
        context_df=context_df,
        neighborhood_df=neighborhood_df,
    )

    assert "U_L1_PU" in feat_df.columns
    assert "VOLTAGE_IMBALANCE_RATIO" in feat_df.columns
    assert "IS_NULL_L1" in feat_df.columns
    assert feat_df.filter(pl.col("UC") == "UC001").get_column("U_L1_PU").item() == 1.0
    assert feat_df.filter(pl.col("UC") == "UC003").get_column("IS_NULL_L1").item() is True


def test_rule_based_voltage_classifier_cases():
    inst_df = pl.DataFrame({
        "UC": ["UC_NORM", "UC_SAG", "UC_SWELL", "UC_MONO_FAULT", "UC_INTERRUPT"],
        "FASE_CIRCUITO": ["TRIFASICO", "TRIFASICO", "TRIFASICO", "TRIFASICO", "TRIFASICO"],
        "TENSAO_BASE": [127.0, 127.0, 127.0, 127.0, 127.0],
        "U_L1": [127.0, 95.0, 145.0, 0.0, 0.0],
        "U_L2": [127.0, 95.0, 145.0, 127.0, 0.0],
        "U_L3": [127.0, 95.0, 145.0, 127.0, 0.0],
        "NEIGHBOR_COVERAGE_RATIO": [1.0, 1.0, 1.0, 1.0, 0.1],
    })

    feat_df = extract_voltage_classifier_features(instantaneous_df=inst_df)
    labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

    labels = dict(zip(labeled_df.get_column("UC").to_list(), labeled_df.get_column("BASELINE_VOLTAGE_LABEL").to_list()))

    assert labels["UC_NORM"] == VoltageDisturbanceType.NORMAL.value
    assert labels["UC_SAG"] == VoltageDisturbanceType.SUBTENSAO.value
    assert labels["UC_SWELL"] == VoltageDisturbanceType.SOBRETENSAO.value
    assert labels["UC_MONO_FAULT"] == VoltageDisturbanceType.FALTA_MONOFASICA.value
    assert labels["UC_INTERRUPT"] == VoltageDisturbanceType.INTERRUPCAO_TOTAL.value


def test_ml_voltage_disturbance_classifier():
    # Build synthetic dataset representing various cases
    rows = []
    for i in range(15):
        rows.append({"UC": f"UC_N_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 127.0, "U_L2": 127.0, "U_L3": 127.0})
        rows.append({"UC": f"UC_S_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 90.0, "U_L2": 92.0, "U_L3": 91.0})
        rows.append({"UC": f"UC_F_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 0.0, "U_L2": 127.0, "U_L3": 127.0})

    df = pl.DataFrame(rows)
    feat_df = extract_voltage_classifier_features(instantaneous_df=df)
    labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

    clf = MLVoltageDisturbanceClassifier(random_state=42)
    clf.fit(labeled_df)

    res = clf.predict(labeled_df)

    assert isinstance(res, ClassificationResult)
    assert "PREDICTED_DISTURBANCE_CLASS" in res.predictions_df.columns
    assert "DISTURBANCE_CONFIDENCE" in res.predictions_df.columns
    assert res.predictions_df.height == 45
