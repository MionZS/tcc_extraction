"""Tests for Composite Multi-Technique Anomaly Detector."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.composite import CompositeAnomalyDetector


def test_composite_detector_fit_and_predict():
    np.random.seed(42)
    # Generate 100 normal samples and 5 extreme outliers
    X_normal = np.random.normal(loc=10.0, scale=1.0, size=(100, 4))
    X_outliers = np.random.normal(loc=50.0, scale=2.0, size=(5, 4))
    X = np.vstack([X_normal, X_outliers])

    detector = CompositeAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(X)

    preds = detector.predict_composite(X)

    assert "consensus_score" in preds
    assert "iso_score" in preds
    assert "lof_score" in preds
    assert "ecod_score" in preds
    assert "pca_score" in preds
    assert "is_anomaly" in preds

    assert len(preds["consensus_score"]) == 105
    # Outliers should have higher consensus scores than normal samples
    outlier_mean_score = np.mean(preds["consensus_score"][100:])
    normal_mean_score = np.mean(preds["consensus_score"][:100])
    assert outlier_mean_score > normal_mean_score
    assert outlier_mean_score > 0.60


def test_composite_detector_rule_violation():
    # Test physical violation elevation
    X = np.ones((10, 3)) * 10.0
    df = pl.DataFrame({
        "VOLTAGE_MEAN": [220.0] * 9 + [150.0],  # 150V is a severe PRODIST violation
    })

    detector = CompositeAnomalyDetector(contamination=0.1, random_state=42)
    detector.fit(X)
    preds = detector.predict_composite(X, raw_dataframe=df)

    assert preds["rule_violation_flag"][-1] is True
    assert preds["is_anomaly"][-1] is True
