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

    assert bool(preds["rule_violation_flag"][-1]) is True
    assert bool(preds["is_anomaly"][-1]) is True


def test_lof_tie_log_compression_prevents_p99_collapse():
    """Verify LOF log1p compression prevents k-distance=0 ties from crushing P01-P99."""
    from src.models.composite import _compute_norm_params, _normalize_with_params

    rng = np.random.default_rng(42)
    # 90 exact duplicate points (k-distance = 0 -> huge raw negative LOF) + 10 dispersed points
    X_ties = np.tile(np.array([[10.0, 2.0, 0.0, 220.0]]), (90, 1))
    X_disp = rng.normal(loc=[12.0, 3.0, 0.5, 220.0], scale=[1.0, 0.5, 0.2, 1.0], size=(10, 4))
    X = np.vstack([X_ties, X_disp])

    detector = CompositeAnomalyDetector(contamination=0.05, lof_neighbors=10, random_state=42)
    detector.fit(X)
    preds = detector.predict_composite(X)

    # Raw LOF should be finite and log-compressed (not ~1e10)
    assert np.all(np.isfinite(preds["raw_lof_score"]))
    assert float(np.max(preds["raw_lof_score"])) < 100.0
    # Dispersed points should still get non-zero LOF scores
    assert float(np.mean(preds["lof_score"][90:])) > 0.0

    # Also test _normalize_with_params P99.9 fallback when P01 == P99
    arr = np.zeros(1000, dtype=float)
    arr[-2:] = [0.5, 1.0]  # P99 == 0.0, P99.9 > 0.0
    params = _compute_norm_params(arr)
    assert params["p01"] == 0.0
    assert params["p99"] == 0.0
    assert params["p999"] > 0.0
    normed = _normalize_with_params(arr, params, mode="frozen_quantile")
    assert normed[-1] == 1.0
    assert 0.0 < normed[-2] < 1.0

    # Constant array fallback returns all zeros
    const_params = _compute_norm_params(np.ones(50, dtype=float))
    const_normed = _normalize_with_params(np.ones(10, dtype=float), const_params, mode="frozen_quantile")
    assert np.allclose(const_normed, 0.0)


def test_pca_internal_standardization_empty_feature_preservation_and_serialization(tmp_path):
    """Verify internal PCA StandardScaler, all-NaN feature preservation, and joblib round-trip."""
    import joblib

    rng = np.random.default_rng(7)
    # Mix vastly different scales + an all-NaN column
    X_train = np.column_stack([
        rng.normal(1000.0, 50.0, size=60),
        rng.normal(0.01, 0.002, size=60),
        np.zeros(60),  # constant zero column (IQR=0)
        np.full(60, np.nan),  # completely empty feature
    ])
    X_eval = np.column_stack([
        rng.normal(1000.0, 50.0, size=15),
        rng.normal(0.01, 0.002, size=15),
        np.zeros(15),
        np.full(15, np.nan),
    ])

    detector = CompositeAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(X_train)
    assert detector.pca.scaler_ is not None

    preds_before = detector.predict_composite(X_eval)
    assert len(preds_before["consensus_score"]) == 15
    assert np.all(np.isfinite(preds_before["pca_score"]))

    model_path = tmp_path / "composite_model.joblib"
    joblib.dump(detector, model_path)
    loaded: CompositeAnomalyDetector = joblib.load(model_path)

    preds_after = loaded.predict_composite(X_eval)
    assert loaded.frozen_threshold_ == detector.frozen_threshold_
    assert np.allclose(preds_before["consensus_score"], preds_after["consensus_score"])
    assert np.allclose(preds_before["pca_score"], preds_after["pca_score"])
    assert np.allclose(preds_before["lof_score"], preds_after["lof_score"])
    assert np.array_equal(preds_before["is_anomaly"], preds_after["is_anomaly"])

