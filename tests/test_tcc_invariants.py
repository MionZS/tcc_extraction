"""Property and invariant tests for TCC ML, temporal leakage, gating, and reproducibility."""

from __future__ import annotations

from datetime import date
import numpy as np
import polars as pl

from src.features.uc_window import build_uc_window_features
from src.models.composite import CompositeAnomalyDetector, select_pure_electrical_features
from src.models.specialized_detectors import ElectricalAnomalyDetector


def test_no_temporal_leakage_in_uc_window():
    """Measurements after cutoff_date must NEVER affect window features."""
    daily_base = pl.DataFrame({
        "UC": ["UC1"] * 5,
        "NIO": ["NIO1"] * 5,
        "REPORT_DAY": [
            date(2026, 8, 28),
            date(2026, 8, 29),
            date(2026, 8, 30),
            date(2026, 8, 31),
            date(2026, 9, 5),  # Future relative to Aug 31 cutoff
        ],
        "INTERVAL_COVERAGE": [1.0, 1.0, 1.0, 1.0, 1.0],
        "FA_INTERVAL_SUM": [10.0, 10.0, 10.0, 10.0, 9999.0],
        "RA_INTERVAL_SUM": [0.0, 0.0, 0.0, 0.0, 5000.0],
        "RA_REVERSAL_RATIO": [0.0, 0.0, 0.0, 0.0, 1.0],
        "LOAD_FACTOR": [0.5, 0.5, 0.5, 0.5, 0.1],
        "VOLTAGE_IMBALANCE_MAX": [1.0, 1.0, 1.0, 1.0, 40.0],
        "CURRENT_IMBALANCE_MAX": [0.1, 0.1, 0.1, 0.1, 1.0],
    })

    win = build_uc_window_features(daily_base, cutoff_date=date(2026, 8, 31), window_days=30)
    assert win.height == 1
    assert float(win["HIST_FA_SUM_MEDIAN"][0]) == 10.0
    assert float(win["HIST_RA_REVERSAL_RATIO_MAX"][0]) == 0.0


def test_subspace_shield_excludes_communication_columns():
    """Pure electrical feature selector must block coverage, latency, alarm, and meter age."""
    cols = [
        "id__uc_id",
        "meta__cutoff_date",
        "x__hist_fa_sum_median",
        "x__hist_current_imbalance_max",
        "x__avg_interval_coverage",
        "x__coverage_reliable_days",
        "x__alarm_count_window",
        "x__avg_alarm_latency_sec",
        "x__meter_age_days",
        "x__meter_changed_30d",
    ]
    selected = select_pure_electrical_features(cols)
    assert selected == ["x__hist_fa_sum_median", "x__hist_current_imbalance_max"]


def test_gating_isolates_ineligible_ucs():
    """Observations below min_quality_threshold must be marked is_reliable=False and not flagged."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(10.0, 1.0, size=(100, 4))
    X_eval = np.vstack([X_train[:5], np.full((2, 4), 100.0)])
    dq = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.30, 0.90])  # Row 5 is extreme but degraded telecom

    det = ElectricalAnomalyDetector(min_quality_threshold=0.70, contamination=0.10, random_state=42)
    det.fit(X_train)
    res = det.predict_electrical(X_eval, data_quality=dq)

    assert res["is_reliable"][5] == False  # noqa: E712
    assert res["is_electrical_anomaly"][5] == False  # noqa: E712
    assert res["is_reliable"][6] == True  # noqa: E712


def test_composite_detector_reproducible_with_fixed_seed_and_batch_order():
    """Fixed seed produces identical scores regardless of row permutation."""
    rng = np.random.default_rng(123)
    X = rng.normal(0.0, 1.0, size=(120, 5))
    perm = rng.permutation(len(X))

    det1 = CompositeAnomalyDetector(random_state=42).fit(X)
    s1 = det1.score_samples(X)

    det2 = CompositeAnomalyDetector(random_state=42).fit(X)
    s2 = det2.score_samples(X[perm])

    inv_perm = np.argsort(perm)
    assert np.allclose(s1, s2[inv_perm], atol=1e-4)
