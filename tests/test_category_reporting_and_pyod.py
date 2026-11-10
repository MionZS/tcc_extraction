"""Unit tests for PyOD-grade Pure Electrical Anomaly Detector, Isolation Rules & Per-Category Reports."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.models.category_reporting import (
    apply_delimited_isolation_and_categories,
    assign_behavioral_consumption_classes,
    build_temporal_class_transition_report,
    export_all_category_reports,
)
from src.models.composite import (
    COPODDetector,
    CompositeAnomalyDetector,
    HBOSDetector,
    select_pure_electrical_features,
)


def test_select_pure_electrical_features_excludes_telecom_and_lifecycle() -> None:
    """Verify that communication coverage, alarms, and meter age are strictly excluded from electrical X."""
    df = pl.DataFrame({
        "id__uc_id": ["100", "101"],
        "x__hist_fa_sum_median": [5.0, 6.0],
        "x__hist_fa_sum_mad": [0.8, 0.9],
        "x__hist_current_imbalance_max": [0.05, 0.45],
        "x__avg_interval_coverage": [0.98, 0.35],
        "x__coverage_reliable_days": [29, 8],
        "x__meter_age_days": [400, 12],
        "x__meter_changed_30d": [0, 1],
        "x__alarm_count_window": [0, 5],
    })
    elec_cols = select_pure_electrical_features(df)
    assert "x__hist_fa_sum_median" in elec_cols
    assert "x__hist_fa_sum_mad" in elec_cols
    assert "x__hist_current_imbalance_max" in elec_cols
    assert "x__avg_interval_coverage" not in elec_cols
    assert "x__coverage_reliable_days" not in elec_cols
    assert "x__meter_age_days" not in elec_cols
    assert "x__meter_changed_30d" not in elec_cols
    assert "x__alarm_count_window" not in elec_cols


def test_pyod_6_detector_composite_scores_extreme_electrical_outliers() -> None:
    """Verify COPOD, HBOS, ECOD, LOF, IsoForest, and PCA ensemble scores electrical outliers highest."""
    rng = np.random.default_rng(42)
    X_normal = rng.normal(loc=5.0, scale=0.5, size=(120, 5))
    X_outliers = np.array([
        [0.01, 0.0, 0.95, -4.5, 0.85],
        [45.0, 12.0, 0.88, 8.0, 0.92],
    ])
    X = np.vstack([X_normal, X_outliers])

    det = CompositeAnomalyDetector(contamination=0.05, random_state=42)
    det.fit(X_normal)
    preds = det.predict_composite(X)

    assert "copod_score" in preds
    assert "hbos_score" in preds
    assert "ecod_score" in preds
    assert preds["consensus_score"][-1] > float(np.median(preds["consensus_score"][:120]))
    assert preds["consensus_score"][-2] > float(np.median(preds["consensus_score"][:120]))


def test_delimited_isolation_rules_and_temporal_class_transitions(tmp_path: Path) -> None:
    """Verify New UC vs. Replaced Meter, Isolated vs. Collective Telecom, and Class Transitions."""
    df = pl.DataFrame({
        "id__uc_id": [
            "UC_NEW_1",       # 1st meter, 10 days old -> INICIACAO_PERFIL_UC_NOVA
            "UC_SWAP_2",      # 2nd meter, 12 days old -> TRANSICAO_TROCA_MEDIDOR
            "UC_COMM_ISO",    # Poor coverage (0.30) on healthy transformer (peers ~0.95) -> ANOMALIA_COMUNICACAO_ISOLADA
            "UC_PEER_OK1",    # Healthy neighbor on TRAFO_A (coverage 0.96)
            "UC_PEER_OK2",    # Healthy neighbor on TRAFO_A (coverage 0.95)
            "UC_MESH_BAD1",   # Poor coverage (0.25) on degraded transformer TRAFO_B -> GARGALO_COLETIVO_RF_MESH
            "UC_MESH_BAD2",   # Poor coverage (0.30) on degraded transformer TRAFO_B
            "UC_MESH_BAD3",   # Poor coverage (0.28) on degraded transformer TRAFO_B
            "UC_SHIFT",       #Cutoff 1: C4 (15 kWh/d)
            "UC_SHIFT",       #Cutoff 2: C1 (0.2 kWh/d) -> Mudança de Classe!
        ],
        "meta__cutoff_date": [
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 8, 31),
            date(2026, 9, 30),
        ],
        "meta__posto": [
            "TRAFO_A", "TRAFO_A", "TRAFO_A", "TRAFO_A", "TRAFO_A",
            "TRAFO_B", "TRAFO_B", "TRAFO_B", "TRAFO_A", "TRAFO_A",
        ],
        "meta__feeder": ["ALIM_1"] * 10,
        "meta__is_clean_for_train": [True, True, False, True, True, False, False, False, True, True],
        "meta__n_meters_historical": [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        "x__meter_age_days": [10, 12, 300, 300, 300, 300, 300, 300, 300, 330],
        "x__meter_changed_30d": [True, True, False, False, False, False, False, False, False, False],
        "x__avg_interval_coverage": [0.85, 0.88, 0.30, 0.96, 0.95, 0.25, 0.30, 0.28, 0.98, 0.97],
        "x__coverage_reliable_days": [10, 12, 5, 29, 28, 4, 5, 4, 29, 29],
        "x__hist_fa_sum_median": [4.5, 6.0, 3.0, 5.0, 5.5, 4.0, 4.2, 3.8, 15.0, 0.20],
        "x__hist_fa_sum_mad": [0.5, 0.8, 0.4, 0.5, 0.6, 0.4, 0.4, 0.4, 1.2, 0.05],
        "x__hist_ra_reversal_ratio_max": [0.0] * 10,
        "x__hist_ra_reversal_days": [0] * 10,
        "x__hist_current_imbalance_max": [0.05] * 10,
        "x__trend_fa_slope": [0.0] * 9 + [-0.8],
        "ANOMALY_SCORE": [0.2] * 10,
        "IS_ANOMALY": [False] * 10,
    })

    classified_df, transitions_df = apply_delimited_isolation_and_categories(df, base_dir=tmp_path)
    by_uc = {
        r["id__uc_id"]: r["CATEGORIA_DIAGNOSTICA"]
        for r in classified_df.filter(pl.col("meta__cutoff_date") == date(2026, 9, 30)).to_dicts()
    }

    assert by_uc["UC_NEW_1"] == "INICIACAO_PERFIL_UC_NOVA"
    assert by_uc["UC_SWAP_2"] == "TRANSICAO_TROCA_MEDIDOR"
    assert by_uc["UC_COMM_ISO"] == "ANOMALIA_COMUNICACAO_ISOLADA"
    assert by_uc["UC_MESH_BAD1"] == "GARGALO_COLETIVO_RF_MESH"
    assert by_uc["UC_SHIFT"] == "MUDANCA_CLASSE_COMPORTAMENTAL"

    assert transitions_df.height == 1
    tr_row = transitions_df.to_dicts()[0]
    assert tr_row["id__uc_id"] == "UC_SHIFT"
    assert tr_row["CLASSE_INICIAL"] == "C4_COMERCIAL_OU_ELEVADO"
    assert tr_row["CLASSE_FINAL"] == "C1_MINIMO_OU_DESOCUPADO"

    manifest = export_all_category_reports(classified_df, transitions_df, reports_dir=tmp_path / "reports")
    assert len(manifest) == 8
    for item in manifest:
        assert Path(item["parquet_path"]).exists()

