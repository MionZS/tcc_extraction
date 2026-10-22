"""Unit tests for the Voltage Disturbance and Imbalance Classifier."""

from __future__ import annotations

from pathlib import Path
import polars as pl
import pytest
from src.models.voltage_classifier import (
    ClassificationResult,
    MLVoltageDisturbanceClassifier,
    RuleBasedVoltageClassifier,
    VoltageDisturbanceType,
    extract_voltage_classifier_features,
    group_train_test_split,
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
    assert "V_SPREAD_PU" in feat_df.columns
    assert "HIST_DELTA_V_PU" in feat_df.columns
    assert feat_df.filter(pl.col("UC") == "UC001").get_column("U_L1_PU").item() == 1.0
    assert feat_df.filter(pl.col("UC") == "UC003").get_column("IS_NULL_L1").item() is True


def test_extract_voltage_classifier_structural_phases():
    # Monofásico: L2 e L3 inativos por estrutura
    inst_df = pl.DataFrame({
        "UC": ["UC_MONO", "UC_BI"],
        "U_L1": [127.0, 127.0],
        "U_L2": [None, 127.0],
        "U_L3": [None, None],
    })

    context_df = pl.DataFrame({
        "UC": ["UC_MONO", "UC_BI"],
        "FASE_CIRCUITO": ["MONOFASICO", "BIFASICO"],
        "TENSAO_BASE": [127.0, 127.0],
    })

    feat_df = extract_voltage_classifier_features(inst_df, context_df=context_df)

    mono_row = feat_df.filter(pl.col("UC") == "UC_MONO")
    assert mono_row.get_column("IS_ACTIVE_L1").item() is True
    assert mono_row.get_column("IS_ACTIVE_L2").item() is False
    assert mono_row.get_column("IS_ACTIVE_L3").item() is False
    assert mono_row.get_column("IS_NULL_L2").item() is False  # Not marked as null anomaly
    assert mono_row.get_column("VOLTAGE_IMBALANCE_RATIO").item() == 0.0

    bi_row = feat_df.filter(pl.col("UC") == "UC_BI")
    assert bi_row.get_column("IS_ACTIVE_L1").item() is True
    assert bi_row.get_column("IS_ACTIVE_L2").item() is True
    assert bi_row.get_column("IS_ACTIVE_L3").item() is False
    assert bi_row.get_column("ACTIVE_VALID_PHASES_COUNT").item() == 2


def test_rule_based_voltage_classifier_cases():
    inst_df = pl.DataFrame({
        "UC": [
            "UC_NORM",
            "UC_SAG",
            "UC_SWELL",
            "UC_MONO_FAULT",
            "UC_BI_FAULT",
            "UC_INTERRUPT",
            "UC_DESBAL",
        ],
        "FASE_CIRCUITO": [
            "TRIFASICO",
            "TRIFASICO",
            "TRIFASICO",
            "TRIFASICO",
            "TRIFASICO",
            "TRIFASICO",
            "TRIFASICO",
        ],
        "TENSAO_BASE": [127.0] * 7,
        "U_L1": [127.0, 95.0, 145.0, 0.0, 0.0, 0.0, 127.0],
        "U_L2": [127.0, 95.0, 145.0, 127.0, 0.0, 0.0, 127.0],
        "U_L3": [127.0, 95.0, 145.0, 127.0, 127.0, 0.0, 118.0],
        "NEIGHBOR_COVERAGE_RATIO": [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0],
    })

    feat_df = extract_voltage_classifier_features(instantaneous_df=inst_df)
    labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

    labels = dict(zip(labeled_df.get_column("UC").to_list(), labeled_df.get_column("BASELINE_VOLTAGE_LABEL").to_list()))

    assert labels["UC_NORM"] == VoltageDisturbanceType.NORMAL.value
    assert labels["UC_SAG"] == VoltageDisturbanceType.SUBTENSAO.value
    assert labels["UC_SWELL"] == VoltageDisturbanceType.SOBRETENSAO.value
    assert labels["UC_MONO_FAULT"] == VoltageDisturbanceType.FALTA_MONOFASICA.value
    assert labels["UC_BI_FAULT"] == VoltageDisturbanceType.FALTA_BIFASICA.value
    assert labels["UC_INTERRUPT"] == VoltageDisturbanceType.INTERRUPCAO_TOTAL.value
    assert labels["UC_DESBAL"] == VoltageDisturbanceType.DESBALANCEAMENTO_CRITICO.value


def test_group_train_test_split():
    df = pl.DataFrame({
        "UC": ["UC1", "UC1", "UC2", "UC2", "UC3", "UC3", "UC4", "UC4"],
        "VAL": [1, 2, 3, 4, 5, 6, 7, 8],
    })

    # 1. Standard canonical baseline split (100% train, test_size=0.0)
    train_full, test_full = group_train_test_split(df, group_col="UC", test_size=0.0)
    assert train_full.height == df.height
    assert test_full.height == df.height

    # 2. Disjoint UC partition split (test_size > 0.0)
    train_df, test_df = group_train_test_split(df, group_col="UC", test_size=0.25, random_state=42)

    train_ucs = set(train_df.get_column("UC").to_list())
    test_ucs = set(test_df.get_column("UC").to_list())

    # Guarantee disjoint group partition
    assert train_ucs.isdisjoint(test_ucs)
    assert len(train_ucs) + len(test_ucs) == 4
    assert train_df.height + test_df.height == 8


def test_ml_voltage_disturbance_classifier_training_evaluation_and_io(tmp_path: Path):
    # Build synthetic balanced dataset representing diverse classes across multiple independent UCs
    rows = []
    classes_to_create = [
        (127.0, 127.0, 127.0, 1.0, False),  # Normal
        (90.0, 92.0, 91.0, 1.0, False),     # Subtensão
        (145.0, 142.0, 144.0, 1.0, False),  # Sobretensão
        (0.0, 127.0, 127.0, 1.0, False),    # Falta monofásica
        (0.0, 0.0, 127.0, 1.0, False),      # Falta bifásica
        (0.0, 0.0, 0.0, 0.1, True),         # Interrupção total
        (127.0, 127.0, 118.0, 1.0, False),  # Desbalanceamento crítico (118V >= 0.90 pu, FD = 4.8%)
    ]

    for cls_idx, (l1, l2, l3, cov, alarm) in enumerate(classes_to_create):
        for uc_idx in range(6):
            uc_name = f"UC_C{cls_idx}_U{uc_idx}"
            for sample_idx in range(5):
                rows.append({
                    "UC": uc_name,
                    "NIO": f"NIO_C{cls_idx}_U{uc_idx}",
                    "FASE_CIRCUITO": "TRIFASICO",
                    "TENSAO_BASE": 127.0,
                    "U_L1": l1,
                    "U_L2": l2,
                    "U_L3": l3,
                    "NEIGHBOR_COVERAGE_RATIO": cov,
                })

    df = pl.DataFrame(rows)
    feat_df = extract_voltage_classifier_features(instantaneous_df=df)
    labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

    train_df, test_df = group_train_test_split(labeled_df, group_col="UC", test_size=0.25, random_state=42)

    clf = MLVoltageDisturbanceClassifier(random_state=42)
    clf.fit(train_df)

    assert clf.is_fitted
    assert len(clf.classes_) >= 4

    # Predict
    res = clf.predict(test_df)
    assert isinstance(res, ClassificationResult)
    assert "PREDICTED_DISTURBANCE_CLASS" in res.predictions_df.columns
    assert "DISTURBANCE_CONFIDENCE" in res.predictions_df.columns
    assert res.predictions_df.height == test_df.height

    # Evaluate
    metrics = clf.evaluate(test_df)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] >= 0.85

    # Persistence (save and load)
    model_path = tmp_path / "voltage_classifier.joblib"
    clf.save(model_path)
    assert model_path.exists()

    loaded_clf = MLVoltageDisturbanceClassifier.load(model_path)
    assert loaded_clf.is_fitted
    loaded_res = loaded_clf.predict(test_df)
    assert (
        loaded_res.predictions_df.get_column("PREDICTED_DISTURBANCE_CLASS").to_list()
        == res.predictions_df.get_column("PREDICTED_DISTURBANCE_CLASS").to_list()
    )
