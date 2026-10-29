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


def test_voltage_classifier_empty_dataframe():
    """Verify graceful handling of empty DataFrames."""
    empty_df = pl.DataFrame()
    feat_df = extract_voltage_classifier_features(empty_df)
    assert feat_df.is_empty()

    classified_df = RuleBasedVoltageClassifier.classify(empty_df)
    assert classified_df.is_empty()


def test_extract_voltage_classifier_filters_all_null_voltages():
    """Verify that records where all voltage channels are null are treated as telemetry absence."""
    inst_df = pl.DataFrame({
        "UC": ["UC_VALID", "UC_NO_TELEMETRY"],
        "NIO": ["NIO_1", "NIO_2"],
        "U_L1": [127.0, None],
        "U_L2": [127.0, None],
        "U_L3": [127.0, None],
    })

    feat_df = extract_voltage_classifier_features(inst_df)
    # The record with all null voltages must be filtered out
    assert feat_df.height == 1
    assert feat_df.get_column("UC").to_list() == ["UC_VALID"]


def test_group_train_test_split_single_group_fallback():
    """Verify group split fallback when all rows belong to the same group."""
    df = pl.DataFrame({
        "UC": ["UC_ONLY"] * 10,
        "VAL": list(range(10)),
    })
    train_df, test_df = group_train_test_split(df, group_col="UC", test_size=0.3, random_state=42)
    assert train_df.height > 0
    assert test_df.height > 0
    assert train_df.height + test_df.height == 10


def test_train_anomaly_model_voltage_mode_routing():
    """Verify that scripts/train_anomaly_model.py supports '--mode voltage'."""
    from scripts.train_anomaly_model import parse_args
    import sys

    orig_argv = sys.argv
    try:
        sys.argv = ["train_anomaly_model.py", "--mode", "voltage"]
        args = parse_args()
        assert args.mode == "voltage"
    finally:
        sys.argv = orig_argv


def test_ml_voltage_classifier_deterministic_alignment():
    """Verify that align_with_rules=True achieves 100% agreement with the deterministic rule."""
    rows = []
    # Mix of distinct classes
    classes = [
        (127.0, 127.0, 127.0),  # Normal
        (92.0, 95.0, 94.0),     # Subtensão
        (144.0, 143.0, 142.0),  # Sobretensão
        (0.0, 127.0, 127.0),    # Falta monofásica
        (127.0, 127.0, 118.0),  # Desbalanceamento
    ]
    for cls_idx, (l1, l2, l3) in enumerate(classes):
        for uc_idx in range(5):
            for sample_idx in range(4):
                rows.append({
                    "UC": f"UC_{cls_idx}_{uc_idx}",
                    "NIO": f"NIO_{cls_idx}_{uc_idx}",
                    "FASE_CIRCUITO": "TRIFASICO",
                    "TENSAO_BASE": 127.0,
                    "U_L1": l1,
                    "U_L2": l2,
                    "U_L3": l3,
                    "NEIGHBOR_COVERAGE_RATIO": 1.0,
                })

    df = pl.DataFrame(rows)
    feat_df = extract_voltage_classifier_features(instantaneous_df=df)
    labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

    # Train model
    clf = MLVoltageDisturbanceClassifier(random_state=42, align_with_rules=True)
    clf.fit(labeled_df)

    # Predict with alignment enabled
    res = clf.predict(labeled_df, align_with_rules=True)
    preds = res.predictions_df.get_column("PREDICTED_DISTURBANCE_CLASS").to_list()
    ground_truth = labeled_df.get_column("BASELINE_VOLTAGE_LABEL").to_list()

    # Must be 100% identical!
    assert preds == ground_truth
    assert "RAW_ML_PREDICTION" in res.predictions_df.columns
    assert "IS_ML_DISCREPANCY" in res.predictions_df.columns
    assert "DISTURBANCE_CONFIDENCE" in res.predictions_df.columns


def test_diagnose_discrepancy_row_noise_sources():
    """Verify forensic diagnosis of root causes across noise source categories."""
    from src.models.voltage_classifier import NoiseSourceCategory, diagnose_discrepancy_row

    # 1. Borderline Subtensão
    row_sag = {
        "BASELINE_VOLTAGE_LABEL": "NORMAL",
        "RAW_ML_PREDICTION": "SUBTENSAO",
        "V_MIN_PU": 0.903,
        "V_MAX_PU": 1.01,
        "VOLTAGE_IMBALANCE_RATIO": 0.01,
        "DISTURBANCE_CONFIDENCE": 0.75,
    }
    src, dist, expl = diagnose_discrepancy_row(row_sag)
    assert src == NoiseSourceCategory.BORDERLINE_SUBTENSAO.value
    assert dist == 0.003
    assert "fronteira difusa" in expl

    # 2. Borderline Sobretensão
    row_swell = {
        "BASELINE_VOLTAGE_LABEL": "SOBRETENSAO",
        "RAW_ML_PREDICTION": "NORMAL",
        "V_MIN_PU": 0.98,
        "V_MAX_PU": 1.052,
        "VOLTAGE_IMBALANCE_RATIO": 0.01,
        "DISTURBANCE_CONFIDENCE": 0.80,
    }
    src, dist, expl = diagnose_discrepancy_row(row_swell)
    assert src == NoiseSourceCategory.BORDERLINE_SOBRETENSAO.value
    assert dist == 0.002

    # 3. Borderline Desbalanceamento
    row_imb = {
        "BASELINE_VOLTAGE_LABEL": "DESBALANCEAMENTO_CRITICO",
        "RAW_ML_PREDICTION": "NORMAL",
        "V_MIN_PU": 0.95,
        "V_MAX_PU": 1.01,
        "VOLTAGE_IMBALANCE_RATIO": 0.029,
        "DISTURBANCE_CONFIDENCE": 0.70,
    }
    src, dist, expl = diagnose_discrepancy_row(row_imb)
    assert src == NoiseSourceCategory.BORDERLINE_DESBALANCEAMENTO.value

    # 4. Zero Voltage Cutoff
    row_cutoff = {
        "BASELINE_VOLTAGE_LABEL": "FALTA_MONOFASICA",
        "RAW_ML_PREDICTION": "SUBTENSAO",
        "U_L1": 4.5,
        "U_L2": 127.0,
        "U_L3": 127.0,
        "DISTURBANCE_CONFIDENCE": 0.65,
    }
    src, dist, expl = diagnose_discrepancy_row(row_cutoff)
    assert src == NoiseSourceCategory.ZERO_VOLTAGE_CUTOFF.value

    # 5. Precedence Conflict
    row_conflict = {
        "BASELINE_VOLTAGE_LABEL": "SUBTENSAO",
        "RAW_ML_PREDICTION": "DESBALANCEAMENTO_CRITICO",
        "V_MIN_PU": 0.85,  # Subtensão
        "V_MAX_PU": 1.00,
        "VOLTAGE_IMBALANCE_RATIO": 0.05,  # Desbalanceamento crítico também presente
        "DISTURBANCE_CONFIDENCE": 0.75,
    }
    src, dist, expl = diagnose_discrepancy_row(row_conflict)
    assert src == NoiseSourceCategory.PRECEDENCE_CONFLICT.value

    # 6. Probability Uncertainty
    row_uncertain = {
        "BASELINE_VOLTAGE_LABEL": "NORMAL",
        "RAW_ML_PREDICTION": "SUBTENSAO",
        "V_MIN_PU": 0.96,
        "V_MAX_PU": 1.01,
        "VOLTAGE_IMBALANCE_RATIO": 0.005,
        "DISTURBANCE_CONFIDENCE": 0.48,  # Alta incerteza
    }
    src, dist, expl = diagnose_discrepancy_row(row_uncertain)
    assert src == NoiseSourceCategory.PROBABILITY_UNCERTAINTY.value


def test_voltage_discrepancy_auditor_csv_and_report_export(tmp_path: Path):
    """Verify that VoltageDiscrepancyAuditor audits mismatches and exports valid CSVs and reports."""
    from src.models.voltage_classifier import VoltageDiscrepancyAuditor

    # Create synthetic prediction DataFrame with intentional discrepancies
    df = pl.DataFrame({
        "UC": ["UC_01", "UC_02", "UC_03", "UC_04"],
        "NIO": ["NIO_01", "NIO_02", "NIO_03", "NIO_04"],
        "BASELINE_VOLTAGE_LABEL": ["NORMAL", "NORMAL", "SUBTENSAO", "DESBALANCEAMENTO_CRITICO"],
        "RAW_ML_PREDICTION": ["NORMAL", "SUBTENSAO", "NORMAL", "DESBALANCEAMENTO_CRITICO"],
        "V_MIN_PU": [1.0, 0.902, 0.898, 0.95],
        "V_MAX_PU": [1.0, 1.02, 0.99, 1.01],
        "V_AVG_PU": [1.0, 0.96, 0.94, 0.98],
        "VOLTAGE_IMBALANCE_RATIO": [0.005, 0.010, 0.012, 0.040],
        "DISTURBANCE_CONFIDENCE": [0.95, 0.72, 0.78, 0.92],
        "FASE_CIRCUITO": ["TRIFASICO"] * 4,
        "TENSAO_BASE": [127.0] * 4,
    })

    auditor = VoltageDiscrepancyAuditor(output_dir=tmp_path)
    report = auditor.audit(df, target_column="BASELINE_VOLTAGE_LABEL")

    assert report.total_samples == 4
    assert report.matched_samples == 2
    assert report.discrepancy_count == 2
    assert report.match_rate == 0.50

    # Verify CSV export
    assert report.export_csv_path is not None
    assert report.export_csv_path.exists()
    latest_csv = tmp_path / "voltage_classification_errors_latest.csv"
    assert latest_csv.exists()

    # Read exported CSV
    exported_csv_df = pl.read_csv(latest_csv, separator=";")
    assert exported_csv_df.height == 2
    assert "NOISE_SOURCE" in exported_csv_df.columns
    assert "DIAGNOSTIC_EXPLANATION" in exported_csv_df.columns
    assert "THRESHOLD_DISTANCE_PU" in exported_csv_df.columns
    assert "BASELINE_RULE_LABEL" in exported_csv_df.columns
    assert "ML_PREDICTED_LABEL" in exported_csv_df.columns

    # Verify markdown report export
    assert report.export_report_path is not None
    assert report.export_report_path.exists()
    report_text = report.export_report_path.read_text(encoding="utf-8")
    assert "BORDERLINE_SUBTENSAO" in report_text
    assert "Resumo Executivo" in report_text


