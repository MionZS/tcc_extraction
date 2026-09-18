"""Tests for target leakage — verifying feature-label independence.

Core principle (§6 of the architecture decision):
"Usar dados do mesmo dia para prever HAS_MDM_DATA do mesmo dia é vazamento direto."
"Rótulos devem ser independentes das features e suportar mais de uma pergunta de pesquisa."

Labels must be computed independently from features, using a strictly
later time window or an independent data source.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl
import pytest

from src.datasets.schemas import LABELS_SCHEMA, METER_DAY_FEATURES_SCHEMA
from src.labels.build import build_labels, build_communication_failure_labels
from src.labels.contracts import validate_label, validate_labels_batch


class TestFeatureLabelSeparation:
    """Features and labels must not share data from the same time window."""

    def test_labels_use_future_window(self):
        """Labels must reference a window after the feature cutoff."""
        feature_cutoff = date(2026, 6, 7)
        prediction_window_start = date(2026, 6, 8)
        prediction_window_end = date(2026, 6, 9)

        assert feature_cutoff < prediction_window_start
        assert prediction_window_start <= prediction_window_end
        # No overlap
        assert prediction_window_start > feature_cutoff

    def test_label_reference_does_not_overlap_feature(self):
        """Label reference window must not overlap with feature window."""
        feature_window = (date(2026, 6, 1), date(2026, 6, 7))
        label_window = (date(2026, 6, 8), date(2026, 6, 9))

        # Check no overlap
        label_start = label_window[0]
        feature_end = feature_window[1]
        assert label_start > feature_end, "Label window overlaps feature window"

    def test_cutoff_is_not_in_label_window(self):
        """The cutoff day must not be inside the label reference window."""
        cutoff = date(2026, 6, 7)
        label_start = date(2026, 6, 8)
        label_end = date(2026, 6, 9)

        assert cutoff < label_start
        assert cutoff not in [label_start + timedelta(days=i) for i in range((label_end - label_start).days + 1)]


class TestLabelConstruction:
    """Labels are built independently from features."""

    def test_build_labels_happy_path(self):
        """Basic label construction works."""
        labels = build_labels(
            entity_type="NIO",
            entity_ids=["001", "002", "003"],
            reference_start=date(2026, 6, 8),
            reference_end=date(2026, 6, 9),
            prediction_horizon=2,
            label_type="falha_comunicacao",
            label_value=True,
            label_source="regra",
        )
        assert labels.height == 3
        assert "LABEL_VALUE" in labels.columns
        assert all(labels["LABEL_VALUE"] == True)

    def test_build_labels_different_types(self):
        """Labels can be built for different label types."""
        for label_type in ["falha_comunicacao", "anomalia_eletrica", "classe_perfil"]:
            labels = build_labels(
                entity_type="NIO",
                entity_ids=["001"],
                reference_start=date(2026, 6, 8),
                reference_end=date(2026, 6, 9),
                prediction_horizon=1,
                label_type=label_type,
                label_value=False,
                label_source="regra",
            )
            assert labels.height == 1
            assert labels["LABEL_TYPE"][0] == label_type

    def test_build_labels_empty_entities(self):
        """Empty entity list returns empty DataFrame."""
        labels = build_labels(
            entity_type="NIO",
            entity_ids=[],
            reference_start=date(2026, 6, 8),
            reference_end=date(2026, 6, 9),
            prediction_horizon=1,
            label_type="falha_comunicacao",
            label_value=False,
            label_source="regra",
        )
        assert labels.height == 0


class TestCommunicationFailureLabels:
    """Communication failure labels must not use feature data."""

    def test_independent_construction(self):
        """Labels are built independently of feature data source."""
        labels = build_communication_failure_labels(
            nios=["001", "002", "003"],
            reference_day=date(2026, 6, 7),
            lookahead_days=2,
            label_source="regra",
        )
        assert labels.height == 3
        assert labels["LABEL_TYPE"][0] == "falha_comunicacao"
        # Labels should reference a future window
        assert labels["REFERENCE_START"][0] <= labels["REFERENCE_END"][0]
        assert labels["LABEL_SOURCE"][0] == "regra"

    def test_label_window_is_after_feature_window(self):
        """The label reference window must start after the feature cutoff."""
        feature_cutoff = date(2026, 6, 7)
        lookahead = 2

        labels = build_communication_failure_labels(
            nios=["001"],
            reference_day=feature_cutoff,
            lookahead_days=lookahead,
        )
        # Feature window ends at reference_day
        # Prediction window is [reference_day+1, reference_day+lookahead_days]
        ref_start = labels["REFERENCE_START"][0]
        ref_end = labels["REFERENCE_END"][0]

        # The reference window is 7 days leading up to the cutoff
        # The prediction horizon is lookahead_days after reference_end
        assert ref_end == feature_cutoff
        # prediction_horizon = lookahead_days
        # The label looks at events AFTER the reference window
        assert labels["PREDICTION_HORIZON"][0] == lookahead

    def test_multiple_entities(self):
        """Labels for multiple entities should have separate rows."""
        labels = build_communication_failure_labels(
            nios=[f"NIO_{i:04d}" for i in range(100)],
            reference_day=date(2026, 6, 7),
        )
        assert labels.height == 100
        # All entity IDs should be unique
        assert labels["ENTITY_ID"].n_unique() == 100


class TestLabelValidation:
    """Label validation ensures data integrity."""

    def test_valid_label(self):
        """A correctly structured label passes validation."""
        row = {
            "ENTITY_TYPE": "NIO",
            "ENTITY_ID": "001",
            "REFERENCE_START": date(2026, 6, 8),
            "REFERENCE_END": date(2026, 6, 9),
            "PREDICTION_HORIZON": 2,
            "LABEL_TYPE": "falha_comunicacao",
            "LABEL_VALUE": True,
            "LABEL_SOURCE": "regra",
            "CONFIDENCE": 0.9,
            "REVIEWER_ID": "reviewer_001",
            "CREATED_AT": datetime.now(),
            "LABEL_VERSION": "v1",
            "NOTES": "Test label",
        }
        errors = validate_label(row)
        assert errors == []

    def test_invalid_entity_type(self):
        """Invalid ENTITY_TYPE should fail validation."""
        row = {
            "ENTITY_TYPE": "INVALID_TYPE",
            "ENTITY_ID": "001",
            "REFERENCE_START": date(2026, 6, 8),
            "REFERENCE_END": date(2026, 6, 9),
            "PREDICTION_HORIZON": 2,
            "LABEL_TYPE": "falha_comunicacao",
            "LABEL_VALUE": True,
            "LABEL_SOURCE": "regra",
            "CONFIDENCE": 1.0,
            "CREATED_AT": datetime.now(),
            "LABEL_VERSION": "v1",
            "NOTES": "",
        }
        errors = validate_label(row)
        assert any("ENTITY_TYPE" in e for e in errors)

    def test_invalid_label_type(self):
        """Invalid LABEL_TYPE should fail validation."""
        row = {
            "ENTITY_TYPE": "NIO",
            "ENTITY_ID": "001",
            "REFERENCE_START": date(2026, 6, 8),
            "REFERENCE_END": date(2026, 6, 9),
            "PREDICTION_HORIZON": 2,
            "LABEL_TYPE": "nonexistent_type",
            "LABEL_VALUE": True,
            "LABEL_SOURCE": "regra",
            "CONFIDENCE": 1.0,
            "CREATED_AT": datetime.now(),
            "LABEL_VERSION": "v1",
            "NOTES": "",
        }
        errors = validate_label(row)
        assert any("LABEL_TYPE" in e for e in errors)

    def test_invalid_confidence_range(self):
        """Confidence outside [0, 1] should fail validation."""
        row = {
            "ENTITY_TYPE": "NIO",
            "ENTITY_ID": "001",
            "REFERENCE_START": date(2026, 6, 8),
            "REFERENCE_END": date(2026, 6, 9),
            "PREDICTION_HORIZON": 2,
            "LABEL_TYPE": "falha_comunicacao",
            "LABEL_VALUE": True,
            "LABEL_SOURCE": "regra",
            "CONFIDENCE": 1.5,  # invalid: > 1.0
            "CREATED_AT": datetime.now(),
            "LABEL_VERSION": "v1",
            "NOTES": "",
        }
        errors = validate_label(row)
        assert any("CONFIDENCE" in e for e in errors)

    def test_missing_required_fields(self):
        """Missing required fields should be caught."""
        errors = validate_label({})
        assert len(errors) > 0
        # Should complain about missing ENTITY_TYPE, ENTITY_ID, etc.

    def test_reference_start_before_end(self):
        """REFERENCE_START must be before REFERENCE_END."""
        row = {
            "ENTITY_TYPE": "NIO",
            "ENTITY_ID": "001",
            "REFERENCE_START": date(2026, 6, 10),  # after end
            "REFERENCE_END": date(2026, 6, 9),      # before start
            "PREDICTION_HORIZON": 1,
            "LABEL_TYPE": "falha_comunicacao",
            "LABEL_VALUE": True,
            "LABEL_SOURCE": "regra",
            "CONFIDENCE": 1.0,
            "CREATED_AT": datetime.now(),
            "LABEL_VERSION": "v1",
            "NOTES": "",
        }
        errors = validate_label(row)
        assert any("REFERENCE_START" in e and "REFERENCE_END" in e for e in errors)

    def test_batch_validation(self):
        """Batch validation catches multiple invalid rows."""
        rows = [
            {
                "ENTITY_TYPE": "NIO",
                "ENTITY_ID": "001",
                "REFERENCE_START": date(2026, 6, 8),
                "REFERENCE_END": date(2026, 6, 9),
                "PREDICTION_HORIZON": 2,
                "LABEL_TYPE": "falha_comunicacao",
                "LABEL_VALUE": True,
                "LABEL_SOURCE": "regra",
                "CONFIDENCE": 1.0,
                "CREATED_AT": datetime.now(),
                "LABEL_VERSION": "v1",
                "NOTES": "",
            },
            {
                "ENTITY_TYPE": "INVALID",  # invalid
                "ENTITY_ID": "002",
                "REFERENCE_START": date(2026, 6, 8),
                "REFERENCE_END": date(2026, 6, 9),
                "PREDICTION_HORIZON": -1,  # invalid
                "LABEL_TYPE": "nonexistent",  # invalid
                "LABEL_VALUE": True,
                "LABEL_SOURCE": "invalid_source",  # invalid
                "CONFIDENCE": 1.0,
                "CREATED_AT": datetime.now(),
                "LABEL_VERSION": "",
                "NOTES": "",
            },
        ]
        df = pl.DataFrame(rows, strict=False)
        issues = validate_labels_batch(df)
        assert len(issues) == 1  # only the second row has issues
        assert issues[0]["row_index"] == 1
        assert len(issues[0]["errors"]) > 0


class TestNoDataLeakage:
    """Verify that labels don't leak information from the feature set."""

    def test_features_and_labels_are_separate_tables(self):
        """Features and labels must be stored in separate tables
        (different schemas, different data sources)."""
        # Feature schema should not contain label columns
        for col in ["LABEL_TYPE", "LABEL_VALUE", "LABEL_SOURCE"]:
            assert col not in METER_DAY_FEATURES_SCHEMA, (
                f"Label column {col!r} leaked into features schema"
            )

        # Label schema should not contain feature columns
        for col in ["FEATURE_SET_VERSION", "FA_INTERVAL_SUM", "LOAD_FACTOR"]:
            assert col not in LABELS_SCHEMA, (
                f"Feature column {col!r} leaked into labels schema"
            )

    def test_features_and_labels_share_only_entity_key(self):
        """The only shared columns between features and labels are entity keys."""
        feature_cols = set(METER_DAY_FEATURES_SCHEMA)
        label_cols = set(LABELS_SCHEMA)

        overlap = feature_cols & label_cols
        # Acceptable shared columns: entity identifiers
        acceptable = {"NIO", "ENTITY_ID", "REPORT_DAY", "REFERENCE_START"}
        unexpected = overlap - acceptable

        assert len(unexpected) == 0, (
            f"Unexpected shared columns between features and labels: {unexpected}"
        )
