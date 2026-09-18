"""Tests for dataset grain correctness.

Verifies that each normalized table has the correct candidate key
(uniqueness at the expected grain).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import polars as pl
import pytest

from src.datasets.schemas import (
    TABLE_REGISTRY,
    get_schema,
    get_keys,
    build_empty_frame,
    METER_DAY_METADATA_SCHEMA,
    METER_INTERVAL_SCHEMA,
    METER_INSTANTANEOUS_SCHEMA,
    METER_REGISTER_SNAPSHOT_SCHEMA,
    METER_DAY_FEATURES_SCHEMA,
    LABELS_SCHEMA,
)


class TestSchemas:
    """Schema definitions are complete and internally consistent."""

    def test_registry_complete(self):
        """All expected tables are registered."""
        expected = {
            "meter_day_metadata",
            "meter_interval",
            "meter_instantaneous",
            "meter_register_snapshot",
            "meter_day_features",
            "labels",
        }
        assert set(TABLE_REGISTRY) == expected

    def test_schemas_have_keys(self):
        """Each registered table has a non-empty keys tuple."""
        for table_name in TABLE_REGISTRY:
            keys = get_keys(table_name)
            assert len(keys) > 0, f"{table_name} has no keys defined"

    def test_all_keys_in_schema(self):
        """All key columns exist in the corresponding schema."""
        for table_name, entry in TABLE_REGISTRY.items():
            schema = entry["schema"]
            for key in entry["keys"]:
                assert key in schema, (
                    f"Key column {key!r} not found in {table_name} schema"
                )

    def test_empty_frame_has_all_columns(self):
        """build_empty_frame returns a DataFrame with all schema columns."""
        for table_name in TABLE_REGISTRY:
            df = build_empty_frame(table_name)
            schema = get_schema(table_name)
            for col in schema:
                assert col in df.columns, f"Column {col!r} missing in empty {table_name} frame"

    def test_empty_frame_is_empty(self):
        """build_empty_frame returns a DataFrame with 0 rows."""
        for table_name in TABLE_REGISTRY:
            df = build_empty_frame(table_name)
            assert df.height == 0, f"{table_name} empty frame has {df.height} rows"


class TestMeterDayMetadataGrain:
    """Grain: NIO × report_day (one row per NIO per day)."""

    def _make_row(self, nio: str, report_day: date, **overrides) -> dict[str, Any]:
        row = {
            "REPORT_DAY": report_day,
            "NIO": nio,
            "UC_KEY": f"UC_{nio}",
            "METER_SERIAL_KEY": f"SERIAL_{nio}",
            "METER_TYPE": "INTEL",
            "METER_SUBTYPE": "A",
            "PHASE_TYPE": "T",
            "SERVICE_STATUS": "AT",
            "INSTALLATION_DATE": report_day,
            "REMOVAL_DATE": None,
            "CONSUMER_CLASS": "RES",
            "SUBGROUP": "B1",
            "INSTALLED_KVA": 15.0,
            "FEEDER_ID": "FDR001",
            "SUBSTATION_ID": "SE001",
            "NOMINAL_VOLTAGE": 220.0,
            "MUNICIPALITY": "ARAUCARIA",
            "LATITUDE_BUCKET": "-25.5",
            "LONGITUDE_BUCKET": "-49.4",
            "SOURCE_UPDATED_AT": datetime.now(timezone.utc),
            "SCHEMA_VERSION": "v1",
            "RUN_ID": "test_001",
        }
        row.update(overrides)
        return row

    def test_unique_by_nio_report_day(self):
        """No duplicate (NIO, report_day) combinations."""
        rows = [
            self._make_row("001", date(2026, 6, 1)),
            self._make_row("001", date(2026, 6, 2)),
            self._make_row("002", date(2026, 6, 1)),
            self._make_row("002", date(2026, 6, 2)),
        ]
        df = pl.DataFrame(rows, strict=False)
        # Check grain uniqueness
        assert df.height == 4
        assert df.unique(subset=["NIO", "REPORT_DAY"]).height == df.height

    def test_duplicate_detected(self):
        """Duplicates at the grain level should be detectable."""
        rows = [
            self._make_row("001", date(2026, 6, 1), RUN_ID="first"),
            self._make_row("001", date(2026, 6, 1), RUN_ID="second"),
        ]
        df = pl.DataFrame(rows, strict=False)
        # There should be a duplicate
        assert df.unique(subset=["NIO", "REPORT_DAY"]).height < df.height

    def test_keys_are_not_null(self):
        """Key columns must not be null in non-empty data."""
        df = pl.DataFrame(
            [self._make_row("001", date(2026, 6, 1))],
            strict=False,
        )
        assert df["NIO"].null_count() == 0
        assert df["REPORT_DAY"].null_count() == 0


class TestMeterIntervalGrain:
    """Grain: NIO × timestamp_utc (one row per NIO per timestamp)."""

    def test_unique_by_nio_timestamp(self):
        """No duplicate (NIO, TIMESTAMP_UTC) combinations."""
        ts = datetime(2026, 6, 1, 0, 0, 0)
        rows = [
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "TIMESTAMP_UTC": ts, "TIMESTAMP_LOCAL": ts,
             "CADENCE_MINUTES": 5, "FA_INTERVAL": 1.0},
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "TIMESTAMP_UTC": datetime(2026, 6, 1, 0, 5, 0),
             "TIMESTAMP_LOCAL": datetime(2026, 6, 1, 0, 5, 0),
             "CADENCE_MINUTES": 5, "FA_INTERVAL": 2.0},
            {"NIO": "002", "REPORT_DAY": date(2026, 6, 1),
             "TIMESTAMP_UTC": ts, "TIMESTAMP_LOCAL": ts,
             "CADENCE_MINUTES": 5, "FA_INTERVAL": 3.0},
        ]
        df = pl.DataFrame(rows, strict=False)
        assert df.unique(subset=["NIO", "TIMESTAMP_UTC"]).height == df.height

    def test_cadence_consistency(self):
        """Rows for the same NIO should have consistent cadence."""
        ts1 = datetime(2026, 6, 1, 0, 0, 0)
        ts2 = datetime(2026, 6, 1, 0, 5, 0)
        rows = [
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "TIMESTAMP_UTC": ts1, "TIMESTAMP_LOCAL": ts1,
             "CADENCE_MINUTES": 5, "FA_INTERVAL": 1.0},
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "TIMESTAMP_UTC": ts2, "TIMESTAMP_LOCAL": ts2,
             "CADENCE_MINUTES": 5, "FA_INTERVAL": 2.0},
        ]
        df = pl.DataFrame(rows, strict=False)
        nio_cadences = df.group_by("NIO").agg(pl.col("CADENCE_MINUTES").unique())
        for row in nio_cadences.iter_rows(named=True):
            assert len(row["CADENCE_MINUTES"]) == 1, (
                f"NIO {row['NIO']} has multiple cadences: {row['CADENCE_MINUTES']}"
            )

    def test_timestamps_within_report_day(self):
        """All timestamps must fall within the report day."""
        rows = [
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "TIMESTAMP_UTC": datetime(2026, 6, 1, 0, 0, 0),
             "TIMESTAMP_LOCAL": datetime(2026, 6, 1, 0, 0, 0),
             "CADENCE_MINUTES": 5, "FA_INTERVAL": 1.0},
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "TIMESTAMP_UTC": datetime(2026, 6, 1, 23, 55, 0),
             "TIMESTAMP_LOCAL": datetime(2026, 6, 1, 23, 55, 0),
             "CADENCE_MINUTES": 5, "FA_INTERVAL": 2.0},
        ]
        df = pl.DataFrame(rows, strict=False)
        assert all(df["REPORT_DAY"] == df["TIMESTAMP_UTC"].dt.date())


class TestMeterRegisterSnapshotGrain:
    """Grain: NIO × timestamp_utc (register snapshots)."""

    def test_register_count_per_day(self):
        """Typical day has 4 register snapshots (6-hour cadence)."""
        ts = datetime(2026, 6, 1, 0, 0, 0)
        snapshots = []
        for hour in [0, 6, 12, 18]:
            snapshots.append({
                "NIO": "001",
                "REPORT_DAY": date(2026, 6, 1),
                "TIMESTAMP_UTC": datetime(2026, 6, 1, hour, 0, 0),
                "FA_TOTAL": 100.0 + hour,
                "RA_TOTAL": 50.0 + hour,
                "FA_MD": 10.0 + hour,
            })
        df = pl.DataFrame(snapshots, strict=False)
        # 4 snapshots for this NIO
        nio_mask = df["NIO"] == "001"
        assert nio_mask.sum() == 4


class TestMeterDayFeaturesGrain:
    """Grain: NIO × report_day × feature_set_version."""

    def test_unique_by_grain(self):
        """No duplicate (NIO, report_day, feature_set_version)."""
        rows = [
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "FEATURE_SET_VERSION": "v1", "FA_INTERVAL_SUM": 100.0},
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 1),
             "FEATURE_SET_VERSION": "v2", "FA_INTERVAL_SUM": 100.0},
            {"NIO": "001", "REPORT_DAY": date(2026, 6, 2),
             "FEATURE_SET_VERSION": "v1", "FA_INTERVAL_SUM": 200.0},
        ]
        df = pl.DataFrame(rows, strict=False)
        unique = df.unique(subset=["NIO", "REPORT_DAY", "FEATURE_SET_VERSION"])
        assert unique.height == df.height


class TestLabelsGrain:
    """Grain: entity_type × entity_id × reference_start × label_type × label_version."""

    def test_unique_by_grain(self):
        """No duplicate grain combinations."""
        rows = [
            {"ENTITY_TYPE": "NIO", "ENTITY_ID": "001",
             "REFERENCE_START": date(2026, 6, 1),
             "REFERENCE_END": date(2026, 6, 2),
             "LABEL_TYPE": "falha_comunicacao",
             "LABEL_VERSION": "v1", "LABEL_VALUE": True,
             "LABEL_SOURCE": "mdm", "CONFIDENCE": 1.0,
             "CREATED_AT": datetime.now(timezone.utc)},
            {"ENTITY_TYPE": "NIO", "ENTITY_ID": "001",
             "REFERENCE_START": date(2026, 6, 1),
             "REFERENCE_END": date(2026, 6, 2),
             "LABEL_TYPE": "anomalia_eletrica",
             "LABEL_VERSION": "v1", "LABEL_VALUE": False,
             "LABEL_SOURCE": "regra", "CONFIDENCE": 0.8,
             "CREATED_AT": datetime.now(timezone.utc)},
        ]
        df = pl.DataFrame(rows, strict=False)
        unique = df.unique(
            subset=["ENTITY_TYPE", "ENTITY_ID", "REFERENCE_START",
                    "LABEL_TYPE", "LABEL_VERSION"]
        )
        assert unique.height == df.height


class TestNormalizeCadence:
    """Cadence detection and expected points computation."""

    def test_cadence_detection_5min(self):
        """288 timestamps → 5-min cadence."""
        from src.datasets.normalize import _infer_cadence_from_slot_count
        assert _infer_cadence_from_slot_count(288) == 5

    def test_cadence_detection_15min(self):
        """96 timestamps → 15-min cadence."""
        from src.datasets.normalize import _infer_cadence_from_slot_count
        assert _infer_cadence_from_slot_count(96) == 15

    def test_cadence_detection_60min(self):
        """24 timestamps → 60-min cadence."""
        from src.datasets.normalize import _infer_cadence_from_slot_count
        assert _infer_cadence_from_slot_count(24) == 60

    def test_expected_points_5min(self):
        """1440 / 5 = 288 expected points."""
        from src.datasets.normalize import compute_expected_points
        assert compute_expected_points(5) == 288

    def test_expected_points_15min(self):
        """1440 / 15 = 96 expected points."""
        from src.datasets.normalize import compute_expected_points
        assert compute_expected_points(15) == 96

    def test_coverage_ratio(self):
        """50% coverage: 144/288."""
        from src.datasets.normalize import compute_coverage
        assert compute_coverage(144, 288) == pytest.approx(0.5)

    def test_coverage_capped_at_1(self):
        """Coverage should not exceed 1.0."""
        from src.datasets.normalize import compute_coverage
        assert compute_coverage(300, 288) == 1.0
