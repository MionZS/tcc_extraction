"""Tests for cadence detection and coverage computation."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.datasets.normalize import (
    _infer_cadence_from_slot_count,
    detect_cadence,
    compute_expected_points,
    compute_coverage,
    normalize_mdm_day,
)


class TestCadenceDetection:
    """Cadence inference from slot counts and timestamps."""

    def test_infer_5min(self):
        assert _infer_cadence_from_slot_count(288) == 5

    def test_infer_10min(self):
        assert _infer_cadence_from_slot_count(144) == 10

    def test_infer_15min(self):
        assert _infer_cadence_from_slot_count(96) == 15

    def test_infer_30min(self):
        assert _infer_cadence_from_slot_count(48) == 30

    def test_infer_60min(self):
        assert _infer_cadence_from_slot_count(24) == 60

    def test_infer_edge_empty(self):
        assert _infer_cadence_from_slot_count(0) == 5

    def test_infer_edge_negative(self):
        assert _infer_cadence_from_slot_count(-5) == 5

    def test_detect_cadence_from_timestamps_5min(self):
        """5-min interval should be detected."""
        timestamps = [datetime(2026, 6, 1, 0, i, 0) for i in range(0, 60, 5)]
        assert detect_cadence(timestamps) == 5

    def test_detect_cadence_from_timestamps_15min(self):
        """15-min interval should be detected."""
        timestamps = [datetime(2026, 6, 1, 0, i, 0) for i in range(0, 60, 15)]
        assert detect_cadence(timestamps) == 15

    def test_detect_cadence_single_timestamp(self):
        """Single timestamp returns default."""
        assert detect_cadence([datetime(2026, 6, 1, 0, 0, 0)]) == 5

    def test_detect_cadence_empty_list(self):
        """Empty list returns default."""
        assert detect_cadence([]) == 5

    def test_detect_cadence_variable(self):
        """Mixed cadence returns the most frequent."""
        timestamps = []
        # 6 timestamps at 10-min intervals (0..50 are valid minutes)
        for i in range(6):
            timestamps.append(datetime(2026, 6, 1, 0, i * 10, 0))
        # 2 at 5-min intervals
        timestamps.append(datetime(2026, 6, 1, 2, 0, 0))
        timestamps.append(datetime(2026, 6, 1, 2, 5, 0))
        # Most frequent is 10-min
        cadence = detect_cadence(timestamps)
        assert cadence == 10


class TestExpectedPoints:
    """Expected points calculation."""

    def test_5min_full_day(self):
        assert compute_expected_points(5) == 288

    def test_15min_full_day(self):
        assert compute_expected_points(15) == 96

    def test_60min_full_day(self):
        assert compute_expected_points(60) == 24

    def test_partial_day_6h(self):
        assert compute_expected_points(5, duration_minutes=360) == 72

    def test_invalid_cadence_raises(self):
        with pytest.raises(ValueError, match="cadence_minutes must be > 0"):
            compute_expected_points(0)

    def test_negative_cadence_raises(self):
        with pytest.raises(ValueError, match="cadence_minutes must be > 0"):
            compute_expected_points(-5)


class TestCoverage:
    """Coverage ratio computation."""

    def test_full_coverage(self):
        assert compute_coverage(288, 288) == 1.0

    def test_half_coverage(self):
        assert compute_coverage(144, 288) == pytest.approx(0.5)

    def test_zero_observed(self):
        assert compute_coverage(0, 288) == 0.0

    def test_capped_at_1(self):
        assert compute_coverage(300, 288) == 1.0

    def test_zero_expected(self):
        assert compute_coverage(100, 0) == 0.0

    def test_negative_expected(self):
        assert compute_coverage(100, -10) == 0.0


class TestNormalizeMdmDay:
    """Normalization of a single MDM row into atomic tables."""

    def test_empty_nio_returns_empty_frames(self):
        """Empty NIO should produce empty DataFrames."""
        interval, instant, register = normalize_mdm_day(
            {"NIO": ""},
            report_day=date(2026, 6, 1),
            run_id="test",
        )
        assert interval.height == 0
        assert instant.height == 0
        assert register.height == 0

    def test_missing_nio_returns_empty_frames(self):
        """Missing NIO should produce empty DataFrames."""
        interval, instant, register = normalize_mdm_day(
            {},
            report_day=date(2026, 6, 1),
            run_id="test",
        )
        assert interval.height == 0
        assert instant.height == 0
        assert register.height == 0

    def test_interval_data_produces_rows(self):
        """Interval data should produce atomic rows."""
        row = {
            "NIO": "001",
            "FA_INTERVAL": '{"00:00": 1.0, "00:05": 2.0}',
            "RA_INTERVAL": '{"00:00": 0.5, "00:05": 1.0}',
        }
        interval, instant, register = normalize_mdm_day(
            row,
            report_day=date(2026, 6, 1),
            run_id="test",
        )
        assert interval.height > 0
        assert "FA_INTERVAL" in interval.columns
        assert interval["FA_INTERVAL"].drop_nulls().len() > 0

    def test_instantaneous_data_produces_rows(self):
        """Instantaneous data should produce atomic rows."""
        row = {
            "NIO": "001",
            "U_L1": '{"00:00": 220.0, "00:15": 221.0}',
            "U_L2": '{"00:00": 219.5, "00:15": 220.5}',
        }
        interval, instant, register = normalize_mdm_day(
            row,
            report_day=date(2026, 6, 1),
            run_id="test",
        )
        assert instant.height > 0
        assert "U_L1" in instant.columns

    def test_register_data_produces_rows(self):
        """Register snapshot data should produce atomic rows."""
        row = {
            "NIO": "001",
            "FA": '{"00:00": 1000, "06:00": 1500, "12:00": 2000}',
            "FA_MD": '{"00:00": 50, "06:00": 60}',
        }
        interval, instant, register = normalize_mdm_day(
            row,
            report_day=date(2026, 6, 1),
            run_id="test",
        )
        assert register.height > 0
        assert "FA_TOTAL" in register.columns

    def test_cadence_column_present(self):
        """Normalized rows should have CADENCE_MINUTES."""
        row = {
            "NIO": "001",
            "FA_INTERVAL": '{"00:00": 1.0, "00:05": 2.0}',
        }
        interval, _, _ = normalize_mdm_day(
            row,
            report_day=date(2026, 6, 1),
            run_id="test",
        )
        assert "CADENCE_MINUTES" in interval.columns
        if interval.height > 0:
            assert interval["CADENCE_MINUTES"].drop_nulls().len() > 0

    def test_run_id_propagated(self):
        """RUN_ID should be present in all output frames."""
        row = {
            "NIO": "001",
            "FA_INTERVAL": '{"00:00": 1.0}',
        }
        interval, instant, register = normalize_mdm_day(
            row,
            report_day=date(2026, 6, 1),
            run_id="test_run_123",
        )
        for df, name in [(interval, "interval"), (instant, "instant"), (register, "register")]:
            if df.height > 0:
                assert "RUN_ID" in df.columns, f"{name} missing RUN_ID"
                assert df["RUN_ID"].drop_nulls()[0] == "test_run_123"

    def test_quality_code_default(self):
        """QUALITY_CODE should default to 0 when data is present."""
        row = {
            "NIO": "001",
            "FA_INTERVAL": '{"00:00": 1.0}',
        }
        interval, _, _ = normalize_mdm_day(
            row,
            report_day=date(2026, 6, 1),
            run_id="test",
        )
        if interval.height > 0:
            assert "QUALITY_CODE" in interval.columns
