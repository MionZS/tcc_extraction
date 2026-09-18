"""Tests for register reset detection.

Register resets occur when accumulated energy values decrease
(negative difference between consecutive readings), which can
indicate meter replacement, rollover, or data quality issues.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from src.checks.register_resets import detect_register_resets


class TestRegisterResetDetection:
    """Register reset detection from consecutive readings."""

    def test_no_reset(self):
        """Monotonically increasing values → no resets."""
        df = pl.DataFrame({
            "NIO": ["001"] * 4,
            "TIMESTAMP_UTC": [
                datetime(2026, 6, 1, 0, 0, 0),
                datetime(2026, 6, 1, 6, 0, 0),
                datetime(2026, 6, 1, 12, 0, 0),
                datetime(2026, 6, 1, 18, 0, 0),
            ],
            "FA_TOTAL": [1000.0, 1500.0, 2000.0, 2500.0],
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        assert resets.height == 0

    def test_single_reset(self):
        """A single negative difference → one reset detected."""
        df = pl.DataFrame({
            "NIO": ["001"] * 4,
            "TIMESTAMP_UTC": [
                datetime(2026, 6, 1, 0, 0, 0),
                datetime(2026, 6, 1, 6, 0, 0),
                datetime(2026, 6, 1, 12, 0, 0),
                datetime(2026, 6, 1, 18, 0, 0),
            ],
            "FA_TOTAL": [1000.0, 1500.0, 500.0, 1000.0],
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        assert resets.height == 1
        # The reset delta should be 1500 - 500 = 1000
        assert abs(resets["FA_TOTAL_RESET_DELTA"][0] - 1000.0) < 0.01

    def test_multiple_resets_different_nios(self):
        """Resets from different NIOs should be detected independently."""
        df = pl.DataFrame({
            "NIO": ["001"] * 3 + ["002"] * 3,
            "TIMESTAMP_UTC": [
                datetime(2026, 6, 1, 0, 0, 0),
                datetime(2026, 6, 1, 6, 0, 0),
                datetime(2026, 6, 1, 12, 0, 0),
                # NIO 002
                datetime(2026, 6, 1, 0, 0, 0),
                datetime(2026, 6, 1, 6, 0, 0),
                datetime(2026, 6, 1, 12, 0, 0),
            ],
            "FA_TOTAL": [
                1000.0, 500.0, 1000.0,  # NIO 001: reset at 06:00
                2000.0, 1800.0, 2000.0,  # NIO 002: reset at 06:00
            ],
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        assert resets.height == 2
        assert set(resets["NIO"].to_list()) == {"001", "002"}

    def test_rollover_small_delta(self):
        """Small negative delta near zero is still a reset."""
        df = pl.DataFrame({
            "NIO": ["001"] * 3,
            "TIMESTAMP_UTC": [
                datetime(2026, 6, 1, 0, 0, 0),
                datetime(2026, 6, 1, 6, 0, 0),
                datetime(2026, 6, 1, 12, 0, 0),
            ],
            "FA_TOTAL": [1000.0, 1000.0001, 999.9999],
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        # The tiny decrease from 1000.0001 to 999.9999 is a reset
        assert resets.height == 1

    def test_empty_dataframe(self):
        """Empty DataFrame → no resets."""
        df = pl.DataFrame(schema={
            "NIO": pl.String,
            "TIMESTAMP_UTC": pl.Datetime,
            "FA_TOTAL": pl.Float64,
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        assert resets.height == 0

    def test_missing_column(self):
        """Missing column → no resets."""
        df = pl.DataFrame({
            "NIO": ["001", "002"],
            "TIMESTAMP_UTC": [datetime(2026, 6, 1, 0, 0, 0)] * 2,
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        assert resets.height == 0

    def test_nulls_do_not_affect(self):
        """Null values between readings should not create false resets."""
        df = pl.DataFrame({
            "NIO": ["001"] * 5,
            "TIMESTAMP_UTC": [
                datetime(2026, 6, 1, 0, 0, 0),
                datetime(2026, 6, 1, 6, 0, 0),
                datetime(2026, 6, 1, 12, 0, 0),
                datetime(2026, 6, 1, 18, 0, 0),
                datetime(2026, 6, 2, 0, 0, 0),
            ],
            "FA_TOTAL": [1000.0, None, 1500.0, None, 2000.0],
        })
        # diff() will compare (1000 → None → 1500 → None → 2000)
        # This should not produce false resets
        resets = detect_register_resets(df, "FA_TOTAL")
        # The diff from None to 1500 = None, not negative
        # The diff from 1500 to None = None
        # The diff from None to 2000 = None
        assert resets.height == 0


class TestResetStatistics:
    """Statistics from register resets."""

    def test_reset_count_per_nio(self):
        """Count resets per NIO."""
        df = pl.DataFrame({
            "NIO": ["001"] * 6,
            "TIMESTAMP_UTC": [
                datetime(2026, 6, 1, h, 0, 0) for h in [0, 6, 12, 18]
            ] + [datetime(2026, 6, 2, h, 0, 0) for h in [0, 6]],
            "FA_TOTAL": [1000.0, 1500.0, 500.0, 1000.0, 500.0, 1000.0],
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        reset_count = (
            resets.group_by("NIO")
            .agg(pl.len().alias("reset_count"))
        )
        assert reset_count.filter(pl.col("NIO") == "001")["reset_count"][0] == 2

    def test_max_reset_delta(self):
        """Maximum reset delta per NIO."""
        df = pl.DataFrame({
            "NIO": ["001"] * 3,
            "TIMESTAMP_UTC": [
                datetime(2026, 6, 1, 0, 0, 0),
                datetime(2026, 6, 1, 6, 0, 0),
                datetime(2026, 6, 1, 12, 0, 0),
            ],
            "FA_TOTAL": [10000.0, 100.0, 200.0],
        })
        resets = detect_register_resets(df, "FA_TOTAL")
        assert abs(resets["FA_TOTAL_RESET_DELTA"][0] - 9900.0) < 0.01
