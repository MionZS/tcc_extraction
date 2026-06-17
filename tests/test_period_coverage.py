"""Tests for period_coverage.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from period_coverage import (
    get_file_dates,
    generate_date_range,
    check_period_coverage,
)


class TestGetFileDates:
    def test_collects_dates(self, dated_cis_dir: Path):
        dates = get_file_dates(dated_cis_dir, suffix=".csv")
        assert len(dates) == 10
        assert dates[0] == date(2026, 6, 6)
        assert dates[-1] == date(2026, 6, 15)

    def test_empty_dir(self, tmp_path: Path):
        assert get_file_dates(tmp_path, suffix=".csv") == []

    def test_prefix_filter(self, tmp_path: Path):
        (tmp_path / "araucaria_cis_20260615.csv").write_text("a")
        (tmp_path / "araucaria_mdm_20260615.csv").write_text("a")
        dates = get_file_dates(tmp_path, suffix=".csv", prefix="araucaria_cis")
        assert dates == [date(2026, 6, 15)]


class TestGenerateDateRange:
    def test_same_day(self):
        d = date(2026, 6, 15)
        assert generate_date_range(d, d) == [d]

    def test_range(self):
        result = generate_date_range(date(2026, 6, 1), date(2026, 6, 5))
        assert len(result) == 5
        assert result[0] == date(2026, 6, 1)
        assert result[-1] == date(2026, 6, 5)

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="posterior"):
            generate_date_range(date(2026, 6, 10), date(2026, 6, 5))


class TestCheckPeriodCoverage:
    def test_complete(self):
        dates = {date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)}
        result = check_period_coverage(dates)
        assert result["complete"] is True
        assert result["missing_dates"] == []
        assert result["coverage_percent"] == 100.0

    def test_gap(self):
        dates = {date(2026, 6, 1), date(2026, 6, 3)}  # missing 6/2
        result = check_period_coverage(dates, start_date=date(2026, 6, 1), end_date=date(2026, 6, 3))
        assert result["complete"] is False
        assert "2026-06-02" in result["missing_dates"]
        assert result["coverage_percent"] < 100.0

    def test_empty_dates(self):
        result = check_period_coverage([])
        assert result["complete"] is True
        assert result["total_days"] == 0

    def test_auto_range(self):
        dates = {date(2026, 6, 1), date(2026, 6, 5)}
        result = check_period_coverage(dates)
        assert result["complete"] is False
        assert result["total_days"] == 5
        assert len(result["missing_dates"]) == 3
