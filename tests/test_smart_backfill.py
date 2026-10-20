"""Unit tests for smart incremental backfill and gap detection."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import pytest

from scripts.run_araucaria_full import (
    determine_smart_extraction_dates,
    find_datalake_date_coverage,
)


def test_find_datalake_date_coverage_empty(tmp_path: Path):
    dates, min_d, max_d = find_datalake_date_coverage(tmp_path)
    assert dates == set()
    assert min_d is None
    assert max_d is None


def test_find_datalake_date_coverage_with_partitions(tmp_path: Path):
    p1 = tmp_path / "measurements" / "ami_instantaneous" / "report_year=2026" / "report_month=09" / "report_day=11"
    p1.mkdir(parents=True, exist_ok=True)
    (p1 / "data.parquet").write_bytes(b"x" * 2000)

    p2 = tmp_path / "measurements" / "ami_instantaneous" / "report_year=2026" / "report_month=09" / "report_day=17"
    p2.mkdir(parents=True, exist_ok=True)
    (p2 / "data.parquet").write_bytes(b"x" * 2000)

    dates, min_d, max_d = find_datalake_date_coverage(tmp_path)
    assert date(2026, 9, 11) in dates
    assert date(2026, 9, 17) in dates
    assert min_d == date(2026, 9, 11)
    assert max_d == date(2026, 9, 17)


def test_determine_smart_extraction_dates_incremental_tail(tmp_path: Path):
    # Setup mock datalake with partition up to 3 days ago
    three_days_ago = date.today() - timedelta(days=3)
    p = (
        tmp_path
        / "measurements"
        / "ami_instantaneous"
        / f"report_year={three_days_ago.year}"
        / f"report_month={three_days_ago.month:02d}"
        / f"report_day={three_days_ago.day:02d}"
    )
    p.mkdir(parents=True, exist_ok=True)
    (p / "data.parquet").write_bytes(b"x" * 2000)

    # Smart extraction should identify from (three_days_ago + 1) to yesterday
    missing_dates, desc = determine_smart_extraction_dates(tmp_path, fill_gaps=False)

    yesterday = date.today() - timedelta(days=1)
    two_days_ago = date.today() - timedelta(days=2)

    assert two_days_ago in missing_dates
    assert yesterday in missing_dates
    assert three_days_ago not in missing_dates
    assert len(missing_dates) == 2
