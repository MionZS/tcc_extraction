"""Unit tests for Data Lake Continuity, Governance & Timeline Calendar."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pytest

from src.visualization.continuity import (
    compute_continuity_segments,
    render_continuity_calendar,
    scan_datalake_dates,
)


def test_compute_continuity_segments_single_contiguous():
    dates = {date(2026, 8, 1), date(2026, 8, 2), date(2026, 8, 3)}
    segs = compute_continuity_segments(date(2026, 8, 1), date(2026, 8, 3), dates)

    assert len(segs) == 1
    assert segs[0]["status"] == "PRESENTE"
    assert segs[0]["days"] == 3
    assert segs[0]["start_date"] == date(2026, 8, 1)
    assert segs[0]["end_date"] == date(2026, 8, 3)


def test_compute_continuity_segments_with_gaps():
    # Gap in the middle: 01 present, 02 missing, 03 present
    dates = {date(2026, 8, 1), date(2026, 8, 3)}
    segs = compute_continuity_segments(date(2026, 8, 1), date(2026, 8, 3), dates)

    assert len(segs) == 3
    assert segs[0]["status"] == "PRESENTE"
    assert segs[0]["days"] == 1

    assert segs[1]["status"] == "LACUNA"
    assert segs[1]["days"] == 1
    assert segs[1]["start_date"] == date(2026, 8, 2)
    assert segs[1]["end_date"] == date(2026, 8, 2)

    assert segs[2]["status"] == "PRESENTE"
    assert segs[2]["days"] == 1


def test_compute_continuity_segments_all_missing():
    segs = compute_continuity_segments(date(2026, 8, 1), date(2026, 8, 10), set())
    assert len(segs) == 1
    assert segs[0]["status"] == "LACUNA"
    assert segs[0]["days"] == 10


def test_scan_datalake_dates_mock(tmp_path: Path):
    # Setup mock partition structure with parquet and csv
    p = tmp_path / "measurements" / "ami_instantaneous" / "report_year=2026" / "report_month=09" / "report_day=11"
    p.mkdir(parents=True, exist_ok=True)
    (p / "data.parquet").write_bytes(b"dummy")

    # Day 12 only has CSV
    p12 = tmp_path / "measurements" / "ami_instantaneous" / "report_year=2026" / "report_month=09" / "report_day=12"
    p12.mkdir(parents=True, exist_ok=True)
    (p12 / "data.csv").write_bytes(b"dummy")

    # 1. Both filter
    found_both = scan_datalake_dates(tmp_path, file_type="both")
    assert date(2026, 9, 11) in found_both
    assert date(2026, 9, 12) in found_both
    assert found_both[date(2026, 9, 11)]["has_parquet"] is True
    assert found_both[date(2026, 9, 12)]["has_csv"] is True

    # 2. Parquet only filter
    found_p = scan_datalake_dates(tmp_path, file_type="parquet")
    assert date(2026, 9, 11) in found_p
    assert date(2026, 9, 12) not in found_p

    # 3. CSV only filter
    found_c = scan_datalake_dates(tmp_path, file_type="csv")
    assert date(2026, 9, 12) in found_c


def test_render_continuity_calendar_smoke(tmp_path: Path):
    # Smoke test rendering without error across formats
    p = tmp_path / "measurements" / "ami_instantaneous" / "report_year=2026" / "report_month=09" / "report_day=11"
    p.mkdir(parents=True, exist_ok=True)
    (p / "data.parquet").write_bytes(b"dummy")
    (p / "data.csv").write_bytes(b"dummy")

    for f_type in ("both", "parquet", "csv"):
        render_continuity_calendar(
            start_date=date(2026, 9, 10),
            end_date=date(2026, 9, 12),
            base_dir=tmp_path,
            file_type=f_type,
        )
