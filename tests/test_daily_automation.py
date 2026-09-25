"""Tests for Daily Automation and Data Lake Health Checker."""

from __future__ import annotations

import tempfile
from pathlib import Path
import polars as pl
import pytest

from src.automation.health_check import DataLakeHealthChecker
from src.automation.daily_runner import DailyAutomationRunner


def test_health_checker_empty_dir(tmp_path: Path):
    checker = DataLakeHealthChecker(base_dir=tmp_path)
    res = checker.run_full_health_audit()

    assert "global_status" in res
    assert res["global_status"] == "INCOMPLETE"


def test_health_checker_with_partitions(tmp_path: Path):
    # Setup test partition
    meas_p = tmp_path / "measurements" / "ami_interval" / "report_year=2026" / "report_month=06" / "report_day=01"
    meas_p.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame({
        "NIO": ["001", "002"],
        "U_L1_AVG": [220.0, 222.0],
    })
    df.write_parquet(meas_p / "data.parquet")

    # Setup context
    ctx_p = tmp_path / "context"
    ctx_p.mkdir(parents=True, exist_ok=True)
    c_df = pl.DataFrame({"UC": ["001"], "NIO": ["001"]})
    c_df.write_parquet(ctx_p / "uc_context.parquet")

    runner = DailyAutomationRunner(base_dir=tmp_path)
    audit = runner.run_daily_audit()

    assert (tmp_path / "reports" / "daily_health_report_latest.md").exists()
    assert audit["measurements"]["ami_interval"]["status"] == "HEALTHY"
    assert audit["measurements"]["ami_interval"]["total_rows"] == 2
