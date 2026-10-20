"""Unit tests for format converter and bidirectional Parquet <-> CSV synchronization."""

from __future__ import annotations

from pathlib import Path
import polars as pl
import pytest

from src.datasets.format_converter import (
    convert_csv_to_parquet,
    convert_parquet_to_csv,
    render_format_audit_table,
    scan_format_discrepancies,
    sync_datalake_formats,
)


def test_convert_parquet_to_csv(tmp_path: Path):
    df = pl.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    p_path = tmp_path / "data.parquet"
    df.write_parquet(p_path)

    csv_path = convert_parquet_to_csv(p_path)
    assert csv_path.exists()
    assert csv_path.suffix == ".csv"

    read_back = pl.read_csv(csv_path, separator=";")
    assert read_back.height == 3
    assert read_back["A"].to_list() == [1, 2, 3]


def test_convert_csv_to_parquet(tmp_path: Path):
    df = pl.DataFrame({"NIO": ["1001", "1002"], "VOLTAGE": [127.5, 128.0]})
    c_path = tmp_path / "data.csv"
    df.write_csv(c_path, separator=";")

    p_path = convert_csv_to_parquet(c_path)
    assert p_path.exists()
    assert p_path.suffix == ".parquet"

    read_back = pl.read_parquet(p_path)
    assert read_back.height == 2
    assert read_back["VOLTAGE"].to_list() == [127.5, 128.0]


def test_scan_and_sync_datalake_formats(tmp_path: Path):
    # Setup mock datalake
    # Case 1: Partition with parquet only (missing CSV)
    p1 = tmp_path / "measurements" / "ami_instantaneous" / "report_year=2026" / "report_month=09" / "report_day=11"
    p1.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"X": [10, 20]}).write_parquet(p1 / "data.parquet")

    # Case 2: Context with CSV only (missing Parquet)
    p2 = tmp_path / "context"
    p2.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"UC": ["UC_1", "UC_2"]}).write_csv(p2 / "uc_context.csv", separator=";")

    # Case 3: Already synchronized
    p3 = tmp_path / "measurements" / "ami_interval" / "report_year=2026" / "report_month=09" / "report_day=11"
    p3.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Z": [1]}).write_parquet(p3 / "data.parquet")
    pl.DataFrame({"Z": [1]}).write_csv(p3 / "data.csv", separator=";")

    # 1. Scan discrepancies
    disc = scan_format_discrepancies(tmp_path)
    assert len(disc["missing_csv"]) == 1
    assert disc["missing_csv"][0]["source"] == p1 / "data.parquet"

    assert len(disc["missing_parquet"]) == 1
    assert disc["missing_parquet"][0]["source"] == p2 / "uc_context.csv"

    # 2. Synchronize
    sync_res = sync_datalake_formats(tmp_path, direction="both")
    assert sync_res["parquet_to_csv"] == 1
    assert sync_res["csv_to_parquet"] == 1
    assert sync_res["errors"] == 0

    assert (p1 / "data.csv").exists()
    assert (p2 / "uc_context.parquet").exists()

    # 3. Re-scan should show 0 discrepancies
    disc_after = scan_format_discrepancies(tmp_path)
    assert len(disc_after["missing_csv"]) == 0
    assert len(disc_after["missing_parquet"]) == 0

    # 4. Smoke test table rendering
    render_format_audit_table(tmp_path)
