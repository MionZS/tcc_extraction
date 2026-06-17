"""Tests for integrity_check.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.checks.integrity_check import (
    check_file_naming,
    check_period,
    check_sizes,
    check_ucs_file,
    run_integrity_check,
    IntegrityReport,
)


@pytest.fixture
def full_output_tree(tmp_path: Path) -> Path:
    """Cria árvore de output completa com arquivos datados."""
    # CIS
    cis_dir = tmp_path / "raw" / "CIS"
    cis_dir.mkdir(parents=True)
    for i in range(1, 11):
        dt = date(2026, 6, 15 - (10 - i))
        (cis_dir / f"araucaria_cis_{dt:%Y%m%d}.csv").write_text(
            "UC;NIO;SITUACAO\nUC001;NIO001;ATIVA\n",
            encoding="utf-8-sig",
        )

    # ORCA
    orca_dir = tmp_path / "raw" / "ORCA"
    orca_dir.mkdir(parents=True)
    for i in range(1, 11):
        dt = date(2026, 6, 15 - (10 - i))
        (orca_dir / f"araucaria_mdm_{dt:%Y%m%d}.csv").write_text(
            "UC;NIO;DATA_LEITURA;VALOR_KWH\n",
            encoding="utf-8-sig",
        )

    # Refined
    refined_dir = tmp_path / "refined" / "reports"
    refined_dir.mkdir(parents=True)

    # UCs
    ucs_dir = tmp_path / "raw" / "CIS" / "daily_cadastrados"
    ucs_dir.mkdir(parents=True)
    # Create a parquet file (just write a placeholder; real parquet test needs polars)
    try:
        import polars as pl
        df = pl.DataFrame({"ucs": ["UC001"], "count": [1], "date": ["2026-06-15"], "created_at": ["2026-06-15T00:00:00"]})
        df.write_parquet(ucs_dir / "ucs_20260615.parquet")
    except ImportError:
        # If polars not available, create a dummy file
        (ucs_dir / "ucs_20260615.parquet").write_bytes(b"dummy")

    return tmp_path


class TestCheckFileNaming:
    def test_all_valid(self, full_output_tree: Path):
        result = check_file_naming(full_output_tree)
        assert result.passed is True

    def test_invalid_naming(self, full_output_tree: Path):
        (full_output_tree / "raw" / "CIS" / "bad_file.csv").write_text("a")
        result = check_file_naming(full_output_tree)
        assert result.passed is False
        assert "bad_file.csv" in result.detail


class TestCheckPeriod:
    def test_incomplete_period(self, full_output_tree: Path):
        result = check_period(full_output_tree, min_days_back=30)
        # Has 10 days, checking 30 → will have gaps
        assert result.passed is False

    def test_missing_dir(self, tmp_path: Path):
        result = check_period(tmp_path, min_days_back=30)
        assert result.passed is True  # No dir = no problem


class TestCheckSizes:
    def test_sizes_ok(self, full_output_tree: Path):
        result = check_sizes(full_output_tree)
        assert result.passed is True

    def test_missing_dir(self, tmp_path: Path):
        result = check_sizes(tmp_path)
        assert result.passed is True


class TestCheckUcsFile:
    def test_ucs_exists(self, full_output_tree: Path):
        result = check_ucs_file(full_output_tree)
        assert result.passed is True

    def test_ucs_missing(self, tmp_path: Path):
        (tmp_path / "raw").mkdir(parents=True)
        result = check_ucs_file(tmp_path)
        assert result.passed is False


class TestIntegrityReport:
    def test_all_pass(self):
        report = IntegrityReport()
        report.add(check_file_naming(Path("/nonexistent")))  # empty = pass
        assert report.passed is True

    def test_one_fail(self):
        report = IntegrityReport()
        report.add(check_file_naming(Path("/nonexistent")))
        from src.checks.integrity_check import CheckResult
        report.add(CheckResult("manual", False, "manual fail"))
        assert report.passed is False

    def test_to_dict(self):
        report = IntegrityReport()
        from src.checks.integrity_check import CheckResult
        report.add(CheckResult("test", True, "ok"))
        d = report.to_dict()
        assert d["passed"] is True
        assert len(d["checks"]) == 1


class TestRunIntegrityCheck:
    def test_full_check(self, full_output_tree: Path):
        report = run_integrity_check(full_output_tree, min_days_back=10)
        assert isinstance(report, IntegrityReport)
        assert report.summary["total_checks"] > 0
