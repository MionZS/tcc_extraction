"""Tests for daily_file_naming.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from daily_file_naming import (
    extract_date_from_filename,
    validate_file_naming,
    check_days_back_consistency,
)


class TestExtractDate:
    def test_valid_csv(self):
        assert extract_date_from_filename("araucaria_cis_20260615.csv") == date(2026, 6, 15)

    def test_valid_parquet(self):
        assert extract_date_from_filename("ucs_20260615.parquet") == date(2026, 6, 15)

    def test_with_sample_tag(self):
        assert extract_date_from_filename("araucaria_cis_sample_200_20260615.csv") == date(2026, 6, 15)

    def test_no_date(self):
        assert extract_date_from_filename("araucaria_cis.csv") is None

    def test_invalid_date(self):
        assert extract_date_from_filename("araucaria_cis_99999999.csv") is None

    def test_empty_string(self):
        assert extract_date_from_filename("") is None


class TestValidateNaming:
    def test_valid_files(self, tmp_path: Path):
        for name in ["araucaria_cis_20260615.csv", "araucaria_mdm_20260615.csv"]:
            (tmp_path / name).write_text("a", encoding="utf-8-sig")
        assert validate_file_naming(tmp_path) == []

    def test_invalid_file(self, tmp_path: Path):
        (tmp_path / "random_file.csv").write_text("a", encoding="utf-8-sig")
        invalid = validate_file_naming(tmp_path)
        assert "random_file.csv" in invalid

    def test_ignores_non_csv(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("a", encoding="utf-8-sig")
        (tmp_path / "araucaria_cis_20260615.csv").write_text("a", encoding="utf-8-sig")
        assert validate_file_naming(tmp_path) == []

    def test_ignores_directories(self, tmp_path: Path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "araucaria_cis_20260615.csv").write_text("a", encoding="utf-8-sig")
        assert validate_file_naming(tmp_path) == []

    def test_prefix_filter(self, tmp_path: Path):
        (tmp_path / "araucaria_cis_20260615.csv").write_text("a", encoding="utf-8-sig")
        (tmp_path / "araucaria_mdm_20260615.csv").write_text("a", encoding="utf-8-sig")
        invalid = validate_file_naming(tmp_path, prefix="cis")
        assert invalid == []

    def test_invalid_prefix_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Prefixo desconhecido"):
            validate_file_naming(tmp_path, prefix="nonexistent")


class TestCheckDaysBackConsistency:
    def test_matching(self, tmp_path: Path):
        csv = tmp_path / "araucaria_cis_20260615.csv"
        csv.write_text("a", encoding="utf-8-sig")
        result = check_days_back_consistency(csv, days_back=2, report_day=date(2026, 6, 15))
        assert result["valid"] is True

    def test_mismatch(self, tmp_path: Path):
        csv = tmp_path / "araucaria_cis_20260615.csv"
        csv.write_text("a", encoding="utf-8-sig")
        result = check_days_back_consistency(csv, days_back=1, report_day=date(2026, 6, 16))
        assert result["valid"] is False

    def test_no_date_in_name(self, tmp_path: Path):
        csv = tmp_path / "araucaria_cis.csv"
        csv.write_text("a", encoding="utf-8-sig")
        result = check_days_back_consistency(csv, days_back=1)
        assert result["valid"] is False
        assert "Não foi possível" in result["message"]
