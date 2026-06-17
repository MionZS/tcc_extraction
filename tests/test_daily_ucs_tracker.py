"""Tests for daily_ucs_tracker.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from src.checks.daily_ucs_tracker import save_daily_ucs, load_daily_ucs, get_ucs_list


pytestmark = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")


@pytest.fixture
def ucs_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "daily_cadastrados"
    d.mkdir()
    return d


class TestSaveAndLoad:
    def test_save_creates_file(self, ucs_output_dir: Path):
        ucs = ["UC001", "UC002", "UC003"]
        result = save_daily_ucs(ucs, ucs_output_dir, date(2026, 6, 15))
        assert result.exists()
        assert result.name == "ucs_20260615.parquet"

    def test_load_roundtrip(self, ucs_output_dir: Path):
        ucs = ["UC001", "UC002"]
        save_daily_ucs(ucs, ucs_output_dir, date(2026, 6, 15))
        df = load_daily_ucs(ucs_output_dir, date(2026, 6, 15))
        assert len(df) == 2
        assert df["ucs"].to_list() == ["UC001", "UC002"]

    def test_load_missing_raises(self, ucs_output_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_daily_ucs(ucs_output_dir, date(2026, 6, 15))

    def test_get_ucs_list(self, ucs_output_dir: Path):
        ucs = ["UC001", "UC002", "UC003"]
        save_daily_ucs(ucs, ucs_output_dir, date(2026, 6, 15))
        result = get_ucs_list(ucs_output_dir, date(2026, 6, 15))
        assert result == ["UC001", "UC002", "UC003"]

    def test_empty_ucs(self, ucs_output_dir: Path):
        result = save_daily_ucs([], ucs_output_dir, date(2026, 6, 15))
        assert result.exists()
        df = load_daily_ucs(ucs_output_dir, date(2026, 6, 15))
        assert len(df) == 0

    def test_creates_parent_dir(self, tmp_path: Path):
        deep_dir = tmp_path / "a" / "b" / "c"
        result = save_daily_ucs(["UC001"], deep_dir, date(2026, 6, 15))
        assert result.exists()
