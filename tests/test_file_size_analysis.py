"""Tests for file_size_analysis.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from file_size_analysis import (
    collect_file_sizes,
    compute_size_stats,
    detect_size_outliers,
    format_size,
    FileSizeRecord,
)


class TestCollectFileSizes:
    def test_collects_all(self, dated_cis_dir: Path):
        records = collect_file_sizes(dated_cis_dir, suffix=".csv")
        assert len(records) == 10
        assert all(r.size_bytes > 0 for r in records)

    def test_empty_dir(self, tmp_path: Path):
        assert collect_file_sizes(tmp_path, suffix=".csv") == []

    def test_prefix_filter(self, tmp_path: Path):
        (tmp_path / "araucaria_cis_20260615.csv").write_text("a" * 100)
        (tmp_path / "araucaria_mdm_20260615.csv").write_text("b" * 200)
        records = collect_file_sizes(tmp_path, suffix=".csv", prefix="araucaria_cis")
        assert len(records) == 1
        assert records[0].size_bytes == 100


class TestComputeSizeStats:
    def test_basic_stats(self):
        records = [
            FileSizeRecord(Path("a.csv"), date(2026, 6, 1), 100),
            FileSizeRecord(Path("b.csv"), date(2026, 6, 2), 200),
            FileSizeRecord(Path("c.csv"), date(2026, 6, 3), 300),
        ]
        stats = compute_size_stats(records)
        assert stats is not None
        assert stats.count == 3
        assert stats.mean_bytes == 200
        assert stats.min_bytes == 100
        assert stats.max_bytes == 300

    def test_single_record(self):
        records = [FileSizeRecord(Path("a.csv"), date(2026, 6, 1), 100)]
        stats = compute_size_stats(records)
        assert stats is not None
        assert stats.std_bytes == 0.0

    def test_empty(self):
        assert compute_size_stats([]) is None


class TestDetectOutliers:
    def test_no_outliers(self):
        records = [
            FileSizeRecord(Path("a.csv"), date(2026, 6, i), 100 + i)
            for i in range(10)
        ]
        outliers = detect_size_outliers(records)
        assert outliers == []

    def test_with_outlier(self):
        records = [
            FileSizeRecord(Path(f"f{i}.csv"), date(2026, 6, i + 1), 1000)
            for i in range(10)
        ]
        # Add a very large file
        records.append(FileSizeRecord(Path("outlier.csv"), date(2026, 6, 11), 1_000_000))
        outliers = detect_size_outliers(records)
        assert len(outliers) == 1
        assert outliers[0].size_bytes == 1_000_000

    def test_empty(self):
        assert detect_size_outliers([]) == []

    def test_single_file(self):
        records = [FileSizeRecord(Path("a.csv"), None, 100)]
        assert detect_size_outliers(records) == []


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_size(2_000_000) == "1.9 MB"
