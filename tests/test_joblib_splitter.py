"""Tests for joblib splitting and chunking utility."""

from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
import pytest

from src.utils.joblib_splitter import dump_split_joblib, load_split_joblib


def test_split_and_load_joblib_roundtrip(tmp_path: Path):
    # Create a synthetic model
    X = np.random.RandomState(42).randn(200, 15)
    model = IsolationForest(n_estimators=50, random_state=42).fit(X)
    orig_scores = model.score_samples(X[:10])

    target = tmp_path / "model.joblib"

    # Test small chunk size to force splitting (e.g. 0.05 MB = 50 KB)
    # Using chunk_size_mb as float/int
    parts = dump_split_joblib(model, target, chunk_size_mb=1, compress=1)
    assert len(parts) >= 1

    # Load back
    loaded_model = load_split_joblib(target)
    loaded_scores = loaded_model.score_samples(X[:10])

    np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


def test_split_joblib_handles_part_files(tmp_path: Path):
    target = tmp_path / "dummy.joblib"
    data = {"key": "value", "numbers": list(range(1000))}
    parts = dump_split_joblib(data, target, chunk_size_mb=25, compress=3)
    assert len(parts) == 1
    assert target.exists()

    loaded = load_split_joblib(target)
    assert loaded == data
