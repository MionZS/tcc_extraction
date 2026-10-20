"""Unit tests for ModelRegistry and model versioning governance."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.registry import ModelMetadata, ModelRegistry


def test_model_registry_lifecycle(tmp_path: Path):
    reg = ModelRegistry(registry_dir=tmp_path / "registry")

    # 1. Register first model version
    dummy_model_v1 = LogisticRegression()
    metrics_v1 = {
        "accuracy": 0.85,
        "f1_macro": 0.80,
        "f1_weighted": 0.84,
        "report": {"NORMAL": {"f1-score": 0.90}},
    }
    data_slice_v1 = {
        "start_date": "2026-08-01",
        "end_date": "2026-08-31",
        "feeder": "TODOS",
        "train_samples": 1000,
        "test_samples": 250,
    }

    v1_id = reg.register_model(
        model_name="voltage_classifier",
        model_obj=dummy_model_v1,
        metrics=metrics_v1,
        data_slice=data_slice_v1,
        hyperparameters={"C": 1.0},
        tag="baseline",
        description="Agosto baseline",
    )

    assert v1_id.startswith("v_")
    models = reg.list_models()
    assert len(models) == 1
    assert models[0]["version_id"] == v1_id
    assert models[0]["accuracy"] == 0.85
    assert models[0]["tag"] == "baseline"

    # 2. Register second improved model version
    dummy_model_v2 = LogisticRegression()
    metrics_v2 = {
        "accuracy": 0.92,
        "f1_macro": 0.89,
        "f1_weighted": 0.91,
        "report": {"NORMAL": {"f1-score": 0.95}},
    }
    data_slice_v2 = {
        "start_date": "2026-08-01",
        "end_date": "2026-10-18",
        "feeder": "TODOS",
        "train_samples": 5000,
        "test_samples": 1250,
    }

    v2_id = reg.register_model(
        model_name="voltage_classifier",
        model_obj=dummy_model_v2,
        metrics=metrics_v2,
        data_slice=data_slice_v2,
        hyperparameters={"C": 2.0},
        tag="production",
        description="Período completo 79 dias",
    )

    models_updated = reg.list_models()
    assert len(models_updated) == 2
    assert models_updated[0]["version_id"] == v2_id  # latest first

    # 3. Load latest model
    loaded_obj, meta, metrics = reg.load_model("voltage_classifier", "latest")
    assert meta["version_id"] == v2_id
    assert metrics["accuracy"] == 0.92

    # 4. Compare v1 vs v2
    comp = reg.compare_models("voltage_classifier", v1_id, v2_id)
    assert comp["model_name"] == "voltage_classifier"
    assert comp["version_a"]["version_id"] == v1_id
    assert comp["version_b"]["version_id"] == v2_id
    assert pytest.approx(comp["deltas"]["accuracy_delta"], 0.001) == 0.07
    assert pytest.approx(comp["deltas"]["f1_macro_delta"], 0.001) == 0.09
