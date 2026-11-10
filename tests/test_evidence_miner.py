"""Unit tests for AnomalyEvidenceMiner and ModelRegistry active promotion."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from src.models.evidence_miner import AnomalyEvidenceMiner
from src.models.registry import ModelRegistry


def test_registry_promote_tag_lifecycle(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir=reg_dir)

    clf1 = LogisticRegression()
    clf2 = LogisticRegression()

    v1 = registry.register_model(
        model_name="test_model",
        model_obj=clf1,
        metrics={"accuracy": 0.85},
        tag="candidate",
        custom_version="v_20261105_001",
    )
    v2 = registry.register_model(
        model_name="test_model",
        model_obj=clf2,
        metrics={"accuracy": 0.92},
        tag="candidate",
        custom_version="v_20261105_002",
    )

    # Initially neither is production
    prod = registry.get_production_model("test_model")
    assert prod is not None  # Fallback to latest
    assert prod[1]["version_id"] == v2

    # Promote v1 to production
    registry.promote_tag("test_model", v1, new_tag="production")
    prod = registry.get_production_model("test_model")
    assert prod is not None
    assert prod[1]["version_id"] == v1
    assert prod[1]["tag"] == "production"

    # Now promote v2 to production -> v1 must be automatically demoted to candidate
    registry.promote_tag("test_model", v2, new_tag="production")
    prod = registry.get_production_model("test_model")
    assert prod is not None
    assert prod[1]["version_id"] == v2
    assert prod[1]["tag"] == "production"

    all_m = registry.list_models("test_model")
    v1_entry = next(m for m in all_m if m["version_id"] == v1)
    assert v1_entry["tag"] == "candidate"


def test_anomaly_evidence_miner_identifies_clusters_and_hypotheses() -> None:
    miner = AnomalyEvidenceMiner(n_clusters=2, random_state=42)

    # Synthetic population: 100 normal meters and 20 anomalous meters
    # Sub-cluster A (10 meters): high active reversal (Solar GD)
    # Sub-cluster B (10 meters): severe current imbalance (Phase bypass)
    rows = []
    for i in range(100):
        rows.append({
            "id__uc_id": f"NORM_{i:03d}",
            "x__hist_ra_reversal_ratio_max": 0.0,
            "x__hist_current_imbalance_max": 0.05,
            "x__fa_sum": 10.0 + np.random.normal(0, 0.5),
            "consensus_score": 0.10,
            "is_anomaly": False,
        })
    for i in range(10):
        rows.append({
            "id__uc_id": f"GD_{i:03d}",
            "x__hist_ra_reversal_ratio_max": 0.85 + np.random.normal(0, 0.05),
            "x__hist_current_imbalance_max": 0.05,
            "x__fa_sum": 5.0,
            "consensus_score": 0.95,
            "is_anomaly": True,
        })
    for i in range(10):
        rows.append({
            "id__uc_id": f"BYPASS_{i:03d}",
            "x__hist_ra_reversal_ratio_max": 0.0,
            "x__hist_current_imbalance_max": 0.90 + np.random.normal(0, 0.02),
            "x__fa_sum": 2.0,
            "consensus_score": 0.92,
            "is_anomaly": True,
        })

    df = pl.DataFrame(rows)
    evidences = miner.mine_evidence(df, score_col="consensus_score", anomaly_flag_col="is_anomaly")

    assert len(evidences) == 2
    names = [e.cluster_name for e in evidences]
    assert any("Injeção Reversa" in n for n in names)
    assert any("Bypass" in n or "Assimetria" in n for n in names)

    # Check that lifts were computed
    for ev in evidences:
        assert ev.sample_count == 10
        assert len(ev.top_divergent_features) > 0
        assert len(ev.sample_ucs) > 0

