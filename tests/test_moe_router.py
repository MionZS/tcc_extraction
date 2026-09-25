"""Tests for Mixture of Experts (MoE) Gating Router."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.moe_router import MoERouter


def test_moe_router_weights_large_feeder():
    router = MoERouter()
    # Feeder with 1500 samples
    weights = router.compute_gating_weights(feeder_sample_count=1500)
    assert weights["w_feeder"] >= 0.65
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)


def test_moe_router_weights_sparse_feeder():
    router = MoERouter()
    # Feeder with only 50 samples
    weights = router.compute_gating_weights(feeder_sample_count=50)
    # Sparse feeder should rely primarily on regional/global regularizer
    assert weights["w_feeder"] <= 0.20
    assert weights["w_global"] >= 0.35
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)


def test_moe_combine_predictions():
    router = MoERouter()
    feeder_scores = np.array([0.8, 0.2])
    global_scores = np.array([0.4, 0.4])

    weights = {"w_feeder": 0.8, "w_global": 0.2}
    combined = router.combine_predictions(feeder_scores, None, global_scores, weights)

    assert len(combined) == 2
    # 0.8*0.8 + 0.4*0.2 = 0.64 + 0.08 = 0.72
    assert pytest.approx(combined[0], abs=0.01) == 0.72
    # 0.2*0.8 + 0.4*0.2 = 0.16 + 0.08 = 0.24
    assert pytest.approx(combined[1], abs=0.01) == 0.24
