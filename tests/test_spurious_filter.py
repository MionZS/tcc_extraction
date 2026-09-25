"""Tests for Spurious Measurement and Grid Event Filter."""

from __future__ import annotations

import polars as pl
import pytest

from src.models.spurious_filter import filter_spurious_anomalies


def test_spurious_filter_grid_event():
    # Transformer TRAFO_1 has 4 UCs, and ALL 4 are marked anomalous (e.g. blackout/voltage drop)
    scores_df = pl.DataFrame({
        "UC": ["UC_1", "UC_2", "UC_3", "UC_4", "UC_SOLO"],
        "anomaly_score": [0.85, 0.82, 0.88, 0.84, 0.90],
        "is_anomaly": [True, True, True, True, True],
    })
    neighborhood_df = pl.DataFrame({
        "UC": ["UC_1", "UC_2", "UC_3", "UC_4", "UC_SOLO"],
        "TRAFO_ID": ["TRAFO_1", "TRAFO_1", "TRAFO_1", "TRAFO_1", "TRAFO_SOLO"],
        "TRAFO_PEER_COUNT": [4, 4, 4, 4, 1],
        "IS_ISOLATED_DEVIATION": [False, False, False, False, True],
    })

    filtered = filter_spurious_anomalies(scores_df, neighborhood_df=neighborhood_df, grid_event_threshold=0.6)
    assert filtered.height == 5

    t1_rows = filtered.filter(pl.col("UC").is_in(["UC_1", "UC_2", "UC_3", "UC_4"]))
    # All 4 in TRAFO_1 should be classified as GRID_DISTURBANCE, not individual theft!
    for r in t1_rows.iter_rows(named=True):
        assert r["IS_GRID_EVENT"] is True
        assert r["ANOMALY_CATEGORY"] == "GRID_DISTURBANCE"
        assert r["FINAL_IS_ANOMALY"] is False

    solo_row = filtered.filter(pl.col("UC") == "UC_SOLO").to_dicts()[0]
    assert solo_row["IS_GRID_EVENT"] is False
    assert solo_row["ANOMALY_CATEGORY"] == "GENUINE_UC_ANOMALY"
    assert solo_row["FINAL_IS_ANOMALY"] is True
