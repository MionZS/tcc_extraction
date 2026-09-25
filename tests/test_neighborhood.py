"""Tests for neighborhood relative deviations across transformer peers."""

from __future__ import annotations

import polars as pl
import pytest

from src.features.neighborhood import compute_neighborhood_features


def test_neighborhood_empty():
    res = compute_neighborhood_features(pl.DataFrame(), pl.DataFrame())
    assert res.is_empty()
    assert "VOLTAGE_DELTA_TO_PEERS" in res.columns


def test_neighborhood_isolated_deviation():
    # Transformer TRAFO_A has 4 UCs: 3 have ~220V, but UC_OUTLIER has 180V (isolated drop)
    features_df = pl.DataFrame({
        "UC": ["UC_1", "UC_2", "UC_3", "UC_OUTLIER"],
        "VOLTAGE_MEAN": [220.0, 220.5, 219.8, 180.0],
        "FA_INTERVAL_SUM": [10.0, 10.2, 9.8, 2.0],
    })
    hierarchy_df = pl.DataFrame({
        "UC": ["UC_1", "UC_2", "UC_3", "UC_OUTLIER"],
        "POSTO_OPERACIONAL": ["TRAFO_A", "TRAFO_A", "TRAFO_A", "TRAFO_A"],
    })

    res = compute_neighborhood_features(features_df, hierarchy_df)
    assert res.height == 4

    outlier_row = res.filter(pl.col("UC") == "UC_OUTLIER").to_dicts()[0]
    assert outlier_row["IS_ISOLATED_DEVIATION"] is True
    assert outlier_row["VOLTAGE_DELTA_TO_PEERS"] < -20.0
    assert outlier_row["TRAFO_PEER_COUNT"] == 4

    normal_row = res.filter(pl.col("UC") == "UC_1").to_dicts()[0]
    assert normal_row["IS_ISOLATED_DEVIATION"] is False


def test_neighborhood_prefixed_columns():
    # Features DF uses new naming convention (id__uc_id and x__ prefixed features)
    features_df = pl.DataFrame({
        "id__uc_id": ["UC_1", "UC_2", "UC_3", "UC_OUTLIER"],
        "x__U_L1_AVG_MEAN": [220.0, 220.5, 219.8, 180.0],
        "x__FA_INTERVAL_SUM_MEDIAN": [10.0, 10.2, 9.8, 2.0],
    })
    hierarchy_df = pl.DataFrame({
        "UC": ["UC_1", "UC_2", "UC_3", "UC_OUTLIER"],
        "POSTO_OPERACIONAL": ["TRAFO_A", "TRAFO_A", "TRAFO_A", "TRAFO_A"],
    })

    res = compute_neighborhood_features(features_df, hierarchy_df)
    assert res.height == 4
    assert "id__uc_id" in res.columns
    assert "UC" in res.columns

    outlier_row = res.filter(pl.col("id__uc_id") == "UC_OUTLIER").to_dicts()[0]
    assert outlier_row["IS_ISOLATED_DEVIATION"] is True

