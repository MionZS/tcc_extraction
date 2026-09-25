"""Tests for circuit profiling and electrical statistical signatures."""

from __future__ import annotations

import polars as pl
import pytest

from src.features.circuit_profile import compute_circuit_profile


def test_circuit_profile_empty():
    res = compute_circuit_profile(pl.DataFrame(), pl.DataFrame())
    assert res.is_empty()
    assert "CIRCUIT_UC_COUNT" in res.columns


def test_circuit_profile_basic():
    features_df = pl.DataFrame({
        "UC": ["UC_1", "UC_2", "UC_3", "UC_4"],
        "VOLTAGE_MEAN": [220.0, 221.0, 219.0, 220.5],
        "VOLTAGE_IMBALANCE_MAX": [1.0, 2.0, 1.5, 0.8],
        "LOAD_FACTOR": [0.6, 0.5, 0.7, 0.4],
        "RA_REVERSAL_RATIO": [0.0, 0.1, 0.0, 0.0],
        "FA_INTERVAL_SUM": [10.0, 20.0, 15.0, 12.0],
    })
    hierarchy_df = pl.DataFrame({
        "UC": ["UC_1", "UC_2", "UC_3", "UC_4"],
        "ALIMENTADOR": ["FONTE_NOVA", "FONTE_NOVA", "THOMAZ_COELHO", "THOMAZ_COELHO"],
        "POSTO_OPERACIONAL": ["P1", "P1", "P2", "P3"],
        "CLASSE_PRINCIPAL": ["RESIDENCIAL", "COMERCIAL", "RESIDENCIAL", "RESIDENCIAL"],
    })

    profiles = compute_circuit_profile(features_df, hierarchy_df)
    assert profiles.height == 2
    assert "ALIMENTADOR" in profiles.columns

    fn_row = profiles.filter(pl.col("ALIMENTADOR") == "FONTE_NOVA").to_dicts()[0]
    assert fn_row["CIRCUIT_UC_COUNT"] == 2
    assert fn_row["CIRCUIT_TRAFO_COUNT"] == 1
    assert fn_row["CIRCUIT_UCS_PER_TRAFO_MEAN"] == 2.0
    assert fn_row["CIRCUIT_RESIDENTIAL_RATIO"] == 0.5
    assert fn_row["CIRCUIT_COMMERCIAL_RATIO"] == 0.5
