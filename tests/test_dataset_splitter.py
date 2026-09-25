"""Unit tests for src/datasets/dataset_splitter.py."""

from __future__ import annotations

import polars as pl
import pytest
from src.datasets.dataset_splitter import DatasetSplitter


def test_dataset_splitter_empty() -> None:
    splitter = DatasetSplitter()
    empty_df = pl.DataFrame()
    res = splitter.split(empty_df)
    assert res.train_df.is_empty()
    assert res.val_df.is_empty()
    assert res.test_df.is_empty()


def test_dataset_splitter_temporal_fallback() -> None:
    df = pl.DataFrame({
        "UC": [f"UC_{i}" for i in range(100)],
        "meta__cutoff_date": [f"2026-01-{i%30+1:02d}" for i in range(100)],
        "x__feature1": [float(i) for i in range(100)],
    })
    splitter = DatasetSplitter(val_ratio=0.15, test_ratio=0.20)
    res = splitter.split(df)

    assert res.train_df.height == 65
    assert res.val_df.height == 15
    assert res.test_df.height == 20
    assert res.train_df.height + res.val_df.height + res.test_df.height == 100


def test_dataset_splitter_feeder_grouping() -> None:
    # 6 feeders in total, 3 in cluster A, 3 in cluster B
    features_data = []
    hierarchy_data = []
    for f_idx in range(6):
        feeder_name = f"FEEDER_{f_idx}"
        for u_idx in range(20):
            uc_id = f"UC_{f_idx}_{u_idx}"
            # Feeder 0,1,2 have high voltage mean, Feeder 3,4,5 have low
            v_val = 230.0 if f_idx < 3 else 110.0
            features_data.append({
                "UC": uc_id,
                "meta__feeder": feeder_name,
                "VOLTAGE_MEAN": v_val,
                "x__val": float(u_idx),
            })
            hierarchy_data.append({
                "UC": uc_id,
                "ALIMENTADOR": feeder_name,
                "POSTO_OPERACIONAL": f"POSTO_{f_idx}_{u_idx//5}",
                "CLASSE_PRINCIPAL": "RESIDENCIAL",
            })

    features_df = pl.DataFrame(features_data)
    hierarchy_df = pl.DataFrame(hierarchy_data)

    splitter = DatasetSplitter(min_cluster_feeders=3)
    res = splitter.split(features_df, hierarchy_df, feeder_col="meta__feeder")

    summary = res.split_summary
    assert "feeder_assignments" in summary
    assignments = summary["feeder_assignments"]

    # Check that test, val, and train got feeders assigned
    assigned_roles = set(assignments.values())
    assert "test" in assigned_roles
    assert "val" in assigned_roles
    assert "train" in assigned_roles

    # Verify zero leakage of test UCs in train set
    test_ucs = set(res.test_df.get_column("UC").to_list())
    train_ucs = set(res.train_df.get_column("UC").to_list())
    assert len(test_ucs.intersection(train_ucs)) == 0
