"""Unit tests for retroactive sanitation, NIO<->UC reconciliation, and decoupled windowing."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from src.datasets.dataset_splitter import DatasetSplitter
from src.datasets.partitioning import (
    ensure_partition_dirs,
    feature_partition_path,
    measurements_partition_path,
)
from src.datasets.retroactive_cleaner import (
    RetroactiveDataCleaner,
    select_non_overlapping_cutoffs,
)


def test_select_non_overlapping_cutoffs_monthly_and_biweekly() -> None:
    days = [
        date(2026, 8, 5),
        date(2026, 8, 14),
        date(2026, 8, 29),
        date(2026, 9, 10),
        date(2026, 9, 30),
        date(2026, 10, 12),
        date(2026, 10, 28),
    ]

    monthly_cutoffs, w_m = select_non_overlapping_cutoffs(days, stride_mode="monthly")
    assert w_m == 30
    assert monthly_cutoffs == [date(2026, 8, 29), date(2026, 9, 30), date(2026, 10, 28)]

    biweekly_cutoffs, w_b = select_non_overlapping_cutoffs(days, stride_mode="biweekly")
    assert w_b == 15
    assert biweekly_cutoffs == [
        date(2026, 8, 14),
        date(2026, 8, 29),
        date(2026, 9, 10),
        date(2026, 9, 30),
        date(2026, 10, 12),
        date(2026, 10, 28),
    ]


def test_retroactive_cleaner_reconciles_nio_filters_noise_and_quarantines_mesh(
    tmp_path: Path,
) -> None:
    """Verifies end-to-end sanitation over CIS 10-digit NIO vs MDM 8-digit NIO and RF Mesh drops."""
    base_dir = tmp_path / "output"
    ctx_dir = base_dir / "context"
    ensure_partition_dirs(ctx_dir)

    # CIS/GEO context stores 10-digit zero-padded NIOs ("0043309810", "0043309811")
    pl.DataFrame({
        "UC": ["10000070", "10000071"],
        "TENSAO_BASE": [127.0, 127.0],
        "CLASSE_PRINCIPAL": ["RESIDENCIAL", "RESIDENCIAL"],
        "FASE_CIRCUITO": ["BI", "BI"],
    }).write_parquet(ctx_dir / "uc_context.parquet")

    pl.DataFrame({
        "UC": ["10000070", "10000071"],
        "NIO": ["0043309810", "0043309811"],
        "DATA_INSTALACAO": [date(2025, 1, 15), date(2025, 3, 10)],
        "DATA_RETIRADA": [None, None],
    }).write_parquet(ctx_dir / "meter_installation_history.parquet")

    pl.DataFrame({
        "UC": ["10000070", "10000071"],
        "NIO": ["0043309810", "0043309811"],
        "ALIMENTADOR": ["Fonte Nova", "Angelica"],
        "POSTO_OPERACIONAL": ["TR_001", "TR_002"],
        "MUNICIPIO_SE": ["ARAUCARIA", "ARAUCARIA"],
    }).write_parquet(ctx_dir / "electrical_hierarchy.parquet")

    # Simulate 3 months (Aug, Sep, Oct) with 4 extracted days per month
    test_days = [
        date(2026, 8, 10),
        date(2026, 8, 15),
        date(2026, 8, 20),
        date(2026, 8, 28),
        date(2026, 9, 10),
        date(2026, 9, 15),
        date(2026, 9, 20),
        date(2026, 9, 28),
        date(2026, 10, 10),
        date(2026, 10, 15),
        date(2026, 10, 20),
        date(2026, 10, 28),
    ]

    for idx_d, d in enumerate(test_days):
        int_dir = measurements_partition_path("ami_interval", d, base_dir=base_dir)
        inst_dir = measurements_partition_path("ami_instantaneous", d, base_dir=base_dir)
        ensure_partition_dirs(int_dir)
        ensure_partition_dirs(inst_dir)

        # Meter 43309810 (8-digit stripped NIO -> UC 10000070):
        # 250 slots (>= 0.70 coverage) on 3 of 4 days, and 1 degraded RF Mesh day (only 10 slots = 3.5% cov)
        is_degraded_day = (idx_d % 4) == 1
        n_slots_clean_uc = 10 if is_degraded_day else 250
        daily_slot_kwh = 0.001 if is_degraded_day else (0.04 + 0.005 * (idx_d % 4))

        # Meter 43309811 (8-digit stripped NIO -> UC 10000071):
        # Chronic silent/intermittent RF Mesh node (only 12 slots every day -> coverage ~4.2%)
        n_slots_mesh_uc = 12

        int_rows = []
        for s in range(n_slots_clean_uc):
            int_rows.append({
                "NIO": "43309810",
                "READ_TIMESTAMP": datetime(d.year, d.month, d.day, (s * 5) // 60, (s * 5) % 60),
                "FA": daily_slot_kwh,
                "RA": 0.0,
                "FR": 0.01,
                "U_L1_AVG": 127.2,
                "U_L2_AVG": 126.8,
                "U_L3_AVG": 0.0,  # Unenergized L3 on BI meter (must NOT trigger 100% voltage unbalance)
            })
        for s in range(n_slots_mesh_uc):
            int_rows.append({
                "NIO": "43309811",
                "READ_TIMESTAMP": datetime(d.year, d.month, d.day, 0, s * 5),
                "FA": 0.05,
                "RA": 0.0,
                "FR": 0.01,
                "U_L1_AVG": 126.5,
                "U_L2_AVG": 126.0,
                "U_L3_AVG": 0.0,
            })
        pl.DataFrame(int_rows).write_parquet(int_dir / "data.parquet")

        # Instantaneous readings: include nocturnal standby (I=0.2A,0.0A -> must be ignored)
        # and loaded regime (I=10.0A, 9.6A -> small realistic unbalance)
        inst_rows = [
            {
                "NIO": "43309810",
                "READ_TIMESTAMP": datetime(d.year, d.month, d.day, 3, 0),
                "U_L1": 127.0,
                "U_L2": 126.8,
                "U_L3": 0.0,  # Dropout/unenergized channel
                "I_L1": 0.25,
                "I_L2": 0.0,  # Standby < 1.0A -> must NOT saturate unbalance to 1.0
                "I_L3": 0.0,
            },
            {
                "NIO": "43309810",
                "READ_TIMESTAMP": datetime(d.year, d.month, d.day, 19, 0),
                "U_L1": 127.0,
                "U_L2": 126.6,
                "U_L3": 0.0,
                "I_L1": 10.0,
                "I_L2": 9.6,
                "I_L3": 0.0,
            },
        ]
        pl.DataFrame(inst_rows).write_parquet(inst_dir / "data.parquet")

    # Inject a broken legacy cached partition on 2026-10-28 where UC == NIO ("43309810")
    broken_f_dir = feature_partition_path("v1", date(2026, 10, 28), base_dir=base_dir)
    ensure_partition_dirs(broken_f_dir)
    pl.DataFrame({
        "UC": ["43309810"],
        "NIO": ["43309810"],
        "REPORT_DAY": [date(2026, 10, 28)],
        "FA_INTERVAL_SUM": [0.0],
    }).write_parquet(broken_f_dir / "data.parquet")

    cleaner = RetroactiveDataCleaner(base_dir=base_dir, min_day_coverage=0.70, min_window_coverage=0.60)
    rep = cleaner.run(stride_mode="monthly", write_csv=True)

    assert rep.cutoff_dates == (date(2026, 8, 28), date(2026, 9, 28), date(2026, 10, 28))
    assert rep.total_uc_windows == 6  # 2 UCs x 3 monthly cutoffs
    assert rep.reconciled_feeder_ratio == 1.0
    assert rep.reconciled_context_ratio == 1.0
    assert rep.clean_train_eligible_rows == 3  # UC 10000070 across 3 months
    assert rep.telecom_quarantine_rows == 3    # UC 10000071 quarantined across 3 months
    assert rep.quarantine_path is not None and rep.quarantine_path.exists()

    ds = pl.read_parquet(rep.dataset_path)
    uc70 = ds.filter(pl.col("id__uc_id") == "10000070")
    uc71 = ds.filter(pl.col("id__uc_id") == "10000071")

    assert uc70.height == 3
    assert uc70.get_column("meta__feeder").to_list() == ["Fonte Nova"] * 3
    assert all(uc70.get_column("meta__is_clean_for_train").to_list())
    # Real intra-window dispersion (MAD > 0) and sanitized unbalances (< 0.10, never 1.0 / 100%)
    assert all(v > 0.0 for v in uc70.get_column("x__hist_fa_sum_mad").to_list())
    assert all(v < 5.0 for v in uc70.get_column("x__hist_voltage_imbalance_max").to_list())
    assert all(v < 0.15 for v in uc70.get_column("x__hist_current_imbalance_max").to_list())

    # Quarantined RF Mesh meter must have IS_CLEAN_FOR_TRAIN == False and realistic low coverage
    assert uc71.height == 3
    assert uc71.get_column("meta__feeder").to_list() == ["Angelica"] * 3
    assert not any(uc71.get_column("meta__is_clean_for_train").to_list())
    assert all(v < 0.10 for v in uc71.get_column("x__avg_interval_coverage").to_list())

    # Verify DatasetSplitter intra-canonical OOT split (Aug+Sep -> Train, Oct -> Val/Test across all feeders)
    splitter = DatasetSplitter()
    split_res = splitter.split(ds)
    assert split_res.split_summary["strategy"] == "intra_canonical_multi_window_oot"
    assert split_res.train_df.height == 4  # Aug + Sep for both UCs
    assert split_res.val_df.height + split_res.test_df.height == 2  # Oct held out OOT

