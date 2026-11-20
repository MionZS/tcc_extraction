"""Automated tests for Cockpit KPI reconciliation and granularity consistency."""

from __future__ import annotations

from datetime import date
import polars as pl
import pytest

from src.eda.cockpit_metrics import (
    compute_reconciled_kpis,
    deduplicate_latest_uc,
    filter_cockpit_dataframe,
)


def _sample_classified_df() -> pl.DataFrame:
    """Create a multi-cutoff classified DataFrame with duplicate UCs across cutoffs."""
    return pl.DataFrame({
        "id__uc_id": ["1001", "1001", "1002", "1002", "1003", "1004", "1004"],
        "meta__cutoff_date": [
            date(2026, 8, 31),
            date(2026, 9, 30),
            date(2026, 8, 31),
            date(2026, 9, 30),
            date(2026, 9, 30),
            date(2026, 8, 31),
            date(2026, 9, 30),
        ],
        "meta__feeder": ["Fonte Nova", "Fonte Nova", "CSN", "CSN", "Polonia", "Fonte Nova", "Fonte Nova"],
        "ANOMALY_SCORE": [0.40, 0.85, 0.60, 0.25, 0.78, 0.30, 0.90],
        "IS_ANOMALY": [False, True, True, False, True, False, True],
        "IS_CANDIDATO_VISTORIA": [False, True, False, False, False, False, True],
        "DIAS_ANOMALIA_SEGUIDOS": [0, 12, 5, 0, 0, 0, 20],
        "CATEGORIA_DIAGNOSTICA": [
            "REGULAR",
            "POSSIVEL_GD_NAO_CADASTRADA",
            "POSSIVEL_GD_NAO_CADASTRADA",
            "REGULAR",
            "TELECOM_QUARANTINE",
            "REGULAR",
            "DESEQUILIBRIO_CARGA_OU_BYPASS",
        ],
    })


def test_deduplicate_latest_uc_keeps_exact_latest_cutoff():
    df = _sample_classified_df()
    uniq = deduplicate_latest_uc(df)
    assert uniq.height == 4
    uc1001 = uniq.filter(pl.col("id__uc_id") == "1001").to_dicts()[0]
    assert uc1001["meta__cutoff_date"] == date(2026, 9, 30)
    assert uc1001["CATEGORIA_DIAGNOSTICA"] == "POSSIVEL_GD_NAO_CADASTRADA"

    # 1002 was GD in August, but REGULAR in September -> latest UC state is REGULAR
    uc1002 = uniq.filter(pl.col("id__uc_id") == "1002").to_dicts()[0]
    assert uc1002["CATEGORIA_DIAGNOSTICA"] == "REGULAR"
    assert uc1002["IS_ANOMALY"] is False


def test_kpis_reconcile_uc_vs_window_counts():
    df = _sample_classified_df()
    kpis = compute_reconciled_kpis(df)

    assert kpis["total_windows"] == 7
    assert kpis["total_unique_ucs"] == 4
    # Windows with anomaly: 1001(Sep), 1002(Aug), 1003(Sep), 1004(Sep) -> 4 windows
    assert kpis["windows_with_anomaly"] == 4
    # Unique UCs with anomaly at latest cutoff: 1001, 1003, 1004 -> 3 UCs
    assert kpis["ucs_with_anomaly"] == 3
    # GD: 2 windows (1001 Sep, 1002 Aug), 1 unique UC at latest cutoff (1001)
    assert kpis["possivel_gd_windows"] == 2
    assert kpis["possivel_gd_ucs"] == 1
    # Telecom: 1 window, 1 UC (1003)
    assert kpis["quarentena_telecom_windows"] == 1
    assert kpis["quarentena_telecom_ucs"] == 1
    # Candidatos: 2 UCs (1001, 1004)
    assert kpis["candidatos_vistoria_ucs"] == 2
    assert kpis["candidatos_vistoria_windows"] == 2


def test_filter_cockpit_matches_reconciled_kpis():
    df = _sample_classified_df()
    kpis = compute_reconciled_kpis(df)

    gd_ucs_df, _ = filter_cockpit_dataframe(df, granularity="unique_ucs", category="POSSIVEL_GD_NAO_CADASTRADA")
    assert gd_ucs_df.height == kpis["possivel_gd_ucs"]

    gd_win_df, _ = filter_cockpit_dataframe(df, granularity="all_windows", category="POSSIVEL_GD_NAO_CADASTRADA")
    assert gd_win_df.height == kpis["possivel_gd_windows"]

    cand_df, sorted_flag = filter_cockpit_dataframe(df, granularity="unique_ucs", category="CANDIDATOS_A_VISTORIA")
    assert sorted_flag is True
    assert cand_df.height == kpis["candidatos_vistoria_ucs"]
    # Sorted by DIAS_ANOMALIA_SEGUIDOS desc -> 1004 (20d) before 1001 (12d)
    assert cand_df["id__uc_id"].to_list() == ["1004", "1001"]
