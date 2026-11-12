"""Unit tests for src.eda.dataset_eda (synthetic data only)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from src.eda import dataset_eda as eda


def _daily(n_days: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    start = date(2026, 8, 1)
    for uc, level in [("1", 2.0), ("2", 2.2), ("3", 20.0), ("4", 21.0), ("5", 22.0)]:
        for i in range(n_days):
            v = float(rng.normal(level, level * 0.05))
            if uc == "5" and i == 10:
                v = level * 8  # injected spike
            rows.append({"UC": uc, "REPORT_DAY": start + timedelta(days=i),
                         "INTERVAL_COVERAGE": 1.0 if uc != "2" else 0.5, "FA_INTERVAL_SUM": v})
    return pl.DataFrame(rows)


def test_completeness_and_histogram():
    comp = eda.meter_completeness(_daily())
    c = dict(zip(comp["UC"], comp["completeness_calendar"]))
    assert c["1"] == 1.0 and abs(c["2"] - 0.5) < 1e-9
    h = eda.completeness_histogram(comp, step_pct=10)
    assert h["meters"].sum() == 5
    assert h.filter(pl.col("bin") == "90-100%")["meters"][0] == 4  # 100 % lands in last closed bin


def test_new_meter_is_not_penalised_for_the_past():
    d = _daily()
    d = d.filter(~((pl.col("UC") == "1") & (pl.col("REPORT_DAY") < date(2026, 8, 20))))
    comp = eda.meter_completeness(d)
    assert comp.filter(pl.col("UC") == "1")["completeness_calendar"][0] == 1.0


def test_outliers_detect_injected_spike():
    d = _daily()
    bands = eda.consumption_bands(d, n_bands=2)
    out = eda.consumption_outliers(d, bands)
    spike = out.filter((pl.col("UC") == "5") & (pl.col("REPORT_DAY") == date(2026, 8, 11)))
    assert spike["out_self"][0] and spike["out_band_high"][0]
    summ = eda.outlier_summary_by_band(out)
    assert summ["meters"].sum() == 4  # UC 2 has only unreliable days (coverage 0.5) -> excluded


def test_delay_heatmap_and_arrival():
    rows = []
    for w in range(288):
        for cat, lab, s, e, c in [("0-60m", "0-5", 0.0, 5.0, 90), ("60-240m", "60-120", 60.0, 120.0, 10)]:
            rows.append({"dt_referencia": "2026-11-01", "window_idx": w, "bin_category": cat, "bin_label": lab,
                         "bin_start": s, "bin_end": e, "bin_width": e - s, "contagem": c, "soma_delay": c * (s + e) / 2})
    h = pl.DataFrame(rows)
    z, wins, bins = eda.delay_heatmap_matrix(h, agg_minutes=60)
    assert z.shape == (2, 24) and wins[0] == "00:00"
    t = eda.delay_arrival_table(h)
    assert abs(t["cumulative_share"][-1] - 1.0) < 1e-9
    s = eda.daily_delay_summary(h)
    assert abs(s["share_le_1h"][0] - 0.9) < 1e-9

