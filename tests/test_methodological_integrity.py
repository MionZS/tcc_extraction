"""Regression, Invariant & Methodological Integrity Tests for TCC AMI Pipeline."""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.eda.uc_timeseries import load_uc_hourly_timeseries
from src.experiments.contracts import (
    ExperimentConfig,
    InvariantReport,
    RunManifest,
    validate_population_partition,
    validate_portfolio_queue_sets,
    validate_score_and_kpi_integrity,
    validate_temporal_no_leakage,
)
from src.experiments.reporting import generate_all_deliverables
from src.models.composite import CompositeAnomalyDetector
from src.models.specialized_detectors import MultiViewAnomalyPipeline
from src.tcc_evidence.ml_experiments import run_ablation_study


def test_no_or_true_in_codebase() -> None:
    """Reject any invalid `assert ... or True` invariant bypass in src/ or scripts/."""
    for folder in (Path("src"), Path("scripts")):
        for py_file in folder.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert) and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or):
                    for val in node.test.values:
                        if isinstance(val, ast.Constant) and val.value is True:
                            pytest.fail(f"Found invalid `assert ... or True` in {py_file}:{node.lineno}")


def test_population_partition_invariants_and_fail_closed() -> None:
    """Verify mutually exclusive partition conservation and fail-closed on overlap or sum mismatch."""
    valid_res = validate_population_partition([
        {
            "cutoff_date": "2026-11-01",
            "total_count": 5,
            "eligible_count": 3,
            "quarantine_count": 1,
            "other_count": 1,
            "eligible_ucs": {"U1", "U2", "U3"},
            "quarantine_ucs": {"U4"},
            "other_ucs": {"U5"},
        }
    ])
    assert valid_res.passed is True

    # Overlapping UC between eligible & quarantine must fail even if arithmetic looks ok
    overlap_res = validate_population_partition([
        {
            "cutoff_date": "2026-11-01",
            "total_count": 4,
            "eligible_count": 2,
            "quarantine_count": 2,
            "other_count": 0,
            "eligible_ucs": {"U1", "U2"},
            "quarantine_ucs": {"U2", "U3"},
            "other_ucs": set(),
        }
    ])
    assert overlap_res.passed is False
    assert any("Overlap" in e for e in overlap_res.errors)


def test_portfolio_queue_set_identities() -> None:
    """Verify C_t == I_t U P_t, C_{t-1} == O_t U P_t by UC identity, catching arithmetic-only errors."""
    c_map = {
        "2026-11-01": {"U1", "U2", "U3"},
        "2026-11-02": {"U2", "U3", "U4"},  # I = {U4}, O = {U1}, P = {U2, U3}
    }
    good = validate_portfolio_queue_sets(
        c_map,
        reported_rows=[
            {"CUTOFF_DATE": "2026-11-01", "CANDIDATOS_VISTORIA": 3, "FILA_ENTRADAS": 3, "FILA_SAIDAS": 0, "FILA_PERMANENCIA": 0},
            {"CUTOFF_DATE": "2026-11-02", "CANDIDATOS_VISTORIA": 3, "FILA_ENTRADAS": 1, "FILA_SAIDAS": 1, "FILA_PERMANENCIA": 2},
        ],
    )
    assert good.passed is True

    bad = validate_portfolio_queue_sets(
        c_map,
        reported_rows=[
            {"CUTOFF_DATE": "2026-11-01", "CANDIDATOS_VISTORIA": 3, "FILA_ENTRADAS": 3, "FILA_SAIDAS": 0, "FILA_PERMANENCIA": 0},
            {"CUTOFF_DATE": "2026-11-02", "CANDIDATOS_VISTORIA": 3, "FILA_ENTRADAS": 2, "FILA_SAIDAS": 2, "FILA_PERMANENCIA": 1},
        ],
    )
    assert bad.passed is False


def test_cadence_aware_hourly_coverage(tmp_path: Path) -> None:
    """Verify 10-minute cadence meters report 100% cadence coverage (6 readings/h) and 50% raw 5-min coverage."""
    base_dir = tmp_path / "output"
    part_dir = base_dir / "measurements" / "ami_interval" / "report_year=2026" / "report_month=11" / "report_day=01"
    part_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = base_dir / "features" / "uc_day_features" / "feature_set_version=v1" / "report_year=2026" / "report_month=11" / "report_day=01"
    feat_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir = base_dir / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Meter installation mapping UC_10M -> NIO_10M
    pl.DataFrame({
        "UC": ["29113296"],
        "NIO": ["42312054"],
        "DATA_INSTALACAO": [date(2026, 1, 1)],
    }).write_parquet(ctx_dir / "meter_installation_history.parquet")

    # 24h of 10-minute intervals = 6 rows per hour * 24h = 144 rows
    base_ts = datetime(2026, 11, 1, 0, 0, 0)
    timestamps = [base_ts + timedelta(minutes=10 * k) for k in range(144)]
    pl.DataFrame({
        "NIO": ["42312054"] * 144,
        "TIMESTAMP_LOCAL": timestamps,
        "FA_INTERVAL": [0.15] * 144,
        "RA_INTERVAL": [0.0] * 144,
        "CADENCE_MINUTES": [10] * 144,
    }).write_parquet(part_dir / "data.parquet")

    pl.DataFrame({
        "UC": ["29113296"],
        "NIO": ["42312054"],
        "REPORT_DAY": [date(2026, 11, 1)],
        "INTERVAL_COVERAGE": [1.0],
        "FA_INTERVAL_SUM": [21.6],
    }).write_parquet(feat_dir / "data.parquet")

    res = load_uc_hourly_timeseries("29113296", base_dir=base_dir, week_days=1, use_cache=False)
    assert res["status"] == "ok"
    assert res["cadence_minutes"] == 10.0
    assert len(res["points"]) == 24
    # Every hour has 6 intervals -> 100% cadence coverage, 50% vs legacy 5-min divisor
    assert all(pt["coverage"] == 1.0 for pt in res["points"])
    assert all(pt["raw_5min_coverage"] == 0.5 for pt in res["points"])
    assert all(pt["flag"] != "GAP_TELECOM" for pt in res["points"])


def test_frozen_normalization_eliminates_batch_shift() -> None:
    """Prove frozen_quantile and frozen_ecdf eliminate the batch_minmax extreme-trim shift."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(loc=20.0, scale=2.0, size=(200, 4))
    X_eval = np.vstack([
        rng.normal(loc=20.0, scale=2.0, size=(95, 4)),
        rng.normal(loc=250.0, scale=5.0, size=(5, 4)),  # extreme outliers in eval batch
    ])

    # Legacy batch_minmax shifts remaining points when extremes are removed
    det_batch = CompositeAnomalyDetector(contamination=0.05, normalization_mode="batch_minmax", random_state=42).fit(X_train)
    sc_b_full = det_batch.score_samples(X_eval)
    keep_mask = np.arange(len(X_eval)) < 95
    sc_b_trim = det_batch.score_samples(X_eval[keep_mask])
    batch_max_shift = float(np.max(np.abs(sc_b_full[keep_mask] - sc_b_trim)))
    assert batch_max_shift > 0.10

    # Frozen quantile & ECDF produce strictly 0.0 shift on all remaining samples
    for mode in ("frozen_quantile", "frozen_ecdf"):
        det_f = CompositeAnomalyDetector(contamination=0.05, normalization_mode=mode, random_state=42).fit(X_train)
        sc_f_full = det_f.score_samples(X_eval)
        sc_f_trim = det_f.score_samples(X_eval[keep_mask])
        assert float(np.max(np.abs(sc_f_full[keep_mask] - sc_f_trim))) == 0.0


def test_oot_frozen_pipeline_produces_valid_candidates() -> None:
    """Regression test for OOT=0 bug: frozen MultiView fit(train) -> score(oot) yields non-empty candidates & aligned taxonomy."""
    rng = np.random.default_rng(7)
    X_tr_comm = rng.uniform(0.85, 1.0, size=(80, 2))
    X_tr_elec = rng.normal(loc=20.0, scale=2.0, size=(80, 4))

    pipe = MultiViewAnomalyPipeline(contamination=0.05, min_quality_threshold=0.70, random_state=42)
    pipe.fit(X_comm=X_tr_comm, X_elec=X_tr_elec)

    # OOT with 15 normal + 5 electrical outliers + 2 degraded telecom
    X_ev_comm = np.vstack([rng.uniform(0.85, 1.0, size=(20, 2)), np.array([[0.1, 0.1], [0.2, 0.2]])])
    X_ev_elec = np.vstack([
        rng.normal(loc=20.0, scale=2.0, size=(15, 4)),
        rng.normal(loc=150.0, scale=5.0, size=(5, 4)),
        rng.normal(loc=150.0, scale=5.0, size=(2, 4)),  # degraded comms must NOT trigger electrical fraud
    ])
    dq = [0.95] * 20 + [0.20, 0.30]
    raw_df = pl.DataFrame({"id__uc_id": [f"UC_{i}" for i in range(22)]})

    scored = pipe.score(
        X_comm=X_ev_comm,
        X_elec=X_ev_elec,
        data_quality=dq,
        tamper_alarms=[0] * 22,
        raw_dataframe=raw_df,
    )
    assert "id__uc_id" in scored.columns
    assert "ANOMALY_SCORE" in scored.columns
    assert "IS_ANOMALY" in scored.columns
    # No "ANOMALIA_ELETRICA_CONFIRMADA" in telemetry labels
    assert not any("CONFIRMADA" in d for d in scored["operational_diagnosis"].to_list())
    # Clean electrical outliers detected, degraded ones gated out
    assert int(scored["is_elec_anomaly"][15:20].sum()) >= 3
    assert int(scored["is_elec_anomaly"][20:22].sum()) == 0


def test_ablation_strict_funnel_inclusion(tmp_path: Path) -> None:
    """Regression test for M4=242 vs M5=274: strict funnel must enforce M5 subset of M4 and M4 <= M3 <= M2 <= M1."""
    df = pl.DataFrame({
        "id__uc_id": ["U1", "U2", "U3", "U4", "U5"],
        "meta__cutoff_date": [date(2026, 11, 19)] * 5,
        "meta__feeder": ["F1", "F1", "F1", None, "F1"],
        "x__avg_interval_coverage": [0.95, 0.90, 0.65, 0.92, 0.88],  # U3 < 0.70!
        "x__hist_fa_sum_median": [50.0, 40.0, 30.0, 25.0, 12.0],
        "IS_ANOMALY": [True, False, True, True, True],
        "IS_CANDIDATO_VISTORIA": [True, False, True, True, False],  # U3 has cov=0.65, U4 has null feeder
    })
    res = run_ablation_study(tmp_path, classified_df=df)
    counts = {r["stage"].split(":")[0]: r["count"] for r in res["stages"]}
    assert counts["M1"] >= counts["M2"] >= counts["M3"] >= counts["M4"] >= counts["M5"]
    assert res["set_reconciliation"]["strict_sequential_inclusion"] is True


def test_temporal_no_leakage_validator() -> None:
    """Verify anti-leakage blocks train >= eval and future observations inside windows."""
    bad = validate_temporal_no_leakage(
        train_max_date=date(2026, 11, 5),
        eval_start_date=date(2026, 11, 1),
    )
    assert bad.passed is False

    leak_df = pl.DataFrame({
        "CUTOFF_DATE": [date(2026, 11, 10)],
        "REPORT_DAY": [date(2026, 11, 12)],
    })
    bad_win = validate_temporal_no_leakage(
        train_max_date=date(2026, 10, 31),
        eval_start_date=date(2026, 11, 1),
        window_df=leak_df,
    )
    assert bad_win.passed is False


def test_manifest_fail_closed_blocks_reporting(tmp_path: Path) -> None:
    """Verify RunManifest sets status='invalid' when invariants fail and reporting blocks publication."""
    cfg = ExperimentConfig(
        experiment_id="test_exp",
        data_source="local_lake",
        scope="test",
        start_date="2026-11-01",
        end_date="2026-11-10",
        timezone="America/Sao_Paulo",
        unit_of_analysis="uc_cutoff",
        pipeline_id="daily_rules",
    )
    run_dir = tmp_path / "run_bad"
    man = RunManifest(run_dir, cfg)
    man.transition("running")

    bad_kpi = validate_score_and_kpi_integrity(np.array([0.2, 1.5]))  # 1.5 outside [0, 1]
    rep = InvariantReport(checks=[bad_kpi])
    ok = man.finalize_with_invariants(rep)
    assert ok is False
    saved = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "invalid"
    assert saved["invariants_verified"] is False

    with pytest.raises(RuntimeError, match="Publish blocked"):
        generate_all_deliverables({"daily_rules": run_dir})


def test_real_november_suite_artifacts_and_uc_90850700_reconciled() -> None:
    """Verify the validated November 2026 suite artifacts, 15 Markdown deliverables, and UC 90850700 reconciliation."""
    suite_idx = Path("experiments/runs/latest_suite.json")
    assert suite_idx.exists(), "Run scripts/run_tcc_experiments.py all first."
    run_map = json.loads(suite_idx.read_text(encoding="utf-8"))
    for key, rpath in run_map.items():
        man = json.loads((Path(rpath) / "manifest.json").read_text(encoding="utf-8"))
        assert man["status"] == "validated", f"Run {key} not validated: {man}"
        assert man["invariants_verified"] is True

    expected_docs = [
        "docs/methodology/DATA_CONTRACTS.md",
        "docs/methodology/EXPERIMENT_DEFINITIONS.md",
        "docs/methodology/SCORE_CONTRACTS.md",
        "docs/methodology/DECISION_POLICY.md",
        "docs/methodology/TEMPORAL_VALIDATION.md",
        "docs/tcc_evidence/AUDITORIA_NOVEMBRO.md",
        "docs/tcc_evidence/CASE_TRACES.md",
        "docs/tcc_evidence/ABLATION_RESULTS.md",
        "docs/tcc_evidence/GATING_SENSITIVITY.md",
        "docs/tcc_evidence/SCORE_STABILITY.md",
        "docs/tcc_evidence/COMPARISON_PROTOCOL.md",
        "docs/tcc_evidence/DATA_PROVENANCE.md",
        "docs/CHANGELOG_METODOLOGICO.md",
        "docs/PENDENCIAS_EXTERNAS.md",
        "docs/GUIA_REPRODUCAO_EXPERIMENTOS.md",
    ]
    for doc in expected_docs:
        p = Path(doc)
        assert p.exists() and p.stat().st_size > 100, f"Missing or empty canonical doc: {doc}"

    # Verify restored classified_ucs_latest schema and UC 90850700 reconciliation
    clf_df = pl.read_parquet("output/reports/classified_ucs_latest.parquet")
    for req_col in ("id__uc_id", "meta__cutoff_date", "meta__feeder", "x__hist_fa_sum_median", "ANOMALY_SCORE", "CATEGORIA_DIAGNOSTICA", "IS_CANDIDATO_VISTORIA", "DIAS_ANOMALIA_SEGUIDOS"):
        assert req_col in clf_df.columns, f"Missing column {req_col} in classified_ucs_latest.parquet"

    u_df = clf_df.filter(pl.col("id__uc_id") == "90850700").sort("meta__cutoff_date")
    assert u_df.height >= 4
    aug_row = u_df.filter(pl.col("meta__cutoff_date") == date(2026, 8, 31)).to_dicts()[0]
    oct_row = u_df.filter(pl.col("meta__cutoff_date") == date(2026, 10, 31)).to_dicts()[0]
    nov_row = u_df.filter(pl.col("meta__cutoff_date") == date(2026, 11, 19)).to_dicts()[0]

    assert aug_row["x__hist_fa_sum_median"] == pytest.approx(17.407, abs=0.05)
    assert int(aug_row["x__hist_ra_reversal_days"]) == 0
    assert oct_row["IS_CANDIDATO_VISTORIA"] is True
    assert oct_row["CATEGORIA_DIAGNOSTICA"] == "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT"
    assert nov_row["x__hist_fa_sum_median"] == pytest.approx(5.717, abs=0.05)
    assert int(nov_row["x__hist_ra_reversal_days"]) == 12
    assert float(nov_row["s_elec"]) > 0.90
    assert int(nov_row["x__meter_age_days"]) == 19
    assert nov_row["CATEGORIA_DIAGNOSTICA"] == "POSSIVEL_GD_NAO_CADASTRADA"
    assert nov_row["IS_CANDIDATO_VISTORIA"] is True
    assert "INJECAO_ATIVA_REVERSA_RA" in str(nov_row["SUBTIPO_DIAGNOSTICO"])

    # In all cutoffs, persistence is strictly bounded inside the 30-day window (never the legacy 93-day bug)
    for row in u_df.to_dicts():
        assert 0 <= int(row["DIAS_ANOMALIA_SEGUIDOS"]) <= 30


def test_contracts_reconciliation_and_fixed_capacity_ablation() -> None:
    """Verify exact numerical contracts: 6.037 OOT vs 5.788 Gating-70% (+249), 1.726 outside M3, and fixed Top-2k ablation."""
    suite_idx = Path("experiments/runs/latest_suite.json")
    assert suite_idx.exists()
    run_map = json.loads(suite_idx.read_text(encoding="utf-8"))

    oot_m = json.loads((Path(run_map["oot_multiview"]) / "metrics.json").read_text(encoding="utf-8"))
    abl_m = json.loads((Path(run_map["ablation"]) / "metrics.json").read_text(encoding="utf-8"))
    gate_m = json.loads((Path(run_map["gating"]) / "metrics.json").read_text(encoding="utf-8"))

    # 1. Total OOT candidates = 6,037 (post-c1539b4 LOF/PCA sanitization)
    assert oot_m["oot_candidates_vistoria"] == 6037

    # 2. Gating at 70% = 5,788 (net difference of 249 UCs)
    gate_70 = next(r for r in gate_m["rows"] if abs(r["coverage_threshold"] - 0.70) < 1e-4)
    assert gate_70["candidates_retained"] == 5788
    assert oot_m["oot_candidates_vistoria"] - gate_70["candidates_retained"] == 249

    # 3. Exactly 1,726 candidates outside topological funnel due to null feeder in corporate hierarchy
    set_recon = abl_m["set_reconciliation"]
    assert set_recon["strict_m5_subset_of_m4"] is True
    assert set_recon["m4_count"] == 4311
    assert set_recon["m5_sequential_count"] == 4311
    assert set_recon["exclusive_m5_unconstrained"] == 1726
    assert 6037 - 4311 == 1726

    # 4. Fixed capacity Top-2k metrics present in subdetector ablation
    sub_abl = abl_m["subdetector_ablation"]
    for row in sub_abl:
        assert "spearman_vs_full6" in row
        assert "top5pct_jaccard_vs_full6" in row
        assert "p95_score" in row


def test_article_and_evidence_numbers_match_active_suite_metrics() -> None:
    """Automated check of every number, table, metric, and neutral phrasing in the article against run artifacts."""
    from src.experiments.article_audit import audit_article_against_metrics

    errors = audit_article_against_metrics()
    assert errors == [], f"Found metric discrepancies in article text: {errors}"



