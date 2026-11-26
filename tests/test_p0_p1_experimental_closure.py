"""Comprehensive verification tests for P0 & P1 experimental closure (Section 23).

Validates:
- Score alignment by UC and detector orientation
- Correlation matrix symmetry, diagonal, and positivity on OOT 25/11
- Ensemble weight normalization and calculation invariants
- Frozen OOT artifacts and zero fit during evaluation
- Eligibility & quarantine partition conservation across 50%-90% gating
- Category consistency (sum=65,686; 6,053 - 16 GD = 6,037)
- Population reconciliation (OOT x Daily Rules exact set identities)
- Controlled perturbation isolation (original series strictly preserved, energy invariance on time shift & shape flatten)
- Multiview contract integrity, LGPD anonymization, and NULL preservation
- Existence and non-zero integrity of all P0 & P1 figures and summary tables
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts.run_controlled_perturbations import (
    _apply_perturbation,
    aggregate_interval_to_daily_vectorized,
)
from src.models.composite import CompositeAnomalyDetector, select_pure_electrical_features
from src.utils.joblib_splitter import load_split_joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P0_DIR = PROJECT_ROOT / "docs" / "tcc_evidence" / "p0_audit"
P1_MV_DIR = PROJECT_ROOT / "docs" / "tcc_evidence" / "p1_multiview"
P1_PERT_DIR = PROJECT_ROOT / "docs" / "tcc_evidence" / "p1_perturbations"
SUITE_PATH = PROJECT_ROOT / "experiments" / "runs" / "latest_suite.json"


def test_p0_score_alignment_orientation_and_correlation_symmetry() -> None:
    """Verify exact score alignment, positive orientation, and symmetric correlation matrices."""
    summary = json.loads((P0_DIR / "p0_summary.json").read_text(encoding="utf-8"))
    corr = summary["correlation_audit"]

    # 1. Exact alignment by UC (max abs diff == 0.0 across all 6 detectors + consensus)
    for k, diff in corr["alignment_max_abs_diff"].items():
        assert diff == pytest.approx(0.0, abs=1e-6), f"Alignment mismatch on {k}: {diff}"

    # 2. Symmetry and unit diagonal
    assert corr["symmetric_spearman_elig"] is True
    assert corr["diagonal_is_one"] is True
    assert corr["any_negative_pairwise"] is False
    assert corr["if_ecod_spearman_elig"] > 0.90

    # 3. Validate CSV matrices directly
    for fname in ("corr_spearman_norm_eligible.csv", "corr_pearson_norm_eligible.csv"):
        df = pl.read_csv(P0_DIR / fname, separator=";")
        mat = df.drop("detector").to_numpy()
        assert mat.shape == (7, 7)
        assert np.allclose(mat, mat.T, atol=1e-4)
        assert np.allclose(np.diag(mat), 1.0, atol=1e-4)
        # All detectors positively oriented with consensus
        assert np.all(mat[:6, 6] > 0.30)


def test_p0_weight_normalization_ensemble_and_stability() -> None:
    """Verify frozen weights sum to 1.0, match code, and ±20% perturbations preserve Top-2000 stability."""
    summary = json.loads((P0_DIR / "p0_summary.json").read_text(encoding="utf-8"))
    w = summary["frozen_model_weights"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert w == {"iso_forest": 0.22, "lof": 0.20, "ecod": 0.18, "copod": 0.18, "hbos": 0.10, "pca": 0.12}

    w_df = pl.read_csv(P0_DIR / "weight_sensitivity.csv", separator=";")
    assert w_df.height >= 20
    # Every weight row sums to 1.0
    w_sums = (
        w_df["w_IF"] + w_df["w_LOF"] + w_df["w_ECOD"]
        + w_df["w_COPOD"] + w_df["w_HBOS"] + w_df["w_PCA"]
    ).to_numpy()
    assert np.allclose(w_sums, 1.0, atol=1e-3)

    # ±20% individual perturbations show high Top-2000 stability (Jaccard >= 0.95)
    pert20 = w_df.filter(pl.col("grupo") == "perturbacao_20pct")
    assert pert20.height == 12
    assert float(pert20["top2k_jaccard"].min()) >= 0.95
    assert float(pert20["spearman_vs_ref"].min()) >= 0.99


def test_p0_frozen_gating_partition_and_population_reconciliation() -> None:
    """Verify exact partition conservation, no-fit frozen gating, and 6053 - 16 = 6037 reconciliation."""
    summary = json.loads((P0_DIR / "p0_summary.json").read_text(encoding="utf-8"))
    rec = summary["reconciliation"]

    assert rec["n_oot"] == 65686
    assert rec["n_oot_elig"] == 62857
    assert rec["n_oot_quar"] == 2829
    assert rec["n_oot_elig"] + rec["n_oot_quar"] == rec["n_oot"]
    assert rec["n_oot_cand"] == 6037
    assert rec["n_oot_elec_anom"] == 6037
    assert rec["n_oot_topo_cand"] == 4311
    assert rec["n_inter_oot_daily"] + rec["n_oot_excl"] == 6037
    assert rec["n_inter_oot_daily"] + rec["n_daily_excl"] == 4978
    assert rec["gd16_outside_queue"]["count"] == 16
    assert rec["gd16_outside_queue"]["s_elec_max"] < rec["gd16_outside_queue"]["frozen_threshold"]

    # Category sum == 65,686 and electrical/GD/metro sum == 6,053
    cat_df = pl.read_csv(P0_DIR / "category_reconciliation.csv", separator=";")
    assert int(cat_df["total_ucs"].sum()) == 65686
    assert int(cat_df["candidatos_vistoria"].sum()) == 6037
    elec_cats = ["ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT", "POSSIVEL_GD_NAO_CADASTRADA", "FALHA_METROLOGICA_MEDIDOR_TRAVADO"]
    assert int(cat_df.filter(pl.col("CATEGORIA_DIAGNOSTICA").is_in(elec_cats))["total_ucs"].sum()) == 6053

    # Frozen gating sensitivity: raw anomalies CONSTANT across all 5 thresholds (proving zero refit on OOT!)
    gat_df = pl.read_csv(P0_DIR / "gating_frozen_sensitivity.csv", separator=";")
    assert gat_df.height == 5
    assert gat_df["raw_anomalies_frozen_model"].n_unique() == 1
    for r in gat_df.to_dicts():
        assert r["eligible_ucs"] + r["quarantine_ucs"] == r["total_ucs"]
        assert r["eligible_ucs"] > r["quarantine_ucs"]  # No inversion at 80% or any threshold!
        assert r["candidates_retained"] + r["anomalies_intercepted_quarantine"] == r["raw_anomalies_frozen_model"]


def test_p0_frozen_oot_no_temporal_leakage() -> None:
    """Confirm OOT frozen pipeline has frozen threshold and training cutoffs strictly precede OOT."""
    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    oot_dir = PROJECT_ROOT / suite["oot_multiview"]
    oot_metrics = json.loads((oot_dir / "metrics.json").read_text(encoding="utf-8"))
    max_train_dt = max(oot_metrics["train_cutoffs"])
    assert max_train_dt < oot_metrics["oot_cutoff"]

    pipe = load_split_joblib(oot_dir / "frozen_multiview_model.joblib")
    assert pipe.elec_detector.detector.is_fitted_ is True
    assert pipe.elec_detector.detector.frozen_threshold_ == pytest.approx(0.513072, abs=1e-4)


def test_p1_controlled_perturbation_isolation_and_energy_invariance() -> None:
    """Ensure perturbation helper never mutates input DataFrame and preserves energy on shift/flatten."""
    # Construct synthetic 2-day 10-min interval frame for 1 UC
    n_pts = 144
    ts_day1 = [date(2026, 11, 4)] * n_pts
    ts_day2 = [date(2026, 11, 6)] * n_pts
    base_fa = np.sin(np.linspace(0, np.pi, n_pts)) * 0.5 + 0.1
    df_orig = pl.DataFrame({
        "UC": ["12345"] * (2 * n_pts),
        "NIO": ["99999"] * (2 * n_pts),
        "REPORT_DAY": ts_day1 + ts_day2,
        "CADENCE_MINUTES": [10] * (2 * n_pts),
        "FA_INTERVAL": list(base_fa) + list(base_fa),
        "RA_INTERVAL": [0.0] * (2 * n_pts),
        "I_L1_AVG": [5.0] * (2 * n_pts),
        "I_L2_AVG": [5.0] * (2 * n_pts),
        "I_L3_AVG": [5.0] * (2 * n_pts),
        "U_L1_AVG": [127.0] * (2 * n_pts),
        "U_L2_AVG": [127.0] * (2 * n_pts),
        "U_L3_AVG": [127.0] * (2 * n_pts),
    }).with_columns(
        (pl.col("REPORT_DAY").cast(pl.Datetime("us")) + pl.duration(minutes=(pl.int_range(0, 2 * n_pts) % n_pts) * 10)).alias("TIMESTAMP_LOCAL")
    )
    orig_sum = float(df_orig["FA_INTERVAL"].sum())

    # 1. Time shift +2h must preserve exact daily FA sum and leave df_orig untouched
    df_shift, _ = _apply_perturbation(df_orig, "time_shift", 2, rng=np.random.default_rng(42))
    assert float(df_orig["FA_INTERVAL"].sum()) == pytest.approx(orig_sum, abs=1e-9)
    assert float(df_shift["FA_INTERVAL"].sum()) == pytest.approx(orig_sum, abs=1e-9)

    # 2. Shape flatten must preserve exact daily FA sum
    df_flat, _ = _apply_perturbation(df_orig, "shape_flatten", 0.80, rng=np.random.default_rng(42))
    assert float(df_flat["FA_INTERVAL"].sum()) == pytest.approx(orig_sum, abs=1e-9)

    # 3. Sustained drop -50% reduces post-onset day by 50% and keeps voltage normal
    df_drop, _ = _apply_perturbation(df_orig, "sustained_drop", 0.50, rng=np.random.default_rng(42))
    d_post = df_drop.filter(pl.col("REPORT_DAY") == date(2026, 11, 6))
    assert float(d_post["FA_INTERVAL"].sum()) == pytest.approx(float(base_fa.sum()) * 0.5, abs=1e-6)
    assert float(d_post["U_L1_AVG"].mean()) == pytest.approx(127.0, abs=1e-6)

    # 4. Comm gap 30% reduces coverage without mutating remaining FA values
    df_gap, _ = _apply_perturbation(df_orig, "comm_gap_random", 0.30, rng=np.random.default_rng(42))
    daily_gap = aggregate_interval_to_daily_vectorized(df_gap)
    assert float(daily_gap["INTERVAL_COVERAGE"].mean()) < 0.85
    assert float(df_orig["FA_INTERVAL"].sum()) == pytest.approx(orig_sum, abs=1e-9)


def test_p1_multiview_contract_lgpd_and_figures_integrity() -> None:
    """Validate P1 multiview summary, LGPD masking on traces, and all 18 required P0+P1 figures."""
    p0_sum = json.loads((P0_DIR / "p0_summary.json").read_text(encoding="utf-8"))
    p1_mv = json.loads((P1_MV_DIR / "p1_multiview_summary.json").read_text(encoding="utf-8"))

    # Check P0 figures (8 figures)
    assert len(p0_sum["figures"]) == 8
    for f_info in p0_sum["figures"].values():
        assert (PROJECT_ROOT / f_info["pdf"]).exists()
        assert (PROJECT_ROOT / f_info["pdf"]).stat().st_size > 5000

    # Check P1 multiview figures (5 figures)
    assert len(p1_mv["figures"]) == 5
    for f_info in p1_mv["figures"].values():
        assert (PROJECT_ROOT / f_info["pdf"]).exists()
        assert (PROJECT_ROOT / f_info["pdf"]).stat().st_size > 5000

    # LGPD check on case studies & contract parquet
    cases_txt = (P1_MV_DIR / "case_studies_multiview.json").read_text(encoding="utf-8").lower()
    for forbidden in ("90850700", "29113296", "68546220", "43570463", "42312054", "43703386", "copel", "polonia", "oleos brasil"):
        assert forbidden not in cases_txt, f"LGPD leak detected: {forbidden}"

    # Verify contract parquet exists in latest p1_multiview run and preserves NULL on ineligible rank
    mv_run_dir = PROJECT_ROOT / "experiments" / "runs" / p1_mv["run_id"]
    contract_p = mv_run_dir / "multiview_evidence_oot.parquet"
    assert contract_p.exists()
    cdf = pl.read_parquet(contract_p)
    assert cdf.height == 65686
    for req_col in ("run_id", "pipeline_id", "uc_id_masked", "cutoff", "coverage", "eligibility",
                    "s_comm", "s_elec", "s_behavior", "s_tamper", "cpd_detected", "physical_rules",
                    "topological_evidence", "priority_score", "rank", "decision_reason"):
        assert req_col in cdf.columns
    # Ineligible UCs MUST have NULL priority_score and NULL rank (not zero!)
    inelig = cdf.filter(~pl.col("eligibility"))
    assert inelig["priority_score"].null_count() == inelig.height
    assert inelig["rank"].null_count() == inelig.height

    # If P1 perturbations completed, check its 6 figures & summary invariants too
    p1_pert_path = P1_PERT_DIR / "p1_perturbations_summary.json"
    if p1_pert_path.exists():
        p1_pert = json.loads(p1_pert_path.read_text(encoding="utf-8"))
        assert p1_pert["sample_selection"]["data_lake_untouched"] is True
        assert len(p1_pert["figures"]) == 6
        for f_info in p1_pert["figures"].values():
            assert (PROJECT_ROOT / f_info["pdf"]).exists()
            assert (PROJECT_ROOT / f_info["pdf"]).stat().st_size > 5000
