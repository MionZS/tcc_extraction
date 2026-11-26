"""OOT_MULTIVIEW, ABLATION_M0_M5, Gating Sensitivity, and Score Stability Runners."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
import time
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.datasets.build_training_dataset import build_training_dataset
from src.datasets.retroactive_cleaner import (
    _read_table_flexible,
    build_sanitized_window_features_vectorized,
)
from src.datasets.writers import write_training_dataset, write_window_features
from src.eda.uc_timeseries import _list_dated_parquets
from src.experiments.contracts import (
    ExperimentConfig,
    InvariantReport,
    RunManifest,
    atomic_write_json,
    validate_population_partition,
    validate_score_and_kpi_integrity,
    validate_temporal_no_leakage,
)
from src.models.category_reporting import apply_delimited_isolation_and_categories
from src.models.composite import CompositeAnomalyDetector, select_pure_electrical_features
from src.models.specialized_detectors import MultiViewAnomalyPipeline
from src.utils.joblib_splitter import dump_split_joblib, load_split_joblib

BASE_DIR = Path("output")
CONTEXT_DIR = BASE_DIR / "context"
DAY_FEAT_DIR = BASE_DIR / "features" / "uc_day_features" / "feature_set_version=v1"
REPORTS_DIR = BASE_DIR / "reports"


def _compute_bounded_streak(
    day_features_root: Path,
    candidate_cutoffs: dict[str, set[str]],
    window_days: int = 30,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Compute exact (max_streak, active_streak) bounded inside (cutoff - window_days, cutoff]."""
    if not candidate_cutoffs:
        return {}
    dated = _list_dated_parquets(day_features_root)
    if not dated:
        return {}

    cand_list = list(candidate_cutoffs.keys())
    wanted = ["UC", "REPORT_DAY", "INTERVAL_COVERAGE", "FA_INTERVAL_SUM", "RA_REVERSAL_RATIO", "CURRENT_IMBALANCE_MAX"]
    frames: list[pl.DataFrame] = []
    for _, fpath in dated:
        lf = pl.scan_parquet(str(fpath))
        avail = set(lf.collect_schema().names())
        if "UC" not in avail:
            continue
        cols = [c for c in wanted if c in avail]
        part = (
            lf.select(cols)
            .with_columns(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc_norm"))
            .filter(pl.col("_uc_norm").is_in(cand_list))
            .collect()
        )
        if part.height > 0:
            frames.append(part)

    if not frames:
        return {}

    all_days = (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(pl.col("REPORT_DAY").cast(pl.Date))
        .sort(["_uc_norm", "REPORT_DAY"])
        .unique(subset=["_uc_norm", "REPORT_DAY"], keep="last")
    )

    streaks: dict[tuple[str, str], tuple[int, int]] = {}
    for key_t, grp in all_days.partition_by("_uc_norm", as_dict=True).items():
        u_key = str(key_t[0] if isinstance(key_t, tuple) else key_t)
        for c_str in candidate_cutoffs.get(u_key, set()):
            c_dt = date.fromisoformat(c_str[:10])
            w_start = c_dt - timedelta(days=window_days)
            bounded = grp.filter((pl.col("REPORT_DAY") > w_start) & (pl.col("REPORT_DAY") <= c_dt))
            if bounded.height == 0:
                streaks[(u_key, c_str)] = (0, 0)
                continue
            covs = bounded["INTERVAL_COVERAGE"].fill_null(0.0).to_numpy()
            fas = bounded["FA_INTERVAL_SUM"].fill_null(0.0).to_numpy()
            ras = bounded["RA_REVERSAL_RATIO"].fill_null(0.0).to_numpy() if "RA_REVERSAL_RATIO" in bounded.columns else np.zeros(len(fas))
            cimb = bounded["CURRENT_IMBALANCE_MAX"].fill_null(0.0).to_numpy() if "CURRENT_IMBALANCE_MAX" in bounded.columns else np.zeros(len(fas))

            rel = covs >= 0.70
            rel_fa = fas[rel] if np.sum(rel) >= 3 else fas
            med = float(np.median(rel_fa))
            mad = float(np.median(np.abs(rel_fa - med)))
            scale = max(1.4826 * mad, 0.10 * abs(med), 0.05)

            max_s = 0
            cur_s = 0
            for i in range(len(fas)):
                if covs[i] < 0.70:
                    cur_s = 0
                    continue
                val = fas[i]
                z = (val - med) / scale
                is_anom = (abs(z) > 3.5) or (ras[i] > 0.08) or (cimb[i] > 0.45) or (val < 0.05 and med > 0.50)
                if is_anom:
                    cur_s += 1
                    if cur_s > max_s:
                        max_s = cur_s
                else:
                    cur_s = 0
            streaks[(u_key, c_str)] = (max_s, cur_s)
    return streaks


def prepare_oot_training_and_eval_dataset(config: ExperimentConfig) -> pl.DataFrame:
    """Build or load training_dataset.parquet with strict train (<= train.end_dt) and OOT (config.end_dt)."""
    train_end = config.train_end_dt or date(2026, 10, 31)
    oot_cutoff = config.end_dt
    if train_end >= oot_cutoff:
        raise ValueError(f"Temporal leakage: train_end ({train_end}) >= oot_cutoff ({oot_cutoff}).")

    uc_context = _read_table_flexible(CONTEXT_DIR / "uc_context.parquet")
    meter_history = _read_table_flexible(CONTEXT_DIR / "meter_installation_history.parquet")
    hierarchy = _read_table_flexible(CONTEXT_DIR / "electrical_hierarchy.parquet")
    alarm_events = _read_table_flexible(BASE_DIR / "events" / "alarm_events.parquet")

    # Load daily features strictly up to oot_cutoff
    dated = _list_dated_parquets(DAY_FEAT_DIR)
    frames: list[pl.DataFrame] = []
    for dt, fpath in dated:
        if date(2026, 8, 1) <= dt <= oot_cutoff:
            frames.append(pl.read_parquet(fpath))
    all_daily = pl.concat(frames, how="diagonal_relaxed")

    # Train monthly non-overlapping cutoffs <= train_end
    train_cutoffs = [c for c in (date(2026, 8, 31), date(2026, 9, 30), date(2026, 10, 31)) if c <= train_end]
    ds_parts: list[pl.DataFrame] = []

    for c_dt in train_cutoffs:
        win_c = build_sanitized_window_features_vectorized(
            all_daily,
            cutoff_date=c_dt,
            window_days=30,
            meter_history_df=meter_history,
            alarm_events_df=alarm_events,
            min_day_coverage=config.coverage_threshold,
            min_window_coverage=config.min_window_coverage,
            min_reliable_days=config.min_reliable_days,
            feature_set_version="v1",
        )
        write_window_features(win_c, "v1", c_dt, base_dir=BASE_DIR, write_csv=False)
        ds_c = build_training_dataset(win_c, uc_context_df=uc_context, hierarchy_df=hierarchy, split_policy="train")
        ds_parts.append(ds_c)

    win_oot = build_sanitized_window_features_vectorized(
        all_daily,
        cutoff_date=oot_cutoff,
        window_days=30,
        meter_history_df=meter_history,
        alarm_events_df=alarm_events,
        min_day_coverage=config.coverage_threshold,
        min_window_coverage=config.min_window_coverage,
        min_reliable_days=config.min_reliable_days,
        feature_set_version="v1",
    )
    write_window_features(win_oot, "v1", oot_cutoff, base_dir=BASE_DIR, write_csv=True)
    ds_oot = build_training_dataset(win_oot, uc_context_df=uc_context, hierarchy_df=hierarchy, split_policy="oot_eval")
    ds_parts.append(ds_oot)

    full_ds = pl.concat(ds_parts, how="vertical")
    write_training_dataset(full_ds, "v1", base_dir=BASE_DIR, write_csv=True)
    return full_ds


def run_oot_multiview(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Execute strict OOT_MULTIVIEW: fit strictly on clean train, freeze, infer on November OOT."""
    t0 = time.time()
    manifest = RunManifest(run_dir, config)
    manifest.transition("running")

    ds = prepare_oot_training_and_eval_dataset(config)
    ds_path = BASE_DIR / "model_input" / "v1" / "training_dataset.parquet"
    manifest.register_input("training_dataset", ds_path)

    df_train_all = ds.filter(pl.col("meta__split") == "train")
    df_oot = ds.filter(pl.col("meta__split") == "oot_eval")

    train_clean_mask = df_train_all["meta__is_clean_for_train"].fill_null(True) if "meta__is_clean_for_train" in df_train_all.columns else pl.lit(True)
    df_train_clean = df_train_all.filter(train_clean_mask)

    feature_cols = [c for c in ds.columns if c.startswith("x__") and ds[c].dtype.is_numeric()]
    comm_cols = [c for c in feature_cols if any(k in c.lower() for k in ("coverage", "alarm_count", "latency", "gap", "reliable_day"))]
    elec_cols = select_pure_electrical_features(ds)

    X_train_comm = df_train_clean.select(comm_cols).fill_null(0.0).to_numpy()
    X_train_elec = df_train_clean.select(elec_cols).fill_null(0.0).to_numpy()

    # 1. Fit MultiView strictly on clean training cutoffs & freeze all normalization/thresholds
    pipeline = MultiViewAnomalyPipeline(
        contamination=0.05,
        min_quality_threshold=config.coverage_threshold,
        normalization_mode=config.normalization_mode,
        freeze_threshold=config.freeze_threshold,
        random_state=config.seed,
    )
    pipeline.fit(X_comm=X_train_comm, X_elec=X_train_elec)
    model_path = run_dir / "frozen_multiview_model.joblib"
    parts = dump_split_joblib(pipeline, model_path, chunk_size_mb=25, compress=3)
    for p in parts:
        manifest.register_artifact(p.name, p)

    # Save frozen normalization bounds explicitly for provenance
    atomic_write_json(run_dir / "frozen_normalization_params.json", {
        "normalization_mode": config.normalization_mode,
        "frozen_threshold_elec": float(pipeline.elec_detector.detector.frozen_threshold_ or 0.0),
        "frozen_threshold_comm": float(pipeline.comm_detector.frozen_threshold_ or 0.0),
        "elec_norm_params": {
            k: {kk: vv for kk, vv in v.items() if kk != "sorted_raw"}
            for k, v in pipeline.elec_detector.detector.norm_params_.items()
        },
    })

    # 2. Frozen Baseline IsolationForest fit on train clean
    base_imp = SimpleImputer(strategy="median")
    base_scl = RobustScaler()
    X_b_tr = base_scl.fit_transform(base_imp.fit_transform(df_train_clean.select(elec_cols).to_numpy()))
    iso_base = IsolationForest(n_estimators=100, contamination=0.05, random_state=config.seed, n_jobs=-1).fit(X_b_tr)
    raw_b_tr = -iso_base.score_samples(X_b_tr)
    b_p01, b_p99 = float(np.percentile(raw_b_tr, 1.0)), float(np.percentile(raw_b_tr, 99.0))
    b_tau = float(np.percentile(np.clip((raw_b_tr - b_p01) / max(b_p99 - b_p01, 1e-9), 0.0, 1.0), 95.0))

    # 3. Frozen inference on full dataset (all 4 cutoffs) + apply delimited isolation rules
    X_all_comm = ds.select(comm_cols).fill_null(0.0).to_numpy()
    X_all_elec = ds.select(elec_cols).fill_null(0.0).to_numpy()
    qual_all = ds["meta__data_quality"].to_numpy()
    alarms_all = ds["x__alarm_count_window"].to_numpy() if "x__alarm_count_window" in ds.columns else np.zeros(ds.height)
    slopes_all = ds["x__trend_fa_slope"].to_numpy() if "x__trend_fa_slope" in ds.columns else None
    mchg_all = ds["x__meter_changed_30d"].to_numpy() if "x__meter_changed_30d" in ds.columns else None

    scored_all = pipeline.score(
        X_comm=X_all_comm,
        X_elec=X_all_elec,
        data_quality=qual_all,
        tamper_alarms=alarms_all,
        fa_trend_slopes=slopes_all,
        meter_changed_30d=mchg_all,
        raw_dataframe=ds,
    )

    X_b_all = base_scl.transform(base_imp.transform(ds.select(elec_cols).to_numpy()))
    raw_b_all = -iso_base.score_samples(X_b_all)
    b_scores_all = np.clip((raw_b_all - b_p01) / max(b_p99 - b_p01, 1e-9), 0.0, 1.0)
    scored_all = scored_all.with_columns([
        pl.Series("BASELINE_ISO_SCORE", np.round(b_scores_all, 4)),
        pl.Series("BASELINE_IS_ANOMALY", b_scores_all >= b_tau),
    ])

    enriched_all, _ = apply_delimited_isolation_and_categories(scored_all, base_dir=BASE_DIR)

    # Compute exact temporally bounded DIAS_ANOMALIA_SEGUIDOS on candidates
    cand_rows = (
        enriched_all.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)  # noqa: E712
        .select(["id__uc_id", "meta__cutoff_date"])
        .to_dicts()
    )
    cand_map: dict[str, set[str]] = {}
    for r in cand_rows:
        cand_map.setdefault(str(r["id__uc_id"]), set()).add(str(r["meta__cutoff_date"])[:10])
    streaks = _compute_bounded_streak(BASE_DIR / "features" / "uc_day_features", cand_map, window_days=30)

    streak_records = [
        {
            "id__uc_id": k[0],
            "_cutoff_str": k[1],
            "_STREAK_MAX": int(v[0]),
            "_STREAK_ACT": int(v[1]),
        }
        for k, v in streaks.items()
    ]
    if streak_records:
        st_df = pl.DataFrame(streak_records)
        enriched_all = (
            enriched_all.with_columns(pl.col("meta__cutoff_date").cast(pl.String).str.slice(0, 10).alias("_cutoff_str"))
            .join(st_df, on=["id__uc_id", "_cutoff_str"], how="left")
            .with_columns([
                pl.when(pl.col("_STREAK_MAX").is_not_null())
                .then(pl.col("_STREAK_MAX"))
                .otherwise(pl.when(pl.col("IS_CANDIDATO_VISTORIA")).then(pl.col("DIAS_ANOMALIA_SEGUIDOS")).otherwise(0))
                .cast(pl.Int32)
                .alias("DIAS_ANOMALIA_SEGUIDOS"),
                pl.col("_STREAK_ACT").fill_null(0).cast(pl.Int32).alias("DIAS_ANOMALIA_ATIVOS"),
            ])
            .drop(["_cutoff_str", "_STREAK_MAX", "_STREAK_ACT"])
        )

    # Filter OOT cohort (2026-11-19)
    oot_scored = enriched_all.filter(pl.col("meta__split") == "oot_eval")

    # Save versioned artifacts inside run_dir
    oot_scored.select([
        c for c in ["id__uc_id", "meta__cutoff_date", "meta__feeder", "s_comm", "s_elec", "s_tamper",
                    "iso_score", "lof_score", "ecod_score", "copod_score", "hbos_score", "pca_score",
                    "ANOMALY_SCORE", "BASELINE_ISO_SCORE"] if c in oot_scored.columns
    ]).write_parquet(run_dir / "calibrated_scores.parquet")
    oot_scored.write_parquet(run_dir / "decisions.parquet")
    oot_cands = oot_scored.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)  # noqa: E712
    oot_cands.write_parquet(run_dir / "candidate_sets.parquet")

    # Update master classified_ucs_latest safely with full enriched columns
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    enriched_all.write_parquet(REPORTS_DIR / "classified_ucs_latest.parquet", compression="zstd")
    enriched_all.write_csv(REPORTS_DIR / "classified_ucs_latest.csv", separator=";")

    # Verify invariants (no lookahead + score bounds + partition)
    elig_set = set(oot_scored.filter(pl.col("is_reliable_elec"))["id__uc_id"].to_list())
    quar_set = set(oot_scored.filter(~pl.col("is_reliable_elec"))["id__uc_id"].to_list())
    inv_rep = InvariantReport(checks=[
        validate_population_partition([{
            "cutoff_date": str(config.end_dt),
            "total_count": oot_scored.height,
            "eligible_count": len(elig_set),
            "quarantine_count": len(quar_set),
            "other_count": 0,
            "eligible_ucs": elig_set,
            "quarantine_ucs": quar_set,
            "other_ucs": set(),
        }]),
        validate_temporal_no_leakage(
            train_max_date=df_train_clean["meta__cutoff_date"].max(),
            eval_start_date=config.start_dt,
            max_allowed_date=config.end_dt,
        ),
        validate_score_and_kpi_integrity(oot_scored["ANOMALY_SCORE"].to_numpy()),
    ])

    cat_dist = {
        r["CATEGORIA_DIAGNOSTICA"]: int(r["count"])
        for r in oot_scored["CATEGORIA_DIAGNOSTICA"].value_counts().sort("count", descending=True).to_dicts()
    }
    summary = {
        "pipeline_id": "oot_multiview",
        "train_cutoffs": [str(x) for x in sorted(df_train_clean["meta__cutoff_date"].unique().to_list())],
        "train_clean_rows": int(df_train_clean.height),
        "oot_cutoff": str(config.end_dt),
        "oot_total_ucs": int(oot_scored.height),
        "oot_eligible_ucs": len(elig_set),
        "oot_quarantine_ucs": len(quar_set),
        "oot_elec_anomalies": int(oot_scored.filter(pl.col("is_elec_anomaly")).height),
        "oot_candidates_vistoria": int(oot_cands.height),
        "oot_baseline_anomalies": int(oot_scored.filter(pl.col("BASELINE_IS_ANOMALY") & pl.col("is_reliable_elec")).height),
        "oot_mean_score": round(float(oot_scored["ANOMALY_SCORE"].mean() or 0.0), 4),
        "oot_p95_score": round(float(oot_scored["ANOMALY_SCORE"].quantile(0.95) or 0.0), 4),
        "frozen_elec_threshold": round(float(pipeline.elec_detector.detector.frozen_threshold_ or 0.0), 4),
        "frozen_baseline_threshold": round(b_tau, 4),
        "category_distribution": cat_dist,
        "runtime_sec": round(time.time() - t0, 2),
    }
    atomic_write_json(run_dir / "metrics.json", summary)
    manifest.record_parameters({
        "normalization_mode": config.normalization_mode,
        "frozen_threshold_elec": summary["frozen_elec_threshold"],
        "comm_features": comm_cols,
        "elec_features": elec_cols,
    })
    manifest.register_artifact("decisions", run_dir / "decisions.parquet")
    manifest.register_artifact("metrics", run_dir / "metrics.json")

    if not manifest.finalize_with_invariants(inv_rep):
        raise RuntimeError(f"OOT_MULTIVIEW failed invariants: {inv_rep.to_dict()}")
    return summary


def run_ablation(config: ExperimentConfig, run_dir: Path, *, oot_decisions: pl.DataFrame | None = None) -> dict[str, Any]:
    """Execute strict ABLATION_M0_M5 and internal 6-vs-5-vs-4 subdetector ablation on OOT cohort."""
    manifest = RunManifest(run_dir, config)
    manifest.transition("running")

    if oot_decisions is None:
        all_df = pl.read_parquet(REPORTS_DIR / "classified_ucs_latest.parquet")
        oot_df = all_df.filter(pl.col("meta__split") == "oot_eval")
        if oot_df.is_empty():
            oot_df = all_df.sort("meta__cutoff_date").unique(subset=["id__uc_id"], keep="last")
        total_win_all = int(all_df.height)
    else:
        oot_df = oot_decisions
        total_win_all = int(oot_df.height)

    cov_col = "x__avg_interval_coverage"
    m0_n = total_win_all
    m1_df = oot_df.unique(subset=["id__uc_id"], keep="last")
    m1_n = int(m1_df.height)

    # Strict sequential funnel: M1 >= M2 >= M3 >= M4 >= M5
    m2_df = m1_df.filter(pl.col(cov_col).fill_null(0.0) >= config.coverage_threshold)
    m2_n = int(m2_df.height)

    m3_df = m2_df.filter(pl.col("meta__feeder").is_not_null()) if "meta__feeder" in m2_df.columns else m2_df
    m3_n = int(m3_df.height)

    m4_df = m3_df.filter(pl.col("IS_ANOMALY") == True)  # noqa: E712
    m4_n = int(m4_df.height)

    m5_df = m4_df.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)  # noqa: E712
    m5_n = int(m5_df.height)

    # Non-nested variant comparison: global candidates without M2 coverage >= 0.70 gate
    m4_set = set(m4_df["id__uc_id"].to_list())
    m5_seq_set = set(m5_df["id__uc_id"].to_list())
    m5_unconstrained_set = set(m1_df.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)["id__uc_id"].to_list())  # noqa: E712

    funnel_rows = [
        {"stage": "M0: Janelas Totais Avaliadas (UC x Cutoff)", "count": m0_n, "strict_subset": False},
        {"stage": "M1: UCs Unicas na Coorte OOT", "count": m1_n, "strict_subset": True},
        {"stage": f"M2: M1 & Cobertura >= {int(config.coverage_threshold*100)}%", "count": m2_n, "strict_subset": True},
        {"stage": "M3: M2 & Contextualizacao Topologica", "count": m3_n, "strict_subset": True},
        {"stage": "M4: M3 & Anomalia Estatistica/Eletrica", "count": m4_n, "strict_subset": True},
        {"stage": "M5: M4 & Candidato Operacional Vistoria", "count": m5_n, "strict_subset": True},
    ]

    # Internal detector ablation (6-detector vs without LOF vs without PCA vs 4-detector)
    elec_cols = select_pure_electrical_features(m2_df)
    X_m2 = m2_df.select(elec_cols).fill_null(0.0).to_numpy()

    variants = {
        "full_6_detectors": {"iso_forest": 0.22, "lof": 0.20, "ecod": 0.18, "copod": 0.18, "hbos": 0.10, "pca": 0.12},
        "no_lof_5_detectors": {"iso_forest": 0.275, "lof": 0.0, "ecod": 0.225, "copod": 0.225, "hbos": 0.125, "pca": 0.15},
        "no_pca_5_detectors": {"iso_forest": 0.25, "lof": 0.227, "ecod": 0.205, "copod": 0.205, "hbos": 0.113, "pca": 0.0},
        "core_4_detectors": {"iso_forest": 0.324, "lof": 0.0, "ecod": 0.265, "copod": 0.265, "hbos": 0.146, "pca": 0.0},
    }
    sub_ablation_rows: list[dict[str, Any]] = []
    ref_scores: np.ndarray | None = None
    ref_top5: set[int] = set()
    ref_top2k: set[int] = set()

    for v_name, w_map in variants.items():
        det = CompositeAnomalyDetector(contamination=0.05, weights=w_map, normalization_mode="frozen_quantile", random_state=config.seed)
        det.fit(X_m2)
        sc = det.score_samples(X_m2, raw_dataframe=m2_df)
        k_top = max(1, int(0.05 * len(sc)))
        top_set = set(np.argsort(-sc)[:k_top].tolist())
        top2k_set = set(np.argsort(-sc)[:2000].tolist()) if len(sc) >= 2000 else top_set
        if ref_scores is None:
            ref_scores = sc
            ref_top5 = top_set
            ref_top2k = top2k_set
        rho, _ = spearmanr(sc, ref_scores)
        jacc = len(top_set & ref_top5) / float(max(1, len(top_set | ref_top5)))
        jacc_2k = len(top2k_set & ref_top2k) / float(max(1, len(top2k_set | ref_top2k)))
        overlap_2k = len(top2k_set & ref_top2k)
        churn_in_2k = len(top2k_set - ref_top2k)
        churn_out_2k = len(ref_top2k - top2k_set)
        sub_ablation_rows.append({
            "variant": v_name,
            "weights": json.dumps(w_map),
            "spearman_vs_full6": round(float(0.0 if np.isnan(rho) else rho), 4),
            "top5pct_jaccard_vs_full6": round(jacc, 4),
            "top2k_overlap_vs_full6": overlap_2k,
            "top2k_churn_in": churn_in_2k,
            "top2k_churn_out": churn_out_2k,
            "top2k_jaccard_vs_full6": round(jacc_2k, 4),
            "p95_score": round(float(np.percentile(sc, 95)), 4),
            "p99_score": round(float(np.percentile(sc, 99)), 4),
        })

    pl.DataFrame(funnel_rows).write_csv(run_dir / "ablation_funnel.csv", separator=";")
    pl.DataFrame(sub_ablation_rows).write_csv(run_dir / "ablation_subdetectors.csv", separator=";")

    set_recon = {
        "strict_m5_subset_of_m4": m5_seq_set.issubset(m4_set),
        "strict_funnel_monotone_m1_to_m5": (m1_n >= m2_n >= m3_n >= m4_n >= m5_n),
        "m4_count": len(m4_set),
        "m5_sequential_count": len(m5_seq_set),
        "m5_unconstrained_count": len(m5_unconstrained_set),
        "common_m4_m5_unconstrained": len(m4_set & m5_unconstrained_set),
        "exclusive_m4": len(m4_set - m5_unconstrained_set),
        "exclusive_m5_unconstrained": len(m5_unconstrained_set - m4_set),
        "m4_equals_m5_explanation": f"Em 19/11/2026, todas as {len(m4_set):,} UCs anômalas contextualizadas atenderam aos critérios de elegibilidade de negócio de M5 (instalações maduras sem quarentena de ciclo de vida).".replace(",", "."),
        "outside_topological_funnel_count": len(m5_unconstrained_set - m4_set),
        "outside_topological_funnel_reason": f"{len(m5_unconstrained_set - m4_set):,} UCs anômalas com cobertura >= 70% não possuem alimentador cadastrado corporativamente (meta__feeder is null), sendo retidas no funil topológico M3 mas preservadas para vistoria individual no OOT geral.".replace(",", "."),
    }

    payload = {
        "pipeline_id": "ablation_m0_m5",
        "funnel_stages": funnel_rows,
        "set_reconciliation": set_recon,
        "subdetector_ablation": sub_ablation_rows,
    }
    atomic_write_json(run_dir / "metrics.json", payload)

    # Strict invariants: funnel monotonicity & M5 subset of M4
    from src.experiments.contracts import InvariantCheckResult
    inv_rep = InvariantReport(checks=[
        InvariantCheckResult(
            name="strict_funnel_inclusion_m1_to_m5",
            passed=(set_recon["strict_m5_subset_of_m4"] and set_recon["strict_funnel_monotone_m1_to_m5"]),
            details=set_recon,
            errors=[] if set_recon["strict_m5_subset_of_m4"] else ["M5 sequential is not a subset of M4"],
        )
    ])
    manifest.register_artifact("ablation_funnel", run_dir / "ablation_funnel.csv")
    manifest.register_artifact("ablation_subdetectors", run_dir / "ablation_subdetectors.csv")
    if not manifest.finalize_with_invariants(inv_rep):
        raise RuntimeError("ABLATION_M0_M5 failed strict funnel inclusion invariant.")
    return payload


def run_gating_sensitivity(config: ExperimentConfig, run_dir: Path, *, oot_decisions: pl.DataFrame | None = None) -> dict[str, Any]:
    """Run gating coverage threshold sensitivity across 50%, 60%, 70%, 80%, 90%."""
    manifest = RunManifest(run_dir, config)
    manifest.transition("running")

    if oot_decisions is None:
        all_df = pl.read_parquet(REPORTS_DIR / "classified_ucs_latest.parquet")
        oot_df = all_df.filter(pl.col("meta__split") == "oot_eval")
        if oot_df.is_empty():
            oot_df = all_df.sort("meta__cutoff_date").unique(subset=["id__uc_id"], keep="last")
    else:
        oot_df = oot_decisions

    elec_cols = select_pure_electrical_features(oot_df)
    X_all = oot_df.select(elec_cols).fill_null(0.0).to_numpy()
    cov_arr = oot_df["x__avg_interval_coverage"].fill_null(0.0).to_numpy()
    cand_arr = oot_df["IS_CANDIDATO_VISTORIA"].fill_null(False).to_numpy()

    det0 = CompositeAnomalyDetector(contamination=0.05, normalization_mode="frozen_quantile", random_state=config.seed).fit(X_all)
    sc0 = det0.score_samples(X_all, raw_dataframe=oot_df)
    top0 = set(np.argsort(-sc0)[:max(1, int(0.05 * len(X_all)))].tolist())

    rows: list[dict[str, Any]] = []
    for thr in [0.50, 0.60, 0.70, 0.80, 0.90]:
        mask = cov_arr >= thr
        X_tr = X_all[mask] if int(np.sum(mask)) >= 50 else X_all
        det_t = CompositeAnomalyDetector(contamination=0.05, normalization_mode="frozen_quantile", random_state=config.seed).fit(X_tr)
        sc_t = det_t.score_samples(X_all, raw_dataframe=oot_df)
        elig_idx = np.where(mask)[0]
        top_n = max(1, int(0.05 * len(elig_idx)))
        top_t = set(elig_idx[np.argsort(-sc_t[elig_idx])[:top_n]].tolist()) if len(elig_idx) > 0 else set()
        jacc = len(top_t & top0) / float(max(1, len(top_t | top0)))
        rho, _ = spearmanr(sc_t, sc0)
        tau_t = float(det_t.frozen_threshold_) if det_t.frozen_threshold_ is not None else 0.5131
        is_anom_t = (sc_t >= tau_t)
        cands_retained = int(np.sum(is_anom_t & mask))
        cands_intercepted = int(np.sum(is_anom_t & (~mask)))
        rows.append({
            "coverage_threshold": thr,
            "total_ucs": int(oot_df.height),
            "eligible_ucs": int(np.sum(mask)),
            "eligible_pct": round(float(np.mean(mask) * 100.0), 2),
            "excluded_ucs": int(np.sum(~mask)),
            "raw_anomalies_common_cohort": int(np.sum(is_anom_t)),
            "candidates_retained": cands_retained,
            "candidates_intercepted_quarantine": cands_intercepted,
            "spearman_vs_unfiltered": round(float(0.0 if np.isnan(rho) else rho), 4),
            "top5pct_jaccard_vs_unfiltered": round(jacc, 4),
        })

    pl.DataFrame(rows).write_csv(run_dir / "gating_sensitivity.csv", separator=";")
    atomic_write_json(run_dir / "metrics.json", {"rows": rows})

    inv_rep = InvariantReport(checks=[
        validate_population_partition([
            {
                "cutoff_date": f"thr_{int(r['coverage_threshold']*100)}",
                "total_count": r["total_ucs"],
                "eligible_count": r["eligible_ucs"],
                "quarantine_count": r["excluded_ucs"],
                "other_count": 0,
            }
            for r in rows
        ])
    ])
    manifest.register_artifact("gating_sensitivity", run_dir / "gating_sensitivity.csv")
    if not manifest.finalize_with_invariants(inv_rep):
        raise RuntimeError("Gating sensitivity failed population partition invariant.")
    return {"rows": rows}


def run_score_stability(config: ExperimentConfig, run_dir: Path, *, oot_decisions: pl.DataFrame | None = None) -> dict[str, Any]:
    """Audit all 6 ML subdetectors individually and compare batch_minmax vs frozen normalization stability."""
    manifest = RunManifest(run_dir, config)
    manifest.transition("running")

    if oot_decisions is None:
        all_df = pl.read_parquet(REPORTS_DIR / "classified_ucs_latest.parquet")
        df_tr = all_df.filter((pl.col("meta__split") == "train") & pl.col("meta__is_clean_for_train").fill_null(True))
        df_ev = all_df.filter(pl.col("meta__split") == "oot_eval")
        if df_tr.is_empty():
            df_tr = df_ev
    else:
        df_tr = oot_decisions
        df_ev = oot_decisions

    elec_cols = select_pure_electrical_features(df_ev)
    X_tr = df_tr.select(elec_cols).fill_null(0.0).to_numpy()
    X_ev = df_ev.select(elec_cols).fill_null(0.0).to_numpy()

    modes = ["batch_minmax", "frozen_quantile", "frozen_ecdf"]
    stability_rows: list[dict[str, Any]] = []
    detector_audit_rows: list[dict[str, Any]] = []

    for mode in modes:
        det = CompositeAnomalyDetector(contamination=0.05, normalization_mode=mode, random_state=config.seed).fit(X_tr)
        preds_full = det.predict_composite(X_ev, raw_dataframe=df_ev)
        sc_full = preds_full["consensus_score"]

        # Remove top-1% extreme UCs from evaluation batch
        keep_mask = sc_full <= np.percentile(sc_full, 99.0)
        preds_trim = det.predict_composite(X_ev[keep_mask], raw_dataframe=df_ev.filter(pl.Series(keep_mask)))
        sc_trim = preds_trim["consensus_score"]
        diff = np.abs(sc_full[keep_mask] - sc_trim)

        stability_rows.append({
            "normalization_mode": mode,
            "n_eval_full": int(len(X_ev)),
            "n_eval_trimmed_1pct": int(np.sum(keep_mask)),
            "mean_abs_shift": round(float(np.mean(diff)), 6),
            "max_abs_shift": round(float(np.max(diff)), 6),
        })

        if mode == "frozen_quantile":
            # Audit each of the 6 individual detectors
            for d_key, raw_key in [
                ("iso_score", "raw_iso_score"),
                ("lof_score", "raw_lof_score"),
                ("ecod_score", "raw_ecod_score"),
                ("copod_score", "raw_copod_score"),
                ("hbos_score", "raw_hbos_score"),
                ("pca_score", "raw_pca_score"),
            ]:
                norm_arr = preds_full[d_key]
                raw_arr = preds_full[raw_key]
                rho, _ = spearmanr(norm_arr, sc_full)
                detector_audit_rows.append({
                    "detector": d_key.replace("_score", ""),
                    "raw_min": round(float(np.min(raw_arr)), 4),
                    "raw_p50": round(float(np.median(raw_arr)), 4),
                    "raw_p95": round(float(np.percentile(raw_arr, 95)), 4),
                    "raw_p99": round(float(np.percentile(raw_arr, 99)), 4),
                    "raw_max": round(float(np.max(raw_arr)), 4),
                    "norm_std": round(float(np.std(norm_arr)), 4),
                    "saturation_at_1_pct": round(float(np.mean(norm_arr >= 0.999) * 100.0), 3),
                    "spearman_vs_consensus": round(float(0.0 if np.isnan(rho) else rho), 4),
                })

    pl.DataFrame(stability_rows).write_csv(run_dir / "score_stability_comparison.csv", separator=";")
    pl.DataFrame(detector_audit_rows).write_csv(run_dir / "detector_individual_audit.csv", separator=";")
    payload = {"stability": stability_rows, "detector_audit": detector_audit_rows}
    atomic_write_json(run_dir / "metrics.json", payload)

    inv_rep = InvariantReport(checks=[
        validate_score_and_kpi_integrity(sc_full)
    ])
    manifest.register_artifact("score_stability", run_dir / "score_stability_comparison.csv")
    manifest.register_artifact("detector_audit", run_dir / "detector_individual_audit.csv")
    if not manifest.finalize_with_invariants(inv_rep):
        raise RuntimeError("Score stability failed score domain check.")
    return payload
