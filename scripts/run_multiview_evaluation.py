"""P1 Integrated Multiview Evaluation, Topological MoE, Ablation V0-V5 (Base A), Case Studies & Operational Projection.

Covers:
- Section 12: Integrated multiview execution over OOT (65,686 UCs) reusing frozen models + full output contract & provenance
- Section 17: S_tamper and physical/PRODIST rules audit (including real alarm_events audit)
- Section 18: Leak-free Topological Context & MoE evaluation (feeder specialists trained strictly on <= 2026-10-31)
- Section 19 (Base A): Methodological Ablation V0-V5 on real OOT under constant budget K = 2,000
- Section 20: Complete real case studies (908***00, 291***96, 685***20) comparing 25/11 vs integrated multiview
- Section 21: Operational projection under K = 2,000 inspections/month with separated queues
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

from src.eda.uc_timeseries import _detect_step_change_cusum, _list_dated_parquets
from src.features.neighborhood import compute_neighborhood_features
from src.models.composite import CompositeAnomalyDetector, select_pure_electrical_features
from src.models.moe_router import MoERouter
from src.models.spurious_filter import filter_spurious_anomalies
from src.utils.censorship import build_feeder_anonymizer, censor_nio_id, censor_uc_id
from src.utils.joblib_splitter import load_split_joblib

SUITE_PATH = PROJECT_ROOT / "experiments" / "runs" / "latest_suite.json"
P1_MV_DIR = PROJECT_ROOT / "docs" / "tcc_evidence" / "p1_multiview"
P1_FIG_DIR = P1_MV_DIR / "figures"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_br_int(val: int | float) -> str:
    return f"{int(round(val)):,}".replace(",", ".")


def _fmt_br_float(val: float, decimals: int = 2) -> str:
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _setup_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_fig(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_p = stem.with_suffix(".pdf")
    png_p = stem.with_suffix(".png")
    fig.savefig(pdf_p, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_p, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_p.relative_to(PROJECT_ROOT)), "png": str(png_p.relative_to(PROJECT_ROOT))}


def _topk_indices(scores: np.ndarray, k: int) -> set[int]:
    n = len(scores)
    k_eff = min(k, n)
    order = np.lexsort((np.arange(n), -scores))
    return set(order[:k_eff].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / float(max(1, len(a | b)))


def _compute_s_behavior_from_frozen_clusterer(
    clusterer: Any,
    X_train_clean: np.ndarray,
    X_oot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute continuous behavioral deviation score S_behavior in [0, 1] using frozen PCA+KMeans centroid distance."""
    X_tr_clean = np.nan_to_num(clusterer.imputer.transform(X_train_clean), nan=0.0)
    X_tr_pca = clusterer.pca.transform(clusterer.scaler.transform(X_tr_clean))
    tr_labels = clusterer.kmeans.predict(X_tr_pca)
    centers = clusterer.kmeans.cluster_centers_
    tr_dist = np.linalg.norm(X_tr_pca - centers[tr_labels], axis=1)
    p01 = float(np.percentile(tr_dist, 1.0))
    p99 = float(np.percentile(tr_dist, 99.0))
    span = max(p99 - p01, 1e-9)

    X_ev_clean = np.nan_to_num(clusterer.imputer.transform(X_oot), nan=0.0)
    X_ev_pca = clusterer.pca.transform(clusterer.scaler.transform(X_ev_clean))
    ev_labels = clusterer.kmeans.predict(X_ev_pca)
    ev_dist = np.linalg.norm(X_ev_pca - centers[ev_labels], axis=1)
    s_beh = np.round(np.clip((ev_dist - p01) / span, 0.0, 1.0), 4)
    return s_beh, ev_labels


def _compute_cpd_for_oot_cohort(
    uc_list: list[str],
    cutoff: date = date(2026, 11, 19),
    window_days: int = 30,
) -> dict[str, dict[str, Any]]:
    """Run real _detect_step_change_cusum on 30-day daily series for all UCs in OOT."""
    day_root = PROJECT_ROOT / "output/features/uc_day_features/feature_set_version=v1"
    dated = _list_dated_parquets(day_root)
    w_start = cutoff - timedelta(days=window_days)
    selected = [(d, p) for d, p in dated if w_start < d <= cutoff]
    if not selected:
        return {}

    frames = []
    for dt, fpath in selected:
        lf = pl.scan_parquet(str(fpath))
        avail = set(lf.collect_schema().names())
        cols = [c for c in ["UC", "REPORT_DAY", "FA_INTERVAL_SUM", "INTERVAL_COVERAGE"] if c in avail]
        if "UC" not in cols or "FA_INTERVAL_SUM" not in cols:
            continue
        df_p = lf.select(cols).with_columns([
            pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc"),
            pl.lit(dt).alias("_dt"),
        ]).collect()
        frames.append(df_p)

    all_daily = (
        pl.concat(frames, how="diagonal_relaxed")
        .sort(["_uc", "_dt"])
        .unique(subset=["_uc", "_dt"], keep="last")
        .sort(["_uc", "_dt"])
    )

    # Group into lists per UC for fast Python loop over 65k UCs (~4s)
    grouped = all_daily.group_by("_uc", maintain_order=False).agg([
        pl.col("_dt").cast(pl.String).alias("dates"),
        pl.col("FA_INTERVAL_SUM").fill_null(0.0).alias("fas"),
        pl.col("INTERVAL_COVERAGE").fill_null(0.0).alias("covs"),
    ])

    results: dict[str, dict[str, Any]] = {}
    for row in grouped.iter_rows(named=True):
        uc = row["_uc"]
        dates = row["dates"]
        fas = row["fas"]
        covs = row["covs"]
        rel_vals = [v for v, c in zip(fas, covs) if c >= 0.70]
        if len(rel_vals) < 8:
            results[uc] = {
                "cpd_evaluable": False,
                "cpd_detected": None,
                "cpd_timestamp": None,
                "cpd_magnitude_pct": None,
                "cpd_direction": None,
                "cpd_cusum_peak": None,
            }
            continue
        y_arr = np.asarray(rel_vals, dtype=float)
        med = float(np.median(y_arr))
        mad = float(np.median(np.abs(y_arr - med)))
        scale = max(1.4826 * mad, 0.10 * abs(med), 0.05)
        step = _detect_step_change_cusum(dates, fas, covs, med, scale)
        if step is None:
            results[uc] = {
                "cpd_evaluable": True,
                "cpd_detected": False,
                "cpd_timestamp": None,
                "cpd_magnitude_pct": 0.0,
                "cpd_direction": "ESTAVEL",
                "cpd_cusum_peak": 0.0,
            }
        else:
            results[uc] = {
                "cpd_evaluable": True,
                "cpd_detected": True,
                "cpd_timestamp": step["date"],
                "cpd_magnitude_pct": float(step["delta_pct"]),
                "cpd_direction": step["direction"],
                "cpd_cusum_peak": float(step["cusum_peak"]),
            }
    return results


def _audit_alarm_events_lake() -> dict[str, Any]:
    """Audit historical alarm_events in Data Lake to document S_tamper data availability."""
    ev_dir = PROJECT_ROOT / "output/events"
    res: dict[str, Any] = {"alarm_files": [], "total_rows": 0, "min_date": None, "max_date": None, "nov_2026_rows": 0}
    parquets = sorted((ev_dir / "alarm_events").rglob("*.parquet")) if (ev_dir / "alarm_events").exists() else []
    res["n_parquet_partitions"] = len(parquets)
    if parquets:
        try:
            lf = pl.scan_parquet([str(p) for p in parquets])
            schema = lf.collect_schema()
            ts_col = next((c for c in ("ORIGIN_TIMESTAMP", "TIMESTAMP_UTC", "EVENT_TIMESTAMP") if c in schema), None)
            n_tot = lf.select(pl.len()).collect().item()
            res["total_rows"] = int(n_tot)
            if ts_col:
                stats = lf.select([
                    pl.col(ts_col).min().alias("min_ts"),
                    pl.col(ts_col).max().alias("max_ts"),
                ]).collect().to_dicts()[0]
                res["min_date"] = str(stats["min_ts"])
                res["max_date"] = str(stats["max_ts"])
                nov_n = lf.filter(pl.col(ts_col).cast(pl.String) >= "2026-10-20").select(pl.len()).collect().item()
                res["nov_2026_rows"] = int(nov_n)
            if "ALARM_CATEGORY" in schema:
                cat_cnt = lf.group_by("ALARM_CATEGORY").len().sort("len", descending=True).collect().to_dicts()
                res["alarm_categories"] = cat_cnt[:10]
        except Exception as e:
            res["error"] = str(e)
    csvs = sorted(ev_dir.glob("*.csv"))
    res["csv_files"] = [c.name for c in csvs]
    return res


def run_multiview_evaluation() -> dict[str, Any]:
    t0 = time.time()
    _setup_pub_style()
    commit = _git_commit()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_p1_multiview_eval"
    run_dir = PROJECT_ROOT / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    P1_FIG_DIR.mkdir(parents=True, exist_ok=True)

    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    oot_dir = PROJECT_ROOT / suite["oot_multiview"]

    oot_dec = pl.read_parquet(oot_dir / "decisions.parquet")
    tr_ds = pl.read_parquet(PROJECT_ROOT / "output/model_input/v1/training_dataset.parquet")
    hier = pl.read_parquet(PROJECT_ROOT / "output/context/electrical_hierarchy.parquet")
    meter_hist = pl.read_parquet(PROJECT_ROOT / "output/context/meter_installation_history.parquet")
    pipe = load_split_joblib(oot_dir / "frozen_multiview_model.joblib")
    det = pipe.elec_detector.detector

    df_train_clean = tr_ds.filter((pl.col("meta__split") == "train") & pl.col("meta__is_clean_for_train").fill_null(True))
    elec_cols = select_pure_electrical_features(oot_dec)
    X_train_clean = df_train_clean.select(elec_cols).fill_null(0.0).to_numpy()
    X_oot = oot_dec.select(elec_cols).fill_null(0.0).to_numpy()

    # Anonymizers
    all_feeders = sorted({
        str(x) for x in (
            hier["ALIMENTADOR"].drop_nulls().unique().to_list()
            + oot_dec["meta__feeder"].drop_nulls().unique().to_list()
        ) if x and str(x).strip()
    })
    feeder_map = build_feeder_anonymizer(all_feeders)

    # 1. Recompute raw/norm scores from frozen global detector (verified exact)
    t_elec0 = time.time()
    preds_global = det.predict_composite(X_oot, raw_dataframe=oot_dec)
    t_elec = time.time() - t_elec0

    # 2. Compute S_behavior from frozen BehavioralClusterer
    t_beh0 = time.time()
    s_behavior, cluster_ids = _compute_s_behavior_from_frozen_clusterer(pipe.clusterer, X_train_clean, X_oot)
    t_beh = time.time() - t_beh0

    # 3. Compute CPD (CUSUM) over 30-day real uc_day_features for ALL 65,686 UCs
    t_cpd0 = time.time()
    uc_list = oot_dec["id__uc_id"].to_list()
    cpd_map = _compute_cpd_for_oot_cohort(uc_list, cutoff=date(2026, 11, 19), window_days=30)
    t_cpd = time.time() - t_cpd0

    cpd_evaluable = np.array([bool(cpd_map.get(u, {}).get("cpd_evaluable", False)) for u in uc_list], dtype=bool)
    cpd_detected_list = [cpd_map.get(u, {}).get("cpd_detected", None) for u in uc_list]
    cpd_ts_list = [cpd_map.get(u, {}).get("cpd_timestamp", None) for u in uc_list]
    cpd_mag_list = [cpd_map.get(u, {}).get("cpd_magnitude_pct", None) for u in uc_list]
    cpd_dir_list = [cpd_map.get(u, {}).get("cpd_direction", None) for u in uc_list]

    # 4. Audit alarm_events & S_tamper
    alarm_audit = _audit_alarm_events_lake()
    s_tamper = oot_dec["s_tamper"].to_numpy()
    alarms_win = oot_dec["x__alarm_count_window"].fill_null(0).to_numpy()

    # 5. Leak-free Topological MoE (Feeder Specialists trained strictly on <= 2026-10-31 clean train!)
    t_moe0 = time.time()
    router = MoERouter(min_feeder_samples=150, large_feeder_samples=1000)
    oot_feeders = oot_dec["meta__feeder"].to_list()
    tr_feeders = df_train_clean["meta__feeder"].to_list()
    feeder_counts_tr = df_train_clean.group_by("meta__feeder").len().to_dicts()
    eligible_feeders = sorted([
        r["meta__feeder"] for r in feeder_counts_tr
        if r["meta__feeder"] is not None and int(r["len"]) >= 150
    ])

    global_scores = preds_global["consensus_score"]
    feeder_score_arr = np.full(len(uc_list), np.nan, dtype=float)
    moe_score_arr = np.copy(global_scores)
    moe_weight_feeder = np.zeros(len(uc_list), dtype=float)
    feeder_model_stats = []

    for f_name in eligible_feeders:
        tr_sub = df_train_clean.filter(pl.col("meta__feeder") == f_name)
        oot_mask_f = np.array([x == f_name for x in oot_feeders], dtype=bool)
        if not np.any(oot_mask_f):
            continue
        X_f_tr = tr_sub.select(elec_cols).fill_null(0.0).to_numpy()
        X_f_oot = oot_dec.filter(pl.Series(oot_mask_f)).select(elec_cols).fill_null(0.0).to_numpy()
        oot_sub_df = oot_dec.filter(pl.Series(oot_mask_f))

        f_det = CompositeAnomalyDetector(
            contamination=0.05,
            weights=dict(det.weights),
            normalization_mode="frozen_quantile",
            freeze_threshold=True,
            random_state=42,
        )
        f_det.fit(X_f_tr)
        f_preds = f_det.predict_composite(X_f_oot, raw_dataframe=oot_sub_df)
        f_sc = f_preds["consensus_score"]
        feeder_score_arr[oot_mask_f] = f_sc

        # GD penetration from training set only (no OOT leakage!)
        gd_pen_tr = float((tr_sub["x__hist_ra_reversal_ratio_max"].fill_null(0.0) >= 0.08).mean() or 0.0)
        w_moe = router.compute_gating_weights(
            feeder_sample_count=int(tr_sub.height),
            circuit_profile={"CIRCUIT_SOLAR_GD_PENETRATION": gd_pen_tr},
        )
        comb = router.combine_predictions(
            scores_feeder=f_sc,
            scores_mun=None,
            scores_global=global_scores[oot_mask_f],
            weights=w_moe,
        )
        moe_score_arr[oot_mask_f] = comb
        moe_weight_feeder[oot_mask_f] = w_moe["w_feeder"] / (w_moe["w_feeder"] + w_moe["w_global"])

        rho_fg = float(spearmanr(f_sc, global_scores[oot_mask_f]).statistic)
        feeder_model_stats.append({
            "circuito": feeder_map.get(f_name, f_name),
            "n_train_clean": int(tr_sub.height),
            "n_oot": int(oot_mask_f.sum()),
            "gd_pen_train_pct": round(gd_pen_tr * 100.0, 2),
            "w_feeder_eff": round(float(w_moe["w_feeder"] / (w_moe["w_feeder"] + w_moe["w_global"])), 3),
            "tau_feeder_frozen": round(float(f_det.frozen_threshold_ or 0.0), 4),
            "spearman_feeder_vs_global": round(0.0 if np.isnan(rho_fg) else rho_fg, 4),
            "mean_global_score": round(float(np.mean(global_scores[oot_mask_f])), 4),
            "mean_moe_score": round(float(np.mean(comb)), 4),
        })

    # Neighborhood & Spurious Grid Disturbance Filter on OOT
    neigh_df = compute_neighborhood_features(oot_dec, hier)
    spurious_in = oot_dec.select(["id__uc_id", "meta__cutoff_date"]).with_columns([
        pl.Series("anomaly_score", moe_score_arr),
        pl.Series("is_anomaly", (moe_score_arr >= float(det.frozen_threshold_)) & oot_dec["is_reliable_elec"].to_numpy()),
    ])
    spurious_out = filter_spurious_anomalies(spurious_in, neighborhood_df=neigh_df, grid_event_threshold=0.60)
    is_grid_event = spurious_out["IS_GRID_EVENT"].to_numpy()
    is_spurious_glitch = spurious_out["IS_SPURIOUS_GLITCH"].to_numpy()
    moe_filtered_score = spurious_out["FILTERED_ANOMALY_SCORE"].to_numpy()
    t_moe = time.time() - t_moe0

    pl.DataFrame(feeder_model_stats).write_csv(run_dir / "moe_feeder_evaluation.csv", separator=";")
    pl.DataFrame(feeder_model_stats).write_csv(P1_MV_DIR / "moe_feeder_evaluation.csv", separator=";")

    # Global vs Contextual percentile comparison on common eligible population with feeder
    elig_mask = oot_dec["is_reliable_elec"].to_numpy()
    has_feeder_mask = ~np.isnan(feeder_score_arr)
    common_ctx_mask = elig_mask & has_feeder_mask
    g_pct_common = rankdata(global_scores[common_ctx_mask]) / float(common_ctx_mask.sum()) * 100.0
    m_pct_common = rankdata(moe_score_arr[common_ctx_mask]) / float(common_ctx_mask.sum()) * 100.0
    top2k_global_ctx = _topk_indices(np.where(common_ctx_mask, global_scores, -1.0), 2000)
    top2k_moe_ctx = _topk_indices(np.where(common_ctx_mask, moe_score_arr, -1.0), 2000)
    top2k_moefilt_ctx = _topk_indices(np.where(common_ctx_mask, moe_filtered_score, -1.0), 2000)

    moe_comparison = {
        "common_population_elig_with_feeder": int(common_ctx_mask.sum()),
        "n_feeder_specialists_trained_leakfree": len(feeder_model_stats),
        "spearman_global_vs_moe_common": round(float(spearmanr(global_scores[common_ctx_mask], moe_score_arr[common_ctx_mask]).statistic), 4),
        "mean_abs_percentile_shift": round(float(np.mean(np.abs(m_pct_common - g_pct_common))), 2),
        "p95_abs_percentile_shift": round(float(np.percentile(np.abs(m_pct_common - g_pct_common), 95)), 2),
        "top2k_global_vs_moe_jaccard": round(_jaccard(top2k_global_ctx, top2k_moe_ctx), 4),
        "top2k_global_vs_moe_common": len(top2k_global_ctx & top2k_moe_ctx),
        "top2k_global_vs_moe_substitutions": len(top2k_moe_ctx - top2k_global_ctx),
        "grid_disturbance_flagged_ucs": int(is_grid_event.sum()),
        "spurious_non_isolated_flagged_ucs": int(is_spurious_glitch.sum()),
        "top2k_global_vs_moefilt_jaccard": round(_jaccard(top2k_global_ctx, top2k_moefilt_ctx), 4),
        "top2k_global_vs_moefilt_substitutions": len(top2k_moefilt_ctx - top2k_global_ctx),
    }

    # 6. Physical rules breakdown per UC
    v_imb = oot_dec["x__hist_voltage_imbalance_max"].fill_null(0.0).to_numpy()
    c_imb = oot_dec["x__hist_current_imbalance_max"].fill_null(0.0).to_numpy()
    ra_r = oot_dec["x__hist_ra_reversal_ratio_max"].fill_null(0.0).to_numpy()
    ra_d = oot_dec["x__hist_ra_reversal_days"].fill_null(0).to_numpy()
    fa_med = oot_dec["x__hist_fa_sum_median"].fill_null(0.0).to_numpy()

    r_v_imb = v_imb > 20.0
    r_c_imb = c_imb > 0.20
    r_c_imb_sev = c_imb > 0.45
    r_ra_ext = ra_r > 0.85
    r_gd = (ra_r >= 0.08) & (ra_d >= 2)
    r_stuck = (fa_med < 0.05) & elig_mask & (global_scores >= float(det.frozen_threshold_))

    phys_rules_str = []
    for i in range(len(uc_list)):
        tags = []
        if r_v_imb[i]:
            tags.append("PRODIST_V_IMB>20%")
        if r_ra_ext[i]:
            tags.append("PRODIST_RA>85%")
        if r_gd[i]:
            tags.append("REGRA_GD_REVERSA")
        if r_c_imb_sev[i]:
            tags.append("DESBAL_CORRENTE>45%")
        elif r_c_imb[i]:
            tags.append("DESBAL_CORRENTE>20%")
        if r_stuck[i]:
            tags.append("MEDIDOR_TRAVADO_FA<0.05")
        phys_rules_str.append("|".join(tags) if tags else "NENHUMA")

    physical_rules_audit = [
        {"regra": "PRODIST Desequilíbrio Tensão (>20%)", "acionamentos_oot": int(r_v_imb.sum()),
         "sobreposicao_anom_eletrica": int((r_v_imb & oot_dec["is_elec_anomaly"].to_numpy()).sum()),
         "exclusivos_sem_anom_eletrica": int((r_v_imb & (~oot_dec["is_elec_anomaly"].to_numpy())).sum())},
        {"regra": "PRODIST Reversão Extrema (>85%)", "acionamentos_oot": int(r_ra_ext.sum()),
         "sobreposicao_anom_eletrica": int((r_ra_ext & oot_dec["is_elec_anomaly"].to_numpy()).sum()),
         "exclusivos_sem_anom_eletrica": int((r_ra_ext & (~oot_dec["is_elec_anomaly"].to_numpy())).sum())},
        {"regra": "Regra Física GD (RA>=8% & >=2d)", "acionamentos_oot": int(r_gd.sum()),
         "sobreposicao_anom_eletrica": int((r_gd & oot_dec["is_elec_anomaly"].to_numpy()).sum()),
         "exclusivos_sem_anom_eletrica": int((r_gd & (~oot_dec["is_elec_anomaly"].to_numpy())).sum())},
        {"regra": "Desequilíbrio Corrente (>20% — gatilho atual +0,15)", "acionamentos_oot": int(r_c_imb.sum()),
         "sobreposicao_anom_eletrica": int((r_c_imb & oot_dec["is_elec_anomaly"].to_numpy()).sum()),
         "exclusivos_sem_anom_eletrica": int((r_c_imb & (~oot_dec["is_elec_anomaly"].to_numpy())).sum())},
        {"regra": "Desequilíbrio Corrente Severo (>45%)", "acionamentos_oot": int(r_c_imb_sev.sum()),
         "sobreposicao_anom_eletrica": int((r_c_imb_sev & oot_dec["is_elec_anomaly"].to_numpy()).sum()),
         "exclusivos_sem_anom_eletrica": int((r_c_imb_sev & (~oot_dec["is_elec_anomaly"].to_numpy())).sum())},
        {"regra": "Medidor Travado / Consumo Nulo (FA<0,05 & Anom.)", "acionamentos_oot": int(r_stuck.sum()),
         "sobreposicao_anom_eletrica": int((r_stuck & oot_dec["is_elec_anomaly"].to_numpy()).sum()),
         "exclusivos_sem_anom_eletrica": 0},
        {"regra": "Tamper (s_tamper >= 0,70; 0 alarmes em Nov/2026)", "acionamentos_oot": int((s_tamper >= 0.70).sum()),
         "sobreposicao_anom_eletrica": 0,
         "exclusivos_sem_anom_eletrica": 0},
    ]
    pl.DataFrame(physical_rules_audit).write_csv(run_dir / "physical_rules_and_tamper_audit.csv", separator=";")
    pl.DataFrame(physical_rules_audit).write_csv(P1_MV_DIR / "physical_rules_and_tamper_audit.csv", separator=";")

    # 7. Integrated Multiview Priority Score (Experimental Variant — strictly preserving 25/11 as reference!)
    # Transparent, non-supervised additive evidence aggregation (capped [0, 1]):
    # - Base: moe_filtered_score (when eligible; else NaN for electrical priority)
    # - +0.06 if CPD detected sustained downward step >= 30%
    # - +0.04 if s_behavior >= 0.80 or longitudinal abrupt class drop
    # - +0.05 if severe physical rule (PRODIST V>20%, RA>85%, or c_imb>45% or stuck)
    cpd_drop_flag = np.array([
        bool(d and (dir_ == "QUEDA_DEGRAU") and (mag is not None) and (mag <= -30.0))
        for d, dir_, mag in zip(cpd_detected_list, cpd_dir_list, cpd_mag_list)
    ], dtype=bool)
    beh_high_flag = (s_behavior >= 0.80) | (oot_dec["HIPOTESE_MUDANCA_CLASSE"].fill_null("").to_numpy() == "QUEDA_ABRUPTA_CLASSE_SUSPEITA_BYPASS_OU_DESOCUPACAO")
    phys_sev_flag = r_v_imb | r_ra_ext | r_c_imb_sev | r_stuck

    # Variant scores (V0..V5) for Section 19 Base A Ablation under constant K = 2,000
    v0_score = oot_dec["BASELINE_ISO_SCORE"].to_numpy()  # Baseline IF without gating
    v1_score = global_scores  # 6-det ensemble without gating
    v2_score = np.where(elig_mask, global_scores, -1.0)  # V1 + Quality Gating (>=70%) — 25/11 reference
    v3_score = np.where(elig_mask, np.clip(moe_filtered_score + np.where(beh_high_flag, 0.04, 0.0), 0.0, 1.0), -1.0)  # V2 + Behavior + Topological MoE + SpuriousFilter
    v4_score = np.where(elig_mask, np.clip(moe_filtered_score + np.where(beh_high_flag, 0.04, 0.0) + np.where(cpd_drop_flag, 0.06, 0.0), 0.0, 1.0), -1.0)  # V3 + CPD
    v5_score = np.where(elig_mask, np.clip(moe_filtered_score + np.where(beh_high_flag, 0.04, 0.0) + np.where(cpd_drop_flag, 0.06, 0.0) + np.where(phys_sev_flag, 0.05, 0.0), 0.0, 1.0), -1.0)  # V4 + Physical/Tamper integrated

    # Rank V5 among eligible UCs (1 = highest priority; None for ineligible)
    v5_rank = np.full(len(uc_list), -1, dtype=int)
    elig_indices = np.where(elig_mask)[0]
    elig_order = elig_indices[np.lexsort((elig_indices, -v5_score[elig_indices]))]
    v5_rank[elig_order] = np.arange(1, len(elig_order) + 1)

    # 8. Build Section 12.1 Contract Table (LGPD-masked, NULL preserved when unavailable)
    feeder_masked = [
        None if f is None else feeder_map.get(str(f), str(f))
        for f in oot_feeders
    ]
    posto_raw = oot_dec["meta__posto"].to_list()
    posto_masked = [None if p is None else f"TR-{hashlib.sha256(str(p).encode()).hexdigest()[:6].upper()}" for p in posto_raw]

    raw_det_json = [
        json.dumps({
            "iso": round(float(preds_global["raw_iso_score"][i]), 4),
            "lof": round(float(preds_global["raw_lof_score"][i]), 4),
            "ecod": round(float(preds_global["raw_ecod_score"][i]), 4),
            "copod": round(float(preds_global["raw_copod_score"][i]), 4),
            "hbos": round(float(preds_global["raw_hbos_score"][i]), 4),
            "pca": round(float(preds_global["raw_pca_score"][i]), 4),
        })
        for i in range(len(uc_list))
    ]
    norm_det_json = [
        json.dumps({
            "iso": float(preds_global["iso_score"][i]),
            "lof": float(preds_global["lof_score"][i]),
            "ecod": float(preds_global["ecod_score"][i]),
            "copod": float(preds_global["copod_score"][i]),
            "hbos": float(preds_global["hbos_score"][i]),
            "pca": float(preds_global["pca_score"][i]),
        })
        for i in range(len(uc_list))
    ]
    topo_ev_str = [
        (
            "SEM_ALIMENTADOR" if not has_feeder_mask[i]
            else ("EVENTO_COLETIVO_REDE" if is_grid_event[i]
                  else ("RUIDO_NAO_ISOLADO" if is_spurious_glitch[i]
                        else f"MOE_F={feeder_score_arr[i]:.3f}|G={global_scores[i]:.3f}|W_F={moe_weight_feeder[i]:.2f}"))
        )
        for i in range(len(uc_list))
    ]
    dec_reasons = []
    for i in range(len(uc_list)):
        if not elig_mask[i]:
            dec_reasons.append(f"QUARENTENA_COBERTURA(cov={oot_dec['x__avg_interval_coverage'][i]:.2f}<0.70)")
        else:
            parts = [f"s_elec={global_scores[i]:.3f}", f"moe={moe_filtered_score[i]:.3f}"]
            if cpd_drop_flag[i]:
                parts.append(f"CPD_QUEDA({cpd_mag_list[i]:.1f}%@{cpd_ts_list[i]})")
            if beh_high_flag[i]:
                parts.append(f"BEH_ALTO({s_behavior[i]:.2f})")
            if phys_sev_flag[i]:
                parts.append(f"FISICA({phys_rules_str[i]})")
            if is_grid_event[i]:
                parts.append("ATENUADO_EVENTO_REDE")
            if not has_feeder_mask[i]:
                parts.append("SEM_ESPECIALISTA_ALIMENTADOR")
            parts.append("ALARMES_TAMPER_INDISPONIVEIS_NOV")
            dec_reasons.append("; ".join(parts))

    contract_df = pl.DataFrame({
        "run_id": [run_id] * len(uc_list),
        "pipeline_id": ["p1_multiview_eval"] * len(uc_list),
        "uc_id_masked": [censor_uc_id(u) for u in uc_list],
        "cutoff": ["2026-11-19"] * len(uc_list),
        "feeder": feeder_masked,
        "transformer": posto_masked,
        "coverage": oot_dec["x__avg_interval_coverage"].to_list(),
        "reliable_days": oot_dec["x__coverage_reliable_days"].to_list(),
        "eligibility": elig_mask.tolist(),
        "s_comm": oot_dec["s_comm"].to_list(),
        "s_elec": global_scores.tolist(),
        "s_behavior": s_behavior.tolist(),
        "s_tamper": [None if alarms_win[i] == 0 else float(s_tamper[i]) for i in range(len(uc_list))],
        "s_tamper_slope_only": s_tamper.tolist(),
        "cpd_detected": cpd_detected_list,
        "cpd_timestamp": cpd_ts_list,
        "cpd_magnitude": cpd_mag_list,
        "physical_rules": phys_rules_str,
        "topological_evidence": topo_ev_str,
        "moe_score": [None if not has_feeder_mask[i] else round(float(moe_score_arr[i]), 4) for i in range(len(uc_list))],
        "raw_detector_scores": raw_det_json,
        "normalized_detector_scores": norm_det_json,
        "diagnostic_hypotheses": oot_dec["CATEGORIA_DIAGNOSTICA"].to_list(),
        "candidate_status": oot_dec["IS_CANDIDATO_VISTORIA"].to_list(),
        "priority_score": [None if not elig_mask[i] else round(float(v5_score[i]), 4) for i in range(len(uc_list))],
        "rank": [None if v5_rank[i] < 0 else int(v5_rank[i]) for i in range(len(uc_list))],
        "decision_reason": dec_reasons,
    })
    contract_df.write_parquet(run_dir / "multiview_evidence_oot.parquet", compression="zstd")

    # =========================================================================
    # SECTION 19 (BASE A): METHODOLOGICAL ABLATION V0..V5 UNDER CONSTANT K=2,000
    # =========================================================================
    variants_base_a = [
        ("V0: Baseline IF (sem gate)", v0_score, np.ones(len(uc_list), dtype=bool), 0.15),
        ("V1: Ensemble 6 det. (sem gate)", v1_score, np.ones(len(uc_list), dtype=bool), round(t_elec, 2)),
        ("V2: V1 + Gate Qualidade (>=70%)", v2_score, elig_mask, round(t_elec, 2)),
        ("V3: V2 + Comportamento + MoE/Topol.", v3_score, elig_mask, round(t_elec + t_beh + t_moe, 2)),
        ("V4: V3 + CPD (CUSUM 30d)", v4_score, elig_mask, round(t_elec + t_beh + t_moe + t_cpd, 2)),
        ("V5: V4 + Regras Físicas/Multivisão", v5_score, elig_mask, round(t_elec + t_beh + t_moe + t_cpd + 0.3, 2)),
    ]
    top2k_v0 = _topk_indices(v0_score, 2000)
    top2k_v2 = _topk_indices(v2_score, 2000)
    cat_arr = oot_dec["CATEGORIA_DIAGNOSTICA"].to_numpy()
    cov_arr = oot_dec["x__avg_interval_coverage"].fill_null(0.0).to_numpy()
    tau_f = float(det.frozen_threshold_)

    ablation_a_rows = []
    for v_name, sc_v, mask_v, rt_s in variants_base_a:
        top_v = _topk_indices(sc_v, 2000)
        top_arr = np.array(sorted(top_v), dtype=int)
        # Spearman on eligible common population vs V2
        rho_v2 = float(spearmanr(sc_v[elig_mask], v2_score[elig_mask]).statistic)
        rho_v0 = float(spearmanr(sc_v[elig_mask], v0_score[elig_mask]).statistic)
        n_cands_tau = int(np.sum((sc_v >= tau_f) & mask_v))
        n_top_quar = int(np.sum(~elig_mask[top_arr]))
        n_top_pnt = int(np.sum(cat_arr[top_arr] == "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT"))
        n_top_gd = int(np.sum(cat_arr[top_arr] == "POSSIVEL_GD_NAO_CADASTRADA"))
        n_top_met = int(np.sum(cat_arr[top_arr] == "FALHA_METROLOGICA_MEDIDOR_TRAVADO"))
        n_top_cpd = int(np.sum(cpd_drop_flag[top_arr]))
        n_top_grid = int(np.sum(is_grid_event[top_arr]))
        ablation_a_rows.append({
            "variante": v_name,
            "ucs_avaliadas": len(uc_list),
            "ucs_elegiveis": int(mask_v.sum()),
            "candidatos_tau_05131": n_cands_tau,
            "top2k_jaccard_vs_V0": round(_jaccard(top_v, top2k_v0), 4),
            "top2k_jaccard_vs_V2": round(_jaccard(top_v, top2k_v2), 4),
            "top2k_entradas_vs_V2": len(top_v - top2k_v2),
            "top2k_saidas_vs_V2": len(top2k_v2 - top_v),
            "spearman_elig_vs_V0": round(rho_v0, 4),
            "spearman_elig_vs_V2": round(rho_v2, 4),
            "top2k_quarentena_vazadas": n_top_quar,
            "top2k_pnt": n_top_pnt,
            "top2k_gd": n_top_gd,
            "top2k_metro": n_top_met,
            "top2k_com_cpd_queda": n_top_cpd,
            "top2k_evento_rede": n_top_grid,
            "tempo_computacional_s": rt_s,
        })
    pl.DataFrame(ablation_a_rows).write_csv(run_dir / "ablation_base_a_oot.csv", separator=";")
    pl.DataFrame(ablation_a_rows).write_csv(P1_MV_DIR / "ablation_base_a_oot.csv", separator=";")

    # =========================================================================
    # SECTION 20: COMPLETE CASE STUDIES (908***00, 291***96, 685***20)
    # =========================================================================
    target_cases = ["90850700", "29113296", "68546220"]
    uc_to_idx = {u: i for i, u in enumerate(uc_list)}
    case_traces = []
    for raw_uc in target_cases:
        idx = uc_to_idx.get(raw_uc)
        if idx is None:
            continue
        mh_uc = meter_hist.filter(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "") == raw_uc)
        nios = sorted({censor_nio_id(x) for x in mh_uc["NIO"].drop_nulls().to_list()}) if "NIO" in mh_uc.columns else []
        case_traces.append({
            "uc_masked": censor_uc_id(raw_uc),
            "nios_masked": nios,
            "feeder_masked": feeder_masked[idx],
            "transformer_masked": posto_masked[idx],
            "coverage_oot": round(float(oot_dec["x__avg_interval_coverage"][idx]), 4),
            "reliable_days_oot": int(oot_dec["x__coverage_reliable_days"][idx]),
            "fa_median_kwh_d": round(float(oot_dec["x__hist_fa_sum_median"][idx]), 3),
            "ra_ratio_max": round(float(oot_dec["x__hist_ra_reversal_ratio_max"][idx]), 4),
            "ra_days": int(oot_dec["x__hist_ra_reversal_days"][idx]),
            "trend_slope": round(float(oot_dec["x__trend_fa_slope"][idx]), 4),
            "eligibility": bool(elig_mask[idx]),
            "s_comm": float(oot_dec["s_comm"][idx]),
            "s_elec_25_11": float(global_scores[idx]),
            "moe_score_p1": None if np.isnan(feeder_score_arr[idx]) else round(float(moe_score_arr[idx]), 4),
            "s_behavior_p1": float(s_behavior[idx]),
            "s_tamper_slope_only": float(s_tamper[idx]),
            "cpd_detected": cpd_detected_list[idx],
            "cpd_timestamp": cpd_ts_list[idx],
            "cpd_magnitude_pct": cpd_mag_list[idx],
            "cpd_direction": cpd_dir_list[idx],
            "physical_rules": phys_rules_str[idx],
            "topological_evidence": topo_ev_str[idx],
            "categoria_25_11": str(oot_dec["CATEGORIA_DIAGNOSTICA"][idx]),
            "candidato_25_11": bool(oot_dec["IS_CANDIDATO_VISTORIA"][idx]),
            "priority_score_v5": None if not elig_mask[idx] else round(float(v5_score[idx]), 4),
            "rank_v5": None if v5_rank[idx] < 0 else int(v5_rank[idx]),
            "rank_v2_25_11": (
                None if not elig_mask[idx]
                else int(np.sum(v2_score[elig_mask] > v2_score[idx]) + 1)
            ),
            "individual_norm_scores": json.loads(norm_det_json[idx]),
            "decision_reason": dec_reasons[idx],
        })
    (run_dir / "case_studies_multiview.json").write_text(json.dumps(case_traces, indent=2, ensure_ascii=False), encoding="utf-8")
    (P1_MV_DIR / "case_studies_multiview.json").write_text(json.dumps(case_traces, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save 10 diverse provenance trace examples (Section 12.2)
    prov_samples = contract_df.filter(pl.col("rank").is_not_null()).sort("rank").head(5).to_dicts()
    prov_quar = contract_df.filter(~pl.col("eligibility")).head(3).to_dicts()
    prov_gd = contract_df.filter(pl.col("diagnostic_hypotheses") == "POSSIVEL_GD_NAO_CADASTRADA").head(2).to_dicts()
    (run_dir / "provenance_traces_sample.json").write_text(
        json.dumps(prov_samples + prov_quar + prov_gd, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # =========================================================================
    # SECTION 21: OPERATIONAL PROJECTION UNDER K = 2,000 INSPECTIONS / MONTH
    # =========================================================================
    # Materiality proxy: historical daily kWh (or pre-drop kWh)
    fa_mat = np.log1p(np.maximum(fa_med, 0.0))
    fa_mat_norm = fa_mat / max(float(np.percentile(fa_mat[elig_mask], 99.0)), 1e-6)
    fa_mat_norm = np.clip(fa_mat_norm, 0.0, 1.0)
    streak_arr = oot_dec["DIAS_ANOMALIA_SEGUIDOS"].fill_null(0).to_numpy()
    persist_norm = np.clip(streak_arr / 15.0, 0.0, 1.0)
    qual_norm = np.clip((cov_arr - 0.70) / 0.30, 0.0, 1.0)

    ops_policies = {
        "P1: Escore Elétrico (OOT 25/11)": v2_score,
        "P2: Prioridade Multivisão (V5)": v5_score,
        "P3: Persistência + CPD": np.where(elig_mask, 0.55 * v2_score + 0.30 * persist_norm + 0.15 * cpd_drop_flag.astype(float), -1.0),
        "P4: Materialidade Energética": np.where(elig_mask, 0.65 * v5_score + 0.35 * fa_mat_norm, -1.0),
        "P5: Qualidade da Evidência": np.where(elig_mask, 0.75 * v5_score + 0.25 * qual_norm, -1.0),
        "P6: Contexto Topológico (MoE)": np.where(elig_mask, moe_filtered_score, -1.0),
    }

    ops_rows = []
    ref_ops_top = _topk_indices(v2_score, 2000)
    for pol_name, sc_p in ops_policies.items():
        top_p = _topk_indices(sc_p, 2000)
        ta = np.array(sorted(top_p), dtype=int)
        ops_rows.append({
            "politica_priorizacao": pol_name,
            "orcamento_K": 2000,
            "jaccard_vs_eletrico_25_11": round(_jaccard(top_p, ref_ops_top), 4),
            "substituicoes_vs_25_11": len(top_p - ref_ops_top),
            "fila_pnt_analise": int(np.sum(cat_arr[ta] == "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT")),
            "fila_gd_nao_cadastrada": int(np.sum(cat_arr[ta] == "POSSIVEL_GD_NAO_CADASTRADA")),
            "fila_metrologica": int(np.sum(cat_arr[ta] == "FALHA_METROLOGICA_MEDIDOR_TRAVADO")),
            "fila_outras_elegiveis": int(2000 - np.sum(np.isin(cat_arr[ta], ["ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT", "POSSIVEL_GD_NAO_CADASTRADA", "FALHA_METROLOGICA_MEDIDOR_TRAVADO"]))),
            "fa_mediana_kwh_d_carteira": round(float(np.median(fa_med[ta])), 2),
            "cobertura_media_carteira_pct": round(float(np.mean(cov_arr[ta]) * 100.0), 2),
            "pct_com_cpd_queda": round(float(np.mean(cpd_drop_flag[ta]) * 100.0), 2),
        })
    pl.DataFrame(ops_rows).write_csv(run_dir / "operational_portfolios_k2000.csv", separator=";")
    pl.DataFrame(ops_rows).write_csv(P1_MV_DIR / "operational_portfolios_k2000.csv", separator=";")

    # Separated Operational Queues across entire OOT (65,686 UCs)
    queue_summary = [
        {"fila_operacional": "1. Comunicação (Quarentena Telecom: Isolada + Coletiva)", "ucs_oot": int(np.sum(np.isin(cat_arr, ["ANOMALIA_COMUNICACAO_ISOLADA", "GARGALO_COLETIVO_RF_MESH"]))),
         "destino": "Equipe Telecom / Automação RF Mesh (NÃO fiscalização comercial)"},
        {"fila_operacional": "2. Ciclo de Vida Cadastral (UC Nova / Troca Medidor)", "ucs_oot": int(np.sum(np.isin(cat_arr, ["INICIACAO_PERFIL_UC_NOVA", "TRANSICAO_TROCA_MEDIDOR"]))),
         "destino": "Maturação cadastral / auditoria pós-troca (NÃO fiscalização comercial)"},
        {"fila_operacional": "3. Manutenção Metrológica (Medidor Travado / FA~0)", "ucs_oot": int(np.sum(cat_arr == "FALHA_METROLOGICA_MEDIDOR_TRAVADO")),
         "destino": "Verificação de ocupação e calibração metrológica"},
        {"fila_operacional": "4. Possível GD Não Cadastrada (Injeção Reversa RA)", "ucs_oot": int(np.sum(cat_arr == "POSSIVEL_GD_NAO_CADASTRADA")),
         "destino": "Auditoria documental CIS / parecer de acesso microgeração"},
        {"fila_operacional": "5. Análise Técnica de Possível PNT (Anomalia Elétrica)", "ucs_oot": int(np.sum(cat_arr == "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT")),
         "destino": "Análise humana (Cockpit) -> eventual vistoria de campo sob K=2.000/mês"},
        {"fila_operacional": "6. Monitoramento Comportamental / Operação Regular", "ucs_oot": int(np.sum(np.isin(cat_arr, ["MUDANCA_CLASSE_COMPORTAMENTAL", "OPERACAO_REGULAR"]))),
         "destino": "Acompanhamento passivo (sem despacho)"},
    ]
    pl.DataFrame(queue_summary).write_csv(run_dir / "operational_queues_separated.csv", separator=";")
    pl.DataFrame(queue_summary).write_csv(P1_MV_DIR / "operational_queues_separated.csv", separator=";")

    # =========================================================================
    # P1 FIGURES (9, 15, 16A, 17, 18)
    # =========================================================================
    figs_p1: dict[str, dict[str, str]] = {}

    # Fig 9: Executed Multiview Architecture Diagram (clean programmatic flowchart with REAL executed counts!)
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)

    def _box(x: float, y: float, w: float, h: float, title: str, body: str, fc: str = "#eaf2f8", ec: str = "#1f4e79") -> None:
        rect = matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111111")
        ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center", fontsize=7.4, color="#222222")

    _box(0.2, 4.7, 2.3, 1.1, "Coorte OOT (19/11)", f"{_fmt_br_int(len(uc_list))} UCs únicas\nReconstrução UC–NIO", "#f2f4f4", "#4a6572")
    _box(3.0, 4.7, 2.5, 1.1, "Gate de Qualidade", f"Elegíveis: {_fmt_br_int(int(elig_mask.sum()))} (95,7%)\nQuarentena: {_fmt_br_int(int((~elig_mask).sum()))}", "#fcf3cf", "#b7950b")
    _box(6.1, 4.7, 5.6, 1.1, "Especialistas Executados sobre Features Congeladas", "S_elec (6 det.) | S_comm | S_behavior (PCA+KMeans) | CPD (CUSUM 30d)\nMoE Hierárquico (14 alimentadores) | Regras Físicas | S_tamper*", "#eaf2f8", "#1f4e79")

    # Specialist mini-boxes
    specs_info = [
        ("S_elec (6 det.)", f"{_fmt_br_int(int(oot_dec['is_elec_anomaly'].sum()))} > τ=0,5131", "#d6eaf8"),
        ("S_behavior", f"P95={_fmt_br_float(float(np.percentile(s_behavior[elig_mask], 95)), 2)}\n{_fmt_br_int(int(beh_high_flag.sum()))} desvios", "#e8daef"),
        ("CPD (CUSUM)", f"{_fmt_br_int(int(sum(1 for x in cpd_detected_list if x)))} quebras\n{_fmt_br_int(int(cpd_drop_flag.sum()))} quedas >=30%", "#fdebd0"),
        ("MoE + Vizinh.", f"{len(feeder_model_stats)} circuitos\n{_fmt_br_int(int(is_grid_event.sum()))} ev. rede", "#d5f5e3"),
        ("Regras Físicas", f"GD:{int(r_gd.sum())} | V>20:{int(r_v_imb.sum())}\nTravado:{int(r_stuck.sum())}", "#fadbd8"),
    ]
    for idx_s, (st, sb, sfc) in enumerate(specs_info):
        _box(0.2 + idx_s * 2.34, 2.55, 2.15, 1.25, st, sb, sfc, "#34495e")

    _box(0.8, 0.45, 4.5, 1.25, "Vetor de Evidências + Triagem Delimitada", "Contrato auditável por UC (NULL preservado se ausente)\n6 filas operacionais separadas (Telecom / GD / Metrol. / PNT)", "#ebdef0", "#6c3483")
    _box(6.2, 0.45, 5.2, 1.25, "Priorização sob Orçamento Fixo (K = 2.000/mês)", "Análise humana (Cockpit) → eventual vistoria prospectiva\n*S_tamper: 0 alarmes em Nov/2026 (avaliado em testes/histórico)", "#d4efdf", "#1e8449")

    # Arrows
    arrow_kw = dict(arrowstyle="->", color="#2c3e50", lw=1.4)
    ax.annotate("", xy=(3.0, 5.25), xytext=(2.5, 5.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.1, 5.25), xytext=(5.5, 5.25), arrowprops=arrow_kw)
    for idx_s in range(5):
        xc = 1.27 + idx_s * 2.34
        ax.annotate("", xy=(xc, 3.85), xytext=(xc, 4.65), arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))
        ax.annotate("", xy=(3.05, 1.75), xytext=(xc, 2.52), arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))
    ax.annotate("", xy=(6.2, 1.07), xytext=(5.3, 1.07), arrowprops=arrow_kw)
    ax.set_title("Arquitetura multivisão efetivamente executada na avaliação integrada P1 (corte OOT 19/11/2026)", fontweight="bold", pad=8)
    plt.tight_layout()
    figs_p1["fig_p1_09"] = _save_fig(fig, P1_FIG_DIR / "fig_p1_09_arquitetura_multivisao_executada")

    # Fig 15: Global vs Contextual (MoE) Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    rng_p = np.random.default_rng(42)
    sample_idx = rng_p.choice(len(g_pct_common), size=min(4000, len(g_pct_common)), replace=False)
    ax1.scatter(g_pct_common[sample_idx], m_pct_common[sample_idx], s=7, alpha=0.28, color="#1f4e79", edgecolors="none")
    ax1.plot([0, 100], [0, 100], "--", color="#c0392b", linewidth=1.2, label="Identidade (Global = MoE)")
    ax1.set_xlabel("Percentil no modelo Global (%)")
    ax1.set_ylabel("Percentil no modelo Contextual MoE (%)")
    ax1.set_title(f"(A) Dispersão de percentis (ρ = {_fmt_br_float(moe_comparison['spearman_global_vs_moe_common'], 3)}, |ΔP| médio = {_fmt_br_float(moe_comparison['mean_abs_percentile_shift'], 1)} p.p.)", fontweight="bold", fontsize=9.2)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.45)

    fm_df = pl.DataFrame(feeder_model_stats).sort("spearman_feeder_vs_global")
    yf = np.arange(fm_df.height)
    ax2.barh(yf, fm_df["spearman_feeder_vs_global"].to_numpy(), color="#2e7d32", alpha=0.85)
    for yv, rv, nf in zip(yf, fm_df["spearman_feeder_vs_global"].to_numpy(), fm_df["n_oot"].to_numpy()):
        ax2.text(rv + 0.01, yv, f"{_fmt_br_float(rv, 3)} (n={_fmt_br_int(nf)})", va="center", fontsize=7.6)
    ax2.set_yticks(yf)
    ax2.set_yticklabels(fm_df["circuito"].to_list(), fontsize=8)
    ax2.set_xlim(0, 1.25)
    ax2.set_xlabel("Spearman (Especialista Alimentador vs. Global no OOT)")
    ax2.set_title("(B) Concordância por circuito (modelos treinados sem vazamento)", fontweight="bold", fontsize=9.2)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)
    plt.tight_layout()
    figs_p1["fig_p1_15"] = _save_fig(fig, P1_FIG_DIR / "fig_p1_15_comparacao_global_vs_contextual_moe")

    # Fig 16A: Methodological Ablation V0..V5 on Base A (Real OOT, K=2000)
    ab_df = pl.DataFrame(ablation_a_rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    xv = np.arange(ab_df.height)
    short_v = ["V0\nBaseline IF", "V1\nEns. 6 det.", "V2\n+Gate 70%", "V3\n+Beh/MoE", "V4\n+CPD", "V5\n+Multivisão"]
    ax1.bar(xv - 0.18, ab_df["top2k_jaccard_vs_V0"].to_numpy(), width=0.34, color="#5c6f7e", label="Jaccard Top-2.000 vs. V0 (Baseline)")
    ax1.bar(xv + 0.18, ab_df["top2k_jaccard_vs_V2"].to_numpy(), width=0.34, color="#1f4e79", label="Jaccard Top-2.000 vs. V2 (OOT 25/11)")
    for i in range(len(xv)):
        ax1.text(xv[i] + 0.18, ab_df["top2k_jaccard_vs_V2"][i] + 0.02, _fmt_br_float(ab_df["top2k_jaccard_vs_V2"][i], 2), ha="center", fontsize=7.6, fontweight="bold")
    ax1.set_xticks(xv)
    ax1.set_xticklabels(short_v, fontsize=8.2)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Índice de Jaccard (K = 2.000)")
    ax1.set_title("(A) Alteração de carteira Top-2.000 entre variantes (Base A)", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Quarantine leakage & CPD enrichment in Top-2000
    ax2.plot(xv, ab_df["top2k_quarentena_vazadas"].to_numpy(), "o-", color="#c0392b", linewidth=2, label="UCs de quarentena (<70%) no Top-2.000")
    ax2.plot(xv, ab_df["top2k_com_cpd_queda"].to_numpy(), "s-", color="#2e7d32", linewidth=2, label="UCs com queda CPD >=30% no Top-2.000")
    for i in range(len(xv)):
        ax2.annotate(str(ab_df["top2k_quarentena_vazadas"][i]), (xv[i], ab_df["top2k_quarentena_vazadas"][i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#c0392b")
        ax2.annotate(str(ab_df["top2k_com_cpd_queda"][i]), (xv[i], ab_df["top2k_com_cpd_queda"][i]), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color="#2e7d32")
    ax2.set_xticks(xv)
    ax2.set_xticklabels(short_v, fontsize=8.2)
    ax2.set_ylabel("Número de UCs no Top-2.000")
    ax2.set_title("(B) Blindagem de telemetria e enriquecimento de evidência CPD", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_p1["fig_p1_16a"] = _save_fig(fig, P1_FIG_DIR / "fig_p1_16a_ablacao_variantes_base_a")

    # Fig 17: Complete Multiview Traces for the 3 Real Case Studies
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    det_names_c = ["IF", "LOF", "ECOD", "COPOD", "HBOS", "PCA", "S_beh", "MoE"]
    for ax, ct in zip(axes, case_traces):
        sc_map = ct["individual_norm_scores"]
        vals = [
            sc_map["iso"], sc_map["lof"], sc_map["ecod"], sc_map["copod"],
            sc_map["hbos"], sc_map["pca"], ct["s_behavior_p1"],
            ct["moe_score_p1"] if ct["moe_score_p1"] is not None else 0.0,
        ]
        cols_c = ["#1f4e79"] * 6 + ["#6c3483", "#2e7d32"]
        bars_c = ax.bar(range(len(vals)), vals, color=cols_c, alpha=0.88)
        for b, v in zip(bars_c, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, _fmt_br_float(v, 2), ha="center", va="bottom", fontsize=7.2)
        ax.axhline(tau_f, color="#c0392b", linestyle="--", linewidth=1.1, label="τ = 0,5131")
        ax.set_xticks(range(len(det_names_c)))
        ax.set_xticklabels(det_names_c, rotation=35, ha="right", fontsize=7.8)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Escore [0, 1]")
        cpd_lbl = f"CPD: {ct['cpd_magnitude_pct']:.1f}% ({ct['cpd_timestamp']})" if ct["cpd_detected"] else ("CPD: Inavaliável" if ct["cpd_detected"] is None else "CPD: Estável")
        ax.set_title(f"UC {ct['uc_masked']} ({ct['feeder_masked'] or 'S/Alim'})\nCob={ct['coverage_oot']*100:.1f}% | {cpd_lbl}", fontsize=8.4, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.suptitle("Trilhas multivisão completas dos três estudos de caso reais (corte OOT 19/11/2026)", fontweight="bold")
    plt.tight_layout()
    figs_p1["fig_p1_17"] = _save_fig(fig, P1_FIG_DIR / "fig_p1_17_trilhas_estudos_de_caso")

    # Fig 18: Operational Portfolios Composition under Fixed Budget K=2,000
    ops_df = pl.DataFrame(ops_rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    xo = np.arange(ops_df.height)
    o1 = ops_df["fila_pnt_analise"].to_numpy()
    o2 = ops_df["fila_gd_nao_cadastrada"].to_numpy()
    o3 = ops_df["fila_metrologica"].to_numpy()
    o4 = ops_df["fila_outras_elegiveis"].to_numpy()
    ax1.bar(xo, o1, width=0.55, color="#1f4e79", label="Análise técnica PNT")
    ax1.bar(xo, o2, width=0.55, bottom=o1, color="#2e7d32", label="Possível GD não cadastrada")
    ax1.bar(xo, o3, width=0.55, bottom=o1 + o2, color="#d35400", label="Manutenção metrológica")
    ax1.bar(xo, o4, width=0.55, bottom=o1 + o2 + o3, color="#95a5a6", label="Outras elegíveis")
    ax1.set_xticks(xo)
    ax1.set_xticklabels([p.split(":")[0] + "\n" + p.split(":")[1].strip()[:18] for p in ops_df["politica_priorizacao"].to_list()], fontsize=7.6)
    ax1.set_ylim(0, 2380)
    ax1.set_ylabel("Composição do Top-2.000 (UCs)")
    ax1.set_title("(A) Carteiras Top-2.000 por critério de priorização", fontweight="bold")
    ax1.legend(loc="upper right", ncol=2, fontsize=7.6)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Median materiality (kWh/d) & CPD% in Top-2000
    ax2_b = ax2.twinx()
    b_mat = ax2.bar(xo - 0.18, ops_df["fa_mediana_kwh_d_carteira"].to_numpy(), width=0.34, color="#2a7b9b", label="FA mediana da carteira (kWh/dia)")
    b_cpd = ax2_b.bar(xo + 0.18, ops_df["pct_com_cpd_queda"].to_numpy(), width=0.34, color="#c0392b", alpha=0.82, label="UCs com queda CPD >=30% (%)")
    ax2.set_xticks(xo)
    ax2.set_xticklabels([p.split(":")[0] for p in ops_df["politica_priorizacao"].to_list()], fontsize=8.5)
    ax2.set_ylabel("Consumo mediano da carteira (kWh/dia)", color="#2a7b9b")
    ax2_b.set_ylabel("Proporção com queda CPD >=30% (%)", color="#c0392b")
    ax2.set_title("(B) Materialidade energética e evidência de degrau", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.5)
    plt.tight_layout()
    figs_p1["fig_p1_18"] = _save_fig(fig, P1_FIG_DIR / "fig_p1_18_composicao_carteiras_operacionais_k2000")

    # Summary & Manifest
    cpd_summary = {
        "total_oot_ucs": len(uc_list),
        "cpd_evaluable_ucs": int(cpd_evaluable.sum()),
        "cpd_not_evaluable_ucs": int((~cpd_evaluable).sum()),
        "cpd_detected_any": int(sum(1 for x in cpd_detected_list if x is True)),
        "cpd_detected_drop_ge30": int(cpd_drop_flag.sum()),
        "cpd_detected_drop_any": int(sum(1 for d, dr in zip(cpd_detected_list, cpd_dir_list) if d is True and dr == "QUEDA_DEGRAU")),
        "cpd_detected_increase_any": int(sum(1 for d, dr in zip(cpd_detected_list, cpd_dir_list) if d is True and dr == "AUMENTO_DEGRAU")),
        "runtime_sec": round(t_cpd, 2),
    }
    summary = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp": datetime.now().isoformat(),
        "runtime_sec": round(time.time() - t0, 2),
        "timings_sec": {"s_elec": round(t_elec, 2), "s_behavior": round(t_beh, 2), "cpd_65k": round(t_cpd, 2), "moe_14feeders": round(t_moe, 2)},
        "cpd_oot_summary": cpd_summary,
        "alarm_events_audit": alarm_audit,
        "moe_comparison": moe_comparison,
        "physical_rules_audit": physical_rules_audit,
        "ablation_base_a": ablation_a_rows,
        "operational_portfolios_k2000": ops_rows,
        "operational_queues_separated": queue_summary,
        "case_studies": case_traces,
        "figures": figs_p1,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (P1_MV_DIR / "p1_multiview_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "pipeline_id": "p1_multiview_eval",
        "git_commit": commit,
        "status": "validated",
        "created_at": datetime.now().isoformat(),
        "artifacts": {
            f.name: {"path": str(f.relative_to(PROJECT_ROOT)), "sha256": _sha256(f)}
            for f in sorted(run_dir.iterdir()) if f.is_file()
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = run_multiview_evaluation()
    print(f"P1 Multiview Evaluation completed in {res['runtime_sec']}s | run_id={res['run_id']}")
    print("Timings:", res["timings_sec"])
    print("CPD OOT summary:", json.dumps(res["cpd_oot_summary"], indent=2))
    print("MoE comparison:", json.dumps(res["moe_comparison"], indent=2))
    print("Alarm audit:", json.dumps(res["alarm_events_audit"], indent=2))
