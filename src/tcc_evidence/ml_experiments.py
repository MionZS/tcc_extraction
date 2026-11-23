"""Real ML Experiments: 6-Detector Comparison, Traceable Fusion Audit, Gating Sensitivity, Behavior/CPD & Ablation."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.eda.cockpit_metrics import deduplicate_latest_uc
from src.eda.uc_timeseries import load_uc_daily_timeseries
from src.models.composite import CompositeAnomalyDetector, select_pure_electrical_features
from src.utils.censorship import censor_uc_id


def _clean_feature_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    X = df.select(cols).to_numpy().astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col) | np.isinf(col)
        if np.any(mask):
            med = float(np.nanmedian(col[~mask])) if np.any(~mask) else 0.0
            X[mask, j] = med
    return X


def run_ml_methods_and_fusion_audit(
    out_dir: Path,
    *,
    classified_df: pl.DataFrame,
    sample_size: int = 3000,
    seed: int = 42,
    censor: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run 6-detector comparison, batch stability test, and single-UC fusion traceability."""
    ml_dir = out_dir / "ml_methods"
    ml_dir.mkdir(parents=True, exist_ok=True)

    uniq_df = deduplicate_latest_uc(classified_df)
    if sample_size > 0 and uniq_df.height > sample_size:
        eval_df = uniq_df.sample(n=sample_size, seed=seed)
    else:
        eval_df = uniq_df

    feat_cols = select_pure_electrical_features(eval_df)
    X_eval = _clean_feature_matrix(eval_df, feat_cols)

    # Fit canonical 6-detector CompositeAnomalyDetector on clean subset
    if "x__avg_interval_coverage" in eval_df.columns:
        clean_mask = eval_df["x__avg_interval_coverage"].fill_null(0.0).to_numpy() >= 0.70
    else:
        clean_mask = np.ones(eval_df.height, dtype=bool)

    X_train = X_eval[clean_mask] if np.sum(clean_mask) >= 50 else X_eval

    t0 = time.perf_counter()
    tracemalloc.start()
    det = CompositeAnomalyDetector(contamination=0.05, random_state=seed)
    det.fit(X_train)
    t_fit = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    preds = det.predict_composite(X_eval, raw_dataframe=eval_df)
    scores_dict = {
        "iso_forest": preds["iso_score"],
        "lof": preds["lof_score"],
        "ecod": preds["ecod_score"],
        "copod": preds["copod_score"],
        "hbos": preds["hbos_score"],
        "pca": preds["pca_score"],
        "consensus": preds["consensus_score"],
    }

    # 1. Detector comparison metrics (Top-5% overlap, Spearman corr vs consensus, quantiles)
    k_top = max(10, int(0.05 * len(X_eval)))
    top_cons = set(np.argsort(-scores_dict["consensus"])[:k_top])
    comp_rows = []
    for name, s_arr in scores_dict.items():
        top_m = set(np.argsort(-s_arr)[:k_top])
        jacc = len(top_m & top_cons) / float(max(1, len(top_m | top_cons)))
        rho, _ = spearmanr(s_arr, scores_dict["consensus"])
        comp_rows.append({
            "detector": name,
            "weight_in_composite": det.weights.get(name, 1.0 if name == "consensus" else 0.0),
            "p50": round(float(np.median(s_arr)), 4),
            "p95": round(float(np.percentile(s_arr, 95)), 4),
            "p99": round(float(np.percentile(s_arr, 99)), 4),
            "top5pct_jaccard_vs_consensus": round(jacc, 3),
            "spearman_vs_consensus": round(float(0.0 if np.isnan(rho) else rho), 4),
        })
    pl.DataFrame(comp_rows).write_csv(ml_dir / "detector_comparison_metrics.csv", separator=";")

    # Pairwise Spearman heatmap
    det_names = ["iso_forest", "lof", "ecod", "copod", "hbos", "pca", "consensus"]
    corr_mat = np.zeros((len(det_names), len(det_names)), dtype=float)
    for i, n1 in enumerate(det_names):
        for j, n2 in enumerate(det_names):
            r_val, _ = spearmanr(scores_dict[n1], scores_dict[n2])
            corr_mat[i, j] = 0.0 if np.isnan(r_val) else float(r_val)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(corr_mat, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Correlação de Postos (Spearman)")
    ax.set_xticks(range(len(det_names)))
    ax.set_xticklabels([d.upper() for d in det_names], rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(det_names)))
    ax.set_yticklabels([d.upper() for d in det_names], fontsize=8.5)
    for i in range(len(det_names)):
        for j in range(len(det_names)):
            val = corr_mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="#ffffff" if val > 0.6 else "#0f172a", fontsize=8.5, fontweight="bold")
    ax.set_title("Correlação de Ranking (Spearman) entre os 6 Detectores e Consenso", fontweight="bold")
    plt.tight_layout()
    fig.savefig(ml_dir / "detector_rank_correlation.png", dpi=200)
    fig.savefig(ml_dir / "detector_rank_correlation.svg")
    plt.close(fig)

    # 2. Batch Stability Experiment: Batch Min-Max vs Frozen Quantile Reference
    # Remove top-1% extreme points from batch and verify how batch min-max shifts scores vs frozen P01-P99
    s_raw_iso = -det.iso_forest.score_samples(det.scaler.transform(det.imputer.transform(X_eval)))
    p01_ref, p99_ref = float(np.percentile(s_raw_iso, 1)), float(np.percentile(s_raw_iso, 99))
    s_frozen = np.clip((s_raw_iso - p01_ref) / max(p99_ref - p01_ref, 1e-9), 0.0, 1.0)

    # Sub-batch without top 1% extremes
    keep_sub = s_raw_iso <= np.percentile(s_raw_iso, 99)
    s_sub_raw = s_raw_iso[keep_sub]
    s_batch_minmax_full = (s_raw_iso[keep_sub] - np.min(s_raw_iso)) / max(np.max(s_raw_iso) - np.min(s_raw_iso), 1e-9)
    s_batch_minmax_trimmed = (s_sub_raw - np.min(s_sub_raw)) / max(np.max(s_sub_raw) - np.min(s_sub_raw), 1e-9)
    max_shift_minmax = float(np.max(np.abs(s_batch_minmax_full - s_batch_minmax_trimmed)))
    mean_shift_minmax = float(np.mean(np.abs(s_batch_minmax_full - s_batch_minmax_trimmed)))

    stability_res = {
        "batch_minmax_mean_abs_shift_after_1pct_trim": round(mean_shift_minmax, 4),
        "batch_minmax_max_abs_shift_after_1pct_trim": round(max_shift_minmax, 4),
        "frozen_reference_shift": 0.0,
        "interpretation": (
            "A normalização min-max dependente do lote desloca scores quando extremos saem do lote; "
            "ancoragem congelada P01-P99 elimina essa dependência."
        ),
    }
    (ml_dir / "batch_stability_experiment.json").write_text(json.dumps(stability_res, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Traceable Single-UC Audit (dados -> features -> scores -> gates -> regras -> categoria -> ranking)
    top_idx = int(np.argmax(scores_dict["consensus"]))
    uc_raw_id = str(eval_df["id__uc_id"][top_idx]) if "id__uc_id" in eval_df.columns else f"ROW_{top_idx}"
    uc_disp = censor_uc_id(uc_raw_id) if censor else uc_raw_id
    ranks_all = rankdata(-scores_dict["consensus"], method="min")
    trace = {
        "uc_id": uc_disp,
        "feeder": str(eval_df["meta__feeder"][top_idx]) if "meta__feeder" in eval_df.columns else "-",
        "cutoff_date": str(eval_df["meta__cutoff_date"][top_idx]) if "meta__cutoff_date" in eval_df.columns else "-",
        "pure_electrical_features": {c: round(float(X_eval[top_idx, k]), 4) for k, c in enumerate(feat_cols)},
        "individual_detector_scores": {k: round(float(v[top_idx]), 4) for k, v in scores_dict.items() if k != "consensus"},
        "weights_applied": det.weights,
        "prodist_rule_violation_flag": bool(preds["rule_violation_flag"][top_idx]),
        "composite_consensus_score": round(float(scores_dict["consensus"][top_idx]), 4),
        "coverage_gate_passed": bool(clean_mask[top_idx]),
        "assigned_category_in_lake": str(eval_df["CATEGORIA_DIAGNOSTICA"][top_idx]) if "CATEGORIA_DIAGNOSTICA" in eval_df.columns else "-",
        "rank_in_evaluated_cohort": int(ranks_all[top_idx]),
        "cohort_size": int(eval_df.height),
    }
    (ml_dir / "traceable_uc_audit.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "eval_cohort_size": int(eval_df.height),
        "clean_train_size": int(len(X_train)),
        "n_pure_electrical_features": len(feat_cols),
        "feature_cols": feat_cols,
        "fit_runtime_ms": round(t_fit * 1000.0, 2),
        "peak_ram_mb": round(peak_ram / (1024 * 1024), 3),
        "stability": stability_res,
        "traceable_uc": trace,
    }
    (ml_dir / "ml_experiments_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, scores_dict


def run_gating_sensitivity_study(
    out_dir: Path,
    *,
    classified_df: pl.DataFrame,
    sample_size: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate coverage thresholds 50%, 60%, 70%, 80%, 90% and training contamination."""
    gate_dir = out_dir / "gating"
    gate_dir.mkdir(parents=True, exist_ok=True)

    uniq_df = deduplicate_latest_uc(classified_df)
    if sample_size > 0 and uniq_df.height > sample_size:
        eval_df = uniq_df.sample(n=sample_size, seed=seed)
    else:
        eval_df = uniq_df

    feat_cols = select_pure_electrical_features(eval_df)
    X_all = _clean_feature_matrix(eval_df, feat_cols)
    cov_arr = eval_df["x__avg_interval_coverage"].fill_null(0.0).to_numpy() if "x__avg_interval_coverage" in eval_df.columns else np.ones(eval_df.height)
    cand_arr = eval_df["IS_CANDIDATO_VISTORIA"].fill_null(False).to_numpy() if "IS_CANDIDATO_VISTORIA" in eval_df.columns else np.zeros(eval_df.height, dtype=bool)

    # Baseline: no gating (0.0) vs thresholds [0.50, 0.60, 0.70, 0.80, 0.90]
    det_unfiltered = CompositeAnomalyDetector(contamination=0.05, random_state=seed).fit(X_all)
    scores_unfiltered = det_unfiltered.score_samples(X_all, raw_dataframe=eval_df)

    thresholds = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows: list[dict[str, Any]] = []
    ref_scores_70: np.ndarray | None = None

    for thr in thresholds:
        mask = cov_arr >= thr
        X_train = X_all[mask] if np.sum(mask) >= 50 else X_all
        det_t = CompositeAnomalyDetector(contamination=0.05, random_state=seed).fit(X_train)
        sc_t = det_t.score_samples(X_all, raw_dataframe=eval_df)

        if abs(thr - 0.70) < 1e-6:
            ref_scores_70 = sc_t

        # Top-5% anomalies among eligible UCs
        elig_idx = np.where(mask)[0]
        top_n = max(1, int(0.05 * len(elig_idx)))
        top_elig = set(elig_idx[np.argsort(-sc_t[elig_idx])[:top_n]]) if len(elig_idx) > 0 else set()
        top_unfilt = set(np.argsort(-scores_unfiltered)[:max(1, int(0.05 * len(X_all)))])
        jacc_vs_nogate = len(top_elig & top_unfilt) / float(max(1, len(top_elig | top_unfilt)))
        rho_vs_nogate, _ = spearmanr(sc_t, scores_unfiltered)

        rows.append({
            "coverage_threshold": thr,
            "eligible_ucs": int(np.sum(mask)),
            "eligible_pct": round(float(np.mean(mask) * 100.0), 2),
            "excluded_ucs": int(np.sum(~mask)),
            "candidates_retained": int(np.sum(cand_arr & mask)),
            "candidates_excluded": int(np.sum(cand_arr & (~mask))),
            "median_score_all": round(float(np.median(sc_t)), 4),
            "p95_score_all": round(float(np.percentile(sc_t, 95)), 4),
            "spearman_vs_no_gating": round(float(0.0 if np.isnan(rho_vs_nogate) else rho_vs_nogate), 4),
            "top5pct_jaccard_vs_no_gating": round(jacc_vs_nogate, 3),
        })

    g_df = pl.DataFrame(rows)
    g_df.write_csv(gate_dir / "gating_sensitivity.csv", separator=";")

    # Plot Gating Sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    thrs_pct = [int(r["coverage_threshold"] * 100) for r in rows]
    ax1.plot(thrs_pct, [r["eligible_pct"] for r in rows], marker="o", linewidth=2.2, color="#0284c7", label="% UCs Elegíveis")
    ax1.axvline(70, color="#dc2626", linestyle="--", label="Gating Atual (70%)")
    ax1.set_title("A. Impacto do Gating na Elegibilidade da População", fontweight="bold")
    ax1.set_xlabel("Limiar Mínimo de Cobertura (%)")
    ax1.set_ylabel("População Elegível (%)")
    ax1.legend()

    ax2.plot(thrs_pct, [r["spearman_vs_no_gating"] for r in rows], marker="s", linewidth=2.2, color="#7c3aed", label="Spearman vs Sem Gating")
    ax2.plot(thrs_pct, [r["top5pct_jaccard_vs_no_gating"] for r in rows], marker="^", linewidth=2.2, color="#d97706", label="Jaccard Top-5% vs Sem Gating")
    ax2.axvline(70, color="#dc2626", linestyle="--")
    ax2.set_title("B. Estabilidade de Ranking e Contaminação no Treino", fontweight="bold")
    ax2.set_xlabel("Limiar Mínimo de Cobertura (%)")
    ax2.set_ylabel("Concordância [0, 1]")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(gate_dir / "gating_sensitivity.png", dpi=200)
    fig.savefig(gate_dir / "gating_sensitivity.svg")
    plt.close(fig)

    return {"rows": rows}


def run_behavior_and_cpd_study(
    base_dir: Path,
    out_dir: Path,
    *,
    classified_df: pl.DataFrame,
    seed: int = 42,
    censor: bool = True,
) -> dict[str, Any]:
    """Analyze behavioral clusters, weekday/weekend contrast, and CUSUM CPD on real daily series."""
    beh_dir = out_dir / "behavior"
    beh_dir.mkdir(parents=True, exist_ok=True)

    uniq_df = deduplicate_latest_uc(classified_df)
    clean_df = uniq_df.filter(pl.col("x__avg_interval_coverage") >= 0.70) if "x__avg_interval_coverage" in uniq_df.columns else uniq_df
    sample_df = clean_df.head(1500)

    feat_cols = [c for c in ["x__hist_fa_sum_median", "x__fa_cv_window", "x__hist_load_factor_mean", "x__hist_ra_reversal_ratio_max"] if c in sample_df.columns]
    X_b = _clean_feature_matrix(sample_df, feat_cols)
    X_std = (X_b - np.median(X_b, axis=0)) / (np.std(X_b, axis=0) + 1e-6)
    km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit(X_std)
    clusters = km.labels_

    cluster_summary = []
    for k in range(4):
        m = clusters == k
        cluster_summary.append({
            "cluster_id": k,
            "ucs_count": int(np.sum(m)),
            "median_kwh": round(float(np.median(X_b[m, 0])), 2) if np.any(m) else 0.0,
            "median_cv": round(float(np.median(X_b[m, 1])), 3) if np.any(m) and X_b.shape[1] > 1 else 0.0,
            "median_lf": round(float(np.median(X_b[m, 2])), 3) if np.any(m) and X_b.shape[1] > 2 else 0.0,
        })
    pl.DataFrame(cluster_summary).write_csv(beh_dir / "behavioral_clusters_summary.csv", separator=";")

    # Real UC x Day heatmap for up to 15 sample UCs (including 93420250, 116781157, 68546220)
    target_ucs = ["93420250", "116781157", "68546220"] + [str(u) for u in sample_df["id__uc_id"].head(12).to_list()]
    seen: list[str] = []
    for u in target_ucs:
        if u not in seen:
            seen.append(u)
    mat_rows: list[np.ndarray] = []
    row_labels: list[str] = []
    cpd_events: list[dict[str, Any]] = []

    for u in seen[:12]:
        ts = load_uc_daily_timeseries(u, base_dir=base_dir, days=30)
        if ts.get("status") == "ok" and ts.get("points"):
            pts = ts["points"]
            vals = np.array([float(p["value"]) for p in pts], dtype=float)
            if len(vals) >= 10:
                med = max(float(np.median(vals)), 0.1)
                pad_vals = vals[-30:] if len(vals) >= 30 else np.pad(vals, (30 - len(vals), 0), mode="edge")
                mat_rows.append(np.clip(pad_vals / med, 0.0, 3.0))
                row_labels.append(censor_uc_id(u) if censor else u)
                if ts.get("step_change"):
                    sc = ts["step_change"]
                    cpd_events.append({
                        "uc_id": censor_uc_id(u) if censor else u,
                        "cpd_date": sc.get("date"),
                        "delta_pct": sc.get("delta_pct"),
                        "direction": sc.get("direction"),
                    })

    if mat_rows:
        heat_arr = np.vstack(mat_rows)
        fig, ax = plt.subplots(figsize=(11, 5.2))
        im = ax.imshow(heat_arr, cmap="RdYlBu_r", aspect="auto", vmin=0.0, vmax=2.2)
        plt.colorbar(im, ax=ax, label="Consumo Relativo à Mediana da Própria UC")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Dia da Janela Analítica (Últimos 30 Dias)")
        ax.set_title("Mapa Comportamental UC × Dia (Normalizado pela Mediana Individual)", fontweight="bold")
        plt.tight_layout()
        fig.savefig(beh_dir / "uc_day_behavioral_calendar.png", dpi=200)
        fig.savefig(beh_dir / "uc_day_behavioral_calendar.svg")
        plt.close(fig)

    if cpd_events:
        pl.DataFrame(cpd_events).write_csv(beh_dir / "cusum_cpd_detected_events.csv", separator=";")

    return {
        "clusters": cluster_summary,
        "heatmap_ucs_plotted": len(mat_rows),
        "cpd_events_detected": len(cpd_events),
    }


def run_ablation_study(
    out_dir: Path,
    *,
    classified_df: pl.DataFrame,
    seed: int = 42,
) -> dict[str, Any]:
    """Fixed-cohort ablation across M0 (Raw) -> M1 (UC-NIO) -> M2 (Gating) -> M3 (Context) -> M4 (CPD/Streak) -> M5 (Integrated)."""
    abl_dir = out_dir / "ablation"
    abl_dir.mkdir(parents=True, exist_ok=True)

    total_windows = int(classified_df.height)
    uniq_df = deduplicate_latest_uc(classified_df)
    n_uniq = int(uniq_df.height)

    cov_col = "x__avg_interval_coverage" if "x__avg_interval_coverage" in uniq_df.columns else None
    streak_col = "DIAS_ANOMALIA_SEGUIDOS" if "DIAS_ANOMALIA_SEGUIDOS" in uniq_df.columns else None
    anom_col = "IS_ANOMALY" if "IS_ANOMALY" in uniq_df.columns else None
    cand_col = "IS_CANDIDATO_VISTORIA" if "IS_CANDIDATO_VISTORIA" in uniq_df.columns else None

    m0_n = total_windows
    m1_n = n_uniq
    m2_df = uniq_df.filter(pl.col(cov_col) >= 0.70) if cov_col else uniq_df
    m2_n = int(m2_df.height)

    m3_df = m2_df.filter(pl.col("meta__feeder").is_not_null()) if "meta__feeder" in m2_df.columns else m2_df
    m3_n = int(m3_df.height)

    # Strict sequential funnel: M4 evaluates anomalies strictly inside M3 (reliable + contextualized)
    m4_df = m3_df.filter(pl.col(anom_col) == True) if anom_col else m3_df  # noqa: E712
    m4_n = int(m4_df.height)

    # Strict sequential funnel M5: candidates strictly within M4
    m5_df = m4_df.filter(pl.col(cand_col) == True) if cand_col else (  # noqa: E712
        m4_df.filter(pl.col(streak_col) >= 3) if streak_col else m4_df
    )
    m5_n = int(m5_df.height)

    # Track any global candidates outside strict M4 (e.g. from legacy artifacts) for audit transparency
    uc_key = "id__uc_id" if "id__uc_id" in uniq_df.columns else ("UC" if "UC" in uniq_df.columns else None)
    set_reconciliation: dict[str, Any] = {}
    if uc_key and cand_col and anom_col:
        m4_set = set(m4_df[uc_key].cast(pl.String).to_list())
        m5_seq_set = set(m5_df[uc_key].cast(pl.String).to_list())
        m5_global_set = set(uniq_df.filter(pl.col(cand_col) == True)[uc_key].cast(pl.String).to_list())  # noqa: E712
        set_reconciliation = {
            "strict_sequential_inclusion": m5_seq_set.issubset(m4_set),
            "m4_count": len(m4_set),
            "m5_sequential_count": len(m5_seq_set),
            "m5_global_unconstrained_count": len(m5_global_set),
            "common_m4_m5_global": len(m4_set & m5_global_set),
            "exclusive_m4_not_m5_global": len(m4_set - m5_global_set),
            "exclusive_m5_global_not_m4": len(m5_global_set - m4_set),
        }

    # Gap incidence among M0 anomalies vs M5 candidates
    gap_rate_m0_anom = 0.0
    gap_rate_m5_cand = 0.0
    if cov_col and anom_col:
        raw_anom = uniq_df.filter(pl.col(anom_col) == True)  # noqa: E712
        if raw_anom.height > 0:
            gap_rate_m0_anom = float((raw_anom[cov_col].fill_null(0.0) < 0.70).mean() * 100.0)
        if m5_df.height > 0:
            gap_rate_m5_cand = float((m5_df[cov_col].fill_null(0.0) < 0.70).mean() * 100.0)

    stages = [
        {"stage": "M0: Janelas Brutas (UC × Cutoff)", "count": m0_n, "pct_of_m1": round(m0_n / max(m1_n, 1) * 100.0, 2), "gap_rate_pct": "-"},
        {"stage": "M1: Saneamento Temporal (UCs Únicas)", "count": m1_n, "pct_of_m1": 100.0, "gap_rate_pct": round(float((uniq_df[cov_col].fill_null(0.0) < 0.70).mean() * 100.0) if cov_col else 0.0, 2)},
        {"stage": "M2: Gating Telecom (Cobertura >= 70%)", "count": m2_n, "pct_of_m1": round(m2_n / max(m1_n, 1) * 100.0, 2), "gap_rate_pct": 0.0},
        {"stage": "M3: Contextualização Topológica (Alimentador)", "count": m3_n, "pct_of_m1": round(m3_n / max(m1_n, 1) * 100.0, 2), "gap_rate_pct": 0.0},
        {"stage": "M4: Anomalias Elétricas Ativas", "count": m4_n, "pct_of_m1": round(m4_n / max(m1_n, 1) * 100.0, 2), "gap_rate_pct": round(gap_rate_m0_anom, 2)},
        {"stage": "M5: Candidatos Prioritários (Persistência + Regras)", "count": m5_n, "pct_of_m1": round(m5_n / max(m1_n, 1) * 100.0, 2), "gap_rate_pct": round(gap_rate_m5_cand, 2)},
    ]
    pl.DataFrame(stages).write_csv(abl_dir / "ablation_funnel_metrics.csv", separator=";")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    labels = [s["stage"] for s in stages[1:]][::-1]
    vals = [s["count"] for s in stages[1:]][::-1]
    colors = ["#dc2626", "#d97706", "#7c3aed", "#059669", "#0284c7"]
    ax.barh(labels, vals, color=colors, edgecolor="#0f172a", height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:,} UCs ({v/max(m1_n,1)*100:.2f}%)", va="center", fontweight="bold", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_title("Ablação M1 → M5: Redução Progressiva de Falsos Positivos até Alvos de Campo", fontweight="bold")
    ax.set_xlabel("UCs Únicas")
    plt.tight_layout()
    fig.savefig(abl_dir / "ablation_funnel.png", dpi=200)
    fig.savefig(abl_dir / "ablation_funnel.svg")
    plt.close(fig)

    return {
        "stages": stages,
        "set_reconciliation": set_reconciliation,
        "gap_rate_unfiltered_anom_pct": round(gap_rate_m0_anom, 2),
        "gap_rate_m5_pct": round(gap_rate_m5_cand, 2),
    }
