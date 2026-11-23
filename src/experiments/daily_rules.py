"""Source Audit & DAILY_RULES Temporal Evaluation Runner."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.datasets.retroactive_cleaner import _read_table_flexible
from src.eda.uc_timeseries import _list_dated_parquets
from src.experiments.contracts import (
    ExperimentConfig,
    InvariantReport,
    RunManifest,
    atomic_write_json,
    validate_population_partition,
    validate_portfolio_queue_sets,
    validate_score_and_kpi_integrity,
    validate_temporal_no_leakage,
)
from src.utils.censorship import censor_uc_id

BASE_DIR = Path("output")
CONTEXT_DIR = BASE_DIR / "context"
MEAS_DIR = BASE_DIR / "measurements"
DAY_FEAT_DIR = BASE_DIR / "features" / "uc_day_features" / "feature_set_version=v1"


def run_audit(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Audit all Data Lake sources, check for future dates, and quantify prior divergences."""
    manifest = RunManifest(run_dir, config)
    manifest.transition("running")

    sources = {
        "ami_interval": MEAS_DIR / "ami_interval",
        "ami_registers": MEAS_DIR / "ami_registers",
        "ami_instantaneous": MEAS_DIR / "ami_instantaneous",
        "uc_day_features": DAY_FEAT_DIR,
        "meter_installation_history": CONTEXT_DIR / "meter_installation_history.parquet",
        "uc_context": CONTEXT_DIR / "uc_context.parquet",
        "electrical_hierarchy": CONTEXT_DIR / "electrical_hierarchy.parquet",
        "training_dataset": BASE_DIR / "model_input" / "v1" / "training_dataset.parquet",
    }

    exec_max_dt = date(2026, 11, 20)
    source_rows: list[dict[str, Any]] = []
    future_violations = 0

    for name, path in sources.items():
        manifest.register_input(name, path)
        info: dict[str, Any] = {
            "source": name,
            "path": str(path),
            "exists": path.exists(),
            "partitions": 0,
            "rows": 0,
            "distinct_ucs": 0,
            "distinct_meters": 0,
            "min_date": None,
            "max_date": None,
            "future_records_gt_2026_11_20": 0,
        }
        if path.exists():
            if path.is_file():
                df = pl.read_parquet(path)
                info["rows"] = int(df.height)
                if "UC" in df.columns:
                    info["distinct_ucs"] = int(df["UC"].n_unique())
                elif "id__uc_id" in df.columns:
                    info["distinct_ucs"] = int(df["id__uc_id"].n_unique())
                if "NIO" in df.columns:
                    info["distinct_meters"] = int(df["NIO"].n_unique())
                for c in ("meta__cutoff_date", "DATA_INSTALACAO", "DATA_REMOCAO"):
                    if c in df.columns:
                        s = df[c].cast(pl.Date, strict=False).drop_nulls()
                        if len(s) > 0:
                            mn, mx = s.min(), s.max()
                            info["min_date"] = str(mn)
                            info["max_date"] = str(mx)
                            if mx > exec_max_dt:
                                cnt = int(df.filter(pl.col(c).cast(pl.Date, strict=False) > exec_max_dt).height)
                                info["future_records_gt_2026_11_20"] += cnt
                                future_violations += cnt
            else:
                dated = _list_dated_parquets(path)
                info["partitions"] = len(dated)
                if dated:
                    info["min_date"] = str(dated[0][0])
                    info["max_date"] = str(dated[-1][0])
                    future_parts = [d for d, _ in dated if d > exec_max_dt]
                    info["future_records_gt_2026_11_20"] = len(future_parts)
                    future_violations += len(future_parts)
                    last_df = pl.read_parquet(dated[-1][1])
                    info["sample_last_day_rows"] = int(last_df.height)
        source_rows.append(info)

    # Reconcile known historical artifacts if present
    reconciliation_findings: list[dict[str, Any]] = []
    old_daily_csv = Path("docs/tcc_evidence/november_experiment/metricas_diarias_populacao_nov2026.csv")
    if old_daily_csv.exists():
        odf = pl.read_csv(old_daily_csv, separator=";")
        reconciliation_findings.append({
            "artifact": str(old_daily_csv),
            "metric": "CANDIDATOS_VISTORIA (19/11 & Mean)",
            "recorded_value": f"19/11={odf['CANDIDATOS_VISTORIA'][-1]:,}; mean={odf['CANDIDATOS_VISTORIA'].mean():.1f}",
            "cause": "Regra simplificada (current_imb_max >= 0.45) sem materialidade nem score ML rotulou ~45% da rede; or True ocultava balanço de fila.",
        })
        reconciliation_findings.append({
            "artifact": str(old_daily_csv),
            "metric": "Total UC x Cutoff windows",
            "recorded_value": f"sum(POPULACAO_TOTAL)={odf['POPULACAO_TOTAL'].sum():,}",
            "cause": "Texto de método embutia literal hardcoded '1.226.702' enquanto soma real do CSV era 1.247.742.",
        })

    audit_payload = {
        "execution_cutoff_guard": str(exec_max_dt),
        "future_violations": future_violations,
        "sources": source_rows,
        "prior_divergences_audit": reconciliation_findings,
    }
    atomic_write_json(run_dir / "audit_sources.json", audit_payload)
    pl.DataFrame(source_rows).write_csv(run_dir / "audit_sources.csv", separator=";")
    manifest.register_artifact("audit_sources", run_dir / "audit_sources.json")

    inv_rep = InvariantReport(checks=[
        validate_temporal_no_leakage(
            train_max_date=None,
            eval_start_date=exec_max_dt,
            max_allowed_date=exec_max_dt,
        )
    ])
    manifest.finalize_with_invariants(inv_rep)
    return audit_payload


def _load_daily_features_range(start_day: date, end_day: date) -> pl.DataFrame:
    """Load `uc_day_features` strictly within [start_day, end_day]."""
    dated = _list_dated_parquets(DAY_FEAT_DIR)
    frames: list[pl.DataFrame] = []
    for dt, fpath in dated:
        if start_day <= dt <= end_day:
            df_d = pl.read_parquet(fpath)
            if not df_d.is_empty() and "UC" in df_d.columns:
                frames.append(
                    df_d.with_columns([
                        pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc_norm"),
                        pl.lit(dt).cast(pl.Date).alias("_dt"),
                    ])
                )
    if not frames:
        raise FileNotFoundError(f"No uc_day_features partitions found between {start_day} and {end_day}.")
    return pl.concat(frames, how="diagonal_relaxed")


def _compute_daily_cut_window(
    all_daily_df: pl.DataFrame,
    meter_history_df: pl.DataFrame,
    c_dt: date,
    *,
    window_days: int = 30,
    min_day_cov: float = 0.70,
    min_win_cov: float = 0.60,
    min_rel_days: int = 15,
) -> tuple[pl.DataFrame, date]:
    """Compute leak-free 30-day rolling features and calibrated rule scores for cutoff c_dt."""
    win_start = c_dt - timedelta(days=window_days)
    win_df = all_daily_df.filter((pl.col("_dt") > win_start) & (pl.col("_dt") <= c_dt))

    max_obs = win_df["_dt"].max()
    if max_obs is not None and max_obs > c_dt:
        raise AssertionError(f"Future leak in daily cut {c_dt}: max_obs={max_obs}")

    n_extracted = max(1, win_df["_dt"].n_unique())
    rel_mask = pl.col("INTERVAL_COVERAGE").fill_null(0.0) >= min_day_cov
    rel_fa = pl.col("FA_INTERVAL_SUM").filter(rel_mask)
    rel_dt = (pl.col("_dt") - pl.lit(win_start)).dt.total_days().cast(pl.Float64).filter(rel_mask)

    agg_df = win_df.sort(["_uc_norm", "_dt"]).group_by("_uc_norm").agg([
        (pl.col("INTERVAL_COVERAGE").fill_null(0.0).sum() / float(n_extracted)).clip(0.0, 1.0).round(4).alias("avg_coverage"),
        rel_mask.sum().cast(pl.Int16).alias("reliable_days"),
        pl.coalesce([rel_fa.median(), pl.col("FA_INTERVAL_SUM").median(), pl.lit(0.0)]).round(4).alias("fa_median"),
        pl.coalesce([
            (rel_fa - rel_fa.median()).abs().median(),
            (pl.col("FA_INTERVAL_SUM") - pl.col("FA_INTERVAL_SUM").median()).abs().median(),
            pl.lit(0.0),
        ]).round(4).alias("fa_mad"),
        (pl.col("RA_REVERSAL_RATIO").fill_null(0.0) > 0.08).sum().cast(pl.Int16).alias("ra_reversal_days"),
        pl.col("RA_REVERSAL_RATIO").fill_null(0.0).max().fill_null(0.0).round(4).alias("ra_reversal_ratio_max"),
        pl.col("CURRENT_IMBALANCE_MAX").fill_null(0.0).quantile(0.90).fill_null(0.0).round(4).alias("current_imb_max"),
        pl.col("VOLTAGE_IMBALANCE_MAX").fill_null(0.0).quantile(0.95).fill_null(0.0).round(4).alias("voltage_imb_max"),
        pl.len().cast(pl.Int16).alias("days_extracted"),
        pl.when((rel_dt.len() >= 3) & (rel_dt.var(ddof=0).fill_null(0.0) > 0.0))
        .then(pl.cov(rel_dt, rel_fa, ddof=0) / rel_dt.var(ddof=0))
        .otherwise(0.0)
        .fill_nan(0.0)
        .fill_null(0.0)
        .round(4)
        .alias("trend_fa_slope"),
    ])

    # Meter replacement status strictly <= c_dt
    if not meter_history_df.is_empty() and "UC" in meter_history_df.columns and "DATA_INSTALACAO" in meter_history_df.columns:
        mh_sub = (
            meter_history_df.select([
                pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc_norm"),
                pl.col("DATA_INSTALACAO").cast(pl.Date, strict=False).alias("_dt_inst"),
                pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_nio"),
            ])
            .filter(pl.col("_dt_inst").is_not_null() & (pl.col("_dt_inst") <= c_dt))
            .group_by("_uc_norm")
            .agg([
                pl.col("_dt_inst").max().alias("_latest_inst"),
                pl.col("_nio").n_unique().alias("n_meters_historical"),
            ])
            .with_columns([
                (pl.lit(c_dt) - pl.col("_latest_inst")).dt.total_days().alias("meter_age_days"),
            ])
            .with_columns((pl.col("meter_age_days") <= 30).alias("meter_changed_30d"))
            .select(["_uc_norm", "meter_age_days", "meter_changed_30d", "n_meters_historical"])
        )
        agg_df = agg_df.join(mh_sub, on="_uc_norm", how="left")

    agg_df = agg_df.with_columns([
        pl.col("meter_age_days").fill_null(999) if "meter_age_days" in agg_df.columns else pl.lit(999).alias("meter_age_days"),
        pl.col("meter_changed_30d").fill_null(False) if "meter_changed_30d" in agg_df.columns else pl.lit(False).alias("meter_changed_30d"),
        pl.col("n_meters_historical").fill_null(1) if "n_meters_historical" in agg_df.columns else pl.lit(1).alias("n_meters_historical"),
    ])

    # Mutually exclusive population partition: ELIGIBLE vs QUARANTINE vs OTHER (recent hardware transition)
    eff_min_rel = min(min_rel_days, max(1, int(np.ceil(0.50 * n_extracted))))
    agg_df = agg_df.with_columns([
        pl.when(pl.col("meter_changed_30d") & (pl.col("meter_age_days") >= 0))
        .then(pl.lit("OTHER_HARDWARE_TRANSITION"))
        .when((pl.col("reliable_days") < eff_min_rel) | (pl.col("avg_coverage") < min_win_cov))
        .then(pl.when(pl.col("avg_coverage") < 0.20).then(pl.lit("QUARANTINE_SILENT_MESH")).otherwise(pl.lit("QUARANTINE_INTERMITTENT_MESH")))
        .otherwise(pl.lit("CLEAN_ELIGIBLE"))
        .alias("population_partition"),
    ]).with_columns([
        (pl.col("population_partition") == "CLEAN_ELIGIBLE").alias("is_eligible_clean"),
    ])

    # Calibrated physical/statistical rule hypotheses (exclusively on is_eligible_clean)
    is_gd_reversal = (pl.col("ra_reversal_days") >= 4) & (pl.col("ra_reversal_ratio_max") >= 0.08)
    is_step_drop = (pl.col("trend_fa_slope") <= -0.25) & (pl.col("fa_median") >= 5.0)
    is_imb_material = (pl.col("current_imb_max") >= 0.60) & (pl.col("fa_median") >= 5.0) & (pl.col("trend_fa_slope") < -0.05)
    is_surge_continuous = (pl.col("trend_fa_slope") >= 0.40) & (pl.col("fa_median") >= 20.0)
    is_legacy_uncalib_imb = pl.col("current_imb_max") >= 0.45

    agg_df = agg_df.with_columns([
        pl.when(pl.col("population_partition") == "OTHER_HARDWARE_TRANSITION")
        .then(pl.lit("TRANSICAO_TROCA_OU_UC_NOVA"))
        .when(~pl.col("is_eligible_clean"))
        .then(pl.lit("TELECOM_QUARANTINE"))
        .when(is_gd_reversal)
        .then(pl.lit("POSSIVEL_GD_NAO_CADASTRADA"))
        .when(is_step_drop)
        .then(pl.lit("DEGRAU_DESCENDENTE_CONSUMO"))
        .when(is_imb_material)
        .then(pl.lit("DESEQUILIBRIO_CARGA_OU_BYPASS"))
        .when(is_surge_continuous)
        .then(pl.lit("ANOMALIA_MULTIVARIADA_CARGA"))
        .otherwise(pl.lit("REGULAR"))
        .alias("categoria_diagnostica"),
        pl.when(~pl.col("is_eligible_clean"))
        .then(0.0)
        .when(is_gd_reversal | is_step_drop | is_imb_material | is_surge_continuous)
        .then(
            (
                0.55
                + 0.15 * is_gd_reversal.cast(pl.Float64)
                + 0.15 * is_step_drop.cast(pl.Float64)
                + 0.10 * is_imb_material.cast(pl.Float64)
                + 0.05 * is_surge_continuous.cast(pl.Float64)
            ).clip(0.0, 1.0)
        )
        .otherwise(0.20)
        .alias("rule_score"),
        (
            pl.col("is_eligible_clean") & (is_gd_reversal | is_step_drop | is_imb_material)
        ).alias("is_candidato_vistoria"),
        (
            pl.col("is_eligible_clean") & (is_gd_reversal | is_step_drop | is_legacy_uncalib_imb | is_surge_continuous)
        ).alias("legacy_uncalibrated_candidate"),
        pl.lit(c_dt).cast(pl.Date).alias("CUTOFF_DATE"),
    ])

    return agg_df, (max_obs if max_obs is not None else c_dt)


def run_daily_rules(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Execute DAILY_RULES calibrated daily monitoring from config.start_dt to config.end_dt."""
    t0 = time.time()
    manifest = RunManifest(run_dir, config)
    manifest.transition("running")

    meter_history = _read_table_flexible(CONTEXT_DIR / "meter_installation_history.parquet")
    manifest.register_input("meter_installation_history", CONTEXT_DIR / "meter_installation_history.parquet")

    # Load from start_dt - 30d strictly up to end_dt (anti-leakage boundary)
    load_start = config.start_dt - timedelta(days=30)
    all_daily_df = _load_daily_features_range(load_start, config.end_dt)

    cutoffs: list[date] = []
    cur = config.start_dt
    while cur <= config.end_dt:
        cutoffs.append(cur)
        cur += timedelta(days=1)

    daily_metrics_rows: list[dict[str, Any]] = []
    partition_records: list[dict[str, Any]] = []
    candidates_by_date: dict[str, set[str]] = {}
    decision_frames: list[pl.DataFrame] = []
    all_scores: list[np.ndarray] = []

    prev_candidates: set[str] | None = None

    for c_dt in cutoffs:
        agg_df, _ = _compute_daily_cut_window(
            all_daily_df,
            meter_history,
            c_dt,
            window_days=30,
            min_day_cov=config.coverage_threshold,
            min_win_cov=config.min_window_coverage,
            min_rel_days=config.min_reliable_days,
        )

        # Sets for strict invariant verification
        elig_set = set(agg_df.filter(pl.col("is_eligible_clean"))["_uc_norm"].to_list())
        quar_set = set(agg_df.filter(pl.col("population_partition").str.starts_with("QUARANTINE_"))["_uc_norm"].to_list())
        other_set = set(agg_df.filter(pl.col("population_partition") == "OTHER_HARDWARE_TRANSITION")["_uc_norm"].to_list())
        cands = set(agg_df.filter(pl.col("is_candidato_vistoria"))["_uc_norm"].to_list())
        legacy_cands = agg_df.filter(pl.col("legacy_uncalibrated_candidate")).height

        dt_str = str(c_dt)
        candidates_by_date[dt_str] = cands

        if prev_candidates is None:
            entries = len(cands)
            exits = 0
            persisted = 0
        else:
            entries = len(cands - prev_candidates)
            exits = len(prev_candidates - cands)
            persisted = len(cands & prev_candidates)

        total_pop = agg_df.height
        elig_cnt = len(elig_set)
        quar_cnt = len(quar_set)
        other_cnt = len(other_set)
        anom_cnt = agg_df.filter(pl.col("is_eligible_clean") & (pl.col("rule_score") >= 0.55)).height
        gd_cnt = agg_df.filter(pl.col("is_eligible_clean") & (pl.col("categoria_diagnostica") == "POSSIVEL_GD_NAO_CADASTRADA")).height
        step_cnt = agg_df.filter(pl.col("is_eligible_clean") & (pl.col("categoria_diagnostica") == "DEGRAU_DESCENDENTE_CONSUMO")).height

        partition_records.append({
            "cutoff_date": dt_str,
            "total_count": total_pop,
            "eligible_count": elig_cnt,
            "quarantine_count": quar_cnt,
            "other_count": other_cnt,
            "eligible_ucs": elig_set,
            "quarantine_ucs": quar_set,
            "other_ucs": other_set,
        })

        daily_metrics_rows.append({
            "CUTOFF_DATE": dt_str,
            "POPULACAO_TOTAL": total_pop,
            "ELEGIVEIS_LIMPOS": elig_cnt,
            "PCT_ELEGIVEIS_SOBRE_TOTAL": round(elig_cnt / max(1, total_pop) * 100.0, 2),
            "TOTAL_QUARENTENA": quar_cnt,
            "PCT_QUARENTENA_SOBRE_TOTAL": round(quar_cnt / max(1, total_pop) * 100.0, 2),
            "OUTROS_TRANSICAO_HARDWARE": other_cnt,
            "PCT_OUTROS_SOBRE_TOTAL": round(other_cnt / max(1, total_pop) * 100.0, 2),
            "ANOMALIAS_DETECTADAS": anom_cnt,
            "PCT_ANOMALIAS_SOBRE_ELEGIVEIS": round(anom_cnt / max(1, elig_cnt) * 100.0, 2),
            "POSSIVEL_GD": gd_cnt,
            "DEGRAU_DESCENDENTE": step_cnt,
            "CANDIDATOS_VISTORIA": len(cands),
            "PCT_CANDIDATOS_SOBRE_ELEGIVEIS": round(len(cands) / max(1, elig_cnt) * 100.0, 2),
            "LEGACY_UNCALIBRATED_CANDIDATES": legacy_cands,
            "FILA_ENTRADAS": entries,
            "FILA_SAIDAS": exits,
            "FILA_PERMANENCIA": persisted,
        })

        all_scores.append(agg_df["rule_score"].to_numpy())
        decision_frames.append(
            agg_df.select([
                "CUTOFF_DATE", "_uc_norm", "population_partition", "is_eligible_clean",
                "avg_coverage", "reliable_days", "fa_median", "fa_mad", "trend_fa_slope",
                "ra_reversal_days", "current_imb_max", "rule_score", "categoria_diagnostica",
                "is_candidato_vistoria",
            ]).rename({"_uc_norm": "id__uc_id"})
        )
        prev_candidates = cands

    # Concatenate outputs
    full_decisions = pl.concat(decision_frames, how="vertical")
    df_daily_metrics = pl.DataFrame(daily_metrics_rows)

    # Save versioned artifacts in run_dir
    full_decisions.select(["CUTOFF_DATE", "id__uc_id", "population_partition"]).write_parquet(run_dir / "population.parquet")
    full_decisions.select(["CUTOFF_DATE", "id__uc_id", "is_eligible_clean", "avg_coverage", "reliable_days"]).write_parquet(run_dir / "quality.parquet")
    full_decisions.write_parquet(run_dir / "decisions.parquet")
    full_decisions.filter(pl.col("is_candidato_vistoria")).select(["CUTOFF_DATE", "id__uc_id", "categoria_diagnostica", "rule_score"]).write_parquet(run_dir / "candidate_sets.parquet")
    df_daily_metrics.write_parquet(run_dir / "daily_metrics.parquet")
    df_daily_metrics.write_csv(run_dir / "daily_metrics.csv", separator=";")

    # Fail-closed Invariants
    flat_scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=float)
    inv_report = InvariantReport(checks=[
        validate_population_partition(partition_records),
        validate_portfolio_queue_sets(candidates_by_date, daily_metrics_rows),
        validate_temporal_no_leakage(
            train_max_date=None,
            eval_start_date=config.start_dt,
            max_allowed_date=config.end_dt,
        ),
        validate_score_and_kpi_integrity(flat_scores, daily_metrics_rows),
    ])

    summary_metrics = {
        "pipeline_id": "daily_rules",
        "start_date": str(config.start_dt),
        "end_date": str(config.end_dt),
        "n_cutoffs": len(cutoffs),
        "total_windows_evaluated": int(full_decisions.height),
        "distinct_ucs": int(full_decisions["id__uc_id"].n_unique()),
        "last_day_total_pop": int(daily_metrics_rows[-1]["POPULACAO_TOTAL"]),
        "last_day_eligible": int(daily_metrics_rows[-1]["ELEGIVEIS_LIMPOS"]),
        "last_day_quarantine": int(daily_metrics_rows[-1]["TOTAL_QUARENTENA"]),
        "last_day_other": int(daily_metrics_rows[-1]["OUTROS_TRANSICAO_HARDWARE"]),
        "last_day_candidates": int(daily_metrics_rows[-1]["CANDIDATOS_VISTORIA"]),
        "last_day_legacy_uncalibrated": int(daily_metrics_rows[-1]["LEGACY_UNCALIBRATED_CANDIDATES"]),
        "mean_daily_candidates": round(float(df_daily_metrics["CANDIDATOS_VISTORIA"].mean()), 2),
        "runtime_sec": round(time.time() - t0, 2),
    }
    atomic_write_json(run_dir / "metrics.json", summary_metrics)

    # Figures
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_daily_rules_charts(df_daily_metrics, fig_dir)

    manifest.record_parameters({
        "rule_definition": "Calibrated deterministic physical rules (no ML fit)",
        "coverage_threshold": config.coverage_threshold,
        "min_window_coverage": config.min_window_coverage,
        "min_reliable_days": config.min_reliable_days,
    })
    manifest.register_artifact("daily_metrics", run_dir / "daily_metrics.csv")
    manifest.register_artifact("decisions", run_dir / "decisions.parquet")
    manifest.register_artifact("metrics", run_dir / "metrics.json")

    ok = manifest.finalize_with_invariants(inv_report)
    if not ok:
        raise RuntimeError(f"DAILY_RULES failed invariant verification in {run_dir}: {inv_report.to_dict()}")
    return summary_metrics


def _plot_daily_rules_charts(df_metrics: pl.DataFrame, fig_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    dates_lbl = [d[8:10] + "/" + d[5:7] for d in df_metrics["CUTOFF_DATE"].to_list()]
    x = np.arange(len(dates_lbl))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.5), dpi=200, sharex=True)
    cands = df_metrics["CANDIDATOS_VISTORIA"].to_numpy()
    ax1.plot(x, cands, marker="o", color="#dc2626", linewidth=2.2, label="Candidatos Calibrados (Política Regras Diárias)")
    for i, v in enumerate(cands):
        ax1.annotate(str(v), (x[i], v + max(5, v * 0.02)), ha="center", fontsize=8, fontweight="bold")
    ax1.set_ylabel("Candidatos Ativos")
    ax1.set_title("Série Temporal DAILY_RULES: Fila de Candidatos e Fluxo Diário (Novembro/2026)", fontweight="bold")
    ax1.legend(loc="upper left")

    entries = df_metrics["FILA_ENTRADAS"].to_numpy()
    exits = df_metrics["FILA_SAIDAS"].to_numpy()
    ax2.bar(x - 0.18, entries, width=0.35, color="#16a34a", label="Entradas (I_t)")
    ax2.bar(x + 0.18, exits, width=0.35, color="#64748b", label="Saídas (O_t)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dates_lbl, rotation=45, ha="right", fontsize=8.5)
    ax2.set_ylabel("Transições")
    ax2.set_xlabel("Data de Corte")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(fig_dir / "daily_rules_queue_dynamics.png", dpi=200)
    plt.close(fig)
