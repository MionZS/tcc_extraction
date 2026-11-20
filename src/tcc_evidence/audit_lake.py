"""Quantitative Data Lake Audit, Pipeline Engineering Benchmark & Cockpit Reconciliation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.eda.cockpit_metrics import KPI_SPECIFICATIONS, compute_reconciled_kpis
from src.eda.uc_timeseries import _list_dated_parquets


def _safe_read_table(parquet_path: Path, csv_path: Path) -> tuple[pl.DataFrame, str]:
    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        return pl.read_parquet(parquet_path), "parquet"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pl.read_csv(csv_path, separator=";", infer_schema_length=5000, ignore_errors=True), "csv"
    return pl.DataFrame(), "missing"


def run_data_lake_audit(
    base_dir: Path,
    out_dir: Path,
    *,
    classified_df: pl.DataFrame,
) -> dict[str, Any]:
    """Audit Population, Temporality, Quality, and Partition Integrity of the local Data Lake."""
    dq_dir = out_dir / "data_quality"
    dq_dir.mkdir(parents=True, exist_ok=True)

    # 1. Context tables
    uc_ctx, ctx_fmt = _safe_read_table(
        base_dir / "context" / "uc_context.parquet",
        base_dir / "context" / "uc_context.csv",
    )
    hier_df, hier_fmt = _safe_read_table(
        base_dir / "context" / "electrical_hierarchy.parquet",
        base_dir / "context" / "electrical_hierarchy.csv",
    )
    inst_df, inst_fmt = _safe_read_table(
        base_dir / "context" / "meter_installation_history.parquet",
        base_dir / "context" / "meter_installation_history.csv",
    )

    pop_metrics: dict[str, Any] = {
        "context_formats": {"uc_context": ctx_fmt, "electrical_hierarchy": hier_fmt, "meter_history": inst_fmt},
        "uc_context_rows": int(uc_ctx.height),
        "electrical_hierarchy_rows": int(hier_df.height),
        "meter_history_rows": int(inst_df.height),
    }

    # Population from meter_installation_history
    if not inst_df.is_empty():
        u_col = "UC" if "UC" in inst_df.columns else ("id__uc_id" if "id__uc_id" in inst_df.columns else None)
        n_col = "NIO" if "NIO" in inst_df.columns else ("id__nio" if "id__nio" in inst_df.columns else None)
        if u_col and n_col:
            clean_inst = inst_df.with_columns([
                pl.col(u_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc"),
                pl.col(n_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_nio"),
            ])
            n_ucs_hist = int(clean_inst["_uc"].n_unique())
            n_nios_hist = int(clean_inst["_nio"].n_unique())
            meters_per_uc = clean_inst.group_by("_uc").agg(pl.col("_nio").n_unique().alias("n_nios"))
            ucs_multi_meter = int(meters_per_uc.filter(pl.col("n_nios") > 1).height)
            ucs_per_nio = clean_inst.group_by("_nio").agg(pl.col("_uc").n_unique().alias("n_ucs"))
            reused_nios = int(ucs_per_nio.filter(pl.col("n_ucs") > 1).height)

            removed_cnt = 0
            if "DATA_REMOCAO" in inst_df.columns:
                removed_cnt = int(inst_df.filter(pl.col("DATA_REMOCAO").is_not_null() & (pl.col("DATA_REMOCAO").cast(pl.String) != "")).height)

            pop_metrics.update({
                "ucs_distintas_cadastro_medidor": n_ucs_hist,
                "nios_distintos_cadastro_medidor": n_nios_hist,
                "ucs_com_multiplos_medidores": ucs_multi_meter,
                "nios_reutilizados_em_multiplas_ucs": reused_nios,
                "medidores_removidos_registros": removed_cnt,
            })

            if "DESCRICAO_TIPO" in inst_df.columns or "FAMILIA_SMART" in inst_df.columns:
                fam_col = "FAMILIA_SMART" if "FAMILIA_SMART" in inst_df.columns else "DESCRICAO_TIPO"
                fam_dist = inst_df.group_by(fam_col).agg(pl.len().alias("count")).sort("count", descending=True).head(10)
                fam_dist.write_csv(dq_dir / "meter_families.csv", separator=";")

    # Population from electrical_hierarchy
    if not hier_df.is_empty():
        if "ALIMENTADOR" in hier_df.columns:
            feed_dist = hier_df.group_by("ALIMENTADOR").agg(pl.len().alias("ucs_cadastro")).sort("ucs_cadastro", descending=True)
            feed_dist.write_csv(dq_dir / "population_by_feeder_hierarchy.csv", separator=";")
            pop_metrics["alimentadores_distintos_hierarquia"] = int(feed_dist.height)
        if "POSTO_OPERACIONAL" in hier_df.columns:
            pop_metrics["trafos_postos_distintos"] = int(hier_df["POSTO_OPERACIONAL"].drop_nulls().n_unique())

    # 2. Partitions audit (ami_interval, uc_day_features, uc_window_features)
    interval_files = _list_dated_parquets(base_dir / "measurements" / "ami_interval")
    day_files = _list_dated_parquets(base_dir / "features" / "uc_day_features")
    win_files = _list_dated_parquets(base_dir / "features" / "uc_window_features")

    pop_metrics["partitions"] = {
        "ami_interval_days": len(interval_files),
        "ami_interval_range": [str(interval_files[0][0]), str(interval_files[-1][0])] if interval_files else [],
        "uc_day_features_days": len(day_files),
        "uc_day_features_range": [str(day_files[0][0]), str(day_files[-1][0])] if day_files else [],
        "uc_window_features_cutoffs": len(win_files),
    }

    # Check CSV vs Parquet divergence on classified_ucs_latest
    pq_p = base_dir / "reports" / "classified_ucs_latest.parquet"
    csv_p = base_dir / "reports" / "classified_ucs_latest.csv"
    if pq_p.exists() and csv_p.exists():
        pq_rows = int(pl.scan_parquet(pq_p).select(pl.len()).collect().item())
        csv_rows = int(pl.scan_csv(csv_p, separator=";", infer_schema_length=1000, ignore_errors=True).select(pl.len()).collect().item())
        pop_metrics["csv_vs_parquet_classified"] = {
            "parquet_rows": pq_rows,
            "csv_rows": csv_rows,
            "row_diff": pq_rows - csv_rows,
            "parquet_mb": round(pq_p.stat().st_size / (1024 * 1024), 2),
            "csv_mb": round(csv_p.stat().st_size / (1024 * 1024), 2),
        }

    # 3. Quality & Eligibility from classified_df
    if not classified_df.is_empty():
        cov_col = "x__avg_interval_coverage" if "x__avg_interval_coverage" in classified_df.columns else None
        rel_col = "x__coverage_reliable_days" if "x__coverage_reliable_days" in classified_df.columns else None
        if cov_col:
            cov_arr = classified_df[cov_col].fill_null(0.0).to_numpy()
            elig_stats = []
            for thr in [0.50, 0.60, 0.70, 0.80, 0.90]:
                n_elig = int(np.sum(cov_arr >= thr))
                elig_stats.append({
                    "coverage_threshold": thr,
                    "eligible_windows": n_elig,
                    "eligible_pct": round(n_elig / max(len(cov_arr), 1) * 100.0, 2),
                    "excluded_windows": int(len(cov_arr) - n_elig),
                })
            pl.DataFrame(elig_stats).write_csv(dq_dir / "eligibility_by_coverage.csv", separator=";")
            pop_metrics["coverage_mean"] = round(float(np.mean(cov_arr)), 4)
            pop_metrics["coverage_median"] = round(float(np.median(cov_arr)), 4)
            pop_metrics["coverage_p10"] = round(float(np.percentile(cov_arr, 10)), 4)

            # Plot coverage histogram
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.hist(cov_arr * 100.0, bins=40, color="#0284c7", edgecolor="#ffffff", alpha=0.85)
            ax.axvline(70.0, color="#dc2626", linestyle="--", linewidth=2.0, label="Limiar Gating Atual (70%)")
            ax.axvline(60.0, color="#f59e0b", linestyle=":", linewidth=1.8, label="Limiar Quarentena (60%)")
            ax.set_title("Distribuição da Cobertura Temporal das Janelas AMI (Araucária)", fontweight="bold")
            ax.set_xlabel("Cobertura Média de Intervalos (%)")
            ax.set_ylabel("Número de Janelas (UC × Cutoff)")
            ax.legend()
            plt.tight_layout()
            fig.savefig(dq_dir / "coverage_distribution.png", dpi=200)
            fig.savefig(dq_dir / "coverage_distribution.svg")
            plt.close(fig)

        if rel_col:
            rel_arr = classified_df[rel_col].fill_null(0).to_numpy()
            pop_metrics["reliable_days_median"] = float(np.median(rel_arr))

    # 4. Daily partition continuity table if uc_day_features exists
    if day_files:
        day_rows = [
            {
                "report_day": str(d),
                "file_kb": round(p.stat().st_size / 1024.0, 1),
            }
            for d, p in day_files
        ]
        pl.DataFrame(day_rows).write_csv(dq_dir / "uc_day_partition_calendar.csv", separator=";")

    (dq_dir / "data_lake_audit.json").write_text(json.dumps(pop_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return pop_metrics


def run_pipeline_engineering_benchmark(base_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Summarize real run manifests and execute a controlled Polars Eager vs Lazy/sink benchmark."""
    pipe_dir = out_dir / "pipeline"
    pipe_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inspect existing run manifests in output/runs/
    runs_dir = base_dir / "runs"
    run_records: list[dict[str, Any]] = []
    if runs_dir.exists():
        for rp in sorted(runs_dir.glob("run_*.json"))[-25:]:
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
                run_records.append({
                    "file": rp.name,
                    "run_id": data.get("run_id", rp.stem),
                    "status": data.get("status", "unknown"),
                    "started_at": data.get("started_at", ""),
                    "rows_written": sum(v for v in (data.get("row_counts") or {}).values() if isinstance(v, (int, float))),
                })
            except Exception:
                continue

    if run_records:
        pl.DataFrame(run_records).write_csv(pipe_dir / "recent_run_manifests.csv", separator=";")

    # 2. Controlled benchmark: Polars Lazy sink_parquet vs Eager read+write on local classified_ucs_latest
    pq_src = base_dir / "reports" / "classified_ucs_latest.parquet"
    bench_res: dict[str, Any] = {"benchmark_executed": False}
    if pq_src.exists() and pq_src.stat().st_size > 0:
        tmp_eager = pipe_dir / "_tmp_bench_eager.parquet"
        tmp_lazy = pipe_dir / "_tmp_bench_lazy.parquet"

        t0 = time.perf_counter()
        df_e = pl.read_parquet(pq_src)
        df_e.write_parquet(tmp_eager, compression="zstd")
        t_eager = time.perf_counter() - t0
        rows_n = int(df_e.height)
        del df_e

        t1 = time.perf_counter()
        pl.scan_parquet(pq_src).sink_parquet(tmp_lazy, compression="zstd")
        t_lazy = time.perf_counter() - t1

        # Clean up temporary benchmark files
        for tmp_f in (tmp_eager, tmp_lazy):
            if tmp_f.exists():
                tmp_f.unlink()

        bench_res = {
            "benchmark_executed": True,
            "dataset_rows": rows_n,
            "source_mb": round(pq_src.stat().st_size / (1024 * 1024), 2),
            "eager_read_write_sec": round(t_eager, 4),
            "lazy_sink_parquet_sec": round(t_lazy, 4),
            "eager_throughput_rows_sec": int(rows_n / max(t_eager, 1e-6)),
            "lazy_throughput_rows_sec": int(rows_n / max(t_lazy, 1e-6)),
        }

    summary = {
        "n_run_manifests_found": len(run_records),
        "controlled_benchmark": bench_res,
    }
    (pipe_dir / "pipeline_engineering_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_cockpit_reconciliation_audit(classified_df: pl.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Generate reconciled Cockpit KPI table and specification audit."""
    cockpit_dir = out_dir / "cockpit"
    cockpit_dir.mkdir(parents=True, exist_ok=True)

    kpis = compute_reconciled_kpis(classified_df)
    spec_rows = [
        {
            "key": s.key,
            "label": s.label,
            "granularity": s.granularity,
            "denominator": s.denominator,
            "formula": s.formula,
            "filters": s.filters,
            "observed_value": kpis.get(s.key, "-"),
        }
        for s in KPI_SPECIFICATIONS
    ]
    pl.DataFrame(spec_rows).write_csv(cockpit_dir / "cockpit_kpi_reconciliation.csv", separator=";")
    (cockpit_dir / "cockpit_kpis_observed.json").write_text(
        json.dumps(kpis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return kpis
