"""Interactive Web Cockpit for Smart Meter Anomaly Inspection and Diagnosis.

Runs an integrated, zero-dependency HTTP server inspired by VETOR (SanPlat AMI System).
Serves:
1. REST APIs:
   - `/api/v1/summary`: Dataset statistics, category counts, model provenance.
   - `/api/v1/anomalies`: Paged, searchable, and filtered table of classified meters.
   - `/api/v1/meter/<uc>`: Detailed multi-feature diagnostics, time-series curves, and root-cause breakdown for a single UC.
   - `/api/v1/categories`: List of identified diagnostic categories and counts.
   - `/api/v1/feeders`: Anomaly rates by electrical feeder.
2. Web UI: Modern React-style SPA with live search, category pills, feeder filter, detailed drawer, and export.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
import urllib.parse
import webbrowser

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda.uc_timeseries import load_uc_daily_timeseries, load_uc_hourly_timeseries

BASE_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = BASE_DIR / "reports"
CLASSIFIED_PARQUET = REPORTS_DIR / "classified_ucs_latest.parquet"
CLASSIFIED_CSV = REPORTS_DIR / "classified_ucs_latest.csv"
SUMMARY_JSON = REPORTS_DIR / "model_training_summary_latest.json"

_DATA_LOCK = threading.Lock()
_CACHED_DF: pl.DataFrame | None = None
_CACHED_FEEDERS: list[str] | None = None
_CACHED_CATEGORIES: list[str] | None = None
_CACHED_SUMMARY: dict | None = None
_CACHED_DASHBOARD: dict | None = None
_ANOMALIES_CACHE: OrderedDict[tuple, bytes] = OrderedDict()
_ANOMALIES_CACHE_MAX = 64


_CATEGORY_META: dict[str, dict[str, str]] = {
    "CANDIDATOS_A_VISTORIA": {
        "label": "🎯 Candidatos a Vistoria (Campo)",
        "mechanism": "Anomalias elétricas confirmadas + persistência temporal (dias seguidos)",
        "color": "#dc2626",
    },
    "DEGRAU_DESCENDENTE_CONSUMO": {
        "label": "Degrau Descendente / Queda",
        "mechanism": "Slope OLS negativo + CUSUM + ECOD/COPOD cauda esquerda",
        "color": "#f59e0b",
    },
    "QUEDA_TENDENCIA_CONSUMO_SLOPE_NEGATIVO": {
        "label": "Queda de Tendência (Slope < 0)",
        "mechanism": "Tendência linear negativa + queda de patamar mediano",
        "color": "#f59e0b",
    },
    "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT": {
        "label": "Anomalia Elétrica / Suspeita PNT",
        "mechanism": "Ensemble PyOD (IF+LOF+ECOD+COPOD+HBOS+PCA) + Guardrails PRODIST",
        "color": "#ef4444",
    },
    "DESEQUILIBRIO_CARGA_OU_BYPASS": {
        "label": "Desequilíbrio / Bypass de Fase",
        "mechanism": "CURRENT_IMBALANCE_MAX elevado + desvio multivariado LOF/IF",
        "color": "#ef4444",
    },
    "ANOMALIA_MULTIVARIADA": {
        "label": "Outlier Multivariado (Picos/Dispersão)",
        "mechanism": "Modified Z-Score MAD > 3.5 + IForest + LOF + HBOS + PCA",
        "color": "#ec4899",
    },
    "POSSIVEL_GD_NAO_CADASTRADA": {
        "label": "Possível GD / Fluxo Reverso",
        "mechanism": "RA_REVERSAL_DAYS / RA_REVERSAL_RATIO_MAX > 0 sem cadastro GD",
        "color": "#06b6d4",
    },
    "MEDIDOR_TRAVADO_OU_CONSUMO_ZERO": {
        "label": "Medidor Travado / Consumo Zero",
        "mechanism": "ZERO_CONSUMPTION_FLAG + tensão presente sem corrente ativa",
        "color": "#a855f7",
    },
    "TELECOM_QUARANTINE": {
        "label": "Quarentena Telecom (Gating)",
        "mechanism": "Cobertura < 60% ou < 15 dias confiáveis (isolado do treino)",
        "color": "#64748b",
    },
    "REGULAR": {
        "label": "Operação Regular",
        "mechanism": "Dentro das bandas MAD e score abaixo do limiar de contaminação",
        "color": "#10b981",
    },
}


def _build_dashboard_stats(df: pl.DataFrame) -> dict:
    """Compute real anomaly counts by category, statistical behavior and feeder."""
    global _CACHED_DASHBOARD
    if _CACHED_DASHBOARD is not None:
        return _CACHED_DASHBOARD

    with _DATA_LOCK:
        if _CACHED_DASHBOARD is not None:
            return _CACHED_DASHBOARD

        cols = set(df.columns)
        uc_col = "id__uc_id" if "id__uc_id" in cols else ("UC" if "UC" in cols else None)
        cat_col = "CATEGORIA_DIAGNOSTICA" if "CATEGORIA_DIAGNOSTICA" in cols else None
        sub_col = "SUBTIPO_DIAGNOSTICO" if "SUBTIPO_DIAGNOSTICO" in cols else None
        anom_col = "IS_ANOMALY" if "IS_ANOMALY" in cols else None
        score_col = "ANOMALY_SCORE" if "ANOMALY_SCORE" in cols else None
        feeder_col = "meta__feeder" if "meta__feeder" in cols else ("ALIMENTADOR" if "ALIMENTADOR" in cols else None)
        slope_col = "x__trend_fa_slope" if "x__trend_fa_slope" in cols else None
        cv_col = "x__fa_cv_window" if "x__fa_cv_window" in cols else None
        cutoff_col = "meta__cutoff_date" if "meta__cutoff_date" in cols else None
        cand_col_name = "IS_CANDIDATO_VISTORIA" if "IS_CANDIDATO_VISTORIA" in cols else None

        total_rows = int(df.height)
        needed_cols = [
            c for c in [uc_col, cat_col, sub_col, anom_col, score_col, feeder_col, slope_col, cv_col, cutoff_col, cand_col_name]
            if c is not None
        ]
        work = df.select(needed_cols) if needed_cols else df
        if uc_col and cutoff_col:
            work = work.sort(cutoff_col).unique(subset=[uc_col], keep="last")
        elif uc_col:
            work = work.unique(subset=[uc_col], keep="last")
        total_unique_ucs = int(work.height)

        total_anomalies = (
            int(work.filter(pl.col(anom_col) == True).height) if anom_col else 0
        )

        categories_out: list[dict] = []
        if cat_col:
            agg_exprs = [pl.len().alias("count")]
            if anom_col:
                agg_exprs.append(pl.col(anom_col).cast(pl.Int64).sum().alias("anomalies"))
            if score_col:
                agg_exprs.append(pl.col(score_col).cast(pl.Float64).mean().alias("avg_score"))

            grp = work.group_by(cat_col).agg(agg_exprs).sort("count", descending=True)
            for row in grp.iter_rows(named=True):
                cat_name = str(row[cat_col] or "OUTROS")
                cnt = int(row["count"])
                meta = _CATEGORY_META.get(
                    cat_name,
                    {
                        "label": cat_name.replace("_", " ").title(),
                        "mechanism": "Classificação diagnóstica do pipeline",
                        "color": "#3b82f6",
                    },
                )
                categories_out.append({
                    "category": cat_name,
                    "label": meta["label"],
                    "mechanism": meta["mechanism"],
                    "color": meta["color"],
                    "count": cnt,
                    "pct": round((cnt / max(total_unique_ucs, 1)) * 100.0, 2),
                    "anomalies": int(row.get("anomalies") or 0),
                    "avg_score": round(float(row.get("avg_score") or 0.0), 4),
                })

        cand_col = "IS_CANDIDATO_VISTORIA" if "IS_CANDIDATO_VISTORIA" in cols else None
        cand_cnt = int(work.filter(pl.col(cand_col) == True).height) if cand_col else 0
        if cand_col and cand_cnt > 0:
            cand_sub = work.filter(pl.col(cand_col) == True)
            avg_c_score = float(cand_sub[score_col].mean() or 0.0) if score_col else 0.0
            categories_out.insert(0, {
                "category": "CANDIDATOS_A_VISTORIA",
                "label": "🎯 Candidatos a Vistoria (Campo)",
                "mechanism": "Anomalias físicas ativas ordenadas por persistência",
                "color": "#dc2626",
                "count": cand_cnt,
                "pct": round((cand_cnt / max(total_unique_ucs, 1)) * 100.0, 2),
                "anomalies": cand_cnt,
                "avg_score": round(avg_c_score, 4),
            })

        subtypes_out: list[dict] = []
        if sub_col:
            sub_grp = (
                work.filter(pl.col(sub_col).is_not_null())
                .group_by(sub_col)
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(12)
            )
            for row in sub_grp.iter_rows(named=True):
                subtypes_out.append({
                    "subtype": str(row[sub_col]),
                    "count": int(row["count"]),
                })

        # Behavioral mechanism counts among anomalous UCs
        anoms_only = work.filter(pl.col(anom_col) == True) if anom_col else work.head(0)
        drop_signal_cnt = (
            int(anoms_only.filter(pl.col(slope_col) < -0.05).height)
            if slope_col and anoms_only.height > 0
            else 0
        )
        high_disp_cnt = (
            int(anoms_only.filter(pl.col(cv_col) > 0.8).height)
            if cv_col and anoms_only.height > 0
            else 0
        )
        both_drop_and_spike_cnt = (
            int(anoms_only.filter((pl.col(slope_col) < -0.05) & (pl.col(cv_col) > 0.8)).height)
            if slope_col and cv_col and anoms_only.height > 0
            else 0
        )

        top_feeders: list[dict] = []
        if feeder_col and anom_col:
            f_grp = (
                work.filter(pl.col(feeder_col).is_not_null())
                .group_by(feeder_col)
                .agg([
                    pl.len().alias("total"),
                    pl.col(anom_col).cast(pl.Int64).sum().alias("anomalies"),
                ])
                .filter(pl.col("anomalies") > 0)
                .sort("anomalies", descending=True)
                .head(8)
            )
            for row in f_grp.iter_rows(named=True):
                tot_f = int(row["total"])
                an_f = int(row["anomalies"])
                top_feeders.append({
                    "feeder": str(row[feeder_col]),
                    "total": tot_f,
                    "anomalies": an_f,
                    "rate_pct": round((an_f / max(tot_f, 1)) * 100.0, 2),
                })

        _CACHED_DASHBOARD = {
            "total_rows": total_rows,
            "total_unique_ucs": total_unique_ucs,
            "total_anomalies": total_anomalies,
            "anomaly_rate_pct": round((total_anomalies / max(total_unique_ucs, 1)) * 100.0, 3),
            "signal_breakdown": {
                "candidatos_vistoria": cand_cnt,
                "drop_slope_neg_anomalies": drop_signal_cnt,
                "high_mad_dispersion_anomalies": high_disp_cnt,
                "combined_drop_and_spike": both_drop_and_spike_cnt,
            },
            "categories": categories_out,
            "subtypes": subtypes_out,
            "top_feeders": top_feeders,
        }
        return _CACHED_DASHBOARD


def _load_data() -> pl.DataFrame:
    """Loads latest classified UCs dataset transparently and caches in memory."""
    global _CACHED_DF
    if _CACHED_DF is not None and not _CACHED_DF.is_empty():
        return _CACHED_DF

    with _DATA_LOCK:
        if _CACHED_DF is not None and not _CACHED_DF.is_empty():
            return _CACHED_DF

        if CLASSIFIED_PARQUET.exists() and CLASSIFIED_PARQUET.stat().st_size > 0:
            try:
                _CACHED_DF = pl.read_parquet(CLASSIFIED_PARQUET)
                return _CACHED_DF
            except Exception:
                pass

        if CLASSIFIED_CSV.exists() and CLASSIFIED_CSV.stat().st_size > 0:
            _CACHED_DF = pl.read_csv(CLASSIFIED_CSV, separator=";", infer_schema_length=5000, ignore_errors=True)
            return _CACHED_DF

        return pl.DataFrame()


def _load_summary() -> dict:
    global _CACHED_SUMMARY
    if _CACHED_SUMMARY is not None:
        return _CACHED_SUMMARY
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
                _CACHED_SUMMARY = json.load(f)
                return _CACHED_SUMMARY
        except Exception:
            pass
    return {}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cockpit Analítico AMI — Detecção de Anomalias Copel Araucária</title>
  <style>
    :root {
      --bg: #0b0f17;
      --card-bg: #161d26;
      --card-border: #232d3b;
      --text: #f1f5f9;
      --text-dim: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
      --accent: #8b5cf6;
      --solar: #eab308;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 40px;
    }
    header {
      background: linear-gradient(180deg, #1e2631 0%, #161d26 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: linear-gradient(135deg, var(--primary), #1d4ed8);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: bold;
    }
    .brand-title { font-size: 17px; font-weight: 700; }
    .brand-subtitle { font-size: 11px; color: var(--text-dim); }
    .header-actions { display: flex; gap: 10px; align-items: center; }
    .btn {
      background-color: var(--primary);
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { background-color: var(--primary-hover); }
    .btn-secondary { background-color: #242f3d; color: var(--text); }
    .btn-secondary:hover { background-color: #334155; }
    
    .container { max-width: 1440px; margin: 0 auto; padding: 24px; }
    
    /* Metrics Row */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .metric-card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .metric-label { font-size: 12px; color: var(--text-dim); font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 26px; font-weight: 800; }
    .metric-sub { font-size: 11px; color: var(--text-dim); }
    
    /* Filters Bar */
    .filter-panel {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }
    .search-box {
      flex: 1;
      min-width: 260px;
      position: relative;
    }
    .search-input {
      width: 100%;
      background-color: #0b0f17;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 9px 12px 9px 34px;
      border-radius: 6px;
      font-size: 13px;
    }
    .search-input:focus { outline: none; border-color: var(--primary); }
    .search-icon {
      position: absolute;
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      font-size: 14px;
    }
    .select-filter {
      background-color: #0b0f17;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 9px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    
    /* Table */
    .table-card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
    }
    .table-responsive { width: 100%; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th {
      background-color: #121820;
      color: var(--text-dim);
      font-weight: 600;
      padding: 12px 14px;
      border-bottom: 1px solid var(--card-border);
      white-space: nowrap;
    }
    td {
      padding: 12px 14px;
      border-bottom: 1px solid #1a222d;
      white-space: nowrap;
    }
    tr:hover td { background-color: rgba(59, 130, 246, 0.05); }
    
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.3px;
      transition: all 0.2s ease;
    }
    .badge:hover {
      filter: brightness(1.25);
      transform: translateY(-1px);
    }
    .badge-fraud { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-solar { background-color: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
    .badge-tamper { background-color: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-telecom { background-color: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
    .badge-meter { background-color: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); }
    .badge-regular { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    .score-bar-bg { width: 60px; height: 6px; background-color: #222d3d; border-radius: 3px; display: inline-block; margin-right: 8px; vertical-align: middle; overflow: hidden; }
    .score-bar-fill { height: 100%; border-radius: 3px; }
    
    .pagination {
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background-color: #121820;
      border-top: 1px solid var(--card-border);
      font-size: 12px;
      color: var(--text-dim);
    }
    
    /* Drawer */
    .drawer {
      position: fixed;
      right: -520px;
      top: 0;
      width: 500px;
      height: 100vh;
      background-color: var(--card-bg);
      border-left: 1px solid var(--card-border);
      box-shadow: -8px 0 24px rgba(0,0,0,0.5);
      transition: right 0.3s ease;
      z-index: 200;
      overflow-y: auto;
      padding: 24px;
    }
    .preset-btn {
      background: #1e2638;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.2s, border-color 0.2s;
    }
    .preset-btn:hover {
      background: #283548;
      border-color: var(--primary);
    }
    .preset-btn.active {
      background: var(--primary);
      border-color: var(--primary);
      color: #fff;
    }
    .custom-filter-box {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      padding: 10px 16px;
      background: #111722;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      margin-top: 10px;
    }
    .col-modal {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      width: 90%;
      max-width: 600px;
      max-height: 80vh;
      display: flex;
      flex-direction: column;
      z-index: 300;
      box-shadow: 0 20px 40px rgba(0,0,0,0.8);
    }
    .col-modal-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .col-modal-body {
      padding: 16px 20px;
      overflow-y: auto;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .col-modal-footer {
      padding: 12px 20px;
      border-top: 1px solid var(--card-border);
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }
    .col-check-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-dim);
      background: #0d121c;
      padding: 6px 10px;
      border-radius: 6px;
      cursor: pointer;
    }
    .col-check-item:hover {
      color: var(--text);
      background: #162030;
    }
    .col-check-item input {
      accent-color: var(--primary);
    }
    .drawer.open { right: 0; }
    .drawer-overlay {
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(0,0,0,0.6);
      z-index: 199;
      display: none;
    }
    .drawer-overlay.open { display: block; }
    .drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
    }
    .drawer-close { background: none; border: none; color: var(--text-dim); font-size: 22px; cursor: pointer; }
    .detail-section { margin-bottom: 20px; }
    .detail-title { font-size: 12px; color: var(--text-dim); font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
    .kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; }
    .kv-box { background: #0b0f17; padding: 10px; border-radius: 6px; border: 1px solid var(--card-border); }
    .kv-k { font-size: 11px; color: var(--text-dim); }
    .kv-v { font-weight: 700; margin-top: 2px; }

    /* Anomaly Breakdown Dashboard & Interactive UC Chart */
    .drawer { width: min(760px, 95vw) !important; }
    .dash-panel { background: var(--card-bg, #111827); border: 1px solid var(--card-border); border-radius: 10px; padding: 16px 18px; margin-bottom: 18px; }
    .dash-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
    .dash-title { font-size: 14px; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 8px; }
    .dash-sub { font-size: 12px; color: var(--text-dim); }
    .dash-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; }
    @media (max-width: 960px) { .dash-grid { grid-template-columns: 1fr; } }
    .dash-bars { display: flex; flex-direction: column; gap: 8px; }
    .dash-bar-row { cursor: pointer; padding: 7px 10px; border-radius: 7px; background: #0b0f17; border: 1px solid #1e293b; transition: border-color .15s, background .15s; }
    .dash-bar-row:hover { border-color: #38bdf8; background: #0f172a; }
    .dash-bar-row.active-filter { border-color: #38bdf8; box-shadow: 0 0 0 1px #38bdf8 inset; }
    .dash-bar-top { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; gap: 8px; }
    .dash-bar-name { font-weight: 600; color: #e2e8f0; }
    .dash-bar-count { font-weight: 700; color: #f8fafc; white-space: nowrap; }
    .dash-bar-track { height: 6px; background: #1e293b; border-radius: 999px; overflow: hidden; }
    .dash-bar-fill { height: 100%; border-radius: 999px; }
    .dash-bar-mech { font-size: 11px; color: var(--text-dim); margin-top: 3px; }
    .dash-side { display: flex; flex-direction: column; gap: 10px; }
    .signal-pills { display: grid; grid-template-columns: 1fr; gap: 8px; }
    .signal-pill { background: #0b0f17; border: 1px solid #1e293b; border-radius: 7px; padding: 9px 11px; display: flex; justify-content: space-between; align-items: center; }
    .feeder-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .feeder-chip { background: #0b0f17; border: 1px solid #334155; color: #cbd5e1; border-radius: 6px; padding: 4px 8px; font-size: 11px; cursor: pointer; }
    .feeder-chip:hover { border-color: #f59e0b; color: #fde68a; }
    .chart-card { background: #0b0f17; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 14px; }
    .chart-toolbar { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    .chart-btn { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 5px; padding: 4px 10px; font-size: 11px; cursor: pointer; font-weight: 600; }
    .chart-btn.active { background: #0284c7; border-color: #38bdf8; color: #ffffff; }
    .chart-legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: #94a3b8; margin-top: 6px; }
    .legend-dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
    .anom-event-list { margin-top: 10px; display: flex; flex-direction: column; gap: 5px; max-height: 150px; overflow-y: auto; }
    .anom-event-item { font-size: 11.5px; padding: 6px 8px; border-radius: 5px; background: #111827; border-left: 3px solid #ef4444; color: #e2e8f0; }
    .anom-event-item.drop { border-left-color: #f59e0b; }

    /* Fast Candidate Navigation & Missingness Telemetry Diagnosis */
    .nav-uc-bar { display: flex; justify-content: space-between; align-items: center; background: #070b12; border: 1px solid #1e293b; border-radius: 7px; padding: 8px 12px; margin-bottom: 12px; gap: 8px; flex-wrap: wrap; }
    .nav-uc-btn { background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 5px 12px; font-size: 12px; cursor: pointer; font-weight: 700; transition: all .15s; }
    .nav-uc-btn:hover:not(:disabled) { background: #0284c7; border-color: #38bdf8; }
    .nav-uc-select { background: #0b0f17; color: #f8fafc; border: 1px solid #334155; border-radius: 5px; padding: 5px 9px; font-size: 11.5px; max-width: 280px; }
    .missing-box { background: #070b12; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 10px; margin-top: 10px; }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">⚡</div>
      <div>
        <div class="brand-title">COCKPIT ANALÍTICO AMI — COPEL ARAUCÁRIA</div>
        <div class="brand-subtitle">Monitoramento Inteligente de Perdas Não-Técnicas, GD e Qualidade de Energia</div>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn btn-secondary" onclick="exportData('csv')">📥 Exportar CSV</button>
      <button class="btn" onclick="fetchData()">🔄 Atualizar Dados</button>
    </div>
  </header>

  <div class="container">
    <!-- Top KPI Cards -->
    <div class="metrics-grid" id="kpiGrid">
      <div class="metric-card">
        <span class="metric-label">Total UCs Analisadas</span>
        <span class="metric-value" id="kpiTotal">-</span>
        <span class="metric-sub">Araucária Completa</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Anomalias Confirmadas</span>
        <span class="metric-value" style="color: var(--danger)" id="kpiAnom">-</span>
        <span class="metric-sub" id="kpiAnomRate">-</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Possível GD Clandestina</span>
        <span class="metric-value" style="color: var(--solar)" id="kpiSolar">-</span>
        <span class="metric-sub">Fluxo Reverso s/ Cadastro</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Quarentena Telecom</span>
        <span class="metric-value" style="color: var(--text-dim)" id="kpiTelecom">-</span>
        <span class="metric-sub">Pacotes Mesh Degradados</span>
      </div>
    </div>

    <!-- Interactive Anomaly Breakdown Dashboard -->
    <div class="dash-panel" id="anomalyDashboard">
      <div class="dash-header">
        <div>
          <div class="dash-title">📊 Painel Analítico de Anomalias Detectadas (Clique em uma classe para filtrar e abrir o gráfico da UC)</div>
          <div class="dash-sub" id="dashMetaSub">Carregando distribuição por categoria diagnóstica, queda (Slope/CUSUM) e picos (MAD)...</div>
        </div>
        <button class="chart-btn" onclick="filterFromDashboardCategory('')">Mostrar Todas</button>
      </div>
      <div class="dash-grid">
        <div class="dash-bars" id="dashCategoryBars"></div>
        <div class="dash-side">
          <div class="signal-pills" id="dashSignalPills"></div>
          <div class="kv-box">
            <div class="kv-k" style="margin-bottom:6px">Top Alimentadores c/ Anomalias (Clique p/ filtrar)</div>
            <div class="feeder-list" id="dashTopFeeders"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Presets and Advanced Filter Bar -->
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; align-items: center;">
      <span style="font-size: 12px; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">Atalhos de Investigação:</span>
      <button class="preset-btn" id="btnPresetCandidates" onclick="applyPreset('candidatos_vistoria')" style="border-color: #ef4444; color: #fca5a5; font-weight:700;">🎯 Candidatos a Vistoria (Campo)</button>
      <button class="preset-btn" id="btnPresetPureFraud" onclick="applyPreset('pure_fraud')">🚨 Fraude / Desbalanço Elétrico</button>
      <button class="preset-btn" id="btnPresetReverseFlow" onclick="applyPreset('reverse_flow')">☀️ Injeção / GD Clandestina</button>
      <button class="preset-btn" id="btnPresetClassShift" onclick="applyPreset('class_shift')">🔄 Mudança de Classe Comportamental</button>
      <button class="preset-btn" id="btnPresetNewUc" onclick="applyPreset('new_uc')">🆕 UCs Novas (Idade ≤ 30d)</button>
      <button class="preset-btn" id="btnPresetReplaced" onclick="applyPreset('meter_replaced')">🔧 Troca Recente de Medidor</button>
      <button class="preset-btn" id="btnPresetMultiMeter" onclick="applyPreset('multi_meter')">📚 UCs c/ >1 Medidor Histórico</button>
      <button class="preset-btn" onclick="clearAllFilters()" style="color: var(--danger);">✖ Limpar Filtros</button>
      
      <div style="margin-left: auto;">
        <button class="btn btn-secondary" onclick="openColumnModal()" style="font-size: 12px; padding: 6px 12px;">⚙️ Configurar Colunas Exibidas</button>
      </div>
    </div>

    <!-- Filter Bar -->
    <div class="filter-panel" style="margin-bottom: 8px;">
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="text" id="searchInput" class="search-input" placeholder="Buscar por UC, NIO, Alimentador ou Bairro..." oninput="onSearch()">
      </div>
      <select id="categorySelect" class="select-filter" onchange="onFilterChange()">
        <option value="">Todas as Categorias Diagnósticas</option>
        <option value="CANDIDATOS_A_VISTORIA">🎯 Candidatos a Vistoria de Campo (Prioridade OS)</option>
        <option value="ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT">🚨 Anomalia Elétrica / Fraude PNT (PyOD ML)</option>
        <option value="POSSIVEL_GD_NAO_CADASTRADA">☀️ Possível GD Não Cadastrada / Injeção</option>
        <option value="FALHA_METROLOGICA_MEDIDOR_TRAVADO">⚠️ Falha Metrológica / Medidor Travado</option>
        <option value="MUDANCA_CLASSE_COMPORTAMENTAL">🔄 Mudança de Classe Comportamental</option>
        <option value="INICIACAO_PERFIL_UC_NOVA">🆕 Iniciação de Perfil (UC Nova <= 30d)</option>
        <option value="TRANSICAO_TROCA_MEDIDOR">🔧 Transição por Troca de Medidor</option>
        <option value="ANOMALIA_COMUNICACAO_ISOLADA">📡 Anomalia Comunicação Isolada</option>
        <option value="GARGALO_COLETIVO_RF_MESH">📶 Gargalo Coletivo Mesh</option>
        <option value="TELECOM_QUARANTINE">📶 Quarentena Telecom (Legado)</option>
        <option value="DESEQUILIBRIO_CARGA_OU_BYPASS">⚡ Desequilíbrio de Carga / Bypass</option>
        <option value="DEGRAU_DESCENDENTE_CONSUMO">📉 Degrau Descendente</option>
        <option value="MEDIDOR_TRAVADO_OU_CONSUMO_ZERO">⚠️ Medidor Travado / Consumo Zero</option>
        <option value="ANOMALIA_MULTIVARIADA">🚨 Anomalia Multivariada</option>
        <option value="OPERACAO_REGULAR">✔ Operação Regular</option>
        <option value="REGULAR">✔ Regular (Legado)</option>
      </select>
      <select id="feederSelect" class="select-filter" onchange="onFilterChange()">
        <option value="">Todos os Alimentadores</option>
      </select>
    </div>

    <!-- Custom Dynamic Field Filter Bar -->
    <div class="custom-filter-box">
      <span style="font-size: 12px; font-weight: 700; color: var(--text-dim); display: flex; align-items: center; gap: 4px;">
        🎯 Filtrar por qualquer campo:
      </span>
      <select id="customColSelect" class="select-filter" style="width: auto; padding: 6px 10px; font-size: 12px;" onchange="onCustomColChange()">
        <option value="">(Selecione um campo...)</option>
      </select>
      <select id="customOpSelect" class="select-filter" style="width: auto; padding: 6px 10px; font-size: 12px;">
        <option value="eq">=</option>
        <option value="ne">≠</option>
        <option value="gt">&gt;</option>
        <option value="gte">≥</option>
        <option value="lt">&lt;</option>
        <option value="lte">≤</option>
        <option value="contains">contém</option>
      </select>
      <input type="text" id="customValInput" class="search-input" style="width: 140px; padding: 6px 10px; font-size: 12px;" placeholder="Valor..." onkeydown="if(event.key==='Enter') applyCustomFilter()">
      <button class="btn btn-secondary" onclick="applyCustomFilter()" style="padding: 6px 12px; font-size: 12px;">Aplicar</button>
      <button class="btn btn-secondary" onclick="clearCustomFilter()" style="padding: 6px 10px; font-size: 12px; color: var(--text-dim);">Limpar</button>
      <span id="activeCustomFilterBadge" style="font-size: 11px; color: var(--primary); font-weight: 600;"></span>
    </div>

    <!-- Table Card -->
    <div class="table-card">
      <div class="table-responsive">
        <table>
          <thead id="tableHead">
            <tr>
              <th>UC</th>
              <th>Alimentador</th>
              <th>Score Anomalia</th>
              <th>Diagnóstico Causal</th>
              <th>Consumo Mediano (kWh)</th>
              <th>MAD Dispersão</th>
              <th>Desb. Corrente</th>
              <th>Reversão Ativa (RA)</th>
              <th>Ações</th>
            </tr>
          </thead>
          <tbody id="tableBody">
            <tr><td colspan="9" style="text-align:center; padding: 24px; color: var(--text-dim)">Carregando dados analíticos...</td></tr>
          </tbody>
        </table>
      </div>
      <div class="pagination">
        <span id="pageInfo">Exibindo 0 de 0 UCs</span>
        <div>
          <button class="btn btn-secondary" onclick="prevPage()" id="prevBtn">Anterior</button>
          <button class="btn btn-secondary" onclick="nextPage()" id="nextBtn">Próxima</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Dynamic Column Picker Modal -->
  <div class="drawer-overlay" id="colModalOverlay" onclick="closeColumnModal()"></div>
  <div class="col-modal" id="colModal" style="display: none;">
    <div class="col-modal-header">
      <h3 style="font-size: 15px;">⚙️ Configurar Colunas da Tabela</h3>
      <button class="drawer-close" onclick="closeColumnModal()">&times;</button>
    </div>
    <div style="padding: 10px 20px; border-bottom: 1px solid var(--card-border); display: flex; gap: 8px;">
      <input type="text" id="colSearchInput" class="search-input" placeholder="Filtrar colunas..." style="padding: 6px 12px; font-size: 12px;" oninput="filterColumnsInModal()">
      <button class="btn btn-secondary" onclick="resetDefaultColumns()" style="font-size: 12px; padding: 6px 10px;">Padrão</button>
      <button class="btn btn-secondary" onclick="selectAllColumns(true)" style="font-size: 12px; padding: 6px 10px;">Marcar Todas</button>
      <button class="btn btn-secondary" onclick="selectAllColumns(false)" style="font-size: 12px; padding: 6px 10px;">Desmarcar</button>
    </div>
    <div class="col-modal-body" id="colCheckboxContainer">
      <!-- Injected dynamically -->
    </div>
    <div class="col-modal-footer">
      <button class="btn btn-secondary" onclick="closeColumnModal()">Cancelar</button>
      <button class="btn" onclick="applyColumnSelection()">Aplicar e Salvar</button>
    </div>
  </div>

  <!-- Detail Drawer -->
  <div class="drawer-overlay" id="drawerOverlay" onclick="closeDrawer()"></div>
  <div class="drawer" id="detailDrawer">
    <div class="drawer-header">
      <div>
        <h3 id="drawerUc">UC 00000000</h3>
        <span id="drawerCat" class="badge">CATEGORIA</span>
      </div>
      <button class="drawer-close" onclick="closeDrawer()">&times;</button>
    </div>
    <div id="drawerContent"></div>
  </div>

  <script>
    let currentFeeders = [];
    let currentCategories = [];
    let allAvailableColumns = [];
    let selectedColumns = [
      "id__uc_id", "meta__feeder", "ANOMALY_SCORE", "CATEGORIA_DIAGNOSTICA", "DIAS_ANOMALIA_SEGUIDOS",
      "x__hist_fa_sum_median", "x__hist_fa_sum_mad", "x__hist_current_imbalance_max", "x__hist_ra_reversal_ratio_max"
    ];
    const DEFAULT_COLUMNS = [...selectedColumns];

    let currentPreset = '';
    let customFilter = { col: '', op: 'eq', val: '' };
    let currentPage = 1;
    const pageSize = 50;
    let totalRecords = 0;
    let debounceTimer = null;
    let activeFetchController = null;
    let activeFetchSeq = 0;
    const queryCache = new Map();
    const QUERY_CACHE_MAX = 50;

    function initCockpit() {
      renderTableHead();
      fetchData();
      fetchColumns();
    }

    async function fetchColumns() {
      try {
        const res = await fetch('/api/v1/columns');
        const json = await res.json();
        allAvailableColumns = json.columns || [];
        populateCustomColSelect(allAvailableColumns);
        buildColumnModalCheckboxes(allAvailableColumns);
      } catch (e) {
        console.error("Falha ao buscar colunas", e);
      }
    }

    function populateCustomColSelect(cols) {
      const sel = document.getElementById('customColSelect');
      sel.innerHTML = '<option value="">(Selecione um campo...)</option>';
      cols.forEach(c => {
        sel.innerHTML += `<option value="${c.name}">${c.name} (${c.type})</option>`;
      });
    }

    function onCustomColChange() {
      const colName = document.getElementById('customColSelect').value;
      const opSel = document.getElementById('customOpSelect');
      const found = allAvailableColumns.find(c => c.name === colName);
      if (found && found.type === 'number') {
        opSel.value = 'lte';
      } else if (found && found.type === 'boolean') {
        opSel.value = 'eq';
        document.getElementById('customValInput').placeholder = 'true / false';
      } else {
        opSel.value = 'contains';
        document.getElementById('customValInput').placeholder = 'Texto...';
      }
    }

    function applyCustomFilter() {
      const col = document.getElementById('customColSelect').value;
      const op = document.getElementById('customOpSelect').value;
      const val = document.getElementById('customValInput').value.trim();
      if (!col || val === '') {
        clearCustomFilter();
        return;
      }
      customFilter = { col, op, val };
      document.getElementById('activeCustomFilterBadge').innerText = `[Ativo: ${col} ${op} ${val}]`;
      currentPage = 1;
      fetchData();
    }

    function clearCustomFilter() {
      customFilter = { col: '', op: 'eq', val: '' };
      document.getElementById('customColSelect').value = '';
      document.getElementById('customValInput').value = '';
      document.getElementById('activeCustomFilterBadge').innerText = '';
      currentPage = 1;
      fetchData();
    }

    function applyPreset(preset) {
      currentPreset = preset;
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      
      // Auto-include relevant columns if not present
      const ensureCol = (c) => { if (!selectedColumns.includes(c)) selectedColumns.push(c); };

      if (preset === 'candidatos_vistoria') {
        document.getElementById('btnPresetCandidates')?.classList.add('active');
        ensureCol('DIAS_ANOMALIA_SEGUIDOS');
        ensureCol('IS_CANDIDATO_VISTORIA');
        ensureCol('CATEGORIA_DIAGNOSTICA');
        ensureCol('ANOMALY_SCORE');
        setCategorySelectValue('CANDIDATOS_A_VISTORIA');
        clearCustomFilter();
        return;
      } else if (preset === 'new_uc') {
        document.getElementById('btnPresetNewUc').classList.add('active');
        ensureCol('x__meter_age_days');
        ensureCol('meta__n_meters_historical');
        ensureCol('meta__telecom_status');
        // Preset: age <= 30
        customFilter = { col: 'x__meter_age_days', op: 'lte', val: '30' };
        document.getElementById('customColSelect').value = 'x__meter_age_days';
        document.getElementById('customOpSelect').value = 'lte';
        document.getElementById('customValInput').value = '30';
        document.getElementById('activeCustomFilterBadge').innerText = `[Ativo: x__meter_age_days <= 30]`;
      } else if (preset === 'meter_replaced') {
        document.getElementById('btnPresetReplaced').classList.add('active');
        ensureCol('x__meter_changed_30d');
        ensureCol('x__meter_age_days');
        ensureCol('meta__n_meters_historical');
        customFilter = { col: 'x__meter_changed_30d', op: 'eq', val: 'true' };
        document.getElementById('customColSelect').value = 'x__meter_changed_30d';
        document.getElementById('customOpSelect').value = 'eq';
        document.getElementById('customValInput').value = 'true';
        document.getElementById('activeCustomFilterBadge').innerText = `[Ativo: x__meter_changed_30d == true]`;
      } else if (preset === 'multi_meter') {
        document.getElementById('btnPresetMultiMeter').classList.add('active');
        ensureCol('meta__n_meters_historical');
        ensureCol('x__meter_age_days');
        customFilter = { col: 'meta__n_meters_historical', op: 'gt', val: '1' };
        document.getElementById('customColSelect').value = 'meta__n_meters_historical';
        document.getElementById('customOpSelect').value = 'gt';
        document.getElementById('customValInput').value = '1';
        document.getElementById('activeCustomFilterBadge').innerText = `[Ativo: meta__n_meters_historical > 1]`;
      } else if (preset === 'class_shift') {
        document.getElementById('btnPresetClassShift').classList.add('active');
        ensureCol('MUDOU_CLASSE_TEMPORAL');
        ensureCol('TRAJETORIA_CLASSES_TEMPORAL');
        customFilter = { col: 'MUDOU_CLASSE_TEMPORAL', op: 'eq', val: 'true' };
        document.getElementById('customColSelect').value = 'MUDOU_CLASSE_TEMPORAL';
        document.getElementById('customOpSelect').value = 'eq';
        document.getElementById('customValInput').value = 'true';
        document.getElementById('activeCustomFilterBadge').innerText = `[Ativo: MUDOU_CLASSE_TEMPORAL == true]`;
      } else if (preset === 'pure_fraud') {
        document.getElementById('btnPresetPureFraud').classList.add('active');
        ensureCol('x__hist_current_imbalance_max');
        ensureCol('x__hist_voltage_imbalance_max');
        customFilter = { col: 'x__hist_current_imbalance_max', op: 'gte', val: '0.70' };
        document.getElementById('customColSelect').value = 'x__hist_current_imbalance_max';
        document.getElementById('customOpSelect').value = 'gte';
        document.getElementById('customValInput').value = '0.70';
        document.getElementById('activeCustomFilterBadge').innerText = `[Ativo: x__hist_current_imbalance_max >= 0.70]`;
      } else if (preset === 'reverse_flow') {
        document.getElementById('btnPresetReverseFlow').classList.add('active');
        ensureCol('x__hist_ra_reversal_ratio_max');
        ensureCol('x__hist_ra_reversal_days');
        customFilter = { col: 'x__hist_ra_reversal_ratio_max', op: 'gt', val: '0.05' };
        document.getElementById('customColSelect').value = 'x__hist_ra_reversal_ratio_max';
        document.getElementById('customOpSelect').value = 'gt';
        document.getElementById('customValInput').value = '0.05';
        document.getElementById('activeCustomFilterBadge').innerText = `[Ativo: x__hist_ra_reversal_ratio_max > 0.05]`;
      }

      currentPage = 1;
      fetchData();
    }

    function clearAllFilters() {
      document.getElementById('searchInput').value = '';
      document.getElementById('categorySelect').value = '';
      document.getElementById('feederSelect').value = '';
      clearCustomFilter();
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      currentPreset = '';
      currentPage = 1;
      fetchData();
    }

    // Modal Column Selector Logic
    function openColumnModal() {
      document.getElementById('colModal').style.display = 'flex';
      document.getElementById('colModalOverlay').classList.add('open');
    }

    function closeColumnModal() {
      document.getElementById('colModal').style.display = 'none';
      document.getElementById('colModalOverlay').classList.remove('open');
    }

    function buildColumnModalCheckboxes(cols) {
      const container = document.getElementById('colCheckboxContainer');
      container.innerHTML = cols.map(c => {
        const isChecked = selectedColumns.includes(c.name) ? 'checked' : '';
        return `
          <label class="col-check-item" data-col="${c.name}">
            <input type="checkbox" value="${c.name}" ${isChecked} onchange="onColCheckboxToggle(this)">
            <span>${c.name}</span>
          </label>
        `;
      }).join('');
    }

    function onColCheckboxToggle(cb) {
      if (cb.checked) {
        if (!selectedColumns.includes(cb.value)) selectedColumns.push(cb.value);
      } else {
        selectedColumns = selectedColumns.filter(c => c !== cb.value);
      }
    }

    function filterColumnsInModal() {
      const term = document.getElementById('colSearchInput').value.toLowerCase();
      document.querySelectorAll('.col-check-item').forEach(el => {
        const name = el.getAttribute('data-col').toLowerCase();
        el.style.display = name.includes(term) ? 'flex' : 'none';
      });
    }

    function resetDefaultColumns() {
      selectedColumns = [...DEFAULT_COLUMNS];
      buildColumnModalCheckboxes(allAvailableColumns);
    }

    function selectAllColumns(check) {
      if (check) {
        selectedColumns = allAvailableColumns.map(c => c.name);
      } else {
        selectedColumns = ["id__uc_id"];
      }
      buildColumnModalCheckboxes(allAvailableColumns);
    }

    function applyColumnSelection() {
      closeColumnModal();
      renderTableHead();
      renderTable(pageRecords);
    }

    function renderTableHead() {
      const thead = document.getElementById('tableHead');
      const ths = selectedColumns.map(col => {
        const label = formatColumnHeader(col);
        return `<th>${label}</th>`;
      }).join('');
      thead.innerHTML = `<tr>${ths}<th>Ações</th></tr>`;
    }

    function formatColumnHeader(col) {
      const map = {
        "id__uc_id": "UC",
        "UC": "UC",
        "meta__feeder": "Alimentador",
        "ALIMENTADOR": "Alimentador",
        "meta__cutoff_date": "Data Cutoff",
        "ANOMALY_SCORE": "Score Anomalia",
        "FINAL_MOE_SCORE": "Score MoE",
        "CATEGORIA_DIAGNOSTICA": "Diagnóstico Causal",
        "IS_CANDIDATO_VISTORIA": "Candidato Vistoria?",
        "DIAS_ANOMALIA_SEGUIDOS": "Dias Seguidos Anomalia",
        "x__hist_fa_sum_median": "Consumo Mediano (kWh)",
        "x__hist_fa_sum_mad": "MAD Dispersão",
        "x__meter_age_days": "Idade Medidor (Dias)",
        "meta__n_meters_historical": "Qtd Medidores Hist.",
        "x__meter_changed_30d": "Troca 30d?",
        "x__hist_current_imbalance_max": "Desb. Corrente",
        "x__hist_voltage_imbalance_max": "Desb. Tensão %",
        "x__hist_ra_reversal_ratio_max": "Reversão Ativa (RA)",
        "meta__peer_median_coverage": "Cobertura Vizinhos",
        "meta__coverage_deficit_vs_peers": "Déficit Telecom vs Vizinhos",
        "CLASSE_COMPORTAMENTAL": "Classe Comportamental",
        "MUDOU_CLASSE_TEMPORAL": "Mudou Classe?",
        "TRAJETORIA_CLASSES_TEMPORAL": "Trajetória de Classes",
        "x__fase_circuito": "Fase",
        "x__tensao_base": "Tensão (V)"
      };
      return map[col] || col;
    }

    function _applyAnomaliesPayload(json) {
      totalRecords = json.total || 0;
      if (json.feeders && json.feeders.length > 0 && currentFeeders.length === 0) {
        currentFeeders = json.feeders;
        populateFeeders(currentFeeders);
      }
      if (json.categories && json.categories.length > 0 && currentCategories.length === 0) {
        currentCategories = json.categories;
        populateCategories(currentCategories);
      }
      updateKPIs(json.summary || {}, json.total_all || totalRecords);
      renderTable(json.data || []);
    }

    async function fetchData() {
      const q = encodeURIComponent(document.getElementById('searchInput').value || '');
      const cat = encodeURIComponent(document.getElementById('categorySelect').value || '');
      const feeder = encodeURIComponent(document.getElementById('feederSelect').value || '');
      const offset = (currentPage - 1) * pageSize;

      let customQuery = '';
      if (customFilter.col && customFilter.val !== '') {
        customQuery = `&filter_col=${encodeURIComponent(customFilter.col)}&filter_op=${encodeURIComponent(customFilter.op)}&filter_val=${encodeURIComponent(customFilter.val)}`;
      }

      const url = `/api/v1/anomalies?limit=${pageSize}&offset=${offset}&q=${q}&category=${cat}&feeder=${feeder}${customQuery}`;

      if (activeFetchController) {
        activeFetchController.abort();
        activeFetchController = null;
      }
      const seq = ++activeFetchSeq;

      renderTableHead();

      if (queryCache.has(url)) {
        const cached = queryCache.get(url);
        queryCache.delete(url);
        queryCache.set(url, cached);
        _applyAnomaliesPayload(cached);
        return;
      }

      const tbody = document.getElementById('tableBody');
      tbody.innerHTML = `<tr><td colspan="${selectedColumns.length + 1}" style="text-align:center; padding: 24px; color: var(--text-dim)">Carregando UCs...</td></tr>`;

      activeFetchController = new AbortController();
      const signal = activeFetchController.signal;

      try {
        const res = await fetch(url, { signal });
        const json = await res.json();
        if (seq !== activeFetchSeq) return;

        queryCache.set(url, json);
        if (queryCache.size > QUERY_CACHE_MAX) {
          const oldestKey = queryCache.keys().next().value;
          queryCache.delete(oldestKey);
        }
        _applyAnomaliesPayload(json);
      } catch (err) {
        if (err && err.name === 'AbortError') return;
        console.error(err);
        if (seq === activeFetchSeq) {
          tbody.innerHTML = `<tr><td colspan="${selectedColumns.length + 1}" style="text-align:center; padding: 24px; color: var(--danger)">Erro ao carregar dados do servidor.</td></tr>`;
        }
      }
    }

    function populateCategories(categories) {
      const sel = document.getElementById('categorySelect');
      const cur = sel.value;
      const allCats = categories.includes('CANDIDATOS_A_VISTORIA')
        ? categories
        : ['CANDIDATOS_A_VISTORIA', ...categories];
      sel.innerHTML = '<option value="">Todas as Categorias Diagnósticas</option>';
      allCats.forEach(c => {
        const icon = getCategoryIcon(c);
        sel.innerHTML += `<option value="${c}">${icon} ${formatCat(c)}</option>`;
      });
      if (cur) sel.value = cur;
    }

    function getCategoryIcon(cat) {
      if (cat.includes('GD')) return '☀️';
      if (cat.includes('FRAUDE') || cat.includes('BYPASS') || cat.includes('DESEQUILIBRIO')) return '🚨';
      if (cat.includes('DEGRAU') || cat.includes('DESCENDENTE')) return '📉';
      if (cat.includes('TRAVADO') || cat.includes('ZERO') || cat.includes('METROLOGICA')) return '⚠️';
      if (cat.includes('TELECOM') || cat.includes('QUARANTINE')) return '📶';
      if (cat.includes('COMUNICACAO')) return '📡';
      if (cat.includes('INICIACAO')) return '🆕';
      if (cat.includes('TRANSICAO')) return '🔧';
      if (cat.includes('MUDANCA')) return '🔄';
      if (cat.includes('REGULAR')) return '✔';
      return '🎯';
    }

    function populateFeeders(feeders) {
      const sel = document.getElementById('feederSelect');
      const cur = sel.value;
      sel.innerHTML = '<option value="">Todos os Alimentadores</option>';
      feeders.forEach(f => {
        sel.innerHTML += `<option value="${f}">${f}</option>`;
      });
      if (cur) sel.value = cur;
    }

    function updateKPIs(sum, totalAll) {
      document.getElementById('kpiTotal').innerText = (totalAll || sum.total_ucs || 0).toLocaleString();
      document.getElementById('kpiAnom').innerText = (sum.anomalies_detected || 0).toLocaleString();
      const rate = totalAll > 0 ? ((sum.anomalies_detected || 0) / totalAll * 100).toFixed(2) : 0;
      document.getElementById('kpiAnomRate').innerText = `${rate}% da população`;

      let gdCount = 0;
      let telecomCount = 0;
      if (sum.categories_distribution) {
        sum.categories_distribution.forEach(c => {
          if (c.CATEGORIA_DIAGNOSTICA === 'POSSIVEL_GD_NAO_CADASTRADA') gdCount += c.count;
          if (c.CATEGORIA_DIAGNOSTICA === 'TELECOM_QUARANTINE' || c.CATEGORIA_DIAGNOSTICA === 'ANOMALIA_COMUNICACAO_ISOLADA' || c.CATEGORIA_DIAGNOSTICA === 'GARGALO_COLETIVO_RF_MESH') telecomCount += c.count;
        });
      }
      document.getElementById('kpiSolar').innerText = gdCount.toLocaleString();
      document.getElementById('kpiTelecom').innerText = telecomCount.toLocaleString();
    }

    function setCategorySelectValue(cat) {
      const sel = document.getElementById('categorySelect');
      if (cat && !Array.from(sel.options).some(o => o.value === cat)) {
        const opt = document.createElement('option');
        opt.value = cat;
        opt.text = `${getCategoryIcon(cat)} ${formatCat(cat)}`;
        sel.appendChild(opt);
      }
      sel.value = cat || '';
    }

    function filterByCategory(cat) {
      setCategorySelectValue(cat);
      currentPage = 1;
      fetchData();
    }

    function filterByFeeder(f) {
      const sel = document.getElementById('feederSelect');
      sel.value = f;
      currentPage = 1;
      fetchData();
    }

    function onSearch() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        currentPage = 1;
        fetchData();
      }, 350);
    }

    function onFilterChange() {
      currentPage = 1;
      fetchData();
    }

    function prevPage() {
      if (currentPage > 1) {
        currentPage--;
        fetchData();
      }
    }

    function nextPage() {
      if ((currentPage * pageSize) < totalRecords) {
        currentPage++;
        fetchData();
      }
    }

    let pageRecords = [];
    function renderTable(data) {
      pageRecords = data;
      const tbody = document.getElementById('tableBody');
      const start = (currentPage - 1) * pageSize;

      if (data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="${selectedColumns.length + 1}" style="text-align:center; padding: 24px; color: var(--text-dim)">Nenhuma UC encontrada com os filtros selecionados.</td></tr>`;
        document.getElementById('pageInfo').innerText = '0 de 0 UCs';
        document.getElementById('prevBtn').disabled = true;
        document.getElementById('nextBtn').disabled = true;
        return;
      }

      tbody.innerHTML = data.map(d => {
        const uc = d.id__uc_id || d.UC || '-';
        const feeder = d.meta__feeder || d.ALIMENTADOR || '-';

        const rowCells = selectedColumns.map(col => {
          const val = d[col];
          if (col === 'id__uc_id' || col === 'UC') {
            return `<td><strong>${val || '-'}</strong></td>`;
          }
          if (col === 'meta__feeder' || col === 'ALIMENTADOR') {
            return `<td><a href="javascript:void(0)" onclick="filterByFeeder('${val}')" style="color: var(--primary); text-decoration: none;">${val || '-'}</a></td>`;
          }
          if (col === 'ANOMALY_SCORE' || col === 'FINAL_MOE_SCORE') {
            const score = Number(val || 0);
            const scorePct = Math.round(score * 100);
            const scoreColor = score >= 0.75 ? 'var(--danger)' : (score >= 0.5 ? 'var(--warning)' : 'var(--primary)');
            return `
              <td>
                <div class="score-bar-bg">
                  <div class="score-bar-fill" style="width: ${scorePct}%; background-color: ${scoreColor}"></div>
                </div>
                <span>${score.toFixed(4)}</span>
              </td>
            `;
          }
          if (col === 'CATEGORIA_DIAGNOSTICA') {
            const cat = val || 'OPERACAO_REGULAR';
            const badgeClass = getBadgeClass(cat);
            return `
              <td>
                <span class="badge ${badgeClass}" onclick="filterByCategory('${cat}')" style="cursor: pointer;" title="Filtrar por esta categoria">
                  ${formatCat(cat)} 🔍
                </span>
              </td>
            `;
          }
          if (col === 'DIAS_ANOMALIA_SEGUIDOS') {
            const streak = Number(val || 0);
            if (streak >= 5) {
              return `<td><span class="badge" style="background:#7f1d1d; color:#fecaca; font-weight:800;" title="${streak} dias consecutivos de anomalia ativa">🔥 ${streak} dias</span></td>`;
            } else if (streak > 0) {
              return `<td><span class="badge" style="background:#78350f; color:#fde68a; font-weight:700;" title="${streak} dias consecutivos de anomalia ativa">⚠️ ${streak} dias</span></td>`;
            }
            return `<td style="color:var(--text-dim); text-align:center;">-</td>`;
          }
          if (col === 'IS_CANDIDATO_VISTORIA') {
            return `<td><span style="color: ${val ? 'var(--danger)' : 'var(--text-dim)'}; font-weight: bold;">${val ? '🎯 SIM' : 'NÃO'}</span></td>`;
          }
          if (typeof val === 'number') {
            return `<td>${Number.isInteger(val) ? val : val.toFixed(2)}</td>`;
          }
          if (typeof val === 'boolean') {
            return `<td><span style="color: ${val ? 'var(--warning)' : 'var(--text-dim)'}; font-weight: bold;">${val ? 'SIM' : 'NÃO'}</span></td>`;
          }
          return `<td>${val !== null && val !== undefined ? val : '-'}</td>`;
        }).join('');

        return `
          <tr>
            ${rowCells}
            <td>
              <button class="btn btn-primary" style="padding: 4px 10px; font-size: 11px; font-weight:700;" onclick="openDetail('${uc}')">📈 Gráfico & Raio-X</button>
            </td>
          </tr>
        `;
      }).join('');

      document.getElementById('pageInfo').innerText = `Exibindo ${start + 1} a ${Math.min(start + data.length, totalRecords)} de ${totalRecords.toLocaleString()} UCs`;
      document.getElementById('prevBtn').disabled = currentPage === 1;
      document.getElementById('nextBtn').disabled = (start + pageSize) >= totalRecords;
    }

    function getBadgeClass(cat) {
      if (cat.includes('GD')) return 'badge-solar';
      if (cat.includes('FRAUDE') || cat.includes('BYPASS') || cat.includes('DESEQUILIBRIO')) return 'badge-fraud';
      if (cat.includes('TRAVADO') || cat.includes('ZERO') || cat.includes('METROLOGICA')) return 'badge-meter';
      if (cat.includes('TELECOM') || cat.includes('COMUNICACAO') || cat.includes('MESH')) return 'badge-telecom';
      if (cat.includes('INICIACAO')) return 'badge-regular';
      if (cat.includes('REGULAR')) return 'badge-regular';
      return 'badge-tamper';
    }

    function formatCat(cat) {
      return cat.replace(/_/g, ' ');
    }

    let activeDashCat = '';
    let activeDrawerUc = '';
    let activeDrawerCutoff = '';
    let activeChartGranularity = 'hourly';
    let activeChartDays = 7;
    let lgpdMaskMode = false;

    function maskUcText(ucStr) {
      if (!lgpdMaskMode || !ucStr) return ucStr;
      const s = String(ucStr).trim();
      if (s.length <= 5) return s.slice(0, 2) + '***';
      return s.slice(0, 3) + '*'.repeat(s.length - 5) + s.slice(-2);
    }

    function toggleLgpdMode() {
      lgpdMaskMode = !lgpdMaskMode;
      const btn = document.getElementById('btnLgpdToggle');
      if (btn) btn.innerText = lgpdMaskMode ? '🔒 LGPD: Censurado' : '🔓 LGPD: Visível';
      if (activeDrawerUc) {
        document.getElementById('drawerUc').innerText = `UC ${maskUcText(activeDrawerUc)}`;
      }
    }

    function navigateDetailUc(delta) {
      if (!pageRecords || pageRecords.length === 0) return;
      const idx = pageRecords.findIndex(x => (x.id__uc_id === activeDrawerUc || x.UC === activeDrawerUc));
      if (idx < 0) {
        openDetail(pageRecords[0].id__uc_id || pageRecords[0].UC);
        return;
      }
      const newIdx = idx + delta;
      if (newIdx >= 0 && newIdx < pageRecords.length) {
        const nextRec = pageRecords[newIdx];
        openDetail(nextRec.id__uc_id || nextRec.UC);
      }
    }

    window.addEventListener('keydown', (e) => {
      const drawer = document.getElementById('detailDrawer');
      if (drawer && drawer.classList.contains('open')) {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          navigateDetailUc(-1);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          navigateDetailUc(1);
        } else if (e.key === 'Escape') {
          closeDrawer();
        }
      }
    });

    async function loadDashboard() {
      try {
        const res = await fetch('/api/v1/dashboard');
        if (!res.ok) return;
        const dash = await res.json();
        document.getElementById('dashMetaSub').innerText =
          `${dash.total_unique_ucs.toLocaleString('pt-BR')} UCs únicas (${dash.total_rows.toLocaleString('pt-BR')} janelas avaliadas) · ` +
          `${dash.total_anomalies.toLocaleString('pt-BR')} anomalias (${dash.anomaly_rate_pct}%)`;

        const barsEl = document.getElementById('dashCategoryBars');
        const maxCnt = Math.max(1, ...(dash.categories || []).map(c => c.count));
        barsEl.innerHTML = (dash.categories || []).map(c => {
          const widthPct = Math.max(2, Math.round((c.count / maxCnt) * 100));
          const isActive = activeDashCat === c.category ? 'active-filter' : '';
          return `
            <div class="dash-bar-row ${isActive}" onclick="filterFromDashboardCategory('${c.category}')">
              <div class="dash-bar-top">
                <span class="dash-bar-name">${c.label} <small style="color:#64748b">(${c.category})</small></span>
                <span class="dash-bar-count">${c.count.toLocaleString('pt-BR')} UCs (${c.pct}%)</span>
              </div>
              <div class="dash-bar-track">
                <div class="dash-bar-fill" style="width:${widthPct}%; background:${c.color}"></div>
              </div>
              <div class="dash-bar-mech">${c.mechanism}</div>
            </div>
          `;
        }).join('');

        const sig = dash.signal_breakdown || {};
        document.getElementById('dashSignalPills').innerHTML = `
          <div class="signal-pill">
            <div>
              <div class="kv-k">Anomalias c/ Queda Sustentada (Slope &lt; -0.05)</div>
              <div style="font-size:11px;color:#94a3b8">Tendência descendente / degrau CUSUM</div>
            </div>
            <div style="font-size:16px;font-weight:800;color:#f59e0b">${(sig.drop_slope_neg_anomalies || 0).toLocaleString('pt-BR')}</div>
          </div>
          <div class="signal-pill">
            <div>
              <div class="kv-k">Anomalias c/ Alta Dispersão (MAD / CV &gt; 0.8)</div>
              <div style="font-size:11px;color:#94a3b8">Picos isolados e instabilidade operacional</div>
            </div>
            <div style="font-size:16px;font-weight:800;color:#ec4899">${(sig.high_mad_dispersion_anomalies || 0).toLocaleString('pt-BR')}</div>
          </div>
          <div class="signal-pill">
            <div>
              <div class="kv-k">Composição Queda + Picos Simultâneos</div>
              <div style="font-size:11px;color:#94a3b8">Slope negativo combinado com picos MAD</div>
            </div>
            <div style="font-size:16px;font-weight:800;color:#ef4444">${(sig.combined_drop_and_spike || 0).toLocaleString('pt-BR')}</div>
          </div>
        `;

        document.getElementById('dashTopFeeders').innerHTML = (dash.top_feeders || []).map(f => `
          <span class="feeder-chip" onclick="filterFromDashboardFeeder('${f.feeder}')">
            ${f.feeder}: <b>${f.anomalies}</b> anom (${f.rate_pct}%)
          </span>
        `).join('') || '<span class="kv-k">Sem dados de alimentador</span>';
      } catch (err) {
        console.error('Erro ao carregar dashboard:', err);
      }
    }

    function filterFromDashboardCategory(cat) {
      activeDashCat = cat;
      const sel = document.getElementById('categorySelect');
      if (sel) sel.value = cat;
      currentPage = 1;
      loadDashboard();
      fetchData();
    }

    function filterFromDashboardFeeder(feeder) {
      const sel = document.getElementById('feederSelect');
      if (sel) sel.value = feeder;
      currentPage = 1;
      fetchData();
    }

    async function loadUcChart(uc, cutoff, granularity, days) {
      activeChartGranularity = granularity || 'hourly';
      activeChartDays = days || (activeChartGranularity === 'hourly' ? 7 : 30);
      const box = document.getElementById('ucChartBody');
      if (!box) return;

      document.getElementById('btnPeriod7h')?.classList.toggle('active', activeChartGranularity === 'hourly' && activeChartDays === 7);
      document.getElementById('btnPeriod14h')?.classList.toggle('active', activeChartGranularity === 'hourly' && activeChartDays === 14);
      document.getElementById('btnPeriod30d')?.classList.toggle('active', activeChartGranularity === 'daily');

      box.innerHTML = '<div style="padding:28px;text-align:center;color:#94a3b8;font-size:12px;">Consultando telemetria real no Data Lake (ami_interval / uc_day_features)...</div>';

      try {
        const params = new URLSearchParams({
          granularity: activeChartGranularity,
          days: activeChartDays
        });
        if (cutoff) params.set('cutoff', cutoff);
        const res = await fetch(`/api/v1/meter/${encodeURIComponent(uc)}/timeseries?${params.toString()}`);
        const data = await res.json();
        renderUcChartSvg(data);
      } catch (err) {
        box.innerHTML = `<div style="padding:18px;color:#fca5a5;font-size:12px;">Erro ao carregar gráfico da UC: ${err.message}</div>`;
      }
    }

    function renderUcChartSvg(data) {
      const box = document.getElementById('ucChartBody');
      if (!box) return;
      if (!data || data.status !== 'ok' || !data.points || data.points.length === 0) {
        box.innerHTML = `
          <div style="padding:16px;border:1px dashed #334155;border-radius:6px;color:#94a3b8;font-size:12px;">
            ${data?.message || 'Partições brutas (uc_day_features / ami_interval) não disponíveis localmente para esta UC/cutoff.'}
          </div>`;
        return;
      }

      const pts = data.points;
      const W = 680;
      const H1 = 220; // Gráfico 1: Consumo & Bandas
      const H2 = 110; // Gráfico 2: Cobertura Telecom & Gaps
      const padL = 46, padR = 16, padT = 16, padB = 26;
      const plotW = W - padL - padR;
      const plotH1 = H1 - padT - padB;
      const plotH2 = H2 - padT - padB;

      const vals = pts.map(p => Number(p.value || 0));
      const uppers = pts.map(p => Number(p.upper_hod !== undefined ? p.upper_hod : data.upper_band || 0));
      const bases = pts.map(p => Number(p.baseline_hod !== undefined ? p.baseline_hod : data.baseline_median || 0));
      const maxY = Math.max(0.5, ...vals, ...uppers) * 1.08;
      const minY = 0;

      const xOf = (i) => padL + (pts.length > 1 ? (i / (pts.length - 1)) * plotW : plotW / 2);
      const yOfVal = (v) => padT + plotH1 - ((v - minY) / Math.max(maxY - minY, 1e-6)) * plotH1;
      const yOfCov = (c) => padT + plotH2 - (Math.max(0, Math.min(1.0, c)) * plotH2);

      // Gap Telecom vertical bands
      const colW = pts.length > 1 ? plotW / (pts.length - 1) : plotW;
      let gapRects1 = '';
      let missingCount = 0;
      pts.forEach((p, i) => {
        if (p.coverage < 0.70 || p.flag === 'GAP_TELECOM') {
          missingCount++;
          gapRects1 += `<rect x="${(xOf(i) - colW / 2).toFixed(1)}" y="${padT}" width="${Math.max(2, colW).toFixed(1)}" height="${plotH1}" fill="#ef4444" opacity="0.16" />`;
        }
      });

      // Step-down CUSUM shading
      let stepOverlay = '';
      if (data.step_change && data.step_change.index >= 0) {
        const sx = xOf(data.step_change.index);
        const wStep = Math.max(0, padL + plotW - sx);
        stepOverlay = `
          <rect x="${sx.toFixed(1)}" y="${padT}" width="${wStep.toFixed(1)}" height="${plotH1}" fill="#f59e0b" opacity="0.11" />
          <line x1="${sx.toFixed(1)}" y1="${padT}" x2="${sx.toFixed(1)}" y2="${padT + plotH1}" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="5,3" />
        `;
      }

      // MAD Band polygon
      const lowerPts = pts.map((p, i) => Number(p.lower_hod !== undefined ? p.lower_hod : data.lower_band || 0));
      const polyUpper = pts.map((_, i) => `${xOf(i).toFixed(1)},${yOfVal(uppers[i]).toFixed(1)}`).join(' ');
      const polyLower = pts.map((_, i) => `${xOf(pts.length - 1 - i).toFixed(1)},${yOfVal(lowerPts[pts.length - 1 - i]).toFixed(1)}`).join(' ');

      const baseLinePts = pts.map((_, i) => `${xOf(i).toFixed(1)},${yOfVal(bases[i]).toFixed(1)}`).join(' ');
      const valLinePts = pts.map((p, i) => `${xOf(i).toFixed(1)},${yOfVal(p.value).toFixed(1)}`).join(' ');

      // Anomaly point circles
      let anomMarkers = '';
      pts.forEach((p, i) => {
        const cx = xOf(i).toFixed(1);
        const cy = yOfVal(p.value).toFixed(1);
        const tip = `${p.label} | ${p.value} kWh | Cob: ${(p.coverage * 100).toFixed(0)}% | z=${p.z_self}`;
        if (p.flag === 'PICO_OUTLIER') {
          anomMarkers += `<circle cx="${cx}" cy="${cy}" r="5" fill="#ef4444" stroke="#ffffff" stroke-width="1.2"><title>PICO: ${tip}</title></circle>`;
        } else if (p.flag === 'QUEDA_OUTLIER' || p.flag === 'DEGRAU_QUEDA') {
          anomMarkers += `<circle cx="${cx}" cy="${cy}" r="4.2" fill="#f59e0b" stroke="#0b0f17" stroke-width="1"><title>QUEDA: ${tip}</title></circle>`;
        } else {
          anomMarkers += `<circle cx="${cx}" cy="${cy}" r="2.3" fill="#38bdf8" opacity="0.65"><title>${tip}</title></circle>`;
        }
      });

      // Y-axis ticks & X-axis labels for Chart 1
      let gridLines1 = '';
      for (let t = 0; t <= 4; t++) {
        const valT = (maxY * t) / 4;
        const yT = yOfVal(valT).toFixed(1);
        gridLines1 += `<line x1="${padL}" y1="${yT}" x2="${padL + plotW}" y2="${yT}" stroke="#1e293b" stroke-width="1" />`;
        gridLines1 += `<text x="${padL - 6}" y="${Number(yT) + 3}" text-anchor="end" fill="#94a3b8" font-size="10">${valT.toFixed(1)}</text>`;
      }
      const xStep = Math.max(1, Math.floor(pts.length / 7));
      let xTicks1 = '';
      for (let i = 0; i < pts.length; i += xStep) {
        xTicks1 += `<text x="${xOf(i).toFixed(1)}" y="${H1 - 7}" text-anchor="middle" fill="#94a3b8" font-size="9.5">${pts[i].label}</text>`;
      }

      // ----------------------------------------------------------------------
      // Gráfico 2: Diagnóstico de Dados Faltantes e Cobertura de Telecomunicação
      // ----------------------------------------------------------------------
      let covBars = '';
      pts.forEach((p, i) => {
        const covVal = Math.max(0, Math.min(1.0, Number(p.coverage || 0)));
        const barH = covVal * plotH2;
        const bx = (xOf(i) - colW * 0.42).toFixed(1);
        const by = (padT + plotH2 - barH).toFixed(1);
        const bw = Math.max(2, colW * 0.84).toFixed(1);
        const color = covVal < 0.70 ? '#ef4444' : (covVal < 0.90 ? '#f59e0b' : '#10b981');
        covBars += `<rect x="${bx}" y="${by}" width="${bw}" height="${Math.max(1, barH).toFixed(1)}" fill="${color}" opacity="0.85"><title>${p.label}: Cobertura ${(covVal*100).toFixed(1)}%</title></rect>`;
      });

      // Linha de Limiar 70% de Cobertura
      const y70 = yOfCov(0.70).toFixed(1);
      const gridLines2 = `
        <line x1="${padL}" y1="${padT}" x2="${padL + plotW}" y2="${padT}" stroke="#1e293b" stroke-width="1" />
        <line x1="${padL}" y1="${y70}" x2="${padL + plotW}" y2="${y70}" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="4,3" />
        <line x1="${padL}" y1="${padT + plotH2}" x2="${padL + plotW}" y2="${padT + plotH2}" stroke="#1e293b" stroke-width="1" />
        <text x="${padL - 6}" y="${padT + 3}" text-anchor="end" fill="#94a3b8" font-size="9.5">100%</text>
        <text x="${padL - 6}" y="${Number(y70) + 3}" text-anchor="end" fill="#ef4444" font-size="9.5" font-weight="bold">70%</text>
        <text x="${padL - 6}" y="${padT + plotH2 + 3}" text-anchor="end" fill="#94a3b8" font-size="9.5">0%</text>
      `;

      let xTicks2 = '';
      for (let i = 0; i < pts.length; i += xStep) {
        xTicks2 += `<text x="${xOf(i).toFixed(1)}" y="${H2 - 6}" text-anchor="middle" fill="#94a3b8" font-size="9.5">${pts[i].label}</text>`;
      }

      const unit = data.granularity === 'hourly' ? 'kWh/h' : 'kWh/dia';
      const avgCovPct = (pts.reduce((acc, p) => acc + (p.coverage || 0), 0) / Math.max(pts.length, 1) * 100).toFixed(1);
      const missingPct = ((missingCount / Math.max(pts.length, 1)) * 100).toFixed(1);

      const missingBadge = missingCount > 0
        ? `<span style="color:#ef4444;font-weight:700;">🚨 ${missingCount} janelas c/ falha de telemetria (&lt;70% cob. — ${missingPct}% do período)</span>`
        : `<span style="color:#10b981;font-weight:700;">✅ Telemetria 100% íntegra (sem gaps críticos)</span>`;

      const stepTxt = data.step_change
        ? `Degrau CUSUM: ${data.step_change.delta_pct > 0 ? '+' : ''}${data.step_change.delta_pct}% em ${data.step_change.date}`
        : 'Sem degrau CUSUM crítico';

      const eventsHtml = (data.anomalies || []).slice(0, 8).map(a => {
        const cls = (a.type || '').includes('QUEDA') || (a.type || '').includes('DEGRAU') ? 'drop' : '';
        return `<div class="anom-event-item ${cls}"><b>${a.label}</b> [${a.type}] — ${a.reason}</div>`;
      }).join('') || '<div class="kv-k">Nenhum ponto isolado acima de ±3.5 sigma MAD nesta janela.</div>';

      box.innerHTML = `
        <div style="font-size:11.5px;color:#cbd5e1;margin-bottom:6px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;">
          <span>Mediana Base: <b>${data.baseline_median} ${unit}</b> · MAD: <b>${data.baseline_mad}</b> · Slope: <b>${data.slope_kwh_day ?? '-'} kWh/dia</b></span>
          <span style="color:#fbbf24">${stepTxt}</span>
        </div>

        <!-- GRÁFICO 1: Curva de Consumo Real & Detecção -->
        <div style="font-size:11px;font-weight:700;color:#94a3b8;margin-bottom:3px;">1. Telemetria Ativa & Envelopes de Anomalia</div>
        <svg viewBox="0 0 ${W} ${H1}" style="width:100%;height:auto;display:block;background:#070b12;border-radius:6px 6px 0 0;border:1px solid #1e293b;border-bottom:none;">
          ${gridLines1}
          ${gapRects1}
          ${stepOverlay}
          <polygon points="${polyUpper} ${polyLower}" fill="#0284c7" opacity="0.14" />
          <polyline fill="none" stroke="#38bdf8" stroke-width="1.3" stroke-dasharray="4,3" points="${baseLinePts}" />
          <polyline fill="none" stroke="#f8fafc" stroke-width="2" points="${valLinePts}" />
          ${anomMarkers}
          ${xTicks1}
        </svg>

        <div class="chart-legend" style="border-left:1px solid #1e293b;border-right:1px solid #1e293b;padding:4px 8px;background:#0a0f18;">
          <span><i class="legend-dot" style="background:#f8fafc"></i>Consumo Real (${unit})</span>
          <span><i class="legend-dot" style="background:#38bdf8"></i>Mediana Baseline</span>
          <span><i class="legend-dot" style="background:#0284c7"></i>Banda MAD (±3.5σ)</span>
          <span><i class="legend-dot" style="background:#ef4444"></i>Pico Anômalo (z &gt; +3.5)</span>
          <span><i class="legend-dot" style="background:#f59e0b"></i>Queda / Degrau CUSUM</span>
        </div>

        <!-- GRÁFICO 2: Diagnóstico de Dados Faltantes & Cobertura -->
        <div style="font-size:11px;font-weight:700;color:#94a3b8;margin:8px 0 3px 0;display:flex;justify-content:space-between;flex-wrap:wrap;">
          <span>2. Diagnóstico de Dados Faltantes (Completude da Telemetria por Janela)</span>
          <span>${missingBadge} · Méd: <b>${avgCovPct}%</b></span>
        </div>
        <svg viewBox="0 0 ${W} ${H2}" style="width:100%;height:auto;display:block;background:#070b12;border-radius:0 0 6px 6px;border:1px solid #1e293b;">
          ${gridLines2}
          ${covBars}
          ${xTicks2}
        </svg>

        <div class="chart-legend" style="margin-top:4px;">
          <span><i class="legend-dot" style="background:#10b981"></i>Cobertura Ótima (≥90%)</span>
          <span><i class="legend-dot" style="background:#f59e0b"></i>Cobertura Parcial (70-90%)</span>
          <span><i class="legend-dot" style="background:#ef4444"></i>Dados Faltantes / Gap Telecom (&lt;70%)</span>
          <span style="color:#64748b;font-size:10.5px;">*Gaps não são computados como queda elétrica</span>
        </div>

        <div class="anom-event-list">${eventsHtml}</div>
      `;
    }

    function openDetail(uc) {
      const idx = pageRecords.findIndex(x => (x.id__uc_id === uc || x.UC === uc));
      const d = idx >= 0 ? pageRecords[idx] : null;
      if (!d) return;

      activeDrawerUc = uc;
      activeDrawerCutoff = d.meta__cutoff_date || d.CUTOFF_DATE || '';
      document.getElementById('drawerUc').innerText = `UC ${maskUcText(uc)}`;
      const catEl = document.getElementById('drawerCat');
      catEl.innerText = formatCat(d.CATEGORIA_DIAGNOSTICA || 'OPERACAO_REGULAR');
      catEl.className = `badge ${getBadgeClass(d.CATEGORIA_DIAGNOSTICA || '')}`;

      const totalCands = pageRecords.length;
      const candNum = idx + 1;
      const prevDisabled = idx <= 0 ? 'disabled style="opacity:0.4;cursor:not-allowed;"' : '';
      const nextDisabled = idx >= totalCands - 1 ? 'disabled style="opacity:0.4;cursor:not-allowed;"' : '';

      const content = `
        <!-- Barra de Navegação Rápida entre Candidatos -->
        <div class="nav-uc-bar">
          <button class="nav-uc-btn" onclick="navigateDetailUc(-1)" ${prevDisabled}>◀ Anterior (←)</button>
          <div style="font-size:12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span>Candidato <b>${candNum}</b> de <b>${totalCands}</b></span>
            <select class="nav-uc-select" onchange="openDetail(this.value)">
              ${pageRecords.map((r, i) => {
                const rUc = r.id__uc_id || r.UC;
                const isSel = rUc === uc ? 'selected' : '';
                const rCat = formatCat(r.CATEGORIA_DIAGNOSTICA || 'REGULAR');
                return `<option value="${rUc}" ${isSel}>#${i + 1}: UC ${maskUcText(rUc)} (${rCat})</option>`;
              }).join('')}
            </select>
          </div>
          <button class="nav-uc-btn" onclick="navigateDetailUc(1)" ${nextDisabled}>Próxima (→) ▶</button>
        </div>

        <div class="kv-grid" style="margin-bottom: 12px;">
          <div class="kv-box" style="border: 1px solid #ef4444; background: #1c1117;">
            <div class="kv-k" style="color:#ef4444; font-weight:700;">🔥 Dias Anomalia Seguidos</div>
            <div class="kv-v" style="color:#fca5a5; font-weight:800; font-size:16px;">${d.DIAS_ANOMALIA_SEGUIDOS || 0} dias</div>
          </div>
          <div class="kv-box">
            <div class="kv-k">Score Consenso</div>
            <div class="kv-v" style="color:var(--primary); font-weight:700;">${Number(d.ANOMALY_SCORE || 0).toFixed(4)}</div>
          </div>
          <div class="kv-box">
            <div class="kv-k">Candidato a Vistoria</div>
            <div class="kv-v" style="color:${d.IS_CANDIDATO_VISTORIA ? '#ef4444' : '#94a3b8'}; font-weight:700;">${d.IS_CANDIDATO_VISTORIA ? '🎯 SIM (Prioritário)' : 'NÃO'}</div>
          </div>
          <div class="kv-box">
            <div class="kv-k">Subtipo Diagnóstico</div>
            <div class="kv-v" style="font-size:11px;">${d.SUBTIPO_DIAGNOSTICO || '-'}</div>
          </div>
        </div>

        <div class="chart-card">
          <div class="chart-toolbar">
            <div class="detail-title" style="margin:0">📈 Acompanhamento Temporal & Diagnóstico</div>
            <div style="display:flex;gap:5px;flex-wrap:wrap;align-items:center;">
              <span style="font-size:11px;color:#94a3b8">Período:</span>
              <button class="chart-btn active" id="btnPeriod7h" onclick="loadUcChart('${uc}', '${activeDrawerCutoff}', 'hourly', 7)">7d Horário</button>
              <button class="chart-btn" id="btnPeriod14h" onclick="loadUcChart('${uc}', '${activeDrawerCutoff}', 'hourly', 14)">14d Horário</button>
              <button class="chart-btn" id="btnPeriod30d" onclick="loadUcChart('${uc}', '${activeDrawerCutoff}', 'daily', 30)">30d Diário</button>
              <button class="chart-btn" id="btnLgpdToggle" onclick="toggleLgpdMode()">${lgpdMaskMode ? '🔒 LGPD: Censurado' : '🔓 LGPD: Visível'}</button>
            </div>
          </div>
          <div id="ucChartBody"></div>
        </div>

        <div class="detail-section">
          <div class="detail-title">Contexto Elétrico e Cadastral</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-k">Alimentador</div><div class="kv-v">${d.meta__feeder || d.ALIMENTADOR || '-'}</div></div>
            <div class="kv-box"><div class="kv-k">Posto Operacional (Trafo)</div><div class="kv-v">${d.meta__posto || d.POSTO_OPERACIONAL || '-'}</div></div>
            <div class="kv-box"><div class="kv-k">Tensão Base</div><div class="kv-v">${d.x__tensao_base || 127} V</div></div>
            <div class="kv-box"><div class="kv-k">Faseamento</div><div class="kv-v">${d.x__fase_circuito || 'MONO'}</div></div>
          </div>
        </div>

        <div class="detail-section">
          <div class="detail-title">Métricas Comportamentais (30 Dias)</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-k">Consumo Diário Mediano</div><div class="kv-v">${Number(d.x__hist_fa_sum_median || 0).toFixed(2)} kWh</div></div>
            <div class="kv-box"><div class="kv-k">Dispersão Robusta (MAD)</div><div class="kv-v">${Number(d.x__hist_fa_sum_mad || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Inclinação Tendência (Slope)</div><div class="kv-v">${Number(d.x__trend_fa_slope || 0).toFixed(4)}</div></div>
            <div class="kv-box"><div class="kv-k">Fator de Carga Médio</div><div class="kv-v">${Number(d.x__hist_load_factor_mean || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Desbalanço de Corrente (P90)</div><div class="kv-v">${Number(d.x__hist_current_imbalance_max || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Desbalanço Tensão (P95)</div><div class="kv-v">${Number(d.x__hist_voltage_imbalance_max || 0).toFixed(2)}%</div></div>
            <div class="kv-box"><div class="kv-k">Reversão Ativa (RA Max)</div><div class="kv-v">${Number(d.x__hist_ra_reversal_ratio_max || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Dias de Reversão</div><div class="kv-v">${d.x__hist_ra_reversal_days || 0} dias</div></div>
          </div>
        </div>

        <div class="detail-section">
          <div class="detail-title">Integridade, Vizinhança Espacial & Ciclo de Vida</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-k">Cobertura da UC</div><div class="kv-v">${(Number(d.x__avg_interval_coverage || 0) * 100).toFixed(1)}%</div></div>
            <div class="kv-box"><div class="kv-k">Cobertura Mediana dos Vizinhos (Trafo/Geo)</div><div class="kv-v">${(Number(d.meta__peer_median_coverage || 0.95) * 100).toFixed(1)}%</div></div>
            <div class="kv-box"><div class="kv-k">Déficit vs. Vizinhos</div><div class="kv-v">${(Number(d.meta__coverage_deficit_vs_peers || 0) * 100).toFixed(1)}%</div></div>
            <div class="kv-box"><div class="kv-k">Dias Confiáveis (>70%)</div><div class="kv-v">${d.x__coverage_reliable_days || 0} dias</div></div>
            <div class="kv-box"><div class="kv-k">Medidores na História da UC</div><div class="kv-v">${d.meta__n_meters_historical || 1} medidor(es)</div></div>
            <div class="kv-box"><div class="kv-k">Idade do Medidor Atual</div><div class="kv-v">${d.x__meter_age_days || 0} dias</div></div>
            <div class="kv-box"><div class="kv-k">Troca de Medidor Recente (30d)</div><div class="kv-v">${d.x__meter_changed_30d ? 'SIM' : 'NÃO'}</div></div>
            <div class="kv-box"><div class="kv-k">Classe Comportamental</div><div class="kv-v">${d.CLASSE_COMPORTAMENTAL || '-'}</div></div>
            <div class="kv-box"><div class="kv-k">Método de Detecção</div><div class="kv-v">${d.METODO_DETECCAO_APLICADO || '-'}</div></div>
          </div>
        </div>
        <div class="detail-section">
          <div class="detail-title">Trajetória Longitudinal de Classes entre Cutoffs</div>
          <div class="kv-box"><div class="kv-k">Evolução Temporal</div><div class="kv-v" style="white-space:normal">${d.TRAJETORIA_CLASSES_TEMPORAL || d.CLASSE_COMPORTAMENTAL || '-'}</div></div>
        </div>
      `;
      document.getElementById('drawerContent').innerHTML = content;
      document.getElementById('detailDrawer').classList.add('open');
      document.getElementById('drawerOverlay').classList.add('open');
      loadUcChart(uc, activeDrawerCutoff, 'hourly', 7);
    }

    function closeDrawer() {
      document.getElementById('detailDrawer').classList.remove('open');
      document.getElementById('drawerOverlay').classList.remove('open');
    }

    function exportData(fmt) {
      window.location.href = '/api/v1/export?format=' + fmt;
    }

    window.onload = () => {
      initCockpit();
      loadDashboard();
    };
  </script>
</body>
</html>
"""


class CockpitHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for Cockpit Web and REST endpoints."""

    def log_message(self, format, *args):
        try:
            super().log_message(format, *args)
        except Exception:
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Static HTML
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        # 2. API: Columns introspection for dynamic picker
        if path == "/api/v1/columns":
            df = _load_data()
            cols_info = []
            if not df.is_empty():
                for col_name, dtype in zip(df.columns, df.dtypes):
                    dtype_str = str(dtype).lower()
                    if "int" in dtype_str or "float" in dtype_str or "decimal" in dtype_str:
                        kind = "number"
                    elif "bool" in dtype_str:
                        kind = "boolean"
                    elif "date" in dtype_str or "time" in dtype_str:
                        kind = "date"
                    else:
                        kind = "string"
                    cols_info.append({"name": col_name, "type": kind, "dtype": str(dtype)})
            self._send_json({"columns": cols_info})
            return

        # 3. API: Anomalies & Classified List (Paginated on Server + Query Filter + LRU Cache)
        if path == "/api/v1/anomalies":
            global _CACHED_FEEDERS, _CACHED_CATEGORIES
            cat_filter = query.get("category", [""])[0]
            feeder_filter = query.get("feeder", [""])[0]
            q_filter = query.get("q", [""])[0].strip().lower()
            limit = int(query.get("limit", [50])[0])
            offset = int(query.get("offset", [0])[0])
            filter_col = query.get("filter_col", [""])[0].strip()
            filter_op = query.get("filter_op", ["eq"])[0].strip()
            filter_val = query.get("filter_val", [""])[0].strip()

            cache_key = (cat_filter, feeder_filter, q_filter, filter_col, filter_op, filter_val, offset, limit)
            with _DATA_LOCK:
                cached_bytes = _ANOMALIES_CACHE.get(cache_key)
                if cached_bytes is not None:
                    _ANOMALIES_CACHE.move_to_end(cache_key)
                    self._send_json_bytes(cached_bytes)
                    return

            df = _load_data()
            summary = _load_summary()
            if df.is_empty():
                payload = {"data": [], "total": 0, "summary": summary, "feeders": [], "categories": []}
                raw_bytes = self._encode_json(payload)
            else:
                with _DATA_LOCK:
                    if _CACHED_FEEDERS is None:
                        _CACHED_FEEDERS = sorted([
                            f for f in (
                                df.get_column("meta__feeder").drop_nulls().unique().to_list()
                                if "meta__feeder" in df.columns
                                else (df.get_column("ALIMENTADOR").drop_nulls().unique().to_list() if "ALIMENTADOR" in df.columns else [])
                            )
                            if f and str(f).strip()
                        ])

                    if _CACHED_CATEGORIES is None:
                        _CACHED_CATEGORIES = sorted([
                            c for c in (
                                df.get_column("CATEGORIA_DIAGNOSTICA").drop_nulls().unique().to_list()
                                if "CATEGORIA_DIAGNOSTICA" in df.columns
                                else []
                            )
                            if c and str(c).strip()
                        ])

                filtered_df = df
                sort_candidates = False
                if cat_filter:
                    if cat_filter == "CANDIDATOS_A_VISTORIA":
                        sort_candidates = True
                        if "IS_CANDIDATO_VISTORIA" in filtered_df.columns:
                            filtered_df = filtered_df.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)
                        elif "IS_ANOMALY" in filtered_df.columns:
                            filtered_df = filtered_df.filter(pl.col("IS_ANOMALY") == True)
                            if "CATEGORIA_DIAGNOSTICA" in filtered_df.columns:
                                filtered_df = filtered_df.filter(
                                    ~pl.col("CATEGORIA_DIAGNOSTICA").is_in([
                                        "NORMAL",
                                        "QUEDA_TELECOM_ISOLADA",
                                        "QUARENTENA_TELECOM",
                                        "MEDIDOR_NOVO_QUARENTENA",
                                    ])
                                )
                    elif "CATEGORIA_DIAGNOSTICA" in filtered_df.columns:
                        filtered_df = filtered_df.filter(
                            (pl.col("CATEGORIA_DIAGNOSTICA") == cat_filter) |
                            (pl.col("CATEGORIA_DIAGNOSTICA").cast(pl.String).str.contains(cat_filter))
                        )
                if feeder_filter:
                    if "meta__feeder" in filtered_df.columns:
                        filtered_df = filtered_df.filter(pl.col("meta__feeder") == feeder_filter)
                    elif "ALIMENTADOR" in filtered_df.columns:
                        filtered_df = filtered_df.filter(pl.col("ALIMENTADOR") == feeder_filter)
                if q_filter:
                    uc_col = "id__uc_id" if "id__uc_id" in filtered_df.columns else "UC"
                    filtered_df = filtered_df.filter(pl.col(uc_col).cast(pl.String).str.to_lowercase().str.contains(q_filter))

                # Arbitrary dynamic filter on any field
                if filter_col and filter_col in filtered_df.columns and filter_val != "":
                    try:
                        col_type = str(filtered_df.schema[filter_col]).lower()
                        if "int" in col_type or "float" in col_type:
                            f_val = float(filter_val)
                            if filter_op == "eq":
                                filtered_df = filtered_df.filter(pl.col(filter_col) == f_val)
                            elif filter_op == "gt":
                                filtered_df = filtered_df.filter(pl.col(filter_col) > f_val)
                            elif filter_op == "gte":
                                filtered_df = filtered_df.filter(pl.col(filter_col) >= f_val)
                            elif filter_op == "lt":
                                filtered_df = filtered_df.filter(pl.col(filter_col) < f_val)
                            elif filter_op == "lte":
                                filtered_df = filtered_df.filter(pl.col(filter_col) <= f_val)
                            elif filter_op == "ne":
                                filtered_df = filtered_df.filter(pl.col(filter_col) != f_val)
                        elif "bool" in col_type:
                            b_val = filter_val.lower() in ("true", "1", "t", "sim", "yes")
                            filtered_df = filtered_df.filter(pl.col(filter_col) == b_val)
                        else:
                            if filter_op == "eq":
                                filtered_df = filtered_df.filter(pl.col(filter_col).cast(pl.String) == filter_val)
                            elif filter_op == "contains":
                                filtered_df = filtered_df.filter(pl.col(filter_col).cast(pl.String).str.to_lowercase().str.contains(filter_val.lower()))
                            elif filter_op == "ne":
                                filtered_df = filtered_df.filter(pl.col(filter_col).cast(pl.String) != filter_val)
                    except Exception as ef:
                        print(f"Erro ao aplicar filtro dinâmico {filter_col} {filter_op} {filter_val}: {ef}")

                if sort_candidates:
                    if "DIAS_ANOMALIA_SEGUIDOS" in filtered_df.columns:
                        filtered_df = filtered_df.sort(["DIAS_ANOMALIA_SEGUIDOS", "ANOMALY_SCORE"], descending=[True, True])
                    elif "ANOMALY_SCORE" in filtered_df.columns:
                        filtered_df = filtered_df.sort("ANOMALY_SCORE", descending=True)

                total_matched = filtered_df.height

                # Fast slice
                paged_slice = filtered_df.slice(offset, limit)
                records = paged_slice.to_dicts()
                payload = {
                    "data": records,
                    "total": total_matched,
                    "total_all": df.height,
                    "offset": offset,
                    "limit": limit,
                    "feeders": _CACHED_FEEDERS,
                    "categories": _CACHED_CATEGORIES,
                    "summary": summary,
                }
                raw_bytes = self._encode_json(payload)
                with _DATA_LOCK:
                    _ANOMALIES_CACHE[cache_key] = raw_bytes
                    if len(_ANOMALIES_CACHE) > _ANOMALIES_CACHE_MAX:
                        _ANOMALIES_CACHE.popitem(last=False)

            self._send_json_bytes(raw_bytes)
            return

        # 3. API: Export Data
        if path == "/api/v1/export":
            if CLASSIFIED_CSV.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=classified_ucs_latest.csv")
                self.end_headers()
                with open(CLASSIFIED_CSV, "rb") as f:
                    self.wfile.write(f.read())
            elif CLASSIFIED_PARQUET.exists():
                df = pl.read_parquet(CLASSIFIED_PARQUET)
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=classified_ucs_latest.csv")
                self.end_headers()
                self.wfile.write(df.write_csv(separator=";").encode("utf-8"))
            else:
                self.send_error(404, "Arquivo de exportação não encontrado.")
            return

        # 4. API: Dashboard Breakdown
        if path == "/api/v1/dashboard":
            df = _load_data()
            if df.is_empty():
                self._send_json({
                    "total_rows": 0,
                    "total_unique_ucs": 0,
                    "total_anomalies": 0,
                    "anomaly_rate_pct": 0.0,
                    "signal_breakdown": {},
                    "categories": [],
                    "subtypes": [],
                    "top_feeders": [],
                })
                return
            self._send_json(_build_dashboard_stats(df))
            return

        # 5. API: Real Per-UC Telemetry & Anomaly Series (/api/v1/meter/<uc>/timeseries)
        if path.startswith("/api/v1/meter/") and path.endswith("/timeseries"):
            raw_uc = path[len("/api/v1/meter/"):-len("/timeseries")].strip("/")
            uc_decoded = urllib.parse.unquote(raw_uc)
            qs = urllib.parse.parse_qs(parsed.query)
            cutoff = (qs.get("cutoff") or [""])[0].strip() or None
            granularity = (qs.get("granularity") or ["daily"])[0].strip().lower()
            days_str = (qs.get("days") or [""])[0].strip()
            days = int(days_str) if days_str.isdigit() else (7 if granularity == "hourly" else 30)
            try:
                if granularity == "hourly":
                    payload = load_uc_hourly_timeseries(
                        uc_decoded, base_dir=BASE_DIR, cutoff_date=cutoff, week_days=days, baseline_days=max(28, days + 7)
                    )
                else:
                    payload = load_uc_daily_timeseries(
                        uc_decoded, base_dir=BASE_DIR, cutoff_date=cutoff, days=max(days, 30)
                    )
                self._send_json(payload)
            except Exception as exc:
                self._send_json({
                    "status": "error",
                    "uc": uc_decoded,
                    "message": f"Falha ao ler série temporal no Data Lake: {exc}",
                    "points": [],
                    "anomalies": [],
                })
            return

        # 6. API: Health Check
        if path == "/api/v1/health":
            self._send_json({"status": "OK", "has_data": CLASSIFIED_PARQUET.exists() or CLASSIFIED_CSV.exists()})
            return

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        self.send_error(404, "Not Found")

    def _encode_json(self, data: Any) -> bytes:
        def _json_default(obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        return json.dumps(data, ensure_ascii=False, default=_json_default).encode("utf-8")

    def _send_json_bytes(self, raw_bytes: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw_bytes)))
        self.end_headers()
        self.wfile.write(raw_bytes)

    def _send_json(self, data: Any):
        self._send_json_bytes(self._encode_json(data))


def _prewarm_caches() -> None:
    """Warm dataset, summary and dashboard stats in background right on startup."""
    try:
        df = _load_data()
        _load_summary()
        if not df.is_empty():
            _build_dashboard_stats(df)
    except Exception as exc:
        print(f"[Cockpit] Aviso no pre-warm: {exc}")


def run_web_cockpit(host: str = "127.0.0.1", port: int = 8050, open_browser: bool = True):
    """Starts the zero-dependency threaded web cockpit server."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, CockpitHandler)
    url = f"http://localhost:{port}"

    threading.Thread(target=_prewarm_caches, daemon=True).start()

    print(f"\n⚡ Cockpit Analítico AMI rodando em: {url}")
    print("Pressione Ctrl+C para encerrar o servidor.\n")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando o servidor web...")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run Cockpit Analítico AMI Web Server.")
    parser.add_argument("--port", type=int, default=8050, help="HTTP server port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically.")
    args = parser.parse_args()

    run_web_cockpit(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

